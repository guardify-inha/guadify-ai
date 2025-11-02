import sys
from pathlib import Path
import json
import streamlit as st

# Ensure project root in path and load .env
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

from scripts.judge_clause import run as judge_run

st.set_page_config(page_title="약관법 위반 판단", page_icon="⚖️", layout="wide")

st.title("약관법 위반 판단 (GraphRAG)")
st.caption("입력한 조항 문구가 약관법을 위반하는지 그래프 기반으로 판단합니다.")

with st.sidebar:
    st.header("입력")
    default_text = "회사는 어떠한 피해배상도 하지않는다"
    user_text = st.text_area("검사할 문장", value=default_text, height=120)
    run_btn = st.button("검사하기", type="primary")
    st.markdown("---")
    st.markdown("환경: Neo4j가 Docker에서 실행 중이어야 합니다.")

col1, col2 = st.columns([1,1])

if run_btn:
    with st.spinner("Neo4j 질의 및 판단 중..."):
        try:
            result = judge_run(user_text)
        except Exception as e:
            st.error(f"실행 오류: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()

    violation = result.get("violation")
    score = result.get("score")
    reasons = result.get("top_reasons", [])
    explanation = result.get("explanation")
    suggestion = result.get("suggestion")

    # Summary card
    status = "위반" if violation else "비위반/불명확"
    status_color = "red" if violation else "green"

    with col1:
        st.subheader("판단 결과")
        st.metric(label="위반 여부", value=status)
        st.write(f"신뢰 점수: **{score:.2f}**")
        st.write(f"조항: **{result.get('article_id', 'N/A')}**")
        st.write(f"심각도: **{result.get('severity', 'N/A')}**")

    with col2:
        st.subheader("설명")
        st.write(explanation)

    st.markdown("---")
    st.subheader("근거 조문/사례")

    if not reasons:
        st.info("근거 후보가 없습니다. 입력 문장을 다시 시도하세요.")
    else:
        for i, r in enumerate(reasons[:5], start=1):
            level = r.get("level", "")
            rid = r.get("id", "")
            article_id = r.get("article_id", "")
            snippet = (r.get("snippet") or "").strip()
            sc = r.get("score", 0)

            header = f"{i}. [{level}] {article_id} - {rid} (유사도: {sc:.2f})".strip()
            with st.expander(header, expanded=(i == 1)):
                st.write(snippet)

    st.markdown("---")
    st.subheader("수정 제안")
    st.write(suggestion)

    st.markdown("---")
    st.subheader("Raw JSON")
    st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")
else:
    st.info("좌측 입력창에 문장을 입력하고 '검사하기'를 눌러 시작하세요.")