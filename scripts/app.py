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

# judge_clause.py (버전 없음) 임포트
try:
    from scripts.judge_clause import run as judge_run
except ImportError:
    st.error("scripts/judge_clause.py 파일을 찾을 수 없습니다.")
    st.stop()


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
    explanation = result.get("explanation")
    suggestion = result.get("suggestion")
    
    # --- (버그 수정) ---
    # 'top_reasons'가 아니라 'best_match_case'를 읽어옵니다.
    best_case = result.get("best_match_case")
    # --- (수정 끝) ---

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

    # --- (버그 수정) ---
    # 'reasons'가 아니라 'best_case'가 비어있는지(None) 확인합니다.
    if not best_case:
        st.info("근거 후보가 없습니다. (유사 사례를 찾지 못했거나 점수 로직에 의해 무시됨)")
    else:
        # 루프를 제거하고 'best_case' 1건만 표시합니다.
        # (v6 이후 버전 호환을 위해 'similarity_violation' 또는 'similarity' 확인)
        if 'similarity_violation' in best_case:
            similarity = best_case.get("similarity_violation", 0.0)
        else:
            similarity = best_case.get("similarity", 0.0)
            
        violation_text = best_case.get("violation_text", "N/A")
        reason = best_case.get("reason", "N/A")
        correction_text = best_case.get("correction_text") # v6인 경우

        header = f"가장 유사한 사례 (유사도: {similarity:.2f})"
        with st.expander(header, expanded=True):
            st.markdown("##### 1. 불공정 원문 (DB)")
            st.info(violation_text)
            st.markdown("##### 2. 시정 요청 사유 (DB)")
            st.warning(reason)
            # v6 로직의 수정본 텍스트가 있다면 함께 표시
            if correction_text:
                st.markdown("##### 3. 공정 수정본 (DB)")
                st.success(correction_text)
    # --- (수정 끝) ---

    st.markdown("---")
    st.subheader("수정 제안")
    st.write(suggestion)

    st.markdown("---")
    st.subheader("Raw JSON")
    st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")
else:
    st.info("좌측 입력창에 문장을 입력하고 '검사하기'를 눌러 시작하세요.")
