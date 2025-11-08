"""
GraphRAG 기반 불공정 약관 판단 시스템 - Streamlit UI
"""
import streamlit as st
import sys
from pathlib import Path
import os
import json

# 프로젝트 루트 추가
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 환경변수 로드
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

# GraphRAG 임포트
from database.neo4j_connector import Neo4jConnector
from rag.hybrid_graphrag import HybridGraphRAG
from judge.graphrag_judge import GraphRAGJudge

# 페이지 설정
st.set_page_config(
    page_title="GraphRAG 약관 판단",
    page_icon="⚖️",
    layout="wide"
)

# --- [수정된 부분 1] ---
# 전역 초기화 (캐싱)
@st.cache_resource
def get_judge():
    """GraphRAG Judge 초기화 - 한번만 실행"""
    
    print("--- [Cache Miss] 🚀 GraphRAG 시스템 초기화를 시작합니다... ---")
    
    try:
        # 1. 커넥터 생성 (이 객체가 내부에 driver를 생성)
        conn = Neo4jConnector()
        
        # 2. [수정] HybridGraphRAG에 driver와 함께 연결 정보도 전달
        from config.settings import settings
        
        rag = HybridGraphRAG(
            driver=conn.driver,
            openai_api_key=os.getenv('OPENAI_API_KEY', ''),
            neo4j_uri=settings.NEO4J_URI,
            neo4j_user=settings.NEO4J_USER,
            neo4j_password=settings.NEO4J_PASSWORD
        )
        
        return GraphRAGJudge(rag, conn)
    
    except Exception as e:
        st.error(f"❌ GraphRAG 초기화 실패: {e}")
        st.stop()
# UI 시작
st.title("⚖️ GraphRAG 기반 불공정 약관 판단 시스템")

st.markdown("""
이 시스템은 **진짜 GraphRAG**를 사용합니다:
- 🕸️ 지식 그래프 네트워크 탐색
- 🔍 벡터 + 그래프 하이브리드 검색  
- 🧠 다단계 추론 판단
""")

# 사이드바
with st.sidebar:
    st.header("🔧 입력")
    
    default_text = "회사는 어떠한 경우에도 책임을 지지 않습니다."
    user_input = st.text_area(
        "검사할 약관 조항:",
        value=default_text,
        height=150,
        placeholder="예: 회사는 고의 또는 중과실이 없는 한 손해배상 책임을 지지 않습니다."
    )
    
    st.markdown("---")
    
    # 옵션
    show_graph = st.checkbox("그래프 분석 상세 정보 표시", value=True)
    show_raw = st.checkbox("Raw JSON 표시", value=False)
    
    st.markdown("---")
    
    analyze_button = st.button("🔍 판단하기", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.caption("💡 **환경 요구사항**")
    st.caption("- Neo4j 실행 중")
    st.caption("- GraphRAG 구축 완료")
    st.caption("- OpenAI API 키 설정")

# 메인 영역
if analyze_button and user_input.strip():
    with st.spinner("🧠 GraphRAG로 다단계 분석 중..."):
        try:
            # --- [수정된 부분 2] ---
            # GraphRAG Judge 가져오기 (불필요한 'conn' 변수 제거)
            judge = get_judge()
            # --- [수정 종료 2] ---
            
            # 판단 실행
            result = judge.judge_clause(user_input)
            
            # === 결과 표시 ===
            st.markdown("---")
            st.subheader("📊 판단 결과")
            
            # 메트릭 3개
            col1, col2, col3 = st.columns(3)
            
            with col1:
                status = "⚠️ 위반" if result['violation'] else "✅ 정상"
                st.metric("위반 여부", status)
            
            with col2:
                confidence_pct = f"{result['confidence']:.1%}"
                st.metric("확신도", confidence_pct)
            
            with col3:
                severity_emoji = {
                    'high': '🔴',
                    'medium': '🟡',
                    'low': '🟢',
                    'none': '⚪'
                }
                severity_display = f"{severity_emoji.get(result['severity'], '⚪')} {result['severity'].upper()}"
                st.metric("심각도", severity_display)
            
            # === 그래프 분석 정보 ===
            if show_graph and 'graph_context' in result:
                st.markdown("---")
                st.subheader("🕸️ 그래프 네트워크 분석")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🔗 연결된 노드**")
                    graph_ctx = result['graph_context']
                    
                    node_info = {
                        "유사 사례": graph_ctx['similar_cases_count'],
                        "관련 법조항": ", ".join(graph_ctx['related_laws']) if graph_ctx['related_laws'] else "없음",
                        "공통 키워드": ", ".join(graph_ctx['keywords'][:5]) if graph_ctx['keywords'] else "없음",
                        "그래프 중심성": f"{graph_ctx['centrality_score']:.3f}"
                    }
                    
                    for key, value in node_info.items():
                        st.metric(key, value)
                
                with col2:
                    st.markdown("**📊 패턴 분석**")
                    if 'patterns' in result:
                        patterns = result['patterns']
                        
                        pattern_info = {
                            "패턴 강도": f"{patterns['strength']:.1%}",
                            "패턴 일관성": f"{patterns.get('pattern_consistency', 0):.1%}",
                        }
                        
                        for key, value in pattern_info.items():
                            st.metric(key, value)
                        
                        if patterns['common_keywords']:
                            st.markdown("**공통 패턴:**")
                            # st.badge는 최신 Streamlit 버전에만 있습니다.
                            # 없는 경우 st.markdown 등으로 대체할 수 있습니다.
                            try:
                                for kw in patterns['common_keywords']:
                                    st.badge(kw)  # type 파라미터 제거
                            except (AttributeError, TypeError):  # TypeError도 추가
                                st.write(", ".join(patterns['common_keywords']))
            
            # === 주요 근거 ===
            st.markdown("---")
            st.subheader("📌 주요 근거")
            
            if 'primary_evidence' in result:
                evidence = result['primary_evidence']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**가장 유사한 사례**")
                    st.info(f"**ID:** {evidence['best_match_id']}\n\n**유사도:** {evidence['similarity']:.3f}")
                
                with col2:
                    st.markdown("**위반 조항**")
                    article = evidence.get('article_id', '없음')
                    st.warning(f"**약관법 {article}**")
            
            # === 상세 설명 ===
            st.markdown("---")
            st.subheader("📝 상세 설명")
            st.write(result['explanation'])
            
            # === 수정 제안 ===
            if result['violation']:
                st.markdown("---")
                st.subheader("💡 수정 제안")
                st.info(result['suggestion'])
            
            # === 유사 사례 ===
            st.markdown("---")
            with st.expander("🔗 유사 사례 보기 (GraphRAG 검색 결과)", expanded=False):
                for i, case in enumerate(result['top_similar_cases'], 1):
                    st.markdown(f"**{i}. 유사도: {case['similarity']:.3f}** (ID: `{case['id']}`)")
                    
                    # 텍스트 미리보기
                    preview = case['text'][:300] + "..." if len(case['text']) > 300 else case['text']
                    st.text(preview)
                    
                    if i < len(result['top_similar_cases']):
                        st.divider()
            
            # === 추론 경로 ===
            if show_graph and 'reasoning_paths' in result and result['reasoning_paths'] > 0:
                with st.expander(f"🧠 추론 경로 ({result['reasoning_paths']}개 경로 탐색)", expanded=False):
                    st.markdown("""
                    다단계 그래프 탐색을 통해 연관 사례들의 네트워크를 발견했습니다.
                    
                    **탐색 방식:**
                    - 1단계: 벡터 유사도 검색으로 초기 사례 발견
                    - 2단계: 그래프 관계(SIMILAR_TO)를 따라 확장 탐색
                    - 3단계: 법조항(VIOLATES) 관계로 근거 강화
                    """)
            
            # === Raw JSON ===
            if show_raw:
                st.markdown("---")
                st.subheader("🔍 Raw JSON")
                st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")
        
        except Exception as e:
            st.error(f"❌ 오류 발생: {e}")
            
            with st.expander("🐛 디버그 정보"):
                import traceback
                st.code(traceback.format_exc())

elif analyze_button:
    st.warning("⚠️ 약관 조항을 입력해주세요.")

else:
    # 초기 화면
    st.info("👈 좌측 사이드바에서 약관 조항을 입력하고 '판단하기'를 클릭하세요.")
    
    # 예시
    with st.expander("💡 테스트 예시", expanded=True):
        st.markdown("""
        ### 위반 가능성 높은 예시:
        
        1. **전면 면책**
            > "회사는 어떠한 경우에도 책임을 지지 않습니다."
        
        2. **일방적 변경**
            > "회사는 이용자의 동의 없이 약관을 언제든지 변경할 수 있습니다."
        
        3. **과도한 손해배상**
            > "계약 위반 시 이용자는 계약금액의 10배를 배상해야 합니다."
        
        4. **부당한 해지 조건**
            > "회사는 이유 없이 언제든지 계약을 해지할 수 있습니다."
        
        ### 정상 예시:
        
        5. **합리적 면책**
            > "회사는 천재지변 등 불가항력으로 인한 손해에 대해서는 책임을 지지 않습니다."
        """)

# 푸터
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🛠️ **GraphRAG 시스템**")
    st.caption("Neo4j + LangChain + OpenAI")

with col2:
    st.caption("📊 **구성요소**")
    st.caption("지식 그래프 · 벡터 검색 · 다단계 추론")

with col3:
    st.caption("⚡ **실시간 판단**")
    st.caption("그래프 네트워크 기반 분석")