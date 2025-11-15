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

# 전역 초기화 (캐싱)
@st.cache_resource
def get_judge():
    """GraphRAG Judge 초기화 - 한번만 실행"""
    
    print("--- [Cache Miss] 🚀 GraphRAG 시스템 초기화를 시작합니다... ---")
    
    try:
        conn = Neo4jConnector()
        
        rag = HybridGraphRAG(
            driver=conn.driver,
            openai_api_key=os.getenv('OPENAI_API_KEY', '')
        )
        
        return GraphRAGJudge(rag, conn)
    
    except Exception as e:
        st.error(f"❌ GraphRAG 초기화 실패: {e}")
        st.stop()

# UI 시작
st.title("⚖️ GraphRAG 기반 불공정 약관 판단 시스템")

st.markdown("""
이 시스템은 **GraphRAG v7**을 사용합니다:
- 🕸️ 지식 그래프 네트워크 탐색
- 🔍 SimCLR Contrastive Learning  
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
            judge = get_judge()
            
            # 판단 실행
            result = judge.judge_clause(user_input)
            
            # === 결과 표시 ===
            st.markdown("---")
            st.subheader("📊 판단 결과")
            
            # 메트릭 3개 (✅ 퍼센트 제거)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                status = "⚠️ 위반" if result['violation'] else "✅ 정상"
                st.metric("위반 여부", status)
            
            with col2:
                # ✅ 퍼센트 제거: 0.853 → "0.853"
                confidence_val = f"{result['confidence']:.3f}"
                st.metric("확신도", confidence_val)
            
            with col3:
                severity_emoji = {
                    'critical': '🔴',
                    'high': '🔴',
                    'medium': '🟡',
                    'low': '🟢',
                    'none': '⚪'
                }
                severity_display = f"{severity_emoji.get(result['severity'], '⚪')} {result['severity'].upper()}"
                st.metric("심각도", severity_display)
            
            # 표현 추가
            if 'confidence_expression' in result:
                st.info(f"💬 {result['confidence_expression']}")
            
            # === 그래프 분석 정보 ===
            if show_graph and 'graph_context' in result:
                st.markdown("---")
                st.subheader("🕸️ 그래프 네트워크 분석")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🔗 연결된 노드**")
                    graph_ctx = result['graph_context']
                    
                    # 키워드 처리
                    keywords = graph_ctx.get('keywords', [])
                    if keywords:
                        keyword_texts = []
                        for kw in keywords[:5]:
                            if isinstance(kw, dict):
                                text = kw.get('text', '')
                                case_count = kw.get('case_count', 0)
                                keyword_texts.append(f"{text} ({case_count}개)")
                            else:
                                keyword_texts.append(str(kw))
                        keyword_display = ", ".join(keyword_texts)
                    else:
                        keyword_display = "없음"
                    
                    st.write(f"**유사 사례:** {graph_ctx.get('similar_cases_count', 0)}개")
                    
                    related_laws = graph_ctx.get('related_laws', [])
                    if related_laws:
                        st.write(f"**관련 법조항:** {', '.join(related_laws)}")
                    else:
                        st.write("**관련 법조항:** 없음")
                    
                    st.write(f"**공통 키워드:** {keyword_display}")
                    
                    # ✅ 네트워크 밀도 (퍼센트 제거)
                    network_density = graph_ctx.get('network_density', 0)
                    st.write(f"**네트워크 밀도:** {network_density:.3f}")
                    
                    # ✅ 구조 점수 (퍼센트 제거)
                    structure_score = graph_ctx.get('structure_score', 0)
                    st.write(f"**구조 점수:** {structure_score:.3f}")
                
                with col2:
                    st.markdown("**📊 패턴 분석**")
                    if 'patterns' in result:
                        patterns = result['patterns']
                        
                        # ✅ 패턴 강도 (퍼센트 제거)
                        pattern_strength = patterns.get('strength', 0)
                        st.write(f"**패턴 강도:** {pattern_strength:.3f}")
                        
                        # ✅ 패턴 일관성 (퍼센트 제거)
                        pattern_consistency = patterns.get('pattern_consistency', 0)
                        st.write(f"**패턴 일관성:** {pattern_consistency:.3f}")
                        
                        # 상위 키워드
                        top_keywords = patterns.get('top_keywords', [])
                        if top_keywords:
                            st.markdown("**상위 키워드:**")
                            for kw, count in top_keywords[:3]:
                                st.caption(f"• {kw}: {count}개 사례")
                        
                        # 공통 패턴
                        if patterns.get('common_keywords'):
                            st.markdown("**공통 패턴:**")
                            common_kw_str = ", ".join(patterns['common_keywords'][:5])
                            st.caption(common_kw_str)
            
            # === 주요 근거 ===
            st.markdown("---")
            st.subheader("📌 주요 근거")
            
            if 'primary_evidence' in result:
                evidence = result['primary_evidence']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**유사도 분석**")
                    
                    unfair_sim = evidence.get('unfair_similarity', 0)
                    st.write(f"불공정 원문 유사도: **{unfair_sim:.3f}**")
                    
                    fair_sim = evidence.get('fair_similarity', 0)
                    if fair_sim > 0:
                        st.write(f"수정본 유사도: **{fair_sim:.3f}**")
                    
                    # ✅ Contrastive 불공정도 (신규)
                    relative_unfairness = evidence.get('relative_unfairness', 0)
                    if relative_unfairness > 0:
                        st.write(f"상대적 불공정도: **{relative_unfairness:.3f}**")
                    
                    # 방법 표시
                    method = evidence.get('contrastive_method', '')
                    if method == 'contrastive_simclr':
                        st.caption("✨ SimCLR Contrastive Learning 적용")
                    
                    st.caption(f"최상위 사례 ID: `{evidence.get('best_match_id', 'N/A')}`")
                
                with col2:
                    st.markdown("**위반 조항**")
                    article = evidence.get('article_id', '없음')
                    st.warning(f"**약관법 {article}**")
                    
                    # 법률 구조 정보
                    if 'law_structure' in result:
                        law_info = result['law_structure']
                        full_path = law_info.get('full_path', '')
                        if full_path and full_path != 'Unknown':
                            st.caption(f"상세 경로: {full_path}")
                        
                        ho_content = law_info.get('ho_content', '')
                        if ho_content:
                            st.caption(f"조항 내용: {ho_content[:100]}...")
            
            # LLM 판단 정보 (접기)
            if 'llm_judgment' in result:
                with st.expander("🤖 LLM 의미 반전 검증", expanded=False):
                    llm_judgment = result['llm_judgment']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        formula_score = llm_judgment.get('formula_score', 0)
                        st.metric("수식 점수", f"{formula_score:.3f}")
                    
                    with col2:
                        adjusted_score = llm_judgment.get('adjusted_score', 0)
                        st.metric("LLM 조정 점수", f"{adjusted_score:.3f}")
                    
                    with col3:
                        is_reversed = llm_judgment.get('is_reversed', False)
                        reversed_display = "✅ 검출" if is_reversed else "❌ 없음"
                        st.metric("의미 반전", reversed_display)
                    
                    reasoning = llm_judgment.get('reasoning', '')
                    if reasoning:
                        st.markdown("**LLM 추론:**")
                        st.write(reasoning)
            
            # === 상세 설명 ===
            st.markdown("---")
            st.subheader("📝 상세 설명")
            explanation = result.get('explanation', '')
            if explanation:
                st.write(explanation)
            else:
                st.caption("설명 없음")
            
            # === 수정 제안 ===
            if result['violation']:
                st.markdown("---")
                st.subheader("💡 수정 제안")
                suggestion = result.get('suggestion', '')
                if suggestion:
                    st.info(suggestion)
                else:
                    st.caption("수정 제안 없음")
            
            # === 유사 사례 ===
            st.markdown("---")
            with st.expander("🔗 유사 사례 보기 (GraphRAG 검색 결과)", expanded=False):
                top_cases = result.get('top_similar_cases', [])
                
                if top_cases:
                    for i, case in enumerate(top_cases, 1):
                        similarity = case.get('similarity', 0)
                        st.markdown(f"**{i}. 유사도: {similarity:.3f}** (ID: `{case.get('id', 'N/A')}`)")
                        st.caption(f"조항: {case.get('article_id', 'N/A')}")
                        
                        # 텍스트 미리보기
                        text = case.get('text', '')
                        preview = text[:300] + "..." if len(text) > 300 else text
                        st.text(preview)
                        
                        if i < len(top_cases):
                            st.divider()
                else:
                    st.caption("유사 사례 없음")
            
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
    st.caption("🛠️ **GraphRAG v7**")
    st.caption("Neo4j + SimCLR + OpenAI")

with col2:
    st.caption("📊 **핵심 기능**")
    st.caption("Contrastive Learning · 그래프 구조 분석")

with col3:
    st.caption("⚡ **실시간 판단**")
    st.caption("다단계 추론 시스템")