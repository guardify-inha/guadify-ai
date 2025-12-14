"""
AI 기반 불공정 약관 판단 시스템 - Streamlit UI (v8.1 Compatible)
"""
import streamlit as st
import sys
from pathlib import Path
import os
import json
import pandas as pd

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
from judge.article_violation_scorer import ArticleViolationScorer

# 페이지 설정
st.set_page_config(
    page_title="AI 기반 불공정 약관 판단",
    page_icon="⚖️",
    layout="wide"
)

# 사이드바 기본 width를 600px로 설정
st.markdown("""
<style>
    /* 사이드바 기본 width 600px 설정 */
    [data-testid="stSidebar"] {
        width: 600px !important;
        min-width: 600px !important;
        max-width: 600px !important;
    }
    
    /* 사용자 제공 선택자에 대한 width 설정 */
    #root > div:nth-child(1) > div.withScreencast > div > div > section > div.st-emotion-cache-1csi29a.e6f82ta2 {
        width: 600px !important;
        min-width: 600px !important;
        max-width: 600px !important;
    }
</style>
""", unsafe_allow_html=True)

# 전역 초기화 (캐싱)
@st.cache_resource
def get_judge():
    """GraphRAG Judge 및 Article Scorer 초기화 - 한번만 실행"""

    print("--- [Cache Miss] 🚀 GraphRAG 시스템 초기화를 시작합니다... ---")

    try:
        conn = Neo4jConnector()

        rag = HybridGraphRAG(
            driver=conn.driver,
            openai_api_key=os.getenv('OPENAI_API_KEY', '')
        )

        judge = GraphRAGJudge(rag, conn)
        scorer = ArticleViolationScorer()

        return judge, scorer

    except Exception as e:
        st.error(f"❌ GraphRAG 초기화 실패: {e}")
        st.stop()

# UI 시작
st.title("⚖️ AI 기반 불공정 약관 판단 시스템")

st.markdown("""
- 🔍 Prototypical Networks 불공정도 계산
- 📊 패턴 기반 위험도 분석  
- 🧠 LLM 의미 반전 검증
""")

# 사이드바
with st.sidebar:
    st.header("🔧 입력")
    
    default_text = "회사는 어떠한 경우에도 책임을 지지 않습니다."
    user_input = st.text_area(
        "검사할 약관 조항:",
        value=default_text,
        height=400,
        placeholder="예: 회사는 고의 또는 중과실이 없는 한 손해배상 책임을 지지 않습니다."
    )
    
    # 옵션 (UI에서 숨김 처리, 초기값 유지)
    show_prototypical = True
    show_raw = False
    
    st.markdown("---")
    
    analyze_button = st.button("🔍 판단하기", type="primary", width='stretch')
    
    st.markdown("---")
    st.caption("💡 **환경 요구사항**")
    st.caption("- Neo4j 실행 중")
    st.caption("- GraphRAG 구축 완료")
    st.caption("- OpenAI API 키 설정")

# 메인 영역
if analyze_button and user_input.strip():
    try:
        judge, scorer = get_judge()

        # 전처리 수행 (모든 분석은 processed_text 기반)
        preprocessing_result = judge._preprocess_user_input(user_input)
        processed_text = preprocessing_result['final_input']

        # 조항별 위반도 점수 계산 (빠른 계산) - processed_text 사용
        with st.spinner("📊 패턴 분석 중..."):
            article_scores = scorer.calculate_article_scores(processed_text)
            primary_violation = scorer.get_primary_violation(article_scores)
            primary_score = primary_violation.get('score', 0.0)

        # 빠른 계산만 수행 (LLM 생성 스킵) - 이미 전처리된 결과 전달
        with st.spinner("🔍 유사도 분석 및 Prototypical Networks 계산 중..."):
            result = judge.judge_clause(user_input, primary_score=primary_score, skip_llm_generation=True, preprocessing_result=preprocessing_result)
        
        # ✅ 전처리 정보 표시 (수정됨: 요약 숨김, 다중 조항 시 원문만 표시)
        if 'preprocessing' in result:
            preprocessing = result['preprocessing']
            
            # 다중 조항인 경우에만 경고 표시
            if preprocessing.get('is_multiple_clauses'):
                st.warning(f"⚠️ **여러 조항이 감지되어 첫 번째 조항만 분석했습니다**")
                
                with st.expander("📝 분석에 사용된 조항 확인", expanded=True):
                    st.markdown("**실제 분석에 사용된 조항 (첫 번째 조항 원문):**")
                    # 요약본(final_input)은 보여주지 않고, 추출된 조항 원문(first_clause_raw)만 보여줍니다.
                    st.text_area("", value=preprocessing['first_clause_raw'], height=80, disabled=True, label_visibility="collapsed")
            
            # needs_summary가 True여도 사용자에게는 알리지 않음 (UI 유지)


        # === 빠른 결과 즉시 표시 ===
        st.markdown("---")
        st.subheader("📊 판단 결과")
        
        # 메트릭 3개
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status = "⚠️ 위반" if result['violation'] else "✅ 정상"
            st.metric("위반 여부", status)
        
        with col2:
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
        
        # === Prototypical Networks 분석 (v8.1 전용) ===
        if show_prototypical and 'primary_evidence' in result and result['primary_evidence'] is not None:
            evidence = result['primary_evidence']
            
            # Prototypical 정보가 있는지 확인
            if evidence and evidence.get('method') in ['prototypical_networks_multi_prototype', 'prototypical_networks_single_prototype']:
                st.markdown("---")
                st.subheader("🧬 Prototypical Networks 분석")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**거리 분석**")
                    
                    unfair_dist = evidence.get('unfair_distance')
                    fair_dist = evidence.get('fair_distance')
                    
                    if unfair_dist is not None:
                        st.write(f"Unfair Prototype 거리: **{unfair_dist:.3f}**")
                    if fair_dist is not None:
                        st.write(f"Fair Prototype 거리: **{fair_dist:.3f}**")
                    
                    distance_ratio = evidence.get('distance_ratio')
                    if distance_ratio is not None:
                        st.write(f"거리 비율: **{distance_ratio:.3f}**")
                
                with col2:
                    st.markdown("**확률 분석**")
                    
                    relative_unfairness = evidence.get('relative_unfairness', 0)
                    st.write(f"P(Unfair): **{relative_unfairness:.3f}**")
                    st.write(f"P(Fair): **{1 - relative_unfairness:.3f}**")
                    
                    temperature = evidence.get('temperature')
                    if temperature:
                        st.caption(f"Temperature τ = {temperature}")
                    
                    # Prototype 개수 표시 (multi-prototype인 경우)
                    num_unfair = evidence.get('num_unfair_prototypes')
                    num_fair = evidence.get('num_fair_prototypes')
                    if num_unfair and num_fair:
                        st.caption(f"사용된 prototype: Unfair {num_unfair}개, Fair {num_fair}개")
                
                # 해석 표시
                interpretation = evidence.get('interpretation', '')
                if interpretation:
                    st.info(f"💡 **해석:** {interpretation}")
        

        # === 조항별 위반도 분석 ===
        st.markdown("---")
        st.subheader("⚖️ 조항별 위반도 분석 (패턴 기반)")

        col1, col2 = st.columns([2, 1])

        with col1:
            # 조항별 점수를 DataFrame으로 변환
            score_data = []
            for article, data in article_scores.items():
                score_data.append({
                    '조항': article,
                    '점수': data['score']
                })

            df_scores = pd.DataFrame(score_data).sort_values('점수', ascending=False)

            # Bar chart 표시
            st.bar_chart(df_scores.set_index('조항')['점수'])

        with col2:
            st.markdown("**최고 위반 조항**")

            # 임계값 표시
            threshold = 0.3
            primary_score = primary_violation['score']

            if primary_score >= threshold:
                st.error(f"**{primary_violation['article']}**")
                st.metric("위반도", f"{primary_score:.3f}")

                # 매칭된 패턴 정보
                details = primary_violation.get('details', {})
                matched_high_risk = details.get('matched_high_risk', [])
                matched_regex = details.get('matched_regex', [])
                matched_exceptions = details.get('matched_exceptions', [])

                if matched_high_risk:
                    st.caption(f"🔴 고위험: {len(matched_high_risk)}개")
                if matched_regex:
                    st.caption(f"🔍 패턴: {len(matched_regex)}개")
                if matched_exceptions:
                    st.caption(f"✅ 예외: {len(matched_exceptions)}개")
            else:
                st.success("위반 없음")
                st.metric("최고 점수", f"{primary_score:.3f}")

            st.caption(f"임계값: {threshold}")

        # 상세 점수 테이블 (접기)
        with st.expander("📋 조항별 상세 점수", expanded=False):
            detailed_data = []
            for article, data in article_scores.items():
                details = data.get('details', {})
                detailed_data.append({
                    '조항': article,
                    '점수': f"{data['score']:.3f}",
                    '키워드': len(details.get('matched_keywords', [])),
                    '고위험': len(details.get('matched_high_risk', [])),
                    'Regex': len(details.get('matched_regex', [])),
                    '예외': len(details.get('matched_exceptions', []))
                })

            # for 루프 밖에서 DataFrame 생성 및 표시
            df_detailed = pd.DataFrame(detailed_data).sort_values('점수', ascending=False)
            st.dataframe(df_detailed, width='stretch')

            # 매칭 상세 (최고 점수 조항만)
            if primary_score >= threshold:
                st.markdown(f"**{primary_violation['article']} 매칭 상세:**")
                details = primary_violation.get('details', {})

                if details.get('matched_high_risk'):
                    st.markdown("🔴 **고위험 키워드:**")
                    for kw in details['matched_high_risk'][:5]:
                        st.caption(f"• {kw}")

                if details.get('matched_regex'):
                    st.markdown("🔍 **매칭 패턴:**")
                    for pattern in details['matched_regex'][:5]:
                        st.caption(f"• {pattern}")

                if details.get('matched_exceptions'):
                    st.markdown("✅ **예외 적용:**")
                    for exc in details['matched_exceptions']:
                        st.caption(f"• {exc}")

        # 약관법 원문 데이터
        LAW_ARTICLES_TEXT = {
            "제7조": """제7조 : (면책조항의 금지) 계약 당사자의 책임에 관하여 정하고 있는 약관의 내용 중 다음 각 호의 어느 하나에 해당하는 내용을 정하고 있는 조항은 무효로 한다.
제1호 : 사업자, 이행 보조자 또는 피고용자의 고의 또는 중대한 과실로 인한 법률상의 책임을 배제하는 조항
제2호 : 상당한 이유 없이 사업자의 손해배상 범위를 제한하거나 사업자가 부담하여야 할 위험을 고객에게 떠넘기는 조항
제3호 : 상당한 이유 없이 사업자의 담보책임을 배제 또는 제한하거나 그 담보책임에 따르는 고객의 권리행사의 요건을 가중하는 조항
제4호 : 상당한 이유 없이 계약목적물에 관하여 견본이 제시되거나 품질ㆍ성능 등에 관한 표시가 있는 경우 그 보장된 내용에 대한 책임을 배제 또는 제한하는 조항""",
            "제8조": """제8조 : (손해배상액의 예정) 고객에게 부당하게 과중한 지연 손해금 등의 손해배상 의무를 부담시키는 약관 조항은 무효로 한다.""",
            "제9조": """제9조 : (계약의 해제ㆍ해지) 계약의 해제ㆍ해지에 관하여 정하고 있는 약관의 내용 중 다음 각 호의 어느 하나에 해당되는 내용을 정하고 있는 조항은 무효로 한다.
제1호 : 법률에 따른 고객의 해제권 또는 해지권을 배제하거나 그 행사를 제한하는 조항
제2호 : 사업자에게 법률에서 규정하고 있지 아니하는 해제권 또는 해지권을 부여하여 고객에게 부당하게 불이익을 줄 우려가 있는 조항
제3호 : 법률에 따른 사업자의 해제권 또는 해지권의 행사 요건을 완화하여 고객에게 부당하게 불이익을 줄 우려가 있는 조항
제4호 : 계약의 해제 또는 해지로 인한 원상회복의무를 상당한 이유 없이 고객에게 과중하게 부담시키거나 고객의 원상회복 청구권을 부당하게 포기하도록 하는 조항
제5호 : 계약의 해제 또는 해지로 인한 사업자의 원상회복의무나 손해배상의무를 부당하게 경감하는 조항
제6호 : 계속적인 채권관계의 발생을 목적으로 하는 계약에서 그 존속기간을 부당하게 단기 또는 장기로 하거나 묵시적인 기간의 연장 또는 갱신이 가능하도록 정하여 고객에게 부당하게 불이익을 줄 우려가 있는 조항""",
            "제10조": """제10조 : (채무의 이행) 채무의 이행에 관하여 정하고 있는 약관의 내용 중 다음 각 호의 어느 하나에 해당하는 내용을 정하고 있는 조항은 무효로 한다.
제1호 : 상당한 이유 없이 급부(給付)의 내용을 사업자가 일방적으로 결정하거나 변경할 수 있도록 권한을 부여하는 조항
제2호 : 상당한 이유 없이 사업자가 이행하여야 할 급부를 일방적으로 중지할 수 있게 하거나 제3자에게 대행할 수 있게 하는 조항""",
            "제11조": """제11조 : (고객의 권익 보호) 고객의 권익에 관하여 정하고 있는 약관의 내용 중 다음 각 호의 어느 하나에 해당하는 내용을 정하고 있는 조항은 무효로 한다.
제1호 : 법률에 따른 고객의 항변권(抗辯權), 상계권(相計權) 등의 권리를 상당한 이유 없이 배제하거나 제한하는 조항
제2호 : 고객에게 주어진 기한의 이익을 상당한 이유 없이 박탈하는 조항
제3호 : 고객이 제3자와 계약을 체결하는 것을 부당하게 제한하는 조항
제4호 : 사업자가 업무상 알게 된 고객의 비밀을 정당한 이유 없이 누설하는 것을 허용하는 조항""",
            "제12조": """제12조 : (의사표시의 의제) 의사표시에 관하여 정하고 있는 약관의 내용 중 다음 각 호의 어느 하나에 해당하는 내용을 정하고 있는 조항은 무효로 한다.
제1호 : 일정한 작위(作爲) 또는 부작위(不作爲)가 있을 경우 고객의 의사표시가 표명되거나 표명되지 아니한 것으로 보는 조항. 다만, 고객에게 상당한 기한 내에 의사표시를 하지 아니하면 의사표시가 표명되거나 표명되지 아니한 것으로 본다는 뜻을 명확하게 따로 고지한 경우이거나 부득이한 사유로 그러한 고지를 할 수 없는 경우에는 그러하지 아니하다.
제2호 : 고객의 의사표시의 형식이나 요건에 대하여 부당하게 엄격한 제한을 두는 조항
제3호 : 고객의 이익에 중대한 영향을 미치는 사업자의 의사표시가 상당한 이유 없이 고객에게 도달된 것으로 보는 조항
제4호 : 고객의 이익에 중대한 영향을 미치는 사업자의 의사표시 기한을 부당하게 길게 정하거나 불확정하게 정하는 조항""",
            "제13조": """제13조 : (대리인의 책임 가중) 고객의 대리인에 의하여 계약이 체결된 경우 고객이 그 의무를 이행하지 아니하는 경우에는 대리인에게 그 의무의 전부 또는 일부를 이행할 책임을 지우는 내용의 약관 조항은 무효로 한다.""",
            "제14조": """제14조 : (소송 제기의 금지 등) 소송 제기 등과 관련된 약관의 내용 중 다음 각 호의 어느 하나에 해당하는 조항은 무효로 한다.
제1호 : 고객에게 부당하게 불리한 소송 제기 금지 조항 또는 재판관할의 합의 조항
제2호 : 상당한 이유 없이 고객에게 입증책임을 부담시키는 약관 조항"""
        }

        # 최고 위반조항의 약관법 원문 표시
        if primary_violation['article'] and primary_violation['article'] in LAW_ARTICLES_TEXT:
            st.markdown("---")
            st.subheader("📜 최고 위반조항 약관법 원문")
            
            article_text = LAW_ARTICLES_TEXT[primary_violation['article']]
            
            # 텍스트를 파싱하여 HTML로 포맷팅
            lines = article_text.split('\n')
            html_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # 조항 제목 (제X조로 시작하고 '조'가 포함된 줄, '호'는 포함하지 않음)
                if line.startswith('제') and '조' in line and '호' not in line and ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        title = parts[0].strip()
                        content = parts[1].strip()
                        html_lines.append(f'<h4 style="color: #1f77b4; margin-top: 1rem; margin-bottom: 0.5rem;">{title}</h4>')
                        html_lines.append(f'<p style="font-weight: 900; margin-bottom: 1rem; color: #333;">{content}</p>')
                    else:
                        html_lines.append(f'<h4 style="color: #1f77b4; margin-top: 1rem; margin-bottom: 0.5rem;">{line}</h4>')
                # 호 항목 (제X호로 시작하는 줄) - h4가 아닌 일반 텍스트로 처리
                elif '호' in line and ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        ho = parts[0].strip()
                        content = parts[1].strip()
                        html_lines.append(f'<div style="margin: 0.3rem 0; padding: 0.3rem 0 0.3rem 0.8rem; border-left: 2px solid #e0e0e0;">')
                        html_lines.append(f'<span style="color: #666; font-weight: 500; font-size: 0.85rem; display: inline;">{ho}</span><span style="color: #444; margin-left: 0.3rem;">{content}</span>')
                        html_lines.append(f'</div>')
                    else:
                        html_lines.append(f'<p style="margin: 0.5rem 0; font-size: 0.85rem;">{line}</p>')
                else:
                    html_lines.append(f'<p style="margin: 0.5rem 0; font-size: 0.85rem;">{line}</p>')
            
            # 카드 형태로 표시
            html_content = '\n'.join(html_lines)
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(to right, #f8f9fa 0%, #ffffff 100%);
                    border-left: 5px solid #1f77b4;
                    padding: 1.5rem;
                    border-radius: 8px;
                    margin: 1rem 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    font-size: 0.85rem;
                    line-height: 1.7;
                ">
                {html_content}
                </div>
                """,
                unsafe_allow_html=True
            )

        # === 주요 근거 ===
        st.markdown("---")
        st.subheader("📌 주요 근거")
        
        if 'primary_evidence' in result and result['primary_evidence'] is not None:
            evidence = result['primary_evidence']

            # 유사도 분석
            st.markdown("**유사도 분석**")

            unfair_sim = evidence.get('unfair_similarity', 0)
            st.write(f"위반사례 유사도: **{unfair_sim:.3f}**")

            # v8.1: fair_similarity는 없고 relative_unfairness만 있음
            relative_unfairness = evidence.get('relative_unfairness', 0)
            if relative_unfairness > 0:
                st.write(f"상대적 불공정도 (P(Unfair)): **{relative_unfairness:.3f}**")

            # 방법 표시
            method = evidence.get('method', '')
            if 'prototypical' in method:
                st.caption("✨ Prototypical Networks 적용")
            
            # 법률 구조 정보 (사용자 요청으로 숨김)
            # st.markdown("**위반 조항**")
            # article_id = evidence.get('article_id', 'Unknown')
            # st.write(f"조항: **{article_id}**")
            #
            # hang = evidence.get('hang')
            # if hang:
            #     st.caption(f"항: {hang}")
            #
            # ho = evidence.get('ho')
            # if ho:
            #     st.caption(f"호: {ho}")

            # st.caption(f"최상위 사례 ID: `{evidence.get('best_match_id', 'N/A')}`")
        else:
            st.info("✅ 정상 판단으로 위반 관련 근거 정보가 없습니다.")
        
        # 패턴 분석 정보 (v8.1에서는 간소화됨)
        if 'patterns' in result:
            with st.expander("📊 패턴 분석 상세", expanded=False):
                patterns = result['patterns']
                
                # 위험 키워드
                matched_keywords = patterns.get('matched_risk_keywords', [])
                if matched_keywords:
                    st.markdown("**매칭된 위험 키워드:**")
                    for kw in matched_keywords[:10]:
                        if isinstance(kw, dict):
                            keyword = kw.get('keyword', '')
                            risk_level = kw.get('risk_level', 'unknown')
                            method = kw.get('method', 'string')
                            st.caption(f"• {keyword} ({risk_level}, {method})")
                        else:
                            st.caption(f"• {kw}")
                
                # 패턴 기반 위험도
                risk_level = patterns.get('risk_level_from_patterns', 'unknown')
                st.write(f"패턴 기반 위험도: **{risk_level}**")
                
                pattern_score = patterns.get('pattern_score', 0)
                st.write(f"패턴 점수: **{pattern_score:.3f}**")
        
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
        
        # === LLM 결과 단계별 표시 (위반 판단 시에만) ===
        if result['violation']:
            # 상세 설명 생성 및 표시
            st.markdown("---")
            st.subheader("📝 상세 설명")
            
            if result.get('explanation'):
                # 이미 생성된 경우
                st.write(result['explanation'])
            else:
                # LLM으로 생성
                with st.spinner("🤖 LLM으로 상세 설명 생성 중..."):
                    explanation = judge.generate_explanation(result, user_input)
                    if explanation:
                        result['explanation'] = explanation
                        st.write(explanation)
                    else:
                        st.caption("설명 생성 실패")
            
            # 수정 제안 생성 및 표시
            st.markdown("---")
            st.subheader("💡 수정 제안")
            
            if result.get('suggestion'):
                # 이미 생성된 경우
                suggestion = result['suggestion']
            else:
                # LLM으로 생성
                with st.spinner("🤖 LLM으로 수정 제안 생성 중..."):
                    suggestion = judge.generate_suggestion(result, user_input)
                    if suggestion:
                        result['suggestion'] = suggestion
            
            if suggestion:
                # 줄바꿈 처리를 위해 마크다운으로 렌더링
                formatted_suggestion = suggestion.replace("수정 전:", "\n\n**수정 전**:") \
                                               .replace("수정 후:", "\n\n**수정 후**:") \
                                               .replace("수정 이유:", "\n\n**수정 이유**:")
                st.markdown(formatted_suggestion)
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
            > "회사는 천재지변 등 불가항력으로 인한 손해에 책임을 지지 않지만, 회사귀책에 인한 손해는 책임을 부담합니다."
        """)

# 푸터
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🛠️ **Guadify AI v8.1**")
    st.caption("Neo4j + Prototypical Networks + OpenAI")

with col2:
    st.caption("📊 **핵심 기능**")
    st.caption("Prototypical Networks · 패턴 분석 · LLM 검증")

with col3:
    st.caption("⚡ **실시간 판단**")
    st.caption("3단계 AI 시스템")