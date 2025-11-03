"""
약관 불공정성 판단 - GraphRAG 기반 (v6 - 위반/수정 비교 로직)
- 핵심 변경:
  1) (v6) find_top_similar_cases:
     - 'similarity_violation' (위반사례 유사도) 계산
     - 'similarity_correction' (수정본 유사도) 계산
     - -> 두 개의 점수를 모두 반환
  2) (v6) calculate_deterministic_score:
     - (similarity_violation - similarity_correction)을 'net_similarity'로 계산
     - 이 net_similarity 점수를 최종 점수 후보(max)에 반영
     - -> "수정본과 비슷하면 공정하다"는 핵심 로직 구현
  3) (v5) 'critical'/'high' 키워드 발견 시 점수 'floor' 보장 (기존 로직 유지)
  4) (v4) best_match_case에 'correction_text' 추가
"""
import re
import sys
import json
import os
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np

# --- 기존 설정 및 임포트 (변경 없음) ---

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

try:
    from database.neo4j_connector import Neo4jConnector
except ImportError:
    print("⚠️ Neo4jConnector를 임포트할 수 없습니다. 'database.neo4j_connector'를 확인하세요.")
    # CI/테스트 환경을 위한 임시 커넥터
    class Neo4jConnector:
        def execute_query(self, query, params): return []
        def close(self): pass

# =============================================================================
# 임베딩 모델 로드
# =============================================================================
try:
    from sentence_transformers import SentenceTransformer
    MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("✓ 임베딩 모델 로드 완료")
except Exception as e:
    print(f"⚠️  임베딩 모델 로드 실패: {e}")
    MODEL = None

# =============================================================================
# LLM 클라이언트 (Claude 또는 OpenAI)
# =============================================================================
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # anthropic or openai

if LLM_PROVIDER == "anthropic":
    try:
        import anthropic
        LLM_CLIENT = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        LLM_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
        print(f"✓ LLM 클라이언트 로드: {LLM_PROVIDER} ({LLM_MODEL})")
    except Exception as e:
        print(f"⚠️  Anthropic 클라이언트 로드 실패: {e}")
        LLM_CLIENT = None
else:
    try:
        import openai
        LLM_CLIENT = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        print(f"✓ LLM 클라이언트 로드: {LLM_PROVIDER} ({LLM_MODEL})")
    except Exception as e:
        print(f"⚠️  OpenAI 클라이언트 로드 실패: {e}")
        LLM_CLIENT = None

# =============================================================================
# v2 패턴 로드
# =============================================================================
def load_patterns_v2():
    """patterns_by_article_v2.json 로드"""
    base_dir = Path(__file__).resolve().parent
    pattern_file = base_dir.parent / 'data' / 'contracts' / 'reference' / 'patterns_by_article_v2.json'
    
    if not pattern_file.exists() and 'PROJECT_ROOT' in globals():
         pattern_file = Path(PROJECT_ROOT) / 'data' / 'contracts' / 'reference' / 'patterns_by_article_v2.json'

    try:
        with open(pattern_file, 'r', encoding='utf-8') as f:
            patterns = json.load(f)
            print(f"✓ v2 패턴 파일 로드: {pattern_file.name}")
            return patterns
    except Exception as e:
        print(f"⚠️  v2 패턴 로드 실패: {e} (경로: {pattern_file})")
        return {}

PATTERNS_V2 = load_patterns_v2()

# =============================================================================
# 범용/복합 위험 키워드 추출 (변경 없음)
# =============================================================================
def get_universal_risk_keywords():
    universal = PATTERNS_V2.get('universal_risk_keywords', {})
    keywords = []
    for item in universal.get('keywords', []):
        keywords.append({
            'keyword': item['keyword'],
            'risk_level': item['risk_level'],
            'description': item['description']
        })
    return keywords

UNIVERSAL_RISKS = get_universal_risk_keywords()

def get_combined_patterns():
    combined = PATTERNS_V2.get('combined_pattern_risks', {})
    return combined.get('patterns', [])

COMBINED_PATTERNS = get_combined_patterns()

# =============================================================================
# 유사도 계산 (변경 없음)
# =============================================================================
def cosine_similarity(v1, v2):
    if v1 is None or v2 is None:
        return 0.0
    v1, v2 = np.array(v1), np.array(v2)
    if v1.size == 0 or v2.size == 0:
        return 0.0
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)

def lexical_jaccard(s1: str, s2: str) -> float:
    set1 = set([t for t in re.findall(r'\w+', s1.lower())])
    set2 = set([t for t in re.findall(r'\w+', s2.lower())])
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

# =============================================================================
# 조항 판별 (v2 패턴 기반) (변경 없음)
# =============================================================================
def detect_best_article(text: str) -> Tuple[str, float]:
    scores = {}
    article_keys = [k for k in PATTERNS_V2.keys() if k.startswith('제') and k.endswith('조')]
    if not article_keys:
        article_keys = ['제6조', '제7조', '제8조', '제9조', '제10조', '제11조', '제12조', '제13조', '제14조']

    for article_id in article_keys:
        if article_id not in PATTERNS_V2:
            continue
        article_data = PATTERNS_V2[article_id]
        score = 0.0
        for pattern in article_data.get('patterns', []):
            keywords = pattern.get('keywords', [])
            high_risk = pattern.get('high_risk_keywords', [])
            keyword_matches = sum(1 for kw in keywords if kw in text)
            score += keyword_matches * 0.1
            high_risk_matches = sum(1 for kw in high_risk if kw in text)
            score += high_risk_matches * 0.3
        scores[article_id] = min(score, 1.0)
    
    if not scores:
        return "제7조", 0.0
    best = max(scores.items(), key=lambda x: x[1])
    return best[0], best[1]

# =============================================================================
# 범용/복합 위험 키워드 체크 (변경 없음)
# =============================================================================
def check_universal_risks(text: str) -> List[Dict]:
    found = []
    for risk in UNIVERSAL_RISKS:
        keyword = risk['keyword']
        if keyword in text:
            found.append({
                'keyword': keyword,
                'risk_level': risk['risk_level'],
                'description': risk['description']
            })
    return found

def check_combined_patterns(text: str) -> List[Dict]:
    found = []
    for pattern in COMBINED_PATTERNS:
        keywords = pattern.get('combination', [])
        if all(kw in text for kw in keywords):
            found.append({
                'combination': keywords,
                'risk_level': pattern.get('risk_level', 'high'),
                'description': pattern.get('description', ''),
                'articles': pattern.get('articles', [])
            })
    return found

# =============================================================================
# Neo4j에서 위반 사례 검색 (변경 없음)
# (이미 correction_embedding을 잘 가져오고 있었음)
# =============================================================================
def query_violation_cases(conn: Neo4jConnector, article_id: str, limit: int = 10) -> List[Dict]:
    """조항별 위반 사례 검색 (위험도, 수정본 포함)"""
    query = """
    MATCH (a:조 {id: $article_id})
    OPTIONAL MATCH (a)-[:HAS_VIOLATION]->(v1:위반사례)
    OPTIONAL MATCH (a)-[:HAS_HANG]->(:항)-[:HAS_VIOLATION]->(v2:위반사례)
    OPTIONAL MATCH (a)-[:HAS_HANG]->(:항)-[:HAS_HO]->(:호)-[:HAS_VIOLATION]->(v3:위반사례)
    WITH collect(v1) + collect(v2) + collect(v3) as all_violations
    UNWIND all_violations as v
    WITH DISTINCT v
    WHERE v IS NOT NULL
    OPTIONAL MATCH (v)-[:HAS_CORRECTION]->(c:수정본)
    RETURN v.id as violation_id,
           v.unfair_text as violation_text,
           v.reason as reason,
           v.risk_level as risk_level,
           v.high_risk_keywords as high_risk_keywords,
           v.pattern_keywords as pattern_keywords,
           v.embedding as violation_embedding,
           c.corrected_text as correction_text,
           c.embedding as correction_embedding
    LIMIT $limit
    """
    return conn.execute_query(query, {"article_id": article_id, "limit": limit})

# =============================================================================
# (수정) 유사도 기반 상위 N개 사례 검색 (v6 - 수정본 유사도 계산 추가)
# =============================================================================
def find_top_similar_cases(user_text: str, cases: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    (v6 수정) 유사도 기반 상위 K개 사례 반환
    - 위반사례(violation)와 수정본(correction) 유사도를 둘 다 계산
    """
    if not cases or MODEL is None:
        return []
    
    user_emb = MODEL.encode(user_text)
    scored_cases = []
    
    for case in cases:
        # --- 1. 위반사례(Violation) 유사도 (기존 로직) ---
        viol_emb = case.get('violation_embedding')
        if viol_emb:
            viol_emb = np.array(viol_emb)
        else:
            viol_emb = MODEL.encode(case.get('violation_text', ''))
        
        sim_viol_semantic = cosine_similarity(user_emb, viol_emb)
        sim_viol_lexical = lexical_jaccard(user_text, case.get('violation_text', ''))
        final_sim_violation = (sim_viol_semantic * 0.7) + (sim_viol_lexical * 0.3)

        # --- 2. 수정본(Correction) 유사도 (v6 신규 로직) ---
        final_sim_correction = 0.0
        corr_emb = case.get('correction_embedding')
        corr_text = case.get('correction_text', '')
        
        # 수정본 텍스트와 임베딩이 모두 존재할 경우에만 계산
        if corr_emb and corr_text:
            try:
                sim_corr_semantic = cosine_similarity(user_emb, np.array(corr_emb))
                sim_corr_lexical = lexical_jaccard(user_text, corr_text)
                final_sim_correction = (sim_corr_semantic * 0.7) + (sim_corr_lexical * 0.3)
            except Exception as e:
                print(f"⚠️ 수정본 유사도 계산 실패: {e}")
                final_sim_correction = 0.0

        # --- 3. 두 개의 점수를 모두 저장 ---
        case_copy = case.copy()
        case_copy['similarity_violation'] = float(final_sim_violation)
        case_copy['similarity_correction'] = float(final_sim_correction)
        scored_cases.append(case_copy)
    
    # 정렬 기준: 여전히 '위반사례'와 가장 유사한 것을 찾는 것이므로, 'similarity_violation'을 기준으로 정렬
    scored_cases.sort(key=lambda x: x['similarity_violation'], reverse=True)
    
    return scored_cases[:top_k]

# =============================================================================
# RAG 컨텍스트 구성 (변경 없음)
# =============================================================================
def build_rag_context(user_text: str, article_id: str, top_cases: List[Dict], 
                      universal_risks: List[Dict], combined_risks: List[Dict]) -> str:
    """RAG를 위한 컨텍스트 구성 (LLM에게 전달할 근거 자료)"""
    
    context_parts = []
    
    article_info = PATTERNS_V2.get(article_id, {})
    context_parts.append(f"## 1. 판별된 관련 조항: {article_id} - {article_info.get('title', '')}")
    
    if top_cases:
        context_parts.append("\n## 2. 근거가 되는 유사 위반 사례 (DB 검색 결과):\n")
        for i, case in enumerate(top_cases, 1):
            risk_emoji = {'critical': '⚫', 'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(
                case.get('risk_level', 'medium'), '⚪'
            )
            # (v6) 표시할 유사도는 'violation' 유사도
            sim_v = case.get('similarity_violation', 0.0)
            sim_c = case.get('similarity_correction', 0.0)
            context_parts.append(f"### 사례 {i} (위반 유사도: {sim_v:.3f} / 수정 유사도: {sim_c:.3f})")
            context_parts.append(f"**- 불공정 원문**: {case.get('violation_text', '')[:200]}...")
            context_parts.append(f"**- 시정 요청 사유**: {case.get('reason', '')[:200]}...")
            if case.get('correction_text'):
                context_parts.append(f"**- 수정 후 약관**: {case.get('correction_text', '')[:200]}...")
            context_parts.append("")
    else:
        context_parts.append("\n## 2. 근거가 되는 유사 위반 사례를 찾을 수 없습니다.\n")
    
    if universal_risks:
        context_parts.append("## 3. 발견된 범용 위험 키워드 (규칙 기반):\n")
        for risk in universal_risks:
            risk_emoji = {'critical': '⚫', 'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(
                risk['risk_level'], '⚪'
            )
            context_parts.append(f"- {risk_emoji} **{risk['keyword']}**: {risk['description']}")
        context_parts.append("")
    
    if combined_risks:
        context_parts.append("## 4. 발견된 복합 패턴 (규칙 기반, 치명적):\n")
        for pattern in combined_risks:
            context_parts.append(f"- ⚫ **{' + '.join(pattern['combination'])}**: {pattern['description']}")
        context_parts.append("")
    
    context_parts.append("\n## 5. 검토 대상 약관 원문:\n")
    context_parts.append(f"```\n{user_text}\n```\n")
    
    return "\n".join(context_parts)

# =============================================================================
# (수정) 결정론적 점수 계산 로직 (v6 - net_similarity 반영)
# =============================================================================
def calculate_deterministic_score(article_score: float, universal_risks: List[Dict], 
                                  combined_risks: List[Dict], top_cases: List[Dict]) -> Dict:
    """
    (v6 수정) LLM 없이, 수집된 데이터를 기반으로 불공정 점수를 결정합니다.
    - 'net_similarity' (위반유사도 - 수정유사도)를 점수 후보로 추가합니다.
    - v5의 '최대값' 로직을 유지하여, 치명적인 근거 하나만으로도 높은 점수를 보장합니다.
    """
    
    WEIGHTS = {
        'article_match': 0.1,
        'universal_risk': {'critical': 0.5, 'high': 0.3, 'medium': 0.15, 'low': 0.05},
        'combined_pattern': 0.7, 
        'top_case_similarity': 0.5 # net_similarity에 적용될 가중치
    }
    
    score_candidates = [0.0] # 기본 점수 0.0
    reasoning = [] 

    # 1. 조항 매칭 점수 (v5와 동일)
    score_candidates.append(article_score * WEIGHTS['article_match'])
    reasoning.append(f"조항 매칭 점수 기여: {article_score * WEIGHTS['article_match']:.2f}")

    # 2. 범용 위험 키워드 점수 (v5와 동일)
    if universal_risks:
        universal_score_sum = sum(WEIGHTS['universal_risk'].get(r.get('risk_level', 'low'), 0.0) for r in universal_risks)
        score_candidates.append(universal_score_sum)
        reasoning.append(f"범용 위험 키워드 합산 기여: {universal_score_sum:.2f}")
        if any(r['risk_level'] == 'critical' for r in universal_risks):
            score_candidates.append(0.8) # 'critical' 키워드 발견 시, 최소 0.8점 보장
            reasoning.append("!! 'critical' 키워드 발견 -> 최소 0.8점 보장")
        elif any(r['risk_level'] == 'high' for r in universal_risks):
            score_candidates.append(0.6) # 'high' 키워드 발견 시, 최소 0.6점 보장
            reasoning.append("!! 'high' 키워드 발견 -> 최소 0.6점 보장")

    # 3. 복합 패턴 점수 (v5와 동일)
    if combined_risks:
        combined_score = len(combined_risks) * WEIGHTS['combined_pattern']
        score_candidates.append(combined_score)
        score_candidates.append(0.8) # 복합 패턴 발견 시, 최소 0.8점 보장
        reasoning.append(f"복합 패턴 {len(combined_risks)}개 발견 기여: {combined_score:.2f}")
        reasoning.append("!! '복합 패턴' 발견 -> 최소 0.8점 보장")
        
    # --- 4. (v6 수정) '위반 vs 수정' 유사도 점수 ---
    if top_cases:
        # 두 개의 유사도를 가져옴
        top_violation_sim = top_cases[0].get('similarity_violation', 0.0)
        top_correction_sim = top_cases[0].get('similarity_correction', 0.0)

        # (핵심 로직) 위반 유사도에서 수정본 유사도를 뺀 '순수 불공정 유사도'
        # (수정본과 더 비슷하면 이 값은 음수가 됨)
        net_similarity = top_violation_sim - top_correction_sim
        
        # 가중치 적용
        similarity_score = net_similarity * WEIGHTS['top_case_similarity']

        # 기존의 '높은 위반 유사도' 보너스는 유지 (불공정한게 확실하므로)
        if top_violation_sim >= 0.9:
            similarity_score += 0.2
        elif top_violation_sim >= 0.8:
            similarity_score += 0.1
        
        # (중요) 점수 후보에는 0보다 클 때만 추가
        # (수정본과 비슷해서 음수가 나온 경우, 다른 점수를 깎아먹지 않도록)
        score_candidates.append(max(0, similarity_score)) 
        
        reasoning.append(f"유사도 점수 (위반 {top_violation_sim:.2f} vs 수정 {top_correction_sim:.2f}) -> Net: {net_similarity:.2f}")
        reasoning.append(f"최종 유사도 기여: {max(0, similarity_score):.2f}")
    
    # 5. (v5) 모든 점수 후보 중 '최대값'을 최종 점수로 선택
    final_score = max(score_candidates)
    final_score = min(final_score, 1.0) # 1.0을 넘지 않도록
    
    # 6. (v4) 점수에 따른 위험도 결정
    severity = 'low'
    violation = False
    if final_score >= 0.7:
        severity = 'critical'
        violation = True
    elif final_score >= 0.5:
        severity = 'high'
        violation = True
    elif final_score >= 0.3:
        severity = 'medium'
        violation = True 
    elif final_score > 0.1:
        severity = 'low'
        violation = False
    else:
        severity = 'none'
        violation = False

    print(f"📈 [자체 로직 점수 계산 v6] 후보 점수: {score_candidates}")
    print(f"📈 [자체 로직 점수 계산 v6] 최종 점수: {final_score:.3f}, 위험도: {severity}")
    for r in reasoning:
        print(f"  - {r}")

    return {
        "score": float(final_score),
        "severity": severity,
        "violation": violation
    }

# =============================================================================
# (수정) LLM에 '설명' 요청 (v6 - 수정본 유사도 언급)
# =============================================================================
def ask_llm_explanation(context: str, user_text: str, score: float, severity: str, 
                      article_id: str, top_cases: List[Dict]) -> Dict:
    """
    (v6 수정) LLM에게 '왜 그런 판단이 나왔는지' 설명을 요청합니다.
    - '비위반'일 경우, '수정본과의 유사도'를 근거로 들 수 있도록 프롬프트 수정
    """
    
    if LLM_CLIENT is None:
        return {
            'explanation': 'LLM이 연결되지 않았습니다. 환경변수를 확인하세요.',
            'suggestion': 'LLM이 연결되지 않아 수정 제안을 생성할 수 없습니다.'
        }
    
    # 상위 케이스에서 수정본 유사도 추출
    top_correction_sim = 0.0
    if top_cases:
        top_correction_sim = top_cases[0].get('similarity_correction', 0.0)

    # 비위반(low/none)일 경우
    if severity in ['low', 'none']:
        # (v6 수정) 만약 수정본 유사도가 높다면, 그것을 '공정한 근거'로 사용
        if top_correction_sim > 0.5:
            explanation_base = f"본 조항은 시스템 분석 결과, DB의 '공정한 수정본' 사례와 높은 유사도(유사도: {top_correction_sim:.2f})를 보여 '비위반(위험도 낮음)'으로 판단됩니다."
        else:
            explanation_base = "본 조항은 시스템 분석 결과, 현행 약관법 기준에 위배되는 명백한 위험 요소가 발견되지 않아 '비위반(위험도 낮음)'으로 판단됩니다."
            
        system_prompt = f"""당신은 법률 비서입니다. 
        시스템이 약관을 '비위반(low/none)'으로 판단했습니다. 
        주어진 판단 근거를 바탕으로 공정하다고 간결하게 설명하고, 수정이 필요 없음을 확인해주세요.
        
        응답 형식 (JSON):
        {{
            "explanation": "{explanation_base}",
            "suggestion": "현재 조항은 공정한 것으로 판단되며, 수정이 필요하지 않습니다."
        }}"""
        
        user_prompt = f"""## 시스템 분석 결과
        - **검토 대상 약관**: "{user_text}"
        - **자체 로직 판단 위험도**: '{severity}'
        - **최상위 수정본 유사도**: {top_correction_sim:.2f}

        ## 당신의 임무
        시스템의 '비위반' 판단을 전달하는 설명을 생성하고, 수정이 필요 없다고 확인해주세요.
        (만약 '최상위 수정본 유사도'가 높다면(>0.5) 그 점을 언급해주세요.)
        
        반드시 JSON 형식으로만 답변해주세요."""

    # 위반(medium/high/critical)일 경우 (v4와 동일)
    else:
        system_prompt = """당신은 약관법 전문가입니다. 
        '판단'은 이미 시스템 로직에 의해 완료되었습니다. 당신의 역할은 그 판단이 왜 타당한지 근거를 들어 논리적으로 설명하고, 명확한 수정안을 제시하는 것입니다.
        'explanation'은 반드시 요청된 3단 구조로 작성해야 합니다.

        응답 형식 (JSON):
        {
            "explanation_structure": {
                "principle": "(원칙) [관련 법률 원칙이나 일반적인 가이드라인을 서술]",
                "violation_details": "(위반) '그러나'로 시작하며, 검토 대상 조항이 '왜' 원칙에 위배되는지 '근거 자료(유사 사례, 위험 키워드)'를 바탕으로 서술",
                "legal_citation": "(법 조항) '이는 약관법 제 O조를 위반할 소지가 있습니다.'와 같이 관련 법 조항 언급"
            },
            "suggestion": "고객에게 유리하고 법적으로 안전한 구체적인 수정 제안."
        }"""
        
        top_reason = "정보 없음"
        if top_cases and 'reason' in top_cases[0]:
            top_reason = top_cases[0]['reason']

        user_prompt = f"""## 1. 시스템 분석 근거 자료
{context}

## 2. 시스템 최종 분석 결과 (자체 로직이 판단 완료)
- **검토 대상 약관**: "{user_text}"
- **자체 로직 판단 점수**: {score:.2f} / 1.0
- **자체 로직 판단 위험도**: '{severity}'
- **판단 근거 (관련 조항)**: {article_id}
- **판단 근거 (최고 유사 사례의 사유)**: {top_reason}

## 3. 당신의 임무 (매우 중요)
위 '1. 시스템 분석 근거 자료'와 '2. 시스템 최종 분석 결과'를 바탕으로, 응답 JSON을 생성해주세요.
'explanation_structure'의 3가지 요소를 반드시 채워야 합니다:

1.  `principle`: {article_id}와 관련된 일반적인 법률 원칙을 서술합니다.
2.  `violation_details`: **반드시 '그러나'로 시작**해야 합니다. '검토 대상 약관'이 '근거 자료'를 바탕으로 왜 '원칙'에 위배되는지 설명합니다.
3.  `legal_citation`: {article_id}를 인용하여 '이는 약관법 {article_id}를 위반할 소지가 있습니다.'라고 마무리합니다.

반드시 JSON 형식으로만 답변해주세요."""
    
    try:
        # --- (이하 LLM 호출 및 JSON 파싱 로직은 v5와 동일) ---
        if LLM_PROVIDER == "anthropic":
            response = LLM_CLIENT.messages.create(
                model=LLM_MODEL, max_tokens=2000, system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            result_text = response.content[0].text
        else:  # openai
            response = LLM_CLIENT.chat.completions.create(
                model=LLM_MODEL, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                max_tokens=2000, response_format={"type": "json_object"}
            )
            result_text = response.choices[0].message.content
        
        result_text = re.sub(r'```json\n', '', result_text)
        result_text = re.sub(r'```', '', result_text)
        result = json.loads(result_text.strip())
        
        final_explanation = ""
        if 'explanation_structure' in result:
            struct = result['explanation_structure']
            final_explanation = f"{struct.get('principle', '')}\n\n{struct.get('violation_details', '')}\n\n{struct.get('legal_citation', '')}"
        elif 'explanation' in result:
            final_explanation = result.get('explanation', '설명 생성에 실패했습니다.')
        else:
            final_explanation = '설명 구조 생성에 실패했습니다.'
            
        return {
            'explanation': final_explanation.strip(),
            'suggestion': result.get('suggestion', '수정 제안 생성에 실패했습니다.')
        }
        
    except Exception as e:
        print(f"⚠️  LLM 설명 생성 실패: {e}")
        return {
            'explanation': f'LLM 설명 생성 중 오류 발생: {str(e)}',
            'suggestion': '오류로 인해 수정 제안을 생성할 수 없습니다.'
        }

# =============================================================================
# (수정) 메인 판단 로직 (v6 - best_match_case 수정)
# =============================================================================
def comprehensive_judgment(user_text: str, conn: Neo4jConnector) -> Dict:
    """GraphRAG 기반 종합 판단 (v6 - 위반/수정 비교 로직 적용)"""
    
    # --- 1단계 ~ 5단계: RAG로 근거 데이터 수집 ---
    article_id, article_score = detect_best_article(user_text)
    print(f"📍 [1/5] 판별된 조항: {article_id} (점수: {article_score:.3f})")
    
    universal_risks = check_universal_risks(user_text)
    print(f"🔍 [2/5] 범용 위험 키워드: {len(universal_risks)}개 발견")
    
    combined_risks = check_combined_patterns(user_text)
    print(f"⚠️  [3/5] 복합 패턴: {len(combined_risks)}개 발견")
    
    all_cases = query_violation_cases(conn, article_id, limit=50)
    print(f"📊 [4/5] 검색된 사례: {len(all_cases)}개")
    
    # (v6) find_top_similar_cases가 이제 similarity_violation, similarity_correction을 반환
    top_cases = find_top_similar_cases(user_text, all_cases, top_k=5)
    print(f"🎯 [5/5] 상위 유사 사례: {len(top_cases)}개")
    
    # --- 6단계: 자체 로직으로 최종 '판단' (v6 로직 적용) ---
    print("\n📈 [6/8] 자체 로직으로 점수 '판단' 중 (v6 가중치)...")
    judgment_result = calculate_deterministic_score(
        article_score, universal_risks, combined_risks, top_cases
    )
    final_score = judgment_result['score']
    final_severity = judgment_result['severity']
    final_violation = judgment_result['violation']
    print(f"✅ [6/8] 자체 '판단' 완료: 점수={final_score:.2f}, 위험도='{final_severity}'")

    # --- 7단계: LLM '설명'을 위한 RAG 컨텍스트 구성 ---
    rag_context = build_rag_context(user_text, article_id, top_cases, universal_risks, combined_risks)
    
    # --- 8단계: LLM에 '설명' 및 '제안' 요청 ---
    print("\n🤖 [7/8] LLM '설명'(v6) 및 '제안' 생성 요청 중...")
    llm_explanation = ask_llm_explanation(
        rag_context, user_text, final_score, final_severity, article_id, top_cases
    )
    
    # --- 9단계: 최종 결과 구성 ---
    print("\n📋 [8/8] 최종 결과 구성 중...")
    
    # (v6 수정) best_match_case 구성
    best_match_case_result = None
    if top_cases:
        best_case = top_cases[0]
        sim_v = best_case.get('similarity_violation', 0.0)
        sim_c = best_case.get('similarity_correction', 0.0)
        
        # 위반 유사도가 0.4 이상일 때만 근거로 채택 (v5와 동일)
        if sim_v >= 0.4:
            best_match_case_result = {
                "similarity_violation": sim_v,
                "similarity_correction": sim_c, # (v6) 수정본 유사도 추가
                "violation_text": best_case.get('violation_text', ''),
                "reason": best_case.get('reason', ''),
                "correction_text": best_case.get('correction_text', '') # (v6) 수정본 원문 추가
            }
        else:
             print(f"ℹ️ 최고 유사도 사례(위반 유사도 {sim_v:.2f})가 0.4 미만으로 근거에서 제외됨.")

    result = {
        # --- '자체 로직'이 결정한 판단 결과 ---
        "violation": final_violation,
        "score": final_score,
        "severity": final_severity,
        "confidence": 1.0,

        # --- 'LLM'이 생성한 서술형 결과 ---
        "explanation": llm_explanation.get('explanation'),
        "suggestion": llm_explanation.get('suggestion'),

        # --- 판단의 근거가 된 데이터 ---
        "article_id": article_id,
        "best_match_case": best_match_case_result, # (v6) Streamlit 표시용 최고 근거
        "top_cases": [ # (참고용) 상위 3개 요약
            {
                "id": case.get('violation_id'),
                "similarity_violation": case.get('similarity_violation', 0.0),
                "similarity_correction": case.get('similarity_correction', 0.0),
                "risk_level": case.get('risk_level', 'medium'),
                "snippet": case.get('violation_text', '')[:200]
            }
            for case in top_cases[:3]
        ],
        "universal_risks": [
            {
                "keyword": r['keyword'],
                "risk_level": r['risk_level']
            }
            for r in universal_risks
        ],
        "combined_patterns": [
            {
                "combination": p['combination'],
                "risk_level": p['risk_level']
            }
            for p in combined_risks
        ],
        "debug": {
            "article_score": float(article_score),
            "cases_found": len(all_cases),
            "top_similarity_violation": top_cases[0].get('similarity_violation', 0.0) if top_cases else 0.0,
            "top_similarity_correction": top_cases[0].get('similarity_correction', 0.0) if top_cases else 0.0,
            "universal_risk_count": len(universal_risks),
            "combined_pattern_count": len(combined_risks),
            "llm_provider": LLM_PROVIDER,
            "llm_model": LLM_MODEL
        }
    }
    
    return result

# =============================================================================
# 실행 엔트리 (변경 없음)
# =============================================================================
def run(user_text: str) -> Dict:
    """메인 실행 함수"""
    conn = Neo4jConnector()
    try:
        return comprehensive_judgment(user_text, conn)
    finally:
        conn.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python scripts/judge_clause_v6.py '검사할 문장'")
        sys.exit(1)
    
    text = sys.argv[1]
    print(f"\n{'='*70}")
    print(f"약관 검토: {text[:100]}...")
    print(f"{'='*70}\n")
    
    result = run(text)
    
    print(f"\n{'='*70}")
    print("📋 최종 판단 결과")
    print(f"{'='*70}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
