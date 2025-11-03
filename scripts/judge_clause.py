"""
약관 불공정성 판단 - GraphRAG 기반 (v3 - 로직/LLM 역할 분리)
- 핵심 변경:
  1) patterns_by_article_v2.json 반영 (high_risk_keywords, universal_risk_keywords, combined_patterns)
  2) RAG + 결정론적 점수 계산:
     - 1단계 (RAG): GraphRAG로 모든 근거 데이터 수집
     - 2단계 (자체 로직): 수집된 데이터로 'calculate_deterministic_score' 함수가 최종 점수/위험도/위반 여부 '판단'
     - 3단계 (LLM): LLM은 '판단된 결과'를 전달받아 '설명'과 '수정 제안'만 생성
  3) 위험도 레벨(critical/high/medium/low) 활용
  4) 복합 패턴 위험도 체크
  5) Claude/OpenAI API 연동 (환경변수로 선택)
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
    # 스크립트 위치 기준으로 data 폴더 경로 설정
    base_dir = Path(__file__).resolve().parent
    pattern_file = base_dir.parent / 'data' / 'contracts' / 'reference' / 'patterns_by_article_v2.json'
    
    # 대체 경로 (PROJECT_ROOT 기준)
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
# 범용 위험 키워드 추출
# =============================================================================
def get_universal_risk_keywords():
    """범용 위험 키워드 추출"""
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

# =============================================================================
# 복합 패턴 추출
# =============================================================================
def get_combined_patterns():
    """복합 패턴 위험도 추출"""
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
    """v2 패턴 기반으로 가장 적합한 조항 판별"""
    scores = {}
    
    # '제N조' 형식의 모든 키를 동적으로 가져옴
    article_keys = [k for k in PATTERNS_V2.keys() if k.startswith('제') and k.endswith('조')]
    if not article_keys:
        # '제N조' 키가 없는 경우, 기본 조항 목록 사용
        article_keys = ['제6조', '제7조', '제8조', '제9조', '제10조', '제11조', '제12조', '제13조', '제14조']

    for article_id in article_keys:
        if article_id not in PATTERNS_V2:
            continue
        
        article_data = PATTERNS_V2[article_id]
        score = 0.0
        
        # 패턴별 점수 계산
        for pattern in article_data.get('patterns', []):
            keywords = pattern.get('keywords', [])
            high_risk = pattern.get('high_risk_keywords', [])
            
            # 기본 키워드 매칭
            keyword_matches = sum(1 for kw in keywords if kw in text)
            score += keyword_matches * 0.1
            
            # 고위험 키워드 매칭 (가중치 높임)
            high_risk_matches = sum(1 for kw in high_risk if kw in text)
            score += high_risk_matches * 0.3
        
        scores[article_id] = min(score, 1.0)
    
    if not scores:
        return "제7조", 0.0 # 기본값
    
    best = max(scores.items(), key=lambda x: x[1])
    return best[0], best[1]

# =============================================================================
# 범용 위험 키워드 체크 (변경 없음)
# =============================================================================
def check_universal_risks(text: str) -> List[Dict]:
    """범용 위험 키워드 체크"""
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

# =============================================================================
# 복합 패턴 체크 (변경 없음)
# =============================================================================
def check_combined_patterns(text: str) -> List[Dict]:
    """복합 패턴 위험도 체크"""
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
# =============================================================================
def query_violation_cases(conn: Neo4jConnector, article_id: str, limit: int = 10) -> List[Dict]:
    """조항별 위반 사례 검색 (위험도 포함)"""
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
# 유사도 기반 상위 N개 사례 검색 (변경 없음)
# =============================================================================
def find_top_similar_cases(user_text: str, cases: List[Dict], top_k: int = 5) -> List[Dict]:
    """유사도 기반 상위 K개 사례 반환"""
    if not cases or MODEL is None:
        return []
    
    user_emb = MODEL.encode(user_text)
    scored_cases = []
    
    for case in cases:
        # Semantic similarity
        viol_emb = case.get('violation_embedding')
        if viol_emb:
            viol_emb = np.array(viol_emb)
        else:
            viol_emb = MODEL.encode(case.get('violation_text', ''))
        
        sim_semantic = cosine_similarity(user_emb, viol_emb)
        
        # Lexical similarity
        sim_lexical = lexical_jaccard(user_text, case.get('violation_text', ''))
        
        # Combined score
        final_sim = (sim_semantic * 0.7) + (sim_lexical * 0.3)
        
        case_copy = case.copy()
        case_copy['similarity'] = float(final_sim)
        scored_cases.append(case_copy)
    
    # Sort by similarity
    scored_cases.sort(key=lambda x: x['similarity'], reverse=True)
    
    return scored_cases[:top_k]

# =============================================================================
# RAG 컨텍스트 구성 (변경 없음)
# =============================================================================
def build_rag_context(user_text: str, article_id: str, top_cases: List[Dict], 
                      universal_risks: List[Dict], combined_risks: List[Dict]) -> str:
    """RAG를 위한 컨텍스트 구성 (LLM에게 전달할 근거 자료)"""
    
    context_parts = []
    
    # 1. 판별된 조항 정보
    article_info = PATTERNS_V2.get(article_id, {})
    context_parts.append(f"## 1. 판별된 관련 조항: {article_id} - {article_info.get('title', '')}")
    
    # 2. 검색된 유사 위반 사례
    if top_cases:
        context_parts.append("\n## 2. 근거가 되는 유사 위반 사례 (DB 검색 결과):\n")
        for i, case in enumerate(top_cases, 1):
            risk_emoji = {'critical': '⚫', 'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(
                case.get('risk_level', 'medium'), '⚪'
            )
            context_parts.append(f"### 사례 {i} (유사도: {case['similarity']:.3f}, 위험도: {risk_emoji} {case.get('risk_level', 'medium')})")
            context_parts.append(f"**- 불공정 원문**: {case.get('violation_text', '')[:200]}...")
            context_parts.append(f"**- 시정 요청 사유**: {case.get('reason', '')[:200]}...")
            if case.get('correction_text'):
                context_parts.append(f"**- 수정 후 약관**: {case.get('correction_text', '')[:200]}...")
            context_parts.append("")
    else:
        context_parts.append("\n## 2. 근거가 되는 유사 위반 사례를 찾을 수 없습니다.\n")
    
    # 3. 범용 위험 키워드
    if universal_risks:
        context_parts.append("## 3. 발견된 범용 위험 키워드 (규칙 기반):\n")
        for risk in universal_risks:
            risk_emoji = {'critical': '⚫', 'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(
                risk['risk_level'], '⚪'
            )
            context_parts.append(f"- {risk_emoji} **{risk['keyword']}**: {risk['description']}")
        context_parts.append("")
    
    # 4. 복합 패턴
    if combined_risks:
        context_parts.append("## 4. 발견된 복합 패턴 (규칙 기반, 치명적):\n")
        for pattern in combined_risks:
            context_parts.append(f"- ⚫ **{' + '.join(pattern['combination'])}**: {pattern['description']}")
        context_parts.append("")
    
    # 5. 검토 대상 약관 원문
    context_parts.append("\n## 5. 검토 대상 약관 원문:\n")
    context_parts.append(f"```\n{user_text}\n```\n")
    
    return "\n".join(context_parts)

# =============================================================================
# (신규) 결정론적 점수 계산 로직 (논문 핵심)
# =============================================================================
def calculate_deterministic_score(article_score: float, universal_risks: List[Dict], 
                                  combined_risks: List[Dict], top_cases: List[Dict]) -> Dict:
    """
    LLM 없이, 수집된 데이터를 기반으로 불공정 점수와 위험도를 결정합니다.
    이 함수의 가중치와 로직이 바로 '내가 만든 모델' (논문 핵심)이 됩니다.
    """
    
    # === 논문에서 중요하게 다룰 가중치 설정 ===
    # 이 가중치를 조절하고 그 근거를 마련하는 것이 종합설계의 핵심입니다.
    WEIGHTS = {
        'article_match': 0.1,  # 조항 매칭 점수는 위험도에 약간의 영향
        'universal_risk': {'critical': 0.4, 'high': 0.25, 'medium': 0.1, 'low': 0.05},
        'combined_pattern': 0.5, # 복합 패턴은 매우 강력한 증거로 간주
        'top_case_similarity': 0.4 # 가장 유사한 사례의 유사도
    }
    # ============================================
    
    final_score = 0.0
    reasoning = [] # 점수 계산 근거 (디버깅용)

    # 1. 조항 매칭 점수 반영
    final_score += article_score * WEIGHTS['article_match']
    reasoning.append(f"조항 매칭 점수 기여: {article_score * WEIGHTS['article_match']:.2f}")

    # 2. 범용 위험 키워드 점수 추가 (가장 높은 위험도 1개만 반영 또는 합산 - 여기서는 합산)
    universal_score = 0.0
    for risk in universal_risks:
        risk_level = risk.get('risk_level', 'low')
        score_to_add = WEIGHTS['universal_risk'].get(risk_level, 0.0)
        universal_score += score_to_add
        reasoning.append(f"범용 위험 키워드 '{risk['keyword']}'({risk_level}) 기여: {score_to_add:.2f}")
    final_score += universal_score
        
    # 3. 복합 패턴 점수 추가 (발견 시 큰 점수 부여)
    combined_score = 0.0
    if combined_risks:
        combined_score = len(combined_risks) * WEIGHTS['combined_pattern']
        reasoning.append(f"복합 패턴 {len(combined_risks)}개 발견 기여: {combined_score:.2f}")
    final_score += combined_score
        
    # 4. 가장 유사한 사례의 유사도 점수 반영
    top_similarity = 0.0
    if top_cases:
        top_similarity = top_cases[0].get('similarity', 0.0)
        similarity_score = top_similarity * WEIGHTS['top_case_similarity']
        # 유사도가 0.9 이상이면 추가 가점 (예시)
        if top_similarity >= 0.9:
            similarity_score += 0.1
        final_score += similarity_score
        reasoning.append(f"최고 유사사례(유사도 {top_similarity:.2f}) 기여: {similarity_score:.2f}")
    
    # 최종 점수는 1.0을 넘지 않도록 조정
    final_score = min(final_score, 1.0)
    
    # 점수에 따른 위험도 및 위반 여부 결정 (이 기준 또한 논문의 핵심)
    severity = 'low'
    violation = False
    if final_score >= 0.8:
        severity = 'critical'
        violation = True
    elif final_score >= 0.6:
        severity = 'high'
        violation = True
    elif final_score >= 0.4:
        severity = 'medium'
        violation = True # medium도 위반으로 간주
    elif final_score > 0.1: # 0.1 초과 low
        severity = 'low'
        violation = False
    else: # 0.1 이하 none
        severity = 'none' # 'low'보다 낮은 'none' 레벨
        violation = False

    print(f"📈 [자체 로직 점수 계산] 최종 점수: {final_score:.3f}, 위험도: {severity}")
    for r in reasoning:
        print(f"  - {r}")

    return {
        "score": float(final_score),
        "severity": severity,
        "violation": violation
    }

# =============================================================================
# (수정) LLM에 '설명' 요청
# =============================================================================
def ask_llm_explanation(context: str, user_text: str, score: float, severity: str) -> Dict:
    """
    LLM에게 '왜 그런 판단이 나왔는지' 설명을 요청합니다. (역할 변경)
    LLM은 판단하지 않고, 주어진 점수와 근거를 바탕으로 설명과 제안만 생성합니다.
    """
    
    if LLM_CLIENT is None:
        return {
            'explanation': 'LLM이 연결되지 않았습니다. 환경변수를 확인하세요.',
            'suggestion': 'LLM이 연결되지 않아 수정 제안을 생성할 수 없습니다.'
        }
    
    system_prompt = """당신은 주어진 분석 데이터를 바탕으로 법률 보고서를 작성하는 유능한 법률 비서입니다.
'판단'은 이미 시스템 로직에 의해 완료되었습니다. 당신의 역할은 그 판단이 왜 타당한지 근거를 들어 논리적으로 설명하고, 명확한 수정안을 제시하는 것입니다.
절대 스스로 판단을 내리거나 점수를 바꾸지 마세요. 주어진 결과를 존중하고 설명에만 집중하세요.

응답 형식 (JSON):
{
    "explanation": "판단 근거에 대한 상세하고 논리적인 설명. (예: '시스템 분석 결과, 본 조항은 {severity} 위험으로 판단되었습니다. 주된 이유는...')",
    "suggestion": "고객에게 유리하고 법적으로 안전한 구체적인 수정 제안. (수정이 필요 없다면 '현재 조항은 공정하며 수정이 필요하지 않습니다.'라고 명시)"
}"""
    
    user_prompt = f"""## 1. 시스템 분석 근거 자료
{context}

## 2. 시스템 최종 분석 결과 (자체 로직이 판단 완료)
- **검토 대상 약관**: "{user_text}"
- **자체 로직 판단 점수**: {score:.2f} / 1.0
- **자체 로직 판단 위험도**: '{severity}'

## 3. 당신의 임무
위 '1. 시스템 분석 근거 자료'를 바탕으로, '2. 시스템 최종 분석 결과'가 왜 타당한지 상세하고 논리적으로 설명해주세요.
(예: '유사 위반 사례 ...와 유사성이 높고, 치명적인 복합 패턴 ...가 발견되어 {severity}로 판단되었습니다.')
그리고 고객에게 실질적인 도움이 될 수 있도록 법적으로 안전한 수정안을 제시해주세요.

반드시 JSON 형식으로만 답변해주세요."""
    
    try:
        if LLM_PROVIDER == "anthropic":
            response = LLM_CLIENT.messages.create(
                model=LLM_MODEL,
                max_tokens=2000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            result_text = response.content[0].text
        else:  # openai
            response = LLM_CLIENT.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,
                response_format={"type": "json_object"} # OpenAI JSON 모드 활용
            )
            result_text = response.choices[0].message.content
        
        # JSON 파싱
        # 코드 블록 제거 (Anthropic은 가끔 코드 블록을 포함)
        result_text = re.sub(r'```json\n', '', result_text)
        result_text = re.sub(r'```', '', result_text)
        result = json.loads(result_text.strip())
        
        # 반환되는 필드가 explanation과 suggestion만 있도록 보장
        return {
            'explanation': result.get('explanation', '설명 생성에 실패했습니다.'),
            'suggestion': result.get('suggestion', '수정 제안 생성에 실패했습니다.')
        }
        
    except Exception as e:
        print(f"⚠️  LLM 설명 생성 실패: {e}")
        return {
            'explanation': f'LLM 설명 생성 중 오류 발생: {str(e)}',
            'suggestion': '오류로 인해 수정 제안을 생성할 수 없습니다.'
        }

# =============================================================================
# (수정) 메인 판단 로직 (GraphRAG + 자체 로직)
# =============================================================================
def comprehensive_judgment(user_text: str, conn: Neo4jConnector) -> Dict:
    """GraphRAG 기반 종합 판단 (결정론적 점수 계산 + LLM 설명 생성)"""
    
    # --- 1단계 ~ 5단계: RAG로 근거 데이터 수집 ---
    # 1. 조항 판별
    article_id, article_score = detect_best_article(user_text)
    print(f"📍 [1/5] 판별된 조항: {article_id} (점수: {article_score:.3f})")
    
    # 2. 범용 위험 키워드 체크
    universal_risks = check_universal_risks(user_text)
    print(f"🔍 [2/5] 범용 위험 키워드: {len(universal_risks)}개 발견")
    
    # 3. 복합 패턴 체크
    combined_risks = check_combined_patterns(user_text)
    print(f"⚠️  [3/5] 복합 패턴: {len(combined_risks)}개 발견")
    
    # 4. DB에서 유사 사례 검색
    all_cases = query_violation_cases(conn, article_id, limit=50)
    print(f"📊 [4/5] 검색된 사례: {len(all_cases)}개")
    
    # 5. 상위 유사 사례 추출
    top_cases = find_top_similar_cases(user_text, all_cases, top_k=5)
    print(f"🎯 [5/5] 상위 유사 사례: {len(top_cases)}개")
    
    # --- 6단계: 자체 로직으로 최종 '판단' ---
    print("\n📈 [6/8] 자체 로직으로 점수 '판단' 중...")
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
    print("\n🤖 [7/8] LLM '설명' 및 '제안' 생성 요청 중...")
    # 판단이 끝난 결과를 LLM에게 전달하여 설명을 요청
    llm_explanation = ask_llm_explanation(rag_context, user_text, final_score, final_severity)
    
    # --- 9단계: 최종 결과 구성 ---
    print("\n📋 [8/8] 최종 결과 구성 중...")
    result = {
        # --- '자체 로직'이 결정한 판단 결과 ---
        "violation": final_violation,
        "score": final_score,
        "severity": final_severity,
        "confidence": 1.0, # 결정론적 로직이므로 확신도는 1.0

        # --- 'LLM'이 생성한 서술형 결과 ---
        "explanation": llm_explanation.get('explanation'),
        "suggestion": llm_explanation.get('suggestion'),

        # --- 판단의 근거가 된 데이터 ---
        "article_id": article_id,
        "top_cases": [
            {
                "id": case.get('violation_id'),
                "similarity": case.get('similarity', 0.0),
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
            "top_similarity": top_cases[0]['similarity'] if top_cases else 0.0,
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
        print("사용법: python scripts/judge_clause.py '검사할 문장'")
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
