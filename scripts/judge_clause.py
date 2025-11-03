"""
약관 불공정성 판단 - GraphRAG 기반 (v2 패턴 반영 + LLM 통합)
- 핵심 변경:
  1) patterns_by_article_v2.json 반영 (high_risk_keywords, universal_risk_keywords, combined_patterns)
  2) 진정한 RAG: 검색 → 컨텍스트 구성 → LLM에 전달 → 최종 판단
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

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

from database.neo4j_connector import Neo4jConnector

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
        LLM_MODEL = "claude-3-5-sonnet-20241022"
        print(f"✓ LLM 클라이언트 로드: {LLM_PROVIDER} ({LLM_MODEL})")
    except Exception as e:
        print(f"⚠️  Anthropic 클라이언트 로드 실패: {e}")
        LLM_CLIENT = None
else:
    try:
        import openai
        LLM_CLIENT = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        LLM_MODEL = "gpt-4-turbo-preview"
        print(f"✓ LLM 클라이언트 로드: {LLM_PROVIDER} ({LLM_MODEL})")
    except Exception as e:
        print(f"⚠️  OpenAI 클라이언트 로드 실패: {e}")
        LLM_CLIENT = None

# =============================================================================
# v2 패턴 로드
# =============================================================================
def load_patterns_v2():
    """patterns_by_article_v2.json 로드"""
    pattern_file = Path(PROJECT_ROOT) / 'data' / 'contracts' / 'reference' / 'patterns_by_article_v2.json'
    try:
        with open(pattern_file, 'r', encoding='utf-8') as f:
            patterns = json.load(f)
            print(f"✓ v2 패턴 파일 로드: {pattern_file.name}")
            return patterns
    except Exception as e:
        print(f"⚠️  v2 패턴 로드 실패: {e}")
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
# 유사도 계산
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
    set1 = set([t for t in re.findall(r'\w+', s1)])
    set2 = set([t for t in re.findall(r'\w+', s2)])
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

# =============================================================================
# 조항 판별 (v2 패턴 기반)
# =============================================================================
def detect_best_article(text: str) -> Tuple[str, float]:
    """v2 패턴 기반으로 가장 적합한 조항 판별"""
    scores = {}
    
    for article_id in ['제6조', '제7조', '제8조', '제9조', '제10조', '제11조', '제12조', '제13조', '제14조']:
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
        return "제7조", 0.0
    
    best = max(scores.items(), key=lambda x: x[1])
    return best[0], best[1]

# =============================================================================
# 범용 위험 키워드 체크
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
# 복합 패턴 체크
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
# Neo4j에서 위반 사례 검색
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
# 유사도 기반 상위 N개 사례 검색
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
# RAG 컨텍스트 구성
# =============================================================================
def build_rag_context(user_text: str, article_id: str, top_cases: List[Dict], 
                      universal_risks: List[Dict], combined_risks: List[Dict]) -> str:
    """RAG를 위한 컨텍스트 구성"""
    
    context_parts = []
    
    # 1. 판별된 조항 정보
    article_info = PATTERNS_V2.get(article_id, {})
    context_parts.append(f"## 판별된 조항: {article_id} - {article_info.get('title', '')}")
    context_parts.append(f"발견된 사례 수: {article_info.get('case_count', 0)}개\n")
    
    # 2. 검색된 유사 위반 사례
    if top_cases:
        context_parts.append("## 유사한 위반 사례 (상위 5개):\n")
        for i, case in enumerate(top_cases, 1):
            risk_emoji = {'critical': '⚫', 'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(
                case.get('risk_level', 'medium'), '⚪'
            )
            context_parts.append(f"### 사례 {i} (유사도: {case['similarity']:.3f}, 위험도: {risk_emoji} {case.get('risk_level', 'medium')})")
            context_parts.append(f"**불공정 원문**: {case.get('violation_text', '')[:200]}...")
            context_parts.append(f"**시정 요청 사유**: {case.get('reason', '')[:200]}...")
            if case.get('correction_text'):
                context_parts.append(f"**수정 후 약관**: {case.get('correction_text', '')[:200]}...")
            context_parts.append("")
    else:
        context_parts.append("## 유사한 위반 사례를 찾을 수 없습니다.\n")
    
    # 3. 범용 위험 키워드
    if universal_risks:
        context_parts.append("## 발견된 범용 위험 키워드:\n")
        for risk in universal_risks:
            risk_emoji = {'critical': '⚫', 'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(
                risk['risk_level'], '⚪'
            )
            context_parts.append(f"- {risk_emoji} **{risk['keyword']}**: {risk['description']}")
        context_parts.append("")
    
    # 4. 복합 패턴
    if combined_risks:
        context_parts.append("## 발견된 복합 패턴 (치명적):\n")
        for pattern in combined_risks:
            context_parts.append(f"- ⚫ **{' + '.join(pattern['combination'])}**: {pattern['description']}")
        context_parts.append("")
    
    # 5. 검토 대상 약관
    context_parts.append("## 검토 대상 약관:\n")
    context_parts.append(f"```\n{user_text}\n```\n")
    
    return "\n".join(context_parts)

# =============================================================================
# LLM에 판단 요청
# =============================================================================
def ask_llm_judgment(context: str, user_text: str) -> Dict:
    """LLM에게 최종 판단 요청"""
    
    if LLM_CLIENT is None:
        return {
            'violation': False,
            'score': 0.5,
            'severity': '불명',
            'explanation': 'LLM이 연결되지 않았습니다. 환경변수를 확인하세요.',
            'suggestion': '',
            'confidence': 0.0
        }
    
    system_prompt = """당신은 약관법 전문가입니다. 
주어진 약관 조항이 불공정한지 판단하고, 근거를 제시해주세요.

판단 기준:
1. 유사한 위반 사례와의 비교
2. 범용 위험 키워드 존재 여부
3. 복합 패턴 (여러 위험 요소가 결합된 경우)
4. 약관법 각 조항의 취지

응답 형식 (JSON):
{
    "violation": true/false,
    "score": 0.0~1.0 (불공정도 점수),
    "severity": "critical/high/medium/low",
    "explanation": "판단 근거 상세 설명",
    "suggestion": "수정 제안",
    "confidence": 0.0~1.0 (판단 확신도)
}"""
    
    user_prompt = f"""{context}

위 정보를 바탕으로 검토 대상 약관의 불공정성을 판단해주세요.

특히 다음을 고려해주세요:
1. 유사 사례와의 일치도
2. 위험 키워드의 심각도
3. 복합 패턴의 존재
4. 고객에게 미치는 실질적 영향

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
                max_tokens=2000
            )
            result_text = response.choices[0].message.content
        
        # JSON 파싱
        # 코드 블록 제거
        result_text = re.sub(r'```json\n', '', result_text)
        result_text = re.sub(r'```', '', result_text)
        result = json.loads(result_text.strip())
        
        return result
        
    except Exception as e:
        print(f"⚠️  LLM 판단 실패: {e}")
        return {
            'violation': False,
            'score': 0.5,
            'severity': '불명',
            'explanation': f'LLM 판단 중 오류 발생: {str(e)}',
            'suggestion': '',
            'confidence': 0.0
        }

# =============================================================================
# 메인 판단 로직 (GraphRAG)
# =============================================================================
def comprehensive_judgment(user_text: str, conn: Neo4jConnector) -> Dict:
    """GraphRAG 기반 종합 판단"""
    
    # 1단계: 조항 판별
    article_id, article_score = detect_best_article(user_text)
    print(f"📍 판별된 조항: {article_id} (점수: {article_score:.3f})")
    
    # 2단계: 범용 위험 키워드 체크
    universal_risks = check_universal_risks(user_text)
    print(f"🔍 범용 위험 키워드: {len(universal_risks)}개 발견")
    
    # 3단계: 복합 패턴 체크
    combined_risks = check_combined_patterns(user_text)
    print(f"⚠️  복합 패턴: {len(combined_risks)}개 발견")
    
    # 4단계: DB에서 유사 사례 검색
    all_cases = query_violation_cases(conn, article_id, limit=50)
    print(f"📊 검색된 사례: {len(all_cases)}개")
    
    # 5단계: 상위 유사 사례 추출
    top_cases = find_top_similar_cases(user_text, all_cases, top_k=5)
    print(f"🎯 상위 유사 사례: {len(top_cases)}개")
    
    # 6단계: RAG 컨텍스트 구성
    rag_context = build_rag_context(user_text, article_id, top_cases, universal_risks, combined_risks)
    
    # 7단계: LLM에 판단 요청
    print("🤖 LLM 판단 요청 중...")
    llm_result = ask_llm_judgment(rag_context, user_text)
    
    # 8단계: 최종 결과 구성
    result = {
        "violation": llm_result.get('violation', False),
        "score": float(llm_result.get('score', 0.5)),
        "severity": llm_result.get('severity', 'medium'),
        "article_id": article_id,
        "explanation": llm_result.get('explanation', ''),
        "suggestion": llm_result.get('suggestion', ''),
        "confidence": float(llm_result.get('confidence', 0.5)),
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
# 실행 엔트리
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