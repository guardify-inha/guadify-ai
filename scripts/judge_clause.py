"""
약관 불공정성 판단 - 다층 유사도 기반 대조 학습 + 규칙 기반 보정
"""
import re
import sys
import json
from typing import List, Dict, Optional
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
# 임베딩 모델
# =============================================================================
try:
    from sentence_transformers import SentenceTransformer
    MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
except Exception as e:
    print(f"Warning: 임베딩 모델 없음: {e}")
    MODEL = None


# =============================================================================
# 조항별 패턴 정의
# =============================================================================
ARTICLE_PATTERNS = {
    "제6조": {"keywords": ["부당", "불리", "예상", "어려운", "본질", "권리", "제한"],
             "patterns": [r"부당.*불리", r"예상.*어려운", r"본질.*권리.*제한"]},
    "제7조": {"keywords": ["면책", "책임", "배상", "배제", "제한", "담보", "고의", "과실", "손해", "피해"],
             "patterns": [r"책임.*없", r"면책", r"배상.*않", r"손해.*책임.*없"]},
    "제8조": {"keywords": ["손해금", "지연", "위약금"], "patterns": [r"손해금", r"지연.*배상", r"위약금"]},
    "제9조": {"keywords": ["해제", "해지", "원상회복"], "patterns": [r"해제", r"해지", r"원상회복"]},
    "제10조": {"keywords": ["급부", "변경", "일방적"], "patterns": [r"일방적.*변경"]},
    "제11조": {"keywords": ["기한", "이익", "박탈", "항변권"], "patterns": [r"기한.*이익", r"항변권"]},
    "제12조": {"keywords": ["의사표시", "간주"], "patterns": [r"간주", r"의제"]},
    "제13조": {"keywords": ["대리인", "책임"], "patterns": [r"대리인.*책임"]},
    "제14조": {"keywords": ["소송", "관할", "입증"], "patterns": [r"소송", r"관할"]},
}

# =============================================================================
# 기본 도우미 함수
# =============================================================================
def cosine_similarity(v1, v2):
    if v1 is None or v2 is None:
        return 0.0
    v1, v2 = np.array(v1), np.array(v2)
    if len(v1) == 0 or len(v2) == 0:
        return 0.0
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def lexical_jaccard(s1, s2):
    set1, set2 = set(s1.split()), set(s2.split())
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

def detect_best_article(text: str) -> str:
    scores = {}
    for article_id, info in ARTICLE_PATTERNS.items():
        score = 0.0
        score += sum(1 for kw in info['keywords'] if kw in text) * 0.1
        score += sum(1 for pat in info['patterns'] if re.search(pat, text)) * 0.3
        scores[article_id] = min(score, 1.0)
    best = max(scores.items(), key=lambda x: x[1])
    return best[0] if best[1] >= 0.2 else "제7조"


# =============================================================================
# DB 조회
# =============================================================================
def query_violation_cases(conn: Neo4jConnector, article_id: str) -> List[Dict]:
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
           v.embedding as violation_embedding,
           c.corrected_text as correction_text,
           c.embedding as correction_embedding
    LIMIT 50
    """
    return conn.execute_query(query, {"article_id": article_id})


# =============================================================================
# 다층 유사도 계산
# =============================================================================
def calculate_multilayer_similarity(user_text: str, case: Dict) -> float:
    """semantic + lexical + context-level 유사도 종합"""
    if MODEL is None:
        return 0.0
    user_emb = MODEL.encode(user_text)
    viol_text = case.get('violation_text', '')
    corr_text = case.get('correction_text', '')

    viol_emb = np.array(case['violation_embedding']) if case.get('violation_embedding') else MODEL.encode(viol_text)
    sim_semantic = cosine_similarity(user_emb, viol_emb)
    sim_lexical = lexical_jaccard(user_text, viol_text)

    # 문장단위 세그먼트 유사도 (context smoothing)
    segs_user = re.split(r'[.?!]', user_text)
    segs_violation = re.split(r'[.?!]', viol_text)
    seg_sims = []
    for s1 in segs_user:
        if not s1.strip():
            continue
        sims = [cosine_similarity(MODEL.encode(s1), MODEL.encode(s2))
                for s2 in segs_violation if s2.strip()]
        if sims:
            seg_sims.append(max(sims))
    context_sim = np.mean(seg_sims) if seg_sims else sim_semantic

    # 최종 가중합
    return (sim_semantic * 0.6) + (sim_lexical * 0.2) + (context_sim * 0.2)


def find_most_similar_violation(user_text: str, cases: List[Dict]) -> Optional[Dict]:
    if not cases:
        return None
    best_case, best_score = None, -1
    for case in cases:
        sim = calculate_multilayer_similarity(user_text, case)
        if sim > best_score:
            best_case = case
            best_score = sim
    if best_case:
        best_case['similarity'] = best_score
    return best_case


# =============================================================================
# 대조 점수
# =============================================================================
def calculate_contrastive_score(user_text: str, best_case: Dict) -> float:
    if not best_case or MODEL is None:
        return 0.5
    user_emb = MODEL.encode(user_text)
    viol_emb = np.array(best_case['violation_embedding']) if best_case.get('violation_embedding') else MODEL.encode(best_case['violation_text'])
    corr_emb = np.array(best_case['correction_embedding']) if best_case.get('correction_embedding') else MODEL.encode(best_case.get('correction_text', ''))

    dist_viol = 1 - cosine_similarity(user_emb, viol_emb)
    dist_corr = 1 - cosine_similarity(user_emb, corr_emb)
    total = dist_viol + dist_corr
    if total == 0:
        return 0.5
    return dist_viol / total


# =============================================================================
# 규칙 기반 보정
# =============================================================================
NEGATION_WORDS = ["않", "없", "아니", "못", "금지", "면책", "배제"]
VIOLATION_PATTERNS = {
    "제7조": [r"어떠한.*책임.*지지.*않", r"일체.*책임.*지지"]
}

def detect_negation(text: str) -> int:
    return sum(1 for neg in NEGATION_WORDS if neg in text)


# =============================================================================
# 종합 판단
# =============================================================================
def comprehensive_judgment(user_text: str, conn: Neo4jConnector) -> Dict:
    article_id = detect_best_article(user_text)
    cases = query_violation_cases(conn, article_id)
    best_case = find_most_similar_violation(user_text, cases) if cases else None

    if not best_case:
        return {"violation": False, "score": 0.3, "explanation": "유사한 위반사례 없음", "article_id": article_id}

    contrastive = calculate_contrastive_score(user_text, best_case)
    unfairness_score = 1 - contrastive
    sim = best_case['similarity']
    final_score = (unfairness_score * 0.7) + (sim * 0.3)

    # 규칙 기반 강화
    if detect_negation(user_text) >= 2:
        final_score = min(final_score * 1.2, 1.0)
    if any(re.search(p, user_text) for p in VIOLATION_PATTERNS.get(article_id, [])):
        final_score = max(final_score, 0.85)

    violation = final_score > 0.5
    severity = "높음" if final_score > 0.8 else "중간" if final_score > 0.6 else "낮음"

    explanation = (
        f"유사한 위반사례 '{best_case['violation_text'][:40]}...' 탐지. "
        f"(유사도: {sim:.2f}, 불공정도: {final_score:.2f})"
    )

    return {
        "violation": violation,
        "score": final_score,
        "severity": severity,
        "explanation": explanation,
        "article_id": article_id,
        "debug": {"semantic_sim": sim, "contrastive": contrastive}
    }


# =============================================================================
# 실행 함수
# =============================================================================
def run(user_text: str):
    conn = Neo4jConnector()
    try:
        return comprehensive_judgment(user_text, conn)
    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python judge_clause.py '검사할 문장'")
        sys.exit(1)
    text = sys.argv[1]
    result = run(text)
    print(json.dumps(result, ensure_ascii=False, indent=2))
