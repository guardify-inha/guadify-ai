"""
약관 불공정성 판단 - 다층 유사도 기반 대조 학습 + 규칙/하드룰 보정
- 핵심 변경:
  1) "어떠한 경우에도 책임을 지지 않습니다" 같은 전면 면책 표현은 즉시 위반으로 처리
  2) 다층 유사도(semantic + lexical + context)로 유사도 계산
  3) 근거 조문/사례(또는 규칙 기반 스니펫)를 항상 반환하도록 보완
  4) 판단 임계값: 0.6
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
# 임베딩 모델 로드
# =============================================================================
try:
    from sentence_transformers import SentenceTransformer
    MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
except Exception as e:
    print(f"Warning: 임베딩 모델 없음: {e}")
    MODEL = None

# =============================================================================
# 조항별 키워드 및 패턴
# =============================================================================
ARTICLE_PATTERNS = {
    "제6조": {"keywords": ["부당", "불리", "예상", "어려운", "본질", "권리", "제한"],
             "patterns": [r"부당.*불리", r"예상.*어려운", r"본질.*권리.*제한"]},
    "제7조": {"keywords": ["면책", "책임", "배상", "배제", "제한", "담보", "고의", "과실", "손해", "피해"],
             "patterns": [r"책임.*없", r"면책", r"배상.*않", r"손해.*책임.*없",
                          r"어떠한.*책임.*지지.*않", r"어떠한.*경우.*책임.*없"]},
    "제8조": {"keywords": ["손해금", "지연", "위약금", "배상"],
             "patterns": [r"손해금", r"지연.*배상", r"위약금"]},
    "제9조": {"keywords": ["해제", "해지", "원상회복", "존속"],
             "patterns": [r"해제", r"해지", r"원상회복", r"존속기간"]},
    "제10조": {"keywords": ["급부", "변경", "일방적", "중지"],
              "patterns": [r"일방적.*변경", r"급부.*변경", r"중지"]},
    "제11조": {"keywords": ["기한", "이익", "박탈", "상실", "항변권", "상계권"],
              "patterns": [r"기한.*이익", r"항변권", r"상계권"]},
    "제12조": {"keywords": ["의사표시", "간주", "의제"],
              "patterns": [r"간주", r"의제", r"의사표시"]},
    "제13조": {"keywords": ["대리인", "책임"],
              "patterns": [r"대리인.*책임"]},
    "제14조": {"keywords": ["소송", "관할", "입증"],
              "patterns": [r"소송", r"관할", r"입증"]}
}

# 하드 룰(명백한 위반으로 간주)
HARD_VIOLATION_PATTERNS = [
    r"어떠한\s*경우에도\s*책임\s*을?\s*지지\s*않습니다?",
    r"어떠한\s*경우에도\s*책임\s*지지\s*않습니다?",
    r"어떠한.*경우.*책임.*지지.*않",
    r"일체.*책임.*지지.*않",
    r"어떠한.*책임.*도.*지지.*않"
]

# =============================================================================
# 도우미: 유사도 계산(semantic + lexical)
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
# 조항 판별
# =============================================================================
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
# Neo4j 조회
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
# 다층 유사도 계산: semantic + lexical + context smoothing
# =============================================================================
def calculate_multilayer_similarity(user_text: str, case: Dict) -> float:
    if MODEL is None:
        return 0.0

    # texts
    viol_text = case.get('violation_text', '') or ''
    # semantic embeddings
    user_emb = MODEL.encode(user_text)
    viol_emb = np.array(case['violation_embedding']) if case.get('violation_embedding') else MODEL.encode(viol_text)

    sim_semantic = cosine_similarity(user_emb, viol_emb)
    sim_lexical = lexical_jaccard(user_text, viol_text)

    # 문장단위 context smoothing: 각 문장-문장 유사도에서 최댓값 평균 사용
    segs_user = [s.strip() for s in re.split(r'[.?!\n]', user_text) if s.strip()]
    segs_viol = [s.strip() for s in re.split(r'[.?!\n]', viol_text) if s.strip()]
    seg_sims = []
    if segs_user and segs_viol:
        # pre-encode shorter segments for speed/consistency
        emb_viol_segs = [MODEL.encode(s) for s in segs_viol]
        for s in segs_user:
            emb_s = MODEL.encode(s)
            sims = [cosine_similarity(emb_s, ev) for ev in emb_viol_segs]
            if sims:
                seg_sims.append(max(sims))
    context_sim = float(np.mean(seg_sims)) if seg_sims else sim_semantic

    # 가중합: semantic 중심 + lexical/context 보정
    final_sim = (sim_semantic * 0.6) + (sim_lexical * 0.2) + (context_sim * 0.2)
    # 안전 범위
    return float(min(max(final_sim, 0.0), 1.0))

def find_most_similar_violation(user_text: str, cases: List[Dict]) -> Optional[Dict]:
    if not cases:
        return None
    best_case, best_score = None, -1.0
    for case in cases:
        sim = calculate_multilayer_similarity(user_text, case)
        if sim > best_score:
            best_case = case.copy()
            best_score = sim
    if best_case is not None:
        best_case['similarity'] = best_score
    return best_case

# =============================================================================
# 대조 점수 (contrastive) 계산 개선
# =============================================================================
def calculate_contrastive_score(user_text: str, best_case: Dict) -> float:
    """0 -> 불공정에 가깝다, 1 -> 수정본에 가깝다"""
    if not best_case or MODEL is None:
        return 0.5

    user_emb = MODEL.encode(user_text)

    if best_case.get('violation_embedding'):
        viol_emb = np.array(best_case['violation_embedding'])
    else:
        viol_emb = MODEL.encode(best_case.get('violation_text', ''))
    if best_case.get('correction_embedding'):
        corr_emb = np.array(best_case['correction_embedding'])
    else:
        # 빈 수정본이면 아주 멀리 있다고 가정(so bias toward violation)
        corr_text = best_case.get('correction_text', '')
        corr_emb = MODEL.encode(corr_text) if corr_text else None

    dist_viol = 1 - cosine_similarity(user_emb, viol_emb)
    dist_corr = 1 - cosine_similarity(user_emb, corr_emb) if corr_emb is not None else 1.0

    total = dist_viol + dist_corr
    if total == 0:
        return 0.5
    position = dist_viol / total
    return float(min(max(position, 0.0), 1.0))

# =============================================================================
# 규칙/하드룰 보정 도구
# =============================================================================
NEGATION_WORDS = ["않", "없", "아니", "못", "금지", "면책", "배제", "제외"]

def detect_negation_count(text: str) -> int:
    return sum(1 for neg in NEGATION_WORDS if neg in text)

def detect_hard_violation(text: str) -> Optional[re.Match]:
    """명백한 전면 면책 표현 매칭(하드룰)"""
    for pat in HARD_VIOLATION_PATTERNS:
        m = re.search(pat, text)
        if m:
            return m
    return None

# =============================================================================
# 종합 판단 (메인 로직)
# =============================================================================
THRESHOLD = 0.6  # 요청대로 0.6

def comprehensive_judgment(user_text: str, conn: Neo4jConnector) -> Dict:
    article_id = detect_best_article(user_text)

    # 1) 하드룰 먼저 체크 (명백한 전면 면책 표현)
    hard_match = detect_hard_violation(user_text)
    hard_snippet = hard_match.group(0) if hard_match else None

    # 2) DB에서 위반 사례 불러오기
    cases = query_violation_cases(conn, article_id)

    # 3) 가장 유사한 사례 찾기 (있다면)
    best_case = find_most_similar_violation(user_text, cases) if cases else None

    # 4) contrastive
    contrastive = calculate_contrastive_score(user_text, best_case) if best_case else 0.5
    unfairness = 1 - contrastive
    base_sim = best_case['similarity'] if best_case else 0.0

    # 5) 기본 final score (embedding 기반)
    final_score = (unfairness * 0.7) + (base_sim * 0.3)
    final_score = float(min(max(final_score, 0.0), 1.0))

    # 6) 규칙 기반 강화
    neg_count = detect_negation_count(user_text)
    if neg_count >= 2:
        final_score = min(final_score * 1.2, 1.0)

    # 하드룰이 존재하면 강제 위반(우선)
    if hard_match:
        final_score = max(final_score, 0.9)

    # 추가 패턴(제7조 전형적 면책 표현) 발견 시 상향
    if any(re.search(pat, user_text) for pat in ARTICLE_PATTERNS.get("제7조", {}).get("patterns", [])):
        final_score = max(final_score, 0.8)

    # 7) 판단 및 심각도
    violation = final_score > THRESHOLD
    if final_score > 0.8:
        severity = "높음"
    elif final_score > 0.6:
        severity = "중간"
    else:
        severity = "낮음"

    # 8) 근거(근거조문/사례) 구성
    top_reasons = []
    if best_case:
        top_reasons.append({
            "level": "위반사례",
            "id": best_case.get('violation_id') or best_case.get('id'),
            "article_id": article_id,
            "snippet": (best_case.get('violation_text') or best_case.get('text') or '')[:300],
            "score": float(final_score)
        })
    # DB 근거가 없지만 하드룰이 잡혔을 때 - 규칙 기반 근거 반환
    if not top_reasons and hard_match:
        top_reasons.append({
            "level": "규칙기반_근거",
            "id": None,
            "article_id": article_id,
            "snippet": hard_snippet,
            "score": float(final_score),
            "note": "명백한 전면 면책 표현(hard rule)"
        })
    # 완전히 근거가 없을 때 (DB 없음, 하드룰 없음)에는 규칙 기반 유추 스니펫 제공
    if not top_reasons:
        # 간단한 추출형 스니펫: 약관 중 부정어 포함 문장 반환
        candidate_snippet = user_text[:300]
        top_reasons.append({
            "level": "추측근거",
            "id": None,
            "article_id": article_id,
            "snippet": candidate_snippet,
            "score": float(final_score),
            "note": "유사 위반사례가 DB에 없으므로 입력문장에서 추출한 후보"
        })

    # 9) 설명 텍스트
    if violation:
        explanation = f"입력문구는 약관법상 불공정한 표현으로 판단됩니다. (유사도: {base_sim:.2f}, 불공정도: {final_score:.2f})"
    else:
        explanation = f"현재 DB/규칙 기준으로는 불공정 여부가 명확하지 않습니다. (유사도: {base_sim:.2f}, 불공정도: {final_score:.2f})"

    # 10) 결과 반환
    return {
        "violation": violation,
        "score": float(final_score),
        "severity": severity,
        "article_id": article_id,
        "explanation": explanation,
        "suggestion": get_suggestion(article_id),
        "top_reasons": top_reasons,
        "debug": {
            "base_similarity": float(base_sim),
            "contrastive": float(contrastive),
            "unfairness": float(unfairness),
            "negation_count": neg_count,
            "hard_rule_matched": bool(hard_match),
        }
    }

# =============================================================================
# 제안 텍스트
# =============================================================================
def get_suggestion(article_id: str) -> str:
    suggestions = {
        "제6조": "고객에게 부당하게 불리한 조항을 삭제하세요.",
        "제7조": "전면 면책 표현을 삭제하고, 고의·중과실 책임을 명시하세요.",
        "제8조": "손해배상액을 과도하게 높게 설정하지 마세요.",
        "제9조": "고객의 해제·해지권을 보장하세요.",
        "제10조": "급부 내용의 일방적 변경을 제한하세요.",
        "제11조": "고객의 항변권, 상계권을 부당하게 제한하지 마세요.",
        "제12조": "의사표시 의제 조항을 신중하게 작성하세요.",
        "제13조": "대리인에게 과도한 책임을 전가하지 마세요.",
        "제14조": "소송 제기를 금지하거나 입증책임을 전가하지 마세요."
    }
    return suggestions.get(article_id, "약관법 취지에 맞게 수정하세요.")

# =============================================================================
# 실행 엔트리
# =============================================================================
def run(user_text: str, article_id: str = None) -> Dict:
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
    result = run(text)
    print(json.dumps(result, ensure_ascii=False, indent=2))
