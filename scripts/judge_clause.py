import re
import sys
import json
from typing import List, Dict, Tuple
from pathlib import Path

# Ensure project root on sys.path and load .env from project root
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

from database.neo4j_connector import Neo4jConnector

# ----------------------------
# 규칙/키워드 사전
# ----------------------------
NEGATION_PATTERNS = [
    r"하지\s*않는다", r"책임\s*없다", r"면책", r"배제", r"제한\s*한다",
    r"어떠한\s*배상도\s*하지\s*않", r"손해배상\s*책임\s*없",
]
RESPONSIBILITY_KEYWORDS = [
    "책임", "손해배상", "배상", "면책", "배제", "제한", "담보책임", "보장",
]
ARTICLE_HINTS = {
    # 약관법 제7조: 면책조항 금지
    "제7조": [
        "면책", "책임 배제", "책임 제한", "손해배상", "담보책임",
        "고의", "중대한 과실", "위험 전가",
    ],
    # 제9조: 해제·해지
    "제9조": ["해제", "해지", "해제권", "해지권", "원상회복", "존속기간", "묵시적", "갱신"],
    # 제10조: 채무 이행
    "제10조": ["급부", "일방적", "변경", "중지", "대행"],
    # 제11조: 권익 보호
    "제11조": ["항변권", "상계권", "기한의 이익", "제3자와 계약", "비밀"],
    # 제12조: 의사표시 의제
    "제12조": ["의사표시", "의제", "간주", "형식", "요건", "도달", "기한"],
    # 제13조: 대리인 책임 가중
    "제13조": ["대리인", "책임", "의무 이행"],
    # 제14조: 소송 제기 금지 등
    "제14조": ["소송", "재판관할", "입증책임"],
}

# 조/항/호 규칙 패턴 (정밀 매칭)
ARTICLE_RULES = {
    "제7조": {
        "본문": [r"면책", r"책임\s*배제", r"책임\s*제한"],
        "제1항": {
            "제1호": [r"고의|중대한\s*과실", r"책임\s*배제"],
            "제2호": [r"손해배상.*제한|제한.*손해배상|배상.*하지\s*않" , r"위험.*고객에게|떠넘기"],
            "제3호": [r"담보책임.*배제|담보책임.*제한|권리행사.*요건.*가중"],
            "제4호": [r"보장된.*내용.*책임.*배제|품질|성능|견본.*책임.*제한"],
        },
    },
    "제9조": {
        "본문": [r"해제|해지"],
        "제1항": {
            "제1호": [r"해제권|해지권.*배제|제한"],
            "제2호": [r"사업자.*해제권|해지권.*부여"],
            "제3호": [r"사업자.*해제권|해지권.*요건.*완화"],
            "제4호": [r"원상회복.*고객.*과중|원상회복.*청구권.*포기"],
            "제5호": [r"사업자.*원상회복.*경감|손해배상.*경감"],
            "제6호": [r"존속기간.*부당|묵시.*연장|갱신"],
        },
    },
    "제10조": {
        "본문": [r"채무|이행|급부"],
        "제1항": {
            "제1호": [r"급부.*일방적.*결정|변경"],
            "제2호": [r"급부.*중지|제3자.*대행"],
        },
    },
    "제11조": {
        "본문": [r"권익"],
        "제1항": {
            "제1호": [r"항변권|상계권.*배제|제한"],
            "제2호": [r"기한의\s*이익.*박탈"],
            "제3호": [r"제3자와\s*계약.*제한"],
            "제4호": [r"비밀.*누설.*허용"],
        },
    },
    "제12조": {
        "본문": [r"의사표시|의제|간주"],
        "제1항": {
            "제1호": [r"작위|부작위.*의사표시.*간주|의제"],
            "제2호": [r"의사표시.*형식|요건.*엄격"],
            "제3호": [r"사업자.*의사표시.*도달.*간주"],
            "제4호": [r"의사표시.*기한.*부당하게.*길게|불확정"],
        },
    },
    "제13조": {
        "본문": [r"대리인"],
        "제1항": {
            "제1호": [r"대리인.*의무.*이행.*책임"],
        },
    },
    "제14조": {
        "본문": [r"소송|재판관할|입증책임"],
        "제1항": {
            "제1호": [r"소송.*금지|재판관할.*합의"],
            "제2호": [r"입증책임.*고객.*부담"],
        },
    },
}

# ----------------------------
# 유틸
# ----------------------------

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def contains_any(text: str, patterns: List[str], regex: bool = True) -> bool:
    if not text:
        return False
    if regex:
        return any(re.search(p, text) for p in patterns)
    return any(p in text for p in patterns)


def count_hits(text: str, keys: List[str]) -> int:
    if not text:
        return 0
    return sum(1 for k in keys if k in text)


# ----------------------------
# Neo4j 검색
# ----------------------------

def query_candidates(conn: Neo4jConnector, user_text: str) -> Dict[str, List[Dict]]:
    """
    입력 문장과 관련 가능성이 높은 조/항/호를 간단한 CONTAINS 기반으로 수집
    반환 형태: {
      "articles": [{id,title,content}],
      "hangs":    [{id,article_id,hang_num,content}],
      "hos":      [{id,hang_id,ho_num,content,article_id,hang_num}]
    }
    """
    text = normalize_text(user_text)

    # 키워드 추출: 규칙/사전 기반 키워드
    base_keys = set(RESPONSIBILITY_KEYWORDS)
    for hints in ARTICLE_HINTS.values():
        base_keys.update(hints)
    # 사용자 문장에 실제 등장하는 키워드만 사용
    keys = [k for k in base_keys if k in text]
    if not keys:
        # 키워드가 하나도 안 잡히면 책임/손해배상 기본 키워드로 최소 탐색
        keys = ["책임", "손해배상", "면책", "배제", "제한"]

    # 동적 WHERE 절
    where_clause = " OR ".join([f"a.title CONTAINS '{k}' OR a.content CONTAINS '{k}'" for k in keys])
    hang_where = " OR ".join([f"h.content CONTAINS '{k}'" for k in keys])
    ho_where = " OR ".join([f"o.content CONTAINS '{k}'" for k in keys])

    # 조 후보
    q_articles = f"""
    MATCH (a:조)
    WHERE {where_clause}
    RETURN a.id AS id, a.title AS title, a.content AS content
    LIMIT 100
    """

    # 항 후보 (조와 연결 정보 포함)
    q_hangs = f"""
    MATCH (a:조)-[:HAS_HANG]->(h:항)
    WHERE {hang_where}
    RETURN a.id AS article_id, h.id AS id, h.hang_num AS hang_num, h.content AS content
    LIMIT 200
    """

    # 호 후보 (조-항-호 구조 + 조-호 직접 연결 구조 모두 포함)
    q_hos = f"""
    MATCH (o:호)
    WHERE {ho_where}
    OPTIONAL MATCH (o)<-[:HAS_HO]-(h:항)<-[:HAS_HANG]-(a_from_hang:조)
    OPTIONAL MATCH (o)<-[:HAS_HO]-(a_direct:조)
    RETURN COALESCE(a_from_hang.id, a_direct.id) AS article_id,
           h.id AS hang_id, h.hang_num AS hang_num,
           o.id AS id, o.ho_num AS ho_num, o.content AS content
    LIMIT 300
    """

    result_hos = conn.execute_query(q_hos)
    
    return {
        "articles": conn.execute_query(q_articles),
        "hangs": conn.execute_query(q_hangs),
        "hos": result_hos if result_hos else [],
    }


# ----------------------------
# 점수화 (규칙 우선)
# ----------------------------

def rule_match_score(user_text: str, article_id: str, hang_num: str = None, ho_num: str = None) -> float:
    text = normalize_text(user_text)
    rules = ARTICLE_RULES.get(article_id, {})
    score = 0.0

    # 조 본문 규칙
    for pat in rules.get("본문", []):
        if re.search(pat, text):
            score += 2.0

    # 항/호 규칙
    if hang_num and isinstance(rules.get("제1항"), dict):
        ho_rules = rules.get("제1항", {}).get(ho_num, [])
        for pat in ho_rules:
            if re.search(pat, text):
                score += 3.0

    return score


def score_article(user_text: str, article: Dict) -> float:
    title = normalize_text(article.get("title"))
    content = normalize_text(article.get("content"))
    aid = article.get("id")

    score = 0.0
    # 규칙 기반 점수(조 본문)
    score += rule_match_score(user_text, aid)
    # 약한 키워드 가점
    score += 0.5 * count_hits(title + " " + content, ARTICLE_HINTS.get(aid, []))
    return score


def score_hang(user_text: str, hang: Dict) -> float:
    content = normalize_text(hang.get("content"))
    aid = hang.get("article_id")
    hnum = hang.get("hang_num")

    score = 0.0
    score += rule_match_score(user_text, aid, hnum)
    # 약한 키워드 가점
    score += 0.5 * count_hits(content, ARTICLE_HINTS.get(aid, []))
    return score


def score_ho(user_text: str, ho: Dict) -> float:
    content = normalize_text(ho.get("content"))
    aid = ho.get("article_id")
    hnum = ho.get("hang_num")
    honum = ho.get("ho_num")

    score = 0.0
    score += rule_match_score(user_text, aid, hnum, honum)
    # 약한 키워드 가점
    score += 0.5 * count_hits(content, ARTICLE_HINTS.get(aid, []))
    return score


def rank_results(user_text: str, cands: Dict[str, List[Dict]]) -> Dict[str, List[Tuple[float, Dict]]]:
    ranked = {
        "articles": [],
        "hangs": [],
        "hos": [],
    }
    for a in cands.get("articles", []):
        ranked["articles"].append((score_article(user_text, a), a))
    for h in cands.get("hangs", []):
        ranked["hangs"].append((score_hang(user_text, h), h))
    for o in cands.get("hos", []):
        ranked["hos"].append((score_ho(user_text, o), o))

    for k in ranked:
        ranked[k].sort(key=lambda x: x[0], reverse=True)
    return ranked


# ----------------------------
# 판단/설명/수정안 생성
# ----------------------------

def decide_and_explain(user_text: str, ranked: Dict[str, List[Tuple[float, Dict]]]) -> Dict:
    # 빈 리스트 체크 후 기본값 제공
    articles_list = ranked.get("articles", [])
    hangs_list = ranked.get("hangs", [])
    hos_list = ranked.get("hos", [])
    
    top_article = (articles_list[0] if articles_list else (0.0, {}))
    top_hang = (hangs_list[0] if hangs_list else (0.0, {}))
    top_ho = (hos_list[0] if hos_list else (0.0, {}))

    # 규칙 기반 점수 최대값 (호>항>조)
    final_score = max(top_ho[0], top_hang[0] * 0.9, top_article[0] * 0.8)

    reasons = []
    if top_ho[0] > 0 and top_ho[1]:
        reasons.append({
            "level": "호",
            "id": top_ho[1].get("id"),
            "article_id": top_ho[1].get("article_id"),
            "hang_id": top_ho[1].get("hang_id"),
            "ho_num": top_ho[1].get("ho_num"),
            "snippet": top_ho[1].get("content"),
            "score": top_ho[0],
        })
    if top_hang[0] > 0 and top_hang[1]:
        reasons.append({
            "level": "항",
            "id": top_hang[1].get("id"),
            "article_id": top_hang[1].get("article_id"),
            "hang_num": top_hang[1].get("hang_num"),
            "snippet": top_hang[1].get("content"),
            "score": top_hang[0],
        })
    if top_article[0] > 0 and top_article[1]:
        reasons.append({
            "level": "조",
            "id": top_article[1].get("id"),
            "title": top_article[1].get("title"),
            "snippet": top_article[1].get("content"),
            "score": top_article[0],
        })

    # 위반 판단: 규칙 매치가 임계치 이상일 때만 True
    violation_threshold = 4.0  # 규칙 점수 기반
    violation = final_score >= violation_threshold

    # 설명/수정안: 최상위 기사 기준으로 동적 생성
    explanation = "규칙 기반 매칭 결과, 입력 문장이 특정 조항(조/항/호)의 금지 유형과 일치합니다."
    suggestion = "조문 취지에 맞게 제한 사유를 합리적으로 한정하고, 전면적 배제 표현은 삭제하세요."

    if reasons and len(reasons) > 0:
        top = reasons[0]
        aid = top.get("article_id") or top.get("id")
        if aid and "제7조" in aid:
            explanation = "면책/책임 배제·제한 표현이 발견되어 제7조(면책조항 금지) 유형과 일치합니다."
            suggestion = (
                "전면적 면책 표현 삭제, 고의·중과실 책임 명시, 불가항력 등 합리적 예외만 한정하세요."
            )
        elif aid and "제9조" in aid:
            explanation = "해제·해지 관련 제한/권리 박탈 표현이 발견되어 제9조 유형과 일치합니다."
            suggestion = "법정 해제·해지권 보장, 원상회복·책임 경감 금지, 존속기간·갱신의 합리화가 필요합니다."
        elif aid and "제12조" in aid:
            explanation = "의사표시 의제/도달 간주/형식 과도 제한 표현이 제12조 유형과 일치합니다."
            suggestion = "의사표시 간주 요건을 법 취지대로 한정하고, 과도한 형식·기한 요구를 완화하세요."
        elif aid and "제11조" in aid:
            explanation = "항변권·상계권 배제, 기한의 이익 박탈 등 제11조 유형과 일치합니다."
            suggestion = "고객 권리 배제·박탈 표현을 삭제·완화하고 정당한 사유를 구체화하세요."
        elif aid and "제10조" in aid:
            explanation = "급부 일방 결정/변경·중지 등 제10조 유형과 일치합니다."
            suggestion = "급부 변경·중지 권한을 합리적 사유와 절차로 한정하세요."
        elif aid and "제14조" in aid:
            explanation = "소송 금지·재판관할 합의·입증책임 전가 등 제14조 유형과 일치합니다."
            suggestion = "소송 금지·부당 관할 합의·입증책임 전가 표현을 삭제하세요."

    return {
        "violation": violation,
        "score": final_score,
        "top_reasons": reasons,
        "explanation": explanation,
        "suggestion": suggestion,
    }


# ----------------------------
# CLI
# ----------------------------

def run(user_text: str) -> Dict:
    conn = Neo4jConnector()
    try:
        cands = query_candidates(conn, user_text)
        ranked = rank_results(user_text, cands)
        result = decide_and_explain(user_text, ranked)
        return result
    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python scripts/judge_clause.py '검사할 문장'", file=sys.stderr)
        sys.exit(1)
    text = sys.argv[1]
    out = run(text)
    print(json.dumps(out, ensure_ascii=False, indent=2))
