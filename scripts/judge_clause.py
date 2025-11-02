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
from typing import List, Dict, Optional, Any
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
from utils.llm_client import get_llm_client
from config.settings import settings

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
    """
    위반 조항 판단 (완전히 개선된 로직)
    - 핵심 키워드 우선 매칭 (조항별 구분력 높은 키워드)
    - 패턴 기반 보조 판단
    - 점수 계산 로직 개선
    """
    # 핵심 키워드 정의 (조항별 구분력 높은 키워드 - 정규식 포함)
    CORE_KEYWORDS = {
        "제7조": [
            r"피해배상.*않", r"손해배상.*않", r"배상.*않", r"책임을 지지 않", 
            r"책임없음", r"책임 없음", r"면책", r"책임.*배제", r"책임.*면제",
            r"어떠한.*책임", r"모든 책임.*없", r"책임.*부담하지",
            "책임을 지지 않습니다", "책임없다", "책임이 없다",
        ],
        "제8조": [
            r"과중한.*손해배상", r"과도한.*손해배상", r"지연.*손해금", 
            r"위약금", r"과중한.*지연", r"과도한.*배상", r"과도한.*손해",
            r"손해배상액", r"지연.*배상", r"지연.*손해",
        ],
        "제9조": [
            r"자동.*연장", r"자동으로.*연장", r"묵시.*연장", r"갱신.*밝히지",
            r"해지.*요청.*않", r"종료.*의사.*밝히지", r"계약.*자동",
            r"해제권", r"해지권", r"원상회복.*고객",
        ],
        "제10조": [
            r"급부.*변경", r"급부를.*변경", r"일방적.*변경", r"사업자.*변경",
            r"급부.*중지", r"급부.*대행", r"사정에.*따라.*변경",
        ],
        "제11조": [
            r"항변권", r"상계권", r"기한.*이익", r"제3자.*거래.*제한",
            r"제3자와.*계약.*금지", r"비밀.*누설",
        ],
        "제12조": [
            r"묵시.*동의", r"동의.*간주", r"의사표시.*간주", r"의사표시.*의제",
            r"묵시적", r"고지.*않.*동의", r"통지.*않.*동의",
        ],
        "제13조": [
            r"대리인.*책임", r"대리인에게.*책임", r"대리인이.*책임", r"대리인의.*책임",
        ],
        "제14조": [
            r"소송.*제기.*금지", r"소송.*제기.*않", r"재판관할", r"입증책임.*고객",
        ]
    }
    
    scores = {}
    
    for article_id, info in ARTICLE_PATTERNS.items():
        score = 0.0
        
        # 1. 핵심 키워드 매칭 (최우선, 매우 높은 점수)
        core_keywords = CORE_KEYWORDS.get(article_id, [])
        core_matched_count = 0
        for pattern in core_keywords:
            try:
                if re.search(pattern, text, re.IGNORECASE):
                    # 핵심 키워드 매칭: 매우 높은 점수 (각 1.0점)
                    core_matched_count += 1
                    score += 1.0
            except re.error:
                continue
        
        # 핵심 키워드 점수 (최대 3.0점 - 여러 개 매칭 가능)
        if core_matched_count > 0:
            score = min(core_matched_count * 1.0, 3.0)
        
        # 2. 일반 키워드 매칭 (핵심 키워드가 없을 때만 보조)
        if core_matched_count == 0:
            keywords_matched = []
            for kw in info.get('keywords', []):
                if kw in text:
                    # 키워드 길이에 따라 가중치
                    if len(kw) >= 15:
                        weight = 0.25
                    elif len(kw) >= 10:
                        weight = 0.15
                    elif len(kw) >= 6:
                        weight = 0.1
                    else:
                        weight = 0.05  # 짧은 키워드는 매우 낮음
                    keywords_matched.append((kw, weight))
            
            # 일반 키워드 점수 (최대 1.0점)
            keyword_score = min(sum(w for _, w in keywords_matched), 1.0)
            score += keyword_score
        
        # 3. 패턴 매칭 (보조)
        patterns_matched = []
        for pat in info.get('patterns', []):
            try:
                if re.search(pat, text):
                    patterns_matched.append(pat)
            except re.error:
                continue
        # 패턴 점수: 핵심 키워드가 있으면 보조(0.2점), 없으면 더 높게(0.4점)
        pattern_weight = 0.2 if core_matched_count > 0 else 0.4
        pattern_score = min(len(patterns_matched) * pattern_weight, 1.0)
        score += pattern_score
        
        # 최종 점수 저장
        scores[article_id] = score
    
    # 최고 점수 조항 선택
    if not scores:
        return "제6조"
    
    best = max(scores.items(), key=lambda x: x[1])
    
    # 핵심 키워드가 매칭된 경우는 바로 반환
    core_keywords_all = CORE_KEYWORDS.get(best[0], [])
    has_core_match = False
    for pattern in core_keywords_all:
        try:
            if re.search(pattern, text, re.IGNORECASE):
                has_core_match = True
                break
        except:
            continue
    
    if has_core_match and best[1] >= 0.5:
        return best[0]
    
    # 점수가 너무 낮으면 (0.3 미만) 기본값 반환
    if best[1] < 0.3:
        top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        if top3[0][1] > 0.15:
            return top3[0][0]
        return "제6조"
    
    return best[0]

# =============================================================================
# Neo4j 조회
# =============================================================================
def query_law_content(conn: Neo4jConnector, article_id: str) -> Dict[str, Any]:
    """
    법률 조문 내용 조회 (조, 항, 호 포함)
    
    Args:
        conn: Neo4j 연결 객체
        article_id: 조항 ID (예: "제7조")
        
    Returns:
        조문 내용 딕셔너리
    """
    query = """
    MATCH (a:조 {id: $article_id})
    OPTIONAL MATCH (a)-[:HAS_HANG]->(h:항)
    OPTIONAL MATCH (a)-[:HAS_HO]->(ho:호)
    OPTIONAL MATCH (h)-[:HAS_HO]->(ho2:호)
    WITH a, 
         collect(DISTINCT {id: h.id, num: h.hang_num, content: h.content}) as hang_list,
         collect(DISTINCT {id: ho.id, num: ho.ho_num, content: ho.content}) as ho_from_article,
         collect(DISTINCT {id: ho2.id, num: ho2.ho_num, content: ho2.content}) as ho_from_hang
    RETURN a.id as article_id,
           a.title as title,
           a.content as content,
           hang_list,
           ho_from_article,
           ho_from_hang
    """
    
    result = conn.execute_query(query, {"article_id": article_id})
    
    if not result:
        return {
            "article_id": article_id,
            "title": "",
            "content": "",
            "hangs": [],
            "hos": []
        }
    
    row = result[0]
    
    # 호 통합 (조에서 직접 연결된 호 + 항에서 연결된 호)
    all_hos = (row.get('ho_from_article', []) or []) + (row.get('ho_from_hang', []) or [])
    # 중복 제거
    unique_hos = []
    seen_ho_ids = set()
    for ho in all_hos:
        if ho and ho.get('id') and ho['id'] not in seen_ho_ids:
            unique_hos.append(ho)
            seen_ho_ids.add(ho['id'])
    
    return {
        "article_id": row.get('article_id', article_id),
        "title": row.get('title', ''),
        "content": row.get('content', ''),
        "hangs": row.get('hang_list', []) or [],
        "hos": unique_hos
    }


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
# Text2Cypher: 자연어 입력을 Cypher 쿼리로 변환
# =============================================================================
def analyze_clause_with_llm(clause_text: str) -> Optional[Dict]:
    """
    LLM으로 입력 조항을 분석하여 관련 조항 및 검색 조건 추출
    
    Args:
        clause_text: 검사할 약관 조항 텍스트
        
    Returns:
        분석 결과 딕셔너리 또는 None
    """
    llm = get_llm_client()
    if not llm:
        return None
    
    prompt = f"""
다음 약관 조항을 분석해서 관련된 약관법 조항과 검색 조건을 판단하세요.

입력 조항:
"{clause_text}"

약관법 조항 설명:
- 제6조: 일반원칙 (부당하게 불리한 조항, 예상하기 어려운 조항, 본질적 권리 제한)
- 제7조: 면책조항의 금지 (책임 면제, 배상 배제, 담보 책임 제한)
- 제8조: 손해배상액의 예정 (과도한 손해배상, 지연 배상, 위약금)
- 제9조: 해제·해지의 제한 (고객의 해제권·해지권 부당 제한)
- 제10조: 급부 내용의 일방적 변경·중지 (일방적 계약 변경, 급부 중지)
- 제11조: 기한의 이익 박탈 및 항변권·상계권 제한
- 제12조: 의사표시의 간주·의제 (고객의 의사표시를 임의로 간주)
- 제13조: 대리인의 책임 (대리인에게 과도한 책임 전가)
- 제14조: 소송 제기의 금지 및 입증책임 전가

JSON 형식으로 응답:
{{
    "article_id": "제7조",
    "confidence": 0.9,
    "violation_types": ["면책", "전면책임배제"],
    "keywords": ["책임", "배상", "면책"],
    "search_strategy": "면책 관련 모든 위반사례 검색",
    "special_conditions": ["고의·중과실 책임 제외 여부 확인"]
}}
"""
    
    result = llm.generate_json(prompt)
    return result


def text2cypher_query(clause_text: str, article_id: str, analysis: Optional[Dict] = None) -> Optional[str]:
    """
    Text2Cypher: 자연어 입력을 기반으로 Cypher 쿼리 생성
    
    Args:
        clause_text: 검사할 약관 조항 텍스트
        article_id: 관련 조항 ID
        analysis: LLM 분석 결과 (선택)
        
    Returns:
        Cypher 쿼리 문자열 또는 None
    """
    llm = get_llm_client()
    if not llm:
        return None
    
    if not analysis:
        analysis = analyze_clause_with_llm(clause_text)
        if not analysis:
            return None
        article_id = analysis.get('article_id', article_id)
    
    system_prompt = """당신은 Neo4j Cypher 쿼리 전문가입니다. 
정확하고 안전한 Cypher 쿼리만 생성하세요. 
설명 없이 쿼리만 반환하세요."""

    prompt = f"""
다음 조건으로 위반사례를 검색하는 Cypher 쿼리를 생성하세요.

조건:
- 관련 조항: {article_id}
- 검사 대상 조항: "{clause_text}"
- 검색 유형: {analysis.get('violation_types', [])}
- 키워드: {analysis.get('keywords', [])}

Neo4j 그래프 스키마:
노드:
- 법률: {{id, name}}
- 조: {{id, title, content}}
- 항: {{id, article_id, hang_num, content}}
- 호: {{id, article_id, hang_id, ho_num, content}}
- 위반사례: {{id, unfair_text, reason, embedding}}
- 수정본: {{id, corrected_text, embedding}}

관계:
- (법률)-[:HAS_ARTICLE]->(조)
- (조)-[:HAS_HANG]->(항)
- (조)-[:HAS_HO]->(호)
- (항)-[:HAS_HO]->(호)
- (조)-[:HAS_VIOLATION]->(위반사례)
- (항)-[:HAS_VIOLATION]->(위반사례)
- (호)-[:HAS_VIOLATION]->(위반사례)
- (위반사례)-[:HAS_CORRECTION]->(수정본)

요구사항:
1. {article_id} 조항과 연결된 모든 위반사례를 찾기
2. 위반사례의 수정본(corrected_text, embedding)도 함께 가져오기
3. 위반사례의 unfair_text, reason, embedding도 포함
4. 위반사례 ID는 violation_id로 반환
5. LIMIT 50으로 제한

반환 형식:
- violation_id
- violation_text (unfair_text)
- reason
- violation_embedding
- correction_text
- correction_embedding

Cypher 쿼리만 반환하세요 (설명 없이):
"""
    
    cypher = llm.generate(prompt, system_prompt, temperature=0.1)
    
    if cypher:
        # 코드 블록 제거
        cypher = cypher.strip()
        if cypher.startswith("```cypher"):
            cypher = cypher[9:]
        elif cypher.startswith("```"):
            cypher = cypher[3:]
        if cypher.endswith("```"):
            cypher = cypher[:-3]
        cypher = cypher.strip()
    
    return cypher


def query_violation_cases_text2cypher(conn: Neo4jConnector, clause_text: str, 
                                     article_id: Optional[str] = None) -> List[Dict]:
    """
    Text2Cypher를 사용하여 위반사례 검색
    
    Args:
        conn: Neo4j 연결 객체
        clause_text: 검사할 약관 조항 텍스트
        article_id: 관련 조항 ID (선택, 없으면 LLM으로 분석)
        
    Returns:
        위반사례 리스트
    """
    # LLM으로 조항 분석
    analysis = analyze_clause_with_llm(clause_text)
    if not analysis:
        # LLM 실패 시 기존 방식으로 폴백
        if article_id:
            return query_violation_cases(conn, article_id)
        return []
    
    if not article_id:
        article_id = analysis.get('article_id')
    
    # Text2Cypher로 쿼리 생성
    cypher = text2cypher_query(clause_text, article_id, analysis)
    if not cypher:
        # 쿼리 생성 실패 시 기존 방식으로 폴백
        return query_violation_cases(conn, article_id)
    
    try:
        # 생성된 쿼리 실행
        result = conn.execute_query(cypher)
        
        # 결과 형식 변환 (일관성 유지)
        formatted_results = []
        for row in result:
            formatted_results.append({
                'violation_id': row.get('violation_id'),
                'violation_text': row.get('violation_text') or row.get('unfair_text'),
                'reason': row.get('reason'),
                'violation_embedding': row.get('violation_embedding'),
                'correction_text': row.get('correction_text'),
                'correction_embedding': row.get('correction_embedding')
            })
        
        return formatted_results
        
    except Exception as e:
        print(f"⚠️ Text2Cypher 쿼리 실행 실패: {e}")
        print(f"생성된 쿼리: {cypher[:200]}...")
        # 실패 시 기존 방식으로 폴백
        return query_violation_cases(conn, article_id)


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
# RAG 컨텍스트 구성 (Augmentation)
# =============================================================================
def format_law_content(law_content: Dict) -> str:
    """법률 조문 내용을 텍스트로 포맷팅"""
    text = f"\n[관련 조문: {law_content.get('article_id', '')}]"
    
    if law_content.get('title'):
        text += f"\n제목: {law_content['title']}"
    
    if law_content.get('content'):
        text += f"\n내용: {law_content['content']}"
    
    # 항들
    hangs = law_content.get('hangs', [])
    if hangs:
        text += "\n\n항:"
        for hang in hangs:
            if hang and hang.get('content'):
                hang_num = hang.get('num', '')
                text += f"\n  {hang_num}: {hang['content']}"
                
                # 항에 연결된 호
                # (호는 별도로 조회되지 않으므로 생략)
    
    # 호들
    hos = law_content.get('hos', [])
    if hos:
        text += "\n\n호:"
        for ho in hos:
            if ho and ho.get('content'):
                ho_num = ho.get('num', '')
                text += f"\n  {ho_num}: {ho['content']}"
    
    return text


def build_rag_context(user_text: str, best_case: Optional[Dict], article_id: str,
                     law_content: Dict, conn: Neo4jConnector) -> str:
    """
    RAG를 위한 컨텍스트 구성 (Augmentation)
    
    Args:
        user_text: 사용자 입력 조항
        best_case: 가장 유사한 위반 사례
        article_id: 관련 조항 ID
        law_content: 법률 조문 내용
        conn: Neo4j 연결 객체
        
    Returns:
        구조화된 컨텍스트 문자열
    """
    context_parts = []
    
    # 1. 검사 대상 조항
    context_parts.append("=" * 60)
    context_parts.append("검사 대상 조항")
    context_parts.append("=" * 60)
    context_parts.append(f'"{user_text}"')
    context_parts.append("")
    
    # 2. 관련 법률 조문
    context_parts.append("=" * 60)
    context_parts.append("관련 약관법 조문")
    context_parts.append("=" * 60)
    context_parts.append(format_law_content(law_content))
    context_parts.append("")
    
    # 3. 유사한 위반 사례 (있는 경우)
    if best_case:
        context_parts.append("=" * 60)
        context_parts.append("유사한 위반 사례")
        context_parts.append("=" * 60)
        
        violation_text = best_case.get('violation_text', '') or best_case.get('unfair_text', '')
        if violation_text:
            context_parts.append(f"원문: {violation_text}")
        
        reason = best_case.get('reason', '')
        if reason:
            context_parts.append(f"시정 사유: {reason}")
        
        correction_text = best_case.get('correction_text', '')
        if correction_text:
            context_parts.append(f"수정 후: {correction_text}")
        
        similarity = best_case.get('similarity', 0.0)
        context_parts.append(f"유사도 점수: {similarity:.2f}")
        context_parts.append("")
    else:
        context_parts.append("=" * 60)
        context_parts.append("유사한 위반 사례")
        context_parts.append("=" * 60)
        context_parts.append("데이터베이스에서 유사한 위반 사례를 찾을 수 없습니다.")
        context_parts.append("")
    
    # 4. 판단 관련 정보
    context_parts.append("=" * 60)
    context_parts.append("판단 정보")
    context_parts.append("=" * 60)
    context_parts.append(f"관련 조항: {article_id}")
    
    if best_case:
        contrastive = calculate_contrastive_score(user_text, best_case)
        unfairness = 1 - contrastive
        context_parts.append(f"불공정도: {unfairness:.2f}")
        context_parts.append(f"유사도: {best_case.get('similarity', 0.0):.2f}")
    
    return "\n".join(context_parts)


# =============================================================================
# LLM 기반 Generation (설명 및 제안 생성)
# =============================================================================
def llm_generate_explanation(user_text: str, rag_context: str, violation: bool,
                             final_score: float, base_sim: float, article_id: str) -> str:
    """
    LLM을 사용하여 불공정 판단 설명 생성 (Generation)
    
    Args:
        user_text: 검사 대상 조항
        rag_context: RAG 컨텍스트 (Augmentation 결과)
        violation: 위반 여부
        final_score: 최종 불공정도 점수
        base_sim: 기본 유사도 점수
        article_id: 관련 조항 ID
        
    Returns:
        설명 텍스트 (LLM 실패 시 기본 설명 반환)
    """
    llm = get_llm_client()
    if not llm:
        # LLM 없으면 기본 설명 반환
        if violation:
            return f"입력문구는 약관법상 불공정한 표현으로 판단됩니다. (유사도: {base_sim:.2f}, 불공정도: {final_score:.2f})"
        else:
            return f"현재 DB/규칙 기준으로는 불공정 여부가 명확하지 않습니다. (유사도: {base_sim:.2f}, 불공정도: {final_score:.2f})"
    
    system_prompt = """당신은 약관법 전문가입니다. 
약관법에 대한 정확한 지식을 바탕으로 조항의 불공정 여부를 판단하고 설명해야 합니다.
한국어로 명확하고 전문적인 설명을 제공하세요."""

    prompt = f"""
다음 정보를 바탕으로 약관 조항의 불공정 여부를 판단하고 상세히 설명해주세요.

{rag_context}

판단 결과:
- 위반 여부: {'위반' if violation else '비위반/불명확'}
- 불공정도 점수: {final_score:.2f} (0.0 = 공정함, 1.0 = 매우 불공정함)
- 유사도 점수: {base_sim:.2f}

다음 사항을 포함하여 설명해주세요:
1. 해당 조항이 약관법의 어떤 조문(특히 {article_id})과 관련있는지
2. 왜 불공정한지 (또는 불공정하지 않은지) 구체적인 이유
3. 유사한 위반 사례와의 비교 (있는 경우)
4. 관련 법률 조문의 어떤 부분이 적용되는지

설명은 전문적이면서도 이해하기 쉬워야 합니다.
"""
    
    explanation = llm.generate(prompt, system_prompt, temperature=0.3, max_tokens=800)
    
    if explanation:
        return explanation.strip()
    else:
        # LLM 실패 시 기본 설명
        if violation:
            return f"입력문구는 약관법상 불공정한 표현으로 판단됩니다. (유사도: {base_sim:.2f}, 불공정도: {final_score:.2f})"
        else:
            return f"현재 DB/규칙 기준으로는 불공정 여부가 명확하지 않습니다. (유사도: {base_sim:.2f}, 불공정도: {final_score:.2f})"


def llm_generate_suggestion(user_text: str, article_id: str, rag_context: str,
                           correction_example: Optional[str] = None) -> str:
    """
    LLM을 사용하여 수정 제안 생성 (Generation)
    
    Args:
        user_text: 검사 대상 조항 (원문)
        article_id: 관련 조항 ID
        rag_context: RAG 컨텍스트
        correction_example: 참고할 수정 사례 (선택)
        
    Returns:
        수정 제안 텍스트 (LLM 실패 시 기본 제안 반환)
    """
    llm = get_llm_client()
    if not llm:
        # LLM 없으면 기본 제안 반환
        return get_suggestion(article_id)
    
    system_prompt = """당신은 약관법 전문가이자 법률 문서 작성 전문가입니다.
약관법에 맞게 조항을 수정하는 제안을 제공해야 합니다.
한국어로 명확하고 실용적인 제안을 제공하세요."""

    prompt = f"""
다음 불공정 약관 조항을 약관법에 맞게 수정하여 제안해주세요.

{rag_context}

원문 조항:
"{user_text}"

"""
    
    if correction_example:
        prompt += f"""
참고할 올바른 수정 사례:
"{correction_example}"

"""
    
    prompt += f"""
다음 사항을 포함하여 수정 제안을 작성해주세요:
1. 수정된 조항 텍스트 (원문의 문제점을 해결한 버전)
2. 왜 그렇게 수정했는지 간단한 설명
3. 약관법의 어떤 원칙을 반영했는지

수정 제안은 구체적이고 실행 가능해야 합니다.
"""
    
    suggestion = llm.generate(prompt, system_prompt, temperature=0.4, max_tokens=600)
    
    if suggestion:
        return suggestion.strip()
    else:
        # LLM 실패 시 기본 제안
        return get_suggestion(article_id)


# =============================================================================
# 종합 판단 (메인 로직)
# =============================================================================
THRESHOLD = 0.6  # 요청대로 0.6

def comprehensive_judgment(user_text: str, conn: Neo4jConnector, 
                         use_text2cypher: Optional[bool] = None) -> Dict:
    """
    종합 판단 함수
    
    Args:
        user_text: 검사할 약관 조항 텍스트
        conn: Neo4j 연결 객체
        use_text2cypher: Text2Cypher 사용 여부 (None이면 설정값 사용)
        
    Returns:
        판단 결과 딕셔너리
    """
    # 설정값 사용
    if use_text2cypher is None:
        use_text2cypher = settings.USE_TEXT2CYPHER
    
    compare_mode = settings.COMPARE_METHODS
    
    # 비교 모드: 두 방식 모두 실행
    if compare_mode:
        return _comprehensive_judgment_compare(user_text, conn)
    
    # 일반 모드: 선택된 방식으로 실행
    if use_text2cypher:
        return _comprehensive_judgment_with_text2cypher(user_text, conn)
    else:
        return _comprehensive_judgment_standard(user_text, conn)


def _judgment_core(user_text: str, cases: List[Dict], article_id: str, 
                  hard_match: Optional[re.Match], hard_snippet: Optional[str]) -> Dict:
    """공통 판단 로직 (점수 계산, 판단 등)"""
    # 가장 유사한 사례 찾기
    best_case = find_most_similar_violation(user_text, cases) if cases else None

    # contrastive 점수
    contrastive = calculate_contrastive_score(user_text, best_case) if best_case else 0.5
    unfairness = 1 - contrastive
    base_sim = best_case['similarity'] if best_case else 0.0

    # 기본 final score (embedding 기반)
    final_score = (unfairness * 0.7) + (base_sim * 0.3)
    final_score = float(min(max(final_score, 0.0), 1.0))

    # 규칙 기반 강화
    neg_count = detect_negation_count(user_text)
    if neg_count >= 2:
        final_score = min(final_score * 1.2, 1.0)

    # 하드룰이 존재하면 강제 위반(우선)
    if hard_match:
        final_score = max(final_score, 0.9)

    # 추가 패턴(제7조 전형적 면책 표현) 발견 시 상향
    if any(re.search(pat, user_text) for pat in ARTICLE_PATTERNS.get("제7조", {}).get("patterns", [])):
        final_score = max(final_score, 0.8)

    # 판단 및 심각도
    violation = final_score > THRESHOLD
    if final_score > 0.8:
        severity = "높음"
    elif final_score > 0.6:
        severity = "중간"
    else:
        severity = "낮음"

    # 근거 구성
    top_reasons = []
    if best_case:
        top_reasons.append({
            "level": "위반사례",
            "id": best_case.get('violation_id') or best_case.get('id'),
            "article_id": article_id,
            "snippet": (best_case.get('violation_text') or best_case.get('text') or '')[:300],
            "score": float(final_score)
        })
    if not top_reasons and hard_match:
        top_reasons.append({
            "level": "규칙기반_근거",
            "id": None,
            "article_id": article_id,
            "snippet": hard_snippet,
            "score": float(final_score),
            "note": "명백한 전면 면책 표현(hard rule)"
        })
    if not top_reasons:
        candidate_snippet = user_text[:300]
        top_reasons.append({
            "level": "추측근거",
            "id": None,
            "article_id": article_id,
            "snippet": candidate_snippet,
            "score": float(final_score),
            "note": "유사 위반사례가 DB에 없으므로 입력문장에서 추출한 후보"
        })

    return {
        "best_case": best_case,
        "contrastive": contrastive,
        "unfairness": unfairness,
        "base_sim": base_sim,
        "final_score": final_score,
        "violation": violation,
        "severity": severity,
        "top_reasons": top_reasons,
        "neg_count": neg_count,
        "hard_match": hard_match
    }


def _comprehensive_judgment_standard(user_text: str, conn: Neo4jConnector) -> Dict:
    """표준 방식: 규칙 기반 조항 판별 + 고정 쿼리"""
    article_id = detect_best_article(user_text)

    # 하드룰 체크
    hard_match = detect_hard_violation(user_text)
    hard_snippet = hard_match.group(0) if hard_match else None

    # DB에서 위반 사례 불러오기 (고정 쿼리)
    cases = query_violation_cases(conn, article_id)
    
    # 법률 조문 조회
    law_content = query_law_content(conn, article_id)

    # 공통 판단 로직
    judgment = _judgment_core(user_text, cases, article_id, hard_match, hard_snippet)

    # RAG 컨텍스트 구성
    rag_context = build_rag_context(
        user_text, judgment["best_case"], article_id, law_content, conn
    )

    # LLM 기반 설명 생성
    explanation = llm_generate_explanation(
        user_text, rag_context, judgment["violation"],
        judgment["final_score"], judgment["base_sim"], article_id
    )

    # LLM 기반 제안 생성
    correction_example = judgment["best_case"].get('correction_text') if judgment["best_case"] else None
    suggestion = llm_generate_suggestion(
        user_text, article_id, rag_context, correction_example
    )

    return {
        "violation": judgment["violation"],
        "score": judgment["final_score"],
        "severity": judgment["severity"],
        "article_id": article_id,
        "law_content": law_content,  # 조, 항, 호 정보 추가
        "explanation": explanation,
        "suggestion": suggestion,
        "top_reasons": judgment["top_reasons"],
        "method": "standard",
        "debug": {
            "base_similarity": judgment["base_sim"],
            "contrastive": judgment["contrastive"],
            "unfairness": judgment["unfairness"],
            "negation_count": judgment["neg_count"],
            "hard_rule_matched": bool(judgment["hard_match"]),
            "cases_found": len(cases),
        }
    }


def _comprehensive_judgment_with_text2cypher(user_text: str, conn: Neo4jConnector) -> Dict:
    """Text2Cypher 방식: LLM으로 조항 분석 + 동적 쿼리 생성"""
    # LLM으로 조항 분석
    analysis = analyze_clause_with_llm(user_text)
    if not analysis:
        # LLM 실패 시 표준 방식으로 폴백
        return _comprehensive_judgment_standard(user_text, conn)
    
    article_id = analysis.get('article_id')
    if not article_id:
        article_id = detect_best_article(user_text)  # 폴백

    # 하드룰 체크
    hard_match = detect_hard_violation(user_text)
    hard_snippet = hard_match.group(0) if hard_match else None

    # Text2Cypher로 위반 사례 검색
    cases = query_violation_cases_text2cypher(conn, user_text, article_id)
    
    # 법률 조문 조회
    law_content = query_law_content(conn, article_id)

    # 공통 판단 로직
    judgment = _judgment_core(user_text, cases, article_id, hard_match, hard_snippet)

    # RAG 컨텍스트 구성
    rag_context = build_rag_context(
        user_text, judgment["best_case"], article_id, law_content, conn
    )

    # LLM 기반 설명 생성
    explanation = llm_generate_explanation(
        user_text, rag_context, judgment["violation"],
        judgment["final_score"], judgment["base_sim"], article_id
    )

    # LLM 기반 제안 생성
    correction_example = judgment["best_case"].get('correction_text') if judgment["best_case"] else None
    suggestion = llm_generate_suggestion(
        user_text, article_id, rag_context, correction_example
    )

    return {
        "violation": judgment["violation"],
        "score": judgment["final_score"],
        "severity": judgment["severity"],
        "article_id": article_id,
        "law_content": law_content,  # 조, 항, 호 정보 추가
        "explanation": explanation,
        "suggestion": suggestion,
        "top_reasons": judgment["top_reasons"],
        "method": "text2cypher",
        "analysis": analysis,  # LLM 분석 결과 포함
        "debug": {
            "base_similarity": judgment["base_sim"],
            "contrastive": judgment["contrastive"],
            "unfairness": judgment["unfairness"],
            "negation_count": judgment["neg_count"],
            "hard_rule_matched": bool(judgment["hard_match"]),
            "cases_found": len(cases),
            "llm_confidence": analysis.get('confidence'),
        }
    }


def _comprehensive_judgment_compare(user_text: str, conn: Neo4jConnector) -> Dict:
    """비교 모드: 두 방식 모두 실행하여 결과 비교"""
    # 두 방식 모두 실행
    result_standard = _comprehensive_judgment_standard(user_text, conn)
    result_text2cypher = _comprehensive_judgment_with_text2cypher(user_text, conn)

    return {
        "violation": result_standard["violation"],  # 표준 방식 결과를 메인으로
        "score": result_standard["score"],
        "severity": result_standard["severity"],
        "article_id": result_standard["article_id"],
        "law_content": result_standard.get("law_content"),  # 조, 항, 호 정보 추가
        "explanation": result_standard["explanation"],
        "suggestion": result_standard["suggestion"],
        "top_reasons": result_standard["top_reasons"],
        "method": "compare",
        "comparison": {
            "standard": {
                "violation": result_standard["violation"],
                "score": result_standard["score"],
                "explanation": result_standard["explanation"],
                "cases_found": result_standard["debug"].get("cases_found", 0),
            },
            "text2cypher": {
                "violation": result_text2cypher["violation"],
                "score": result_text2cypher["score"],
                "explanation": result_text2cypher["explanation"],
                "cases_found": result_text2cypher["debug"].get("cases_found", 0),
                "llm_confidence": result_text2cypher.get("analysis", {}).get("confidence"),
            },
            "differences": {
                "score_diff": abs(result_standard["score"] - result_text2cypher["score"]),
                "violation_match": result_standard["violation"] == result_text2cypher["violation"],
                "cases_diff": abs(result_standard["debug"].get("cases_found", 0) - 
                                 result_text2cypher["debug"].get("cases_found", 0)),
            }
        },
        "debug": result_standard["debug"]
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
