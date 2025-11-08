"""약관법 조항별 핵심 키워드 매핑"""
from typing import Dict, List, Tuple
import re


# 각 조항별 핵심 키워드 정의
ARTICLE_KEYWORDS: Dict[str, Dict] = {
    "제6조": {
        "priority": 0,  # 가장 일반적
        "primary_keywords": [
            "일반원칙", "신의성실", "공정성", "부당하게 불리",
            "예상하기 어려운", "본질적 권리 제한", "약관 변경",
            "운영정책", "회사의 사정", "기타", "포괄적", "추상적"
        ],
        "secondary_keywords": [
            "공정성을 잃은", "무효", "신의성실 원칙 위반",
            "동의한 것으로 간주", "승인한 것으로 봄", "의제",
            "자의적", "불명확", "예측하기 어려운"
        ],
        "patterns": [
            r".*부당하게.*불리",
            r".*예상.*어려운",
            r".*본질적.*권리.*제한",
            r".*신의성실.*원칙.*위반",
            r".*공정성.*잃",
            r".*약관.*변경.*동의.*간주",
            r".*운영정책",
            r".*회사의.*사정",
            r".*기타.*경우",
            r".*포괄적",
            r".*추상적",
            r".*자의적"
        ],
        "core_concept": "다른 구체적 조항에 해당하지 않는 일반적인 불공정 조항"
    },
    "제7조": {
        "priority": 1,
        "primary_keywords": [
            "면책", "책임 배제", "책임을 지지 않는다", "책임을 부담하지 않는다",
            "책임 면제", "손해배상 범위 제한", "담보책임 배제",
            "일체의 책임", "어떠한 책임", "책임을 지지 아니", "책임을 지지 않음",
            "책임 없음", "책임 부담하지 않음"
        ],
        "secondary_keywords": [
            "고의 중과실 책임 배제", "위험을 떠넘기기", "책임을 제한",
            "책임 부담하지 않음", "책임 없음", "책임 면제",
            "배상 범위 제한", "배상 책임 제한"
        ],
        "patterns": [
            r".*책임.*지지.*않",
            r".*책임.*배제",
            r".*면책",
            r".*책임.*면제",
            r".*손해배상.*범위.*제한",
            r".*담보책임.*배제",
            r".*고의.*중과실.*책임.*배제",
            r".*일체.*책임.*지지",
            r".*어떠한.*책임.*지지",
            r".*책임.*부담.*않",
            r".*배상.*범위.*제한",
            r".*배상.*책임.*제한"
        ],
        "exclude_patterns": [
            r".*위약금",
            r".*지연손해금",
            r".*\d+.*배.*지급",
            r".*\d+.*배.*배상",
            r".*과중한.*손해"
        ],
        "core_concept": "사업자의 책임 자체를 배제하거나 제한하는 조항 (손해배상액의 예정이 아닌 책임 자체의 배제)"
    },
    "제8조": {
        "priority": 1,
        "primary_keywords": [
            "위약금", "지연손해금", "손해배상액", "손해배상 의무",
            "배상", "배", "배수", "과중한", "손해배상액의 예정",
            "지급해야", "배상해야", "지급하여야", "배상하여야"
        ],
        "secondary_keywords": [
            "손해배상액의 예정", "과중한 손해배상", "배상 의무 부담",
            "과중한 지연 손해금", "부당하게 과중한"
        ],
        "patterns": [
            r".*위약금.*지급",
            r".*위약금.*배상",
            r".*지연손해금",
            r".*손해배상액",
            r".*손해배상.*의무.*부담",
            r".*배상.*\d+.*배",
            r".*계약금액.*\d+.*배",
            r".*\d+.*배.*지급",
            r".*\d+.*배.*배상",
            r".*과중한.*손해",
            r".*과중.*지연.*손해",
            r".*배상.*해야",
            r".*지급.*해야",
            r".*\d+.*억원.*배상",
            r".*\d+.*원.*배상",
            r".*부당하게.*과중.*손해",
            r".*부당하게.*과중.*지연"
        ],
        "core_concept": "고객에게 손해배상 의무를 부담시키는 조항 (위약금, 지연손해금, 배수 표현, 구체적 금액 등)"
    },
    "제9조": {
        "priority": 1,
        "primary_keywords": [
            "해제", "해지", "원상회복", "해제권", "해지권",
            "계약 해제", "계약 해지", "서비스 해지", "계약 해지권",
            "서비스 제공을 중지", "시정조치", "해지 사유", "계약 해지 사유"
        ],
        "secondary_keywords": [
            "해제권 배제", "해지권 배제", "원상회복의무", "존속기간",
            "계약 취소", "계약 종료", "서비스 중지", "서비스 종료",
            "시정 기간", "해지권 행사", "해제권 행사", "계약 종료"
        ],
        "patterns": [
            r".*계약.*해제",
            r".*계약.*해지",
            r".*해제권.*배제",
            r".*해지권.*배제",
            r".*원상회복",
            r".*존속기간",
            r".*서비스.*해지",
            r".*계약.*취소",
            r".*계약.*종료",
            r".*서비스.*제공.*중지",
            r".*시정조치",
            r".*해지.*사유",
            r".*계약.*해지.*사유",
            r".*해지권.*행사",
            r".*해제권.*행사"
        ],
        "exclude_patterns": [
            r".*서비스.*변경",
            r".*서비스.*제한",
            r".*급부.*변경",
            r".*이자율.*변경",
            r".*수수료.*변경"
        ],
        "core_concept": "계약의 해제/해지와 관련된 조항 (급부 변경/중지가 아닌 계약 자체의 해제/해지)"
    },
    "제10조": {
        "priority": 1,
        "primary_keywords": [
            "급부", "일방적 변경", "일방적 결정", "급부 중지", "제3자 대행",
            "서비스 변경", "서비스 중지", "서비스 제한", "이자율 변경",
            "수수료 변경", "여신거래조건 변경", "서비스 내용 변경",
            "일임", "처분", "임의로 결정", "임의로 변경", "임의 처분"
        ],
        "secondary_keywords": [
            "급부 내용 변경", "이자율 변경", "수수료 변경",
            "회사가 정하는", "회사가 변경", "임의로 변경",
            "일방적으로 변경", "일방적으로 결정", "일방적으로 중지",
            "저당물건의 처분", "처분방법", "처분대금", "회사에게 일임",
            "임의로 결정", "임의로 변경", "임의 처분"
        ],
        "patterns": [
            r".*급부.*일방적.*변경",
            r".*급부.*일방적.*결정",
            r".*급부.*중지",
            r".*제3자.*대행",
            r".*이자율.*변경",
            r".*수수료.*변경",
            r".*서비스.*변경",
            r".*서비스.*중지",
            r".*서비스.*제한",
            r".*여신거래조건.*변경",
            r".*회사가.*정하는.*기준",
            r".*회사가.*변경.*수.*있",
            r".*일방적.*변경.*수.*있",
            r".*일방적.*결정.*수.*있",
            r".*일방적.*중지.*수.*있",
            r".*일체.*회사에게.*일임",
            r".*회사에게.*일임",
            r".*처분.*방법.*시기.*가격",
            r".*저당물건.*처분",
            r".*임의.*처분",
            r".*임의로.*결정",
            r".*임의로.*변경"
        ],
        "exclude_patterns": [
            r".*계약.*해제",
            r".*계약.*해지",
            r".*원상회복"
        ],
        "core_concept": "급부의 내용을 일방적으로 변경/중지하는 조항"
    },
    "제11조": {
        "priority": 1,
        "primary_keywords": [
            "항변권", "상계권", "기한의 이익", "제3자 계약 제한", "비밀 누설",
            "이의를 제기할 수 없다", "이의제기 금지", "이의 제기 불가"
        ],
        "secondary_keywords": [
            "항변권 배제", "상계권 배제", "기한의 이익 박탈",
            "이의를 제기할 수 없음", "이의제기 불가", "이의 제기 금지"
        ],
        "patterns": [
            r".*항변권.*배제",
            r".*상계권.*배제",
            r".*기한.*이익.*박탈",
            r".*제3자.*계약.*제한",
            r".*비밀.*누설",
            r".*이의.*제기.*할.*수.*없",
            r".*이의제기.*금지",
            r".*이의.*제기.*불가",
            r".*이의.*제기.*불가능"
        ],
        "core_concept": "고객의 권리를 배제/제한하는 조항"
    },
    "제12조": {
        "priority": 1,
        "primary_keywords": [
            "의사표시", "의제", "간주", "고지", "형식", "서면",
            "서면에 의하여", "서면으로만", "서면 제한"
        ],
        "secondary_keywords": [
            "의사표시 간주", "의사표시 의제", "서면만 제한", "형식 엄격 제한",
            "서면에 의하여", "서면으로만 제한", "서면 통보로만"
        ],
        "patterns": [
            r".*의사표시.*간주",
            r".*의사표시.*의제",
            r".*서면.*만.*제한",
            r".*형식.*엄격.*제한",
            r".*고지.*도달.*간주",
            r".*서면.*의하여",
            r".*서면.*으로만",
            r".*서면.*제한",
            r".*서면.*통보.*만"
        ],
        "core_concept": "의사표시를 임의로 간주하거나 형식을 엄격히 제한하는 조항"
    },
    "제13조": {
        "priority": 1,
        "primary_keywords": [
            "대리인", "책임", "이행"
        ],
        "secondary_keywords": [
            "대리인 책임", "대리인 이행"
        ],
        "patterns": [
            r".*대리인.*책임",
            r".*대리인.*이행"
        ],
        "core_concept": "대리인에게 책임을 지우는 조항"
    },
    "제14조": {
        "priority": 1,
        "primary_keywords": [
            "소송 제기 금지", "재판관할", "입증책임",
            "소송 제기", "소송 제기 제한", "소송 제기 배제",
            "소송", "재판", "입증"
        ],
        "secondary_keywords": [
            "소송 제기 금지", "재판관할 합의", "입증책임 부담",
            "소송 제한", "소송 배제", "재판관할 제한",
            "소송 제기 불가", "소송 제기 불가능"
        ],
        "patterns": [
            r".*소송.*제기.*금지",
            r".*소송.*제기.*제한",
            r".*소송.*제기.*배제",
            r".*재판관할",
            r".*입증책임.*부담",
            r".*입증책임.*가중",
            r".*소송.*제기.*불가",
            r".*소송.*제기.*불가능"
        ],
        "core_concept": "소송 제기나 입증책임 관련 조항"
    }
}


def analyze_clause_keywords(clause: str) -> List[Tuple[str, float]]:
    """
    입력 조항의 키워드를 분석하여 각 조항과의 매칭 점수 계산
    
    Args:
        clause: 분석할 조항 텍스트
    
    Returns:
        [(조항, 점수), ...] 형태의 리스트, 점수 순으로 정렬됨
    """
    clause_lower = clause.lower()
    scores = []
    max_specific_score = 0.0  # 구체적 조항(제7조~제14조)의 최고 점수
    
    for article, keywords_info in ARTICLE_KEYWORDS.items():
        score = 0.0
        
        # Primary keywords 매칭 (가중치 1.0)
        for keyword in keywords_info["primary_keywords"]:
            if keyword in clause_lower:
                score += 1.0
        
        # Secondary keywords 매칭 (가중치 0.5)
        for keyword in keywords_info["secondary_keywords"]:
            if keyword in clause_lower:
                score += 0.5
        
        # Pattern 매칭 (가중치 1.5)
        for pattern in keywords_info["patterns"]:
            if re.search(pattern, clause_lower, re.IGNORECASE):
                score += 1.5
        
        # Exclude patterns (특정 조항 제외 패턴)
        if "exclude_patterns" in keywords_info:
            for exclude_pattern in keywords_info["exclude_patterns"]:
                if re.search(exclude_pattern, clause_lower, re.IGNORECASE):
                    score = 0.0  # 제외 패턴이 매칭되면 점수 0으로 설정
                    break
        
        # 우선순위 보정 (priority가 낮을수록 일반적이므로 점수 감소)
        priority_penalty = keywords_info["priority"] * 0.1
        score = max(0, score - priority_penalty)
        
        # 구체적 조항(제7조~제14조)의 최고 점수 추적
        if article != "제6조" and score > max_specific_score:
            max_specific_score = score
        
        if score > 0:
            scores.append((article, score))
    
    # 구체적 조항의 점수가 모두 낮으면(0.5 미만) 제6조에 기본 점수 부여
    if max_specific_score < 0.5 and "제6조" not in [a for a, s in scores]:
        # 제6조 키워드 재확인
        article_6_info = ARTICLE_KEYWORDS["제6조"]
        score_6 = 0.0
        for keyword in article_6_info["primary_keywords"]:
            if keyword in clause_lower:
                score_6 += 1.0
        for keyword in article_6_info["secondary_keywords"]:
            if keyword in clause_lower:
                score_6 += 0.5
        for pattern in article_6_info["patterns"]:
            if re.search(pattern, clause_lower, re.IGNORECASE):
                score_6 += 1.5
        
        if score_6 > 0:
            scores.append(("제6조", score_6))
        else:
            # 다른 조항에 해당하지 않으면 제6조로 분류 (낮은 점수)
            scores.append(("제6조", 0.3))
    
    # 점수 순으로 정렬
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def get_top_matching_articles(clause: str, top_k: int = 3) -> List[str]:
    """
    입력 조항과 가장 관련성 높은 조항 반환
    
    Args:
        clause: 분석할 조항 텍스트
        top_k: 반환할 상위 조항 개수
    
    Returns:
        조항 리스트 (예: ["제8조", "제7조", "제6조"])
    """
    scores = analyze_clause_keywords(clause)
    return [article for article, score in scores[:top_k]]


def get_article_core_concept(article: str) -> str:
    """
    조항의 핵심 개념 반환
    
    Args:
        article: 조항 (예: "제8조")
    
    Returns:
        핵심 개념 설명
    """
    if article in ARTICLE_KEYWORDS:
        return ARTICLE_KEYWORDS[article]["core_concept"]
    return ""

