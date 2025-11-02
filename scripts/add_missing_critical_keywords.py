"""
주요 혼동 패턴에 대한 핵심 구분 키워드 직접 추가
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
pattern_file = PROJECT_ROOT / 'data' / 'contracts' / 'violation_cases' / 'filtered_article_patterns.json'

with open(pattern_file, 'r', encoding='utf-8') as f:
    patterns = json.load(f)

# 주요 혼동 패턴에 대한 구분 키워드 직접 추가 (더 구체적으로)
critical_keywords = {
    "제8조": [
        "손해배상액",
        "지연 손해금",
        "위약금",
        "과중한 손해배상",
        "손해배상 의무",
        "지연으로 인한 손해배상",
        "과도한 손해배상",
        "손해배상 범위",
        "과중한 지연 손해금",
        "손해배상 의무를 부담",
    ],
    "제9조": [
        "자동 연장",
        "자동으로 연장",
        "갱신 의사",
        "계약 종료 의사",
        "해지 요청",
        "원상회복 비용",
        "원상회복 의무",
        "환불 절차",
        "회원 탈퇴",
        "계약 해지",
        "해지로 인한",
        "계약 기간 연장",
        "계약은 자동으로",
        "의사를 밝히지 않으면",
        "종료 의사를 밝히지",
        "갱신 의사를 밝히지",
    ],
    "제10조": [
        "급부 내용",
        "급부를 변경",
        "급부의 내용",
        "사전 고지 없이 서비스",
        "대행사를 통해",
        "서비스를 중지",
        "서비스를 일시 중단",
        "서비스를 중단",
        "급부 내용을",
        "급부를 변경할 수",
        "회사의 결정에 따라 변경",
    ],
    "제6조": [
        "제3자에게 판매",
        "정보를 수집하고 제3자",
        "계정 정보를 제3자",
        "재량에 따라 변경",
        "임의로 변경",
        "임의로 삭제",
        "임의로 해지",
        "사전 동의 없이",
        "이용 기록을 제3자",
        "서비스 이용 기록을",
        "재량에 따라 거래 조건",
    ],
    "제12조": [
        "의사표시가 없을 경우",
        "의사표시를 하지 않으면",
        "동의한 것으로 간주",
        "특정 조치를 취한 것으로",
        "본인 명의의",
        "본인 명의로만",
        "의사표시가 없을 경우 특정",
        "간주됩니다",
        "동의한 것으로 처리",
    ],
    "제14조": [
        "증빙을 제시하지 않을 경우",
        "입증책임",
        "재판관할",
        "소송 제기 금지",
        "분쟁 발생 시",
        "분쟁 시",
        "재판관할에서 해결",
        "회사가 지정한 재판관할",
        "입증책임은 고객",
        "증빙을 요청할 수 있으며",
    ],
    "제11조": [
        "제3자와의 거래",
        "제3자와 거래",
        "당사의 승인을 얻지",
        "당사와의 계약을 무시",
        "외부 업체와 협력",
        "외부 전문가들과",
        "항변권을 배제",
        "항변권을 제한",
        "상계권을 주장",
        "기한 내 이익을 박탈",
    ],
}

# 키워드 추가
for article_id, keywords_to_add in critical_keywords.items():
    if article_id in patterns:
        existing = set(patterns[article_id].get('keywords', []))
        new_keywords = [kw for kw in keywords_to_add if kw not in existing]
        if new_keywords:
            patterns[article_id]['keywords'].extend(new_keywords)
            print(f"{article_id}: {len(new_keywords)}개 핵심 키워드 추가")

# 저장
with open(pattern_file, 'w', encoding='utf-8') as f:
    json.dump(patterns, f, ensure_ascii=False, indent=2)

print(f"\n✅ 핵심 구분 키워드 추가 완료!")
print(f"   저장: {pattern_file}")

