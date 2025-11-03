"""
종합 테스트 케이스 생성: 조항별 골고루 120개
"""
import json
from pathlib import Path

# 조항별 테스트 케이스 (총 120개)
test_cases = []

# 제6조 (일반원칙) - 13개
test_cases.extend([
    {"query": "고객에게 부당하게 불리한 조항", "expected": "제6조"},
    {"query": "공정성을 잃은 약관 조항", "expected": "제6조"},
    {"query": "고객이 예상하기 어려운 조항", "expected": "제6조"},
    {"query": "계약의 본질적 권리를 제한하는 조항", "expected": "제6조"},
    {"query": "부당하게 불리한 내용을 정하고 있는 조항", "expected": "제6조"},
    {"query": "고객에게 불공정한 약관", "expected": "제6조"},
    {"query": "신의성실의 원칙을 위반한 조항", "expected": "제6조"},
    {"query": "공정성을 잃은 불리한 조항", "expected": "제6조"},
    {"query": "고객이 예상할 수 없는 부당한 조항", "expected": "제6조"},
    {"query": "본질적 권리를 과도하게 제한하는 조항", "expected": "제6조"},
    {"query": "부당하게 불리한 거래조건", "expected": "제6조"},
    {"query": "공정성을 해치는 약관", "expected": "제6조"},
    {"query": "고객에게 불공정한 계약 조건", "expected": "제6조"},
])

# 제7조 (면책조항의 금지) - 15개
test_cases.extend([
    {"query": "회사는 어떠한 피해배상도 하지않는다", "expected": "제7조"},
    {"query": "사업자의 고의 또는 중대한 과실로 인한 책임을 배제한다", "expected": "제7조"},
    {"query": "상당한 이유 없이 사업자의 손해배상 범위를 제한한다", "expected": "제7조"},
    {"query": "사업자가 부담하여야 할 위험을 고객에게 떠넘긴다", "expected": "제7조"},
    {"query": "상당한 이유 없이 사업자의 담보책임을 배제한다", "expected": "제7조"},
    {"query": "어떠한 경우에도 회사는 책임을 지지 않는다", "expected": "제7조"},
    {"query": "면책 조항으로 모든 책임을 배제한다", "expected": "제7조"},
    {"query": "손해배상 책임을 전면적으로 배제한다", "expected": "제7조"},
    {"query": "고의 또는 중대한 과실에 대한 법률상 책임을 배제한다", "expected": "제7조"},
    {"query": "사업자의 담보책임을 제한하거나 배제한다", "expected": "제7조"},
    {"query": "품질 보장에 대한 책임을 배제한다", "expected": "제7조"},
    {"query": "견본이나 표시에 대한 책임을 배제한다", "expected": "제7조"},
    {"query": "어떠한 손해에 대해서도 배상하지 않는다", "expected": "제7조"},
    {"query": "전면 면책 조항", "expected": "제7조"},
    {"query": "책임을 완전히 배제하는 조항", "expected": "제7조"},
])

# 제8조 (손해배상액의 예정) - 10개
test_cases.extend([
    {"query": "과도한 손해배상금을 부담해야 합니다", "expected": "제8조"},
    {"query": "고객에게 부당하게 과중한 지연 손해금을 부담시킨다", "expected": "제8조"},
    {"query": "과중한 위약금을 지급해야 한다", "expected": "제8조"},
    {"query": "지나치게 높은 손해배상금을 예정한다", "expected": "제8조"},
    {"query": "부당하게 과중한 배상금을 부담시킨다", "expected": "제8조"},
    {"query": "과도한 지연 손해금을 청구한다", "expected": "제8조"},
    {"query": "부당하게 높은 손해금을 예정한다", "expected": "제8조"},
    {"query": "과중한 배상 의무를 부담시킨다", "expected": "제8조"},
    {"query": "지연 이자와 손해금을 과도하게 청구한다", "expected": "제8조"},
    {"query": "부당하게 과중한 손해배상액을 예정한다", "expected": "제8조"},
])

# 제9조 (계약의 해제·해지) - 15개
test_cases.extend([
    {"query": "계약을 해지할 수 없습니다", "expected": "제9조"},
    {"query": "법률에 따른 고객의 해제권을 배제한다", "expected": "제9조"},
    {"query": "고객의 해지권 행사를 제한한다", "expected": "제9조"},
    {"query": "사업자에게 법률에서 규정하지 않은 해제권을 부여한다", "expected": "제9조"},
    {"query": "사업자의 해제권 행사 요건을 완화한다", "expected": "제9조"},
    {"query": "계약 해제 시 원상회복의무를 고객에게 과중하게 부담시킨다", "expected": "제9조"},
    {"query": "고객의 원상회복 청구권을 부당하게 포기하도록 한다", "expected": "제9조"},
    {"query": "해제 또는 해지로 인한 사업자의 원상회복의무를 부당하게 경감한다", "expected": "제9조"},
    {"query": "존속기간을 부당하게 단기 또는 장기로 한다", "expected": "제9조"},
    {"query": "묵시적인 기간 연장이 가능하도록 정하여 고객에게 불이익을 준다", "expected": "제9조"},
    {"query": "고객의 해제권을 완전히 배제한다", "expected": "제9조"},
    {"query": "계약 해지가 불가능하도록 정한다", "expected": "제9조"},
    {"query": "사업자에게만 일방적인 해제권을 부여한다", "expected": "제9조"},
    {"query": "원상회복 의무를 고객에게만 부담시킨다", "expected": "제9조"},
    {"query": "계약 기간을 부당하게 자동 연장한다", "expected": "제9조"},
])

# 제10조 (채무의 이행) - 12개
test_cases.extend([
    {"query": "회사가 일방적으로 계약을 변경할 수 있습니다", "expected": "제10조"},
    {"query": "상당한 이유 없이 급부의 내용을 사업자가 일방적으로 결정할 수 있다", "expected": "제10조"},
    {"query": "사업자가 이행하여야 할 급부를 일방적으로 중지할 수 있다", "expected": "제10조"},
    {"query": "제3자에게 대행할 수 있게 하는 조항", "expected": "제10조"},
    {"query": "급부 내용을 사업자가 임의로 변경할 수 있다", "expected": "제10조"},
    {"query": "일방적으로 계약 내용을 수정할 수 있다", "expected": "제10조"},
    {"query": "이행 급부를 중단할 수 있는 권한을 부여한다", "expected": "제10조"},
    {"query": "급부를 제3자에게 위임할 수 있다", "expected": "제10조"},
    {"query": "사업자가 일방적으로 서비스 내용을 변경한다", "expected": "제10조"},
    {"query": "계약 조건을 사업자가 임의로 수정할 수 있다", "expected": "제10조"},
    {"query": "급부의 이행을 중지할 수 있는 권한을 가진다", "expected": "제10조"},
    {"query": "일방적으로 계약 이행을 변경하거나 중지할 수 있다", "expected": "제10조"},
])

# 제11조 (고객의 권익 보호) - 15개
test_cases.extend([
    {"query": "고객은 기한의 이익을 상실합니다", "expected": "제11조"},
    {"query": "고객의 항변권과 상계권을 배제합니다", "expected": "제11조"},
    {"query": "법률에 따른 고객의 항변권을 상당한 이유 없이 배제한다", "expected": "제11조"},
    {"query": "고객의 상계권을 제한한다", "expected": "제11조"},
    {"query": "기한의 이익을 박탈한다", "expected": "제11조"},
    {"query": "고객이 제3자와 계약을 체결하는 것을 부당하게 제한한다", "expected": "제11조"},
    {"query": "사업자가 업무상 알게 된 고객의 비밀을 정당한 이유 없이 누설하는 것을 허용한다", "expected": "제11조"},
    {"query": "고객의 기한의 이익을 상실시킨다", "expected": "제11조"},
    {"query": "항변권을 배제하거나 제한한다", "expected": "제11조"},
    {"query": "상계권을 상당한 이유 없이 배제한다", "expected": "제11조"},
    {"query": "기한의 이익을 상당한 이유 없이 박탈한다", "expected": "제11조"},
    {"query": "제3자와의 계약 체결을 부당하게 금지한다", "expected": "제11조"},
    {"query": "고객의 비밀을 누설할 수 있도록 허용한다", "expected": "제11조"},
    {"query": "고객의 권리를 부당하게 제한한다", "expected": "제11조"},
    {"query": "기한의 이익을 상실하게 하는 조항", "expected": "제11조"},
])

# 제12조 (의사표시의 의제) - 12개
test_cases.extend([
    {"query": "고객이 답변하지 않으면 동의한 것으로 간주합니다", "expected": "제12조"},
    {"query": "일정한 작위 또는 부작위가 있을 경우 고객의 의사표시가 표명된 것으로 본다", "expected": "제12조"},
    {"query": "고객의 의사표시 형식에 대하여 부당하게 엄격한 제한을 둔다", "expected": "제12조"},
    {"query": "고객의 이익에 중대한 영향을 미치는 사업자의 의사표시가 도달된 것으로 본다", "expected": "제12조"},
    {"query": "사업자의 의사표시 기한을 부당하게 길게 정한다", "expected": "제12조"},
    {"query": "부작위를 의사표시로 간주한다", "expected": "제12조"},
    {"query": "답변 없음을 동의로 간주한다", "expected": "제12조"},
    {"query": "의사표시 요건을 부당하게 엄격하게 한다", "expected": "제12조"},
    {"query": "사업자의 통지를 도달된 것으로 간주한다", "expected": "제12조"},
    {"query": "의사표시 기한을 불확정하게 정한다", "expected": "제12조"},
    {"query": "작위나 부작위를 의사표시로 의제한다", "expected": "제12조"},
    {"query": "고객의 의사표시를 부당하게 제한한다", "expected": "제12조"},
])

# 제13조 (대리인의 책임 가중) - 8개
test_cases.extend([
    {"query": "고객의 대리인이 의무를 이행할 책임을 집니다", "expected": "제13조"},
    {"query": "대리인에 의하여 계약이 체결된 경우 대리인에게 의무의 전부를 이행할 책임을 지운다", "expected": "제13조"},
    {"query": "고객이 의무를 이행하지 아니하는 경우 대리인에게 그 의무를 이행할 책임을 지운다", "expected": "제13조"},
    {"query": "대리인에게 고객의 의무를 부담시킨다", "expected": "제13조"},
    {"query": "대리인이 계약 의무를 이행할 책임을 진다", "expected": "제13조"},
    {"query": "대리인에게 책임을 가중시킨다", "expected": "제13조"},
    {"query": "고객의 대리인에게 의무 이행 책임을 부과한다", "expected": "제13조"},
    {"query": "대리인에게 고객의 채무를 이행할 책임을 지운다", "expected": "제13조"},
])

# 제14조 (소송 제기의 금지 등) - 10개
test_cases.extend([
    {"query": "이 계약에 관한 소송은 회사 본사 소재지 관할법원으로 합니다", "expected": "제14조"},
    {"query": "고객에게 부당하게 불리한 소송 제기 금지 조항", "expected": "제14조"},
    {"query": "재판관할의 합의 조항이 고객에게 불리하다", "expected": "제14조"},
    {"query": "상당한 이유 없이 고객에게 입증책임을 부담시킨다", "expected": "제14조"},
    {"query": "소송 제기를 금지하는 조항", "expected": "제14조"},
    {"query": "관할법원을 사업자에게 유리하게 정한다", "expected": "제14조"},
    {"query": "고객에게 불리한 재판관할 합의", "expected": "제14조"},
    {"query": "입증책임을 고객에게 부담시킨다", "expected": "제14조"},
    {"query": "소송 제기 금지 또는 관할법원 지정", "expected": "제14조"},
    {"query": "고객에게 부당하게 불리한 관할 합의", "expected": "제14조"},
])

# 총 120개 확인
print(f"총 테스트 케이스: {len(test_cases)}개")
print(f"조항별 분포:")
from collections import Counter
article_counts = Counter([tc["expected"] for tc in test_cases])
for article, count in sorted(article_counts.items()):
    print(f"  {article}: {count}개")

# JSON 파일로 저장
output_path = Path(__file__).parent / "comprehensive_test_cases.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(test_cases, f, ensure_ascii=False, indent=2)

print(f"\n테스트 케이스 저장 완료: {output_path}")

