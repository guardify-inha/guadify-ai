"""
각 최말단 조항별 위반 문장 생성 스크립트
"""
import json
import os
import sys
from pathlib import Path
from typing import List, Dict

# 프로젝트 루트 경로 추가
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 조항 데이터
ARTICLES = {
    "제6조": [
        {"id": "제6조1항", "content": "신의성실의 원칙을 위반하여 공정성을 잃은 약관 조항은 무효이다."},
        {"id": "제6조2항1호", "content": "고객에게 부당하게 불리한 조항"},
        {"id": "제6조2항2호", "content": "고객이 계약의 거래형태 등 관련된 모든 사정에 비추어 예상하기 어려운 조항"},
        {"id": "제6조2항3호", "content": "계약의 목적을 달성할 수 없을 정도로 계약에 따르는 본질적 권리를 제한하는 조항"},
    ],
    "제7조": [
        {"id": "제7조1호", "content": "사업자, 이행 보조자 또는 피고용자의 고의 또는 중대한 과실로 인한 법률상의 책임을 배제하는 조항"},
        {"id": "제7조2호", "content": "상당한 이유 없이 사업자의 손해배상 범위를 제한하거나 사업자가 부담하여야 할 위험을 고객에게 떠넘기는 조항"},
        {"id": "제7조3호", "content": "상당한 이유 없이 사업자의 담보책임을 배제 또는 제한하거나 그 담보책임에 따르는 고객의 권리행사의 요건을 가중하는 조항"},
        {"id": "제7조4호", "content": "상당한 이유 없이 계약목적물에 관하여 견본이 제시되거나 품질ㆍ성능 등에 관한 표시가 있는 경우 그 보장된 내용에 대한 책임을 배제 또는 제한하는 조항"},
    ],
    "제8조": [
        {"id": "제8조", "content": "고객에게 부당하게 과중한 지연 손해금 등의 손해배상 의무를 부담시키는 약관 조항은 무효로 한다."},
    ],
    "제9조": [
        {"id": "제9조1호", "content": "법률에 따른 고객의 해제권 또는 해지권을 배제하거나 그 행사를 제한하는 조항"},
        {"id": "제9조2호", "content": "사업자에게 법률에서 규정하고 있지 아니하는 해제권 또는 해지권을 부여하여 고객에게 부당하게 불이익을 줄 우려가 있는 조항"},
        {"id": "제9조3호", "content": "법률에 따른 사업자의 해제권 또는 해지권의 행사 요건을 완화하여 고객에게 부당하게 불이익을 줄 우려가 있는 조항"},
        {"id": "제9조4호", "content": "계약의 해제 또는 해지로 인한 원상회복의무를 상당한 이유 없이 고객에게 과중하게 부담시키거나 고객의 원상회복 청구권을 부당하게 포기하도록 하는 조항"},
        {"id": "제9조5호", "content": "계약의 해제 또는 해지로 인한 사업자의 원상회복의무나 손해배상의무를 부당하게 경감하는 조항"},
        {"id": "제9조6호", "content": "계속적인 채권관계의 발생을 목적으로 하는 계약에서 그 존속기간을 부당하게 단기 또는 장기로 하거나 묵시적인 기간의 연장 또는 갱신이 가능하도록 정하여 고객에게 부당하게 불이익을 줄 우려가 있는 조항"},
    ],
    "제10조": [
        {"id": "제10조1호", "content": "상당한 이유 없이 급부(給付)의 내용을 사업자가 일방적으로 결정하거나 변경할 수 있도록 권한을 부여하는 조항"},
        {"id": "제10조2호", "content": "상당한 이유 없이 사업자가 이행하여야 할 급부를 일방적으로 중지할 수 있게 하거나 제3자에게 대행할 수 있게 하는 조항"},
    ],
    "제11조": [
        {"id": "제11조1호", "content": "법률에 따른 고객의 항변권(抗辯權), 상계권(相計權) 등의 권리를 상당한 이유 없이 배제하거나 제한하는 조항"},
        {"id": "제11조2호", "content": "고객에게 주어진 기한의 이익을 상당한 이유 없이 박탈하는 조항"},
        {"id": "제11조3호", "content": "고객이 제3자와 계약을 체결하는 것을 부당하게 제한하는 조항"},
        {"id": "제11조4호", "content": "사업자가 업무상 알게 된 고객의 비밀을 정당한 이유 없이 누설하는 것을 허용하는 조항"},
    ],
    "제12조": [
        {"id": "제12조1호", "content": "일정한 작위(作爲) 또는 부작위(不作爲)가 있을 경우 고객의 의사표시가 표명되거나 표명되지 아니한 것으로 보는 조항. 다만, 고객에게 상당한 기한 내에 의사표시를 하지 아니하면 의사표시가 표명되거나 표명되지 아니한 것으로 본다는 뜻을 명확하게 따로 고지한 경우이거나 부득이한 사유로 그러한 고지를 할 수 없는 경우에는 그러하지 아니하다."},
        {"id": "제12조2호", "content": "고객의 의사표시의 형식이나 요건에 대하여 부당하게 엄격한 제한을 두는 조항"},
        {"id": "제12조3호", "content": "고객의 이익에 중대한 영향을 미치는 사업자의 의사표시가 상당한 이유 없이 고객에게 도달된 것으로 보는 조항"},
        {"id": "제12조4호", "content": "고객의 이익에 중대한 영향을 미치는 사업자의 의사표시 기한을 부당하게 길게 정하거나 불확정하게 정하는 조항"},
    ],
    "제13조": [
        {"id": "제13조", "content": "고객의 대리인에 의하여 계약이 체결된 경우 고객이 그 의무를 이행하지 아니하는 경우에는 대리인에게 그 의무의 전부 또는 일부를 이행할 책임을 지우는 내용의 약관 조항은 무효로 한다."},
    ],
    "제14조": [
        {"id": "제14조1호", "content": "고객에게 부당하게 불리한 소송 제기 금지 조항 또는 재판관할의 합의 조항"},
        {"id": "제14조2호", "content": "상당한 이유 없이 고객에게 입증책임을 부담시키는 약관 조항"},
    ],
}


def generate_violation_cases_for_article(article_id: str, content: str, count: int = 30) -> List[str]:
    """
    LLM을 사용하여 특정 조항에 대한 위반 문장 생성
    """
    try:
        from utils.llm_client import get_llm_client
        llm = get_llm_client()
        if not llm:
            return []
        
        prompt = f"""
다음 약관법 조항을 위반할 수 있는 구체적인 약관 문장을 {count}개 생성하세요.

조항 내용:
{content}

요구사항:
1. 위반 문장은 실제 약관에서 사용될 수 있는 실제적인 표현이어야 합니다
2. 각 문장은 서로 중복되지 않아야 합니다
3. 다양한 변형과 표현 방식을 사용하세요
4. 직접적인 위반 표현과 간접적인 위반 표현을 모두 포함하세요

출력 형식: JSON 배열
예시:
["문장1", "문장2", ..., "문장{count}"]

JSON 배열만 반환하세요:
"""
        
        result = llm.generate_json(prompt)
        if isinstance(result, list):
            return result[:count]
        elif isinstance(result, dict) and 'cases' in result:
            return result['cases'][:count]
        else:
            return []
            
    except Exception as e:
        print(f"   ⚠️ 생성 실패 ({article_id}): {e}")
        return []


def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("각 최말단 조항별 위반 문장 생성")
    print("=" * 70)
    
    output_dir = "data/contracts/violation_cases"
    os.makedirs(output_dir, exist_ok=True)
    
    all_cases = {}
    total_articles = sum(len(items) for items in ARTICLES.values())
    current = 0
    
    for article_group, items in ARTICLES.items():
        print(f"\n{article_group} 처리 중...")
        for item in items:
            current += 1
            article_id = item['id']
            content = item['content']
            
            print(f"  [{current}/{total_articles}] {article_id}...", end=" ", flush=True)
            
            cases = generate_violation_cases_for_article(article_id, content, count=30)
            
            if cases:
                all_cases[article_id] = {
                    "article_group": article_group,
                    "content": content,
                    "violation_cases": cases,
                    "count": len(cases)
                }
                print(f"✅ {len(cases)}개 생성")
            else:
                print(f"❌ 생성 실패")
    
    # 결과 저장
    output_file = os.path.join(output_dir, "generated_violation_cases.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ 완료!")
    print(f"   총 {len(all_cases)}개 조항 처리")
    print(f"   총 {sum(v['count'] for v in all_cases.values())}개 문장 생성")
    print(f"   저장 위치: {output_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

