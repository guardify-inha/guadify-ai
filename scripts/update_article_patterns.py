"""
추출된 키워드와 패턴으로 ARTICLE_PATTERNS 업데이트
"""
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def aggregate_by_article_group(data: dict) -> dict:
    """조항 그룹별로 키워드와 패턴 통합"""
    groups = defaultdict(lambda: {'keywords': set(), 'patterns': []})
    
    for article_id, item in data.items():
        group = item['article_group']
        
        # 키워드 통합
        for kw in item.get('keywords', []):
            groups[group]['keywords'].add(kw)
        
        # 패턴 통합
        for pat in item.get('patterns', []):
            if pat not in groups[group]['patterns']:
                groups[group]['patterns'].append(pat)
    
    # set을 list로 변환하고 정렬
    result = {}
    for group, item in groups.items():
        result[group] = {
            'keywords': sorted(list(item['keywords'])),
            'patterns': item['patterns']
        }
    
    return result


def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("ARTICLE_PATTERNS 업데이트 데이터 생성")
    print("=" * 70)
    
    input_file = "data/contracts/violation_cases/extracted_keywords_patterns.json"
    
    if not os.path.exists(input_file):
        print(f"❌ 입력 파일 없음: {input_file}")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 조항 그룹별 통합
    aggregated = aggregate_by_article_group(data)
    
    # Python 코드 형식으로 출력
    print("\n=== 업데이트된 ARTICLE_PATTERNS ===\n")
    print("ARTICLE_PATTERNS = {")
    
    for group in ['제6조', '제7조', '제8조', '제9조', '제10조', '제11조', '제12조', '제13조', '제14조']:
        if group in aggregated:
            item = aggregated[group]
            keywords = item['keywords'][:30]  # 상위 30개만
            patterns = item['patterns'][:15]  # 상위 15개만
            
            print(f'    "{group}": {{')
            print(f'        "keywords": {json.dumps(keywords, ensure_ascii=False)},')
            print(f'        "patterns": {json.dumps(patterns, ensure_ascii=False)}')
            print(f'    }},')
    
    print("}")
    
    # JSON 파일로도 저장
    output_file = "data/contracts/violation_cases/updated_article_patterns.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ 완료!")
    print(f"   JSON 저장: {output_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

