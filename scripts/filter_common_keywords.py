"""
너무 일반적인 키워드 필터링 및 조항별 고유 키워드 선별
"""
import json
import os
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def filter_common_keywords():
    """일반적인 키워드 필터링"""
    input_file = "data/contracts/violation_cases/updated_article_patterns.json"
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 모든 조항에서 공통으로 나타나는 키워드 찾기
    all_keywords = {}
    for article_id, item in data.items():
        for kw in item.get('keywords', []):
            if kw not in all_keywords:
                all_keywords[kw] = []
            all_keywords[kw].append(article_id)
    
    # 너무 일반적인 키워드 (단일 단어이거나 3개 이상 조항에서 나타나는 것)
    too_common = set()
    too_short = set()
    
    common_words = {'고객', '회사', '서비스', '계약', '책임', '손해', '이용', '변경', '제한', '동의', 
                   '통지', '사전', '해지', '배상', '이익', '의무', '권리', '분쟁', '입증', '소송'}
    
    for kw, articles in all_keywords.items():
        # 너무 짧은 키워드 (1-2글자 단어 또는 공통 단어 포함)
        if len(kw) <= 2 or any(cw in kw for cw in common_words) and len(kw) <= 5:
            too_short.add(kw)
        
        # 3개 이상 조항에서 공통으로 나타나는 키워드
        if len(articles) >= 3:
            too_common.add(kw)
    
    print(f"=== 필터링 통계 ===")
    print(f"너무 짧은 키워드: {len(too_short)}개")
    print(f"너무 일반적인 키워드 (3개 이상 조항): {len(too_common)}개")
    print()
    
    # 필터링된 데이터 생성
    filtered_data = {}
    
    for article_id, item in data.items():
        # 키워드 필터링 (너무 일반적인 것만 제거, 최소 키워드 수 보장)
        keywords = item.get('keywords', [])
        
        # 1단계: 너무 짧은 키워드만 제거 (단일 단어가 아닌 2글자 이하만)
        filtered_keywords = [
            kw for kw in keywords 
            if not (len(kw) <= 2 and kw in common_words)
        ]
        
        # 2단계: 5개 이상 조항에서 공통인 키워드만 제거 (너무 일반적인 것)
        very_common = {kw for kw, articles in all_keywords.items() if len(articles) >= 5}
        filtered_keywords = [
            kw for kw in filtered_keywords 
            if kw not in very_common
        ]
        
        # 3단계: 구체적인 키워드 우선 정렬 (긴 키워드 우선)
        filtered_keywords.sort(key=lambda x: (len(x), -keywords.index(x) if x in keywords else 0), reverse=True)
        
        # 최소 키워드 수 보장 (각 조항별 최소 10개)
        min_keywords = 10
        if len(filtered_keywords) < min_keywords:
            # 필터링되지 않은 키워드에서 추가
            remaining = [kw for kw in keywords if kw not in filtered_keywords]
            remaining.sort(key=lambda x: len(x), reverse=True)
            filtered_keywords.extend(remaining[:min_keywords - len(filtered_keywords)])
        
        # 최대 30개만 유지
        filtered_keywords = filtered_keywords[:30]
        
        filtered_data[article_id] = {
            "keywords": filtered_keywords,
            "patterns": item.get('patterns', [])[:15]  # 패턴도 최대 15개
        }
        
        print(f"{article_id}: {len(keywords)}개 → {len(filtered_keywords)}개")
    
    # 저장
    output_file = "data/contracts/violation_cases/filtered_article_patterns.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 필터링 완료!")
    print(f"   저장: {output_file}")
    
    return filtered_data


if __name__ == "__main__":
    filter_common_keywords()

