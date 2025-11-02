"""
생성된 위반 문장들에서 키워드와 패턴 추출
"""
import json
import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Set
from collections import Counter

# 프로젝트 루트 경로 추가
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from utils.llm_client import get_llm_client
    LLM_AVAILABLE = True
except:
    LLM_AVAILABLE = False


def extract_keywords_llm(article_id: str, cases: List[str]) -> List[str]:
    """LLM을 사용하여 키워드 추출"""
    if not LLM_AVAILABLE:
        return []
    
    try:
        llm = get_llm_client()
        if not llm:
            return []
        
        prompt = f"""
다음은 {article_id}를 위반하는 약관 문장들입니다. 이 문장들에서 반복적으로 나타나는 핵심 키워드를 추출하세요.

위반 문장 예시 (일부):
{chr(10).join(cases[:10])}

요구사항:
1. 위반 판단에 핵심적인 단어나 구문을 추출
2. 실제 약관에서 자주 사용되는 표현의 핵심 단어
3. 10-20개 정도의 핵심 키워드 추출
4. 한국어 단어/구문

출력 형식: JSON 배열
예시: ["키워드1", "키워드2", ...]

JSON 배열만 반환:
"""
        
        result = llm.generate_json(prompt)
        if isinstance(result, list):
            return result
        elif isinstance(result, dict) and 'keywords' in result:
            return result['keywords']
        else:
            return []
    except Exception as e:
        print(f"   ⚠️ 키워드 추출 실패: {e}")
        return []


def extract_patterns_llm(article_id: str, cases: List[str]) -> List[str]:
    """LLM을 사용하여 패턴 추출"""
    if not LLM_AVAILABLE:
        return []
    
    try:
        llm = get_llm_client()
        if not llm:
            return []
        
        prompt = f"""
다음은 {article_id}를 위반하는 약관 문장들입니다. 이 문장들에서 반복적으로 나타나는 정규표현식 패턴을 추출하세요.

위반 문장 예시 (일부):
{chr(10).join(cases[:10])}

요구사항:
1. 위반 판단에 유용한 정규표현식 패턴 추출
2. Python re 모듈에서 사용 가능한 형식
3. 5-15개 정도의 핵심 패턴
4. 다양한 표현 방식을 포함하는 패턴

출력 형식: JSON 배열
예시: ["패턴1", "패턴2", ...]

JSON 배열만 반환:
"""
        
        result = llm.generate_json(prompt)
        if isinstance(result, list):
            return result
        elif isinstance(result, dict) and 'patterns' in result:
            return result['patterns']
        else:
            return []
    except Exception as e:
        print(f"   ⚠️ 패턴 추출 실패: {e}")
        return []


def extract_keywords_statistical(cases: List[str], top_n: int = 20) -> List[str]:
    """통계적 방법으로 키워드 추출 (LLM 실패 시 폴백)"""
    # 일반적인 불용어 (약관 판단에 유용하지 않은 단어)
    stopwords = {'은', '는', '이', '가', '을', '를', '의', '에', '와', '과', '로', '으로', 
                 '으로서', '로서', '에게', '에게서', '도', '만', '까지', '부터', '에서', 
                 '의', '한', '다', '것', '수', '등', '및', '또한', '또는', '또한', '그', 
                 '이', '저', '이런', '그런', '저런', '때문', '경우', '때', '때문에'}
    
    # 명사/동사 추출을 위한 간단한 방법
    all_words = []
    for case in cases:
        # 한글 단어 추출 (2글자 이상)
        words = re.findall(r'[가-힣]{2,}', case)
        all_words.extend([w for w in words if w not in stopwords and len(w) >= 2])
    
    # 빈도수 상위 N개
    counter = Counter(all_words)
    return [word for word, count in counter.most_common(top_n)]


def extract_patterns_statistical(cases: List[str]) -> List[str]:
    """통계적 방법으로 패턴 추출 (간단한 패턴)"""
    patterns = []
    
    # 부정 표현 패턴
    negation_patterns = [
        r"책임.*지지.*않",
        r"배상.*않",
        r"면책",
        r"제한.*금지",
        r"부당.*불리"
    ]
    
    # 공통 패턴 찾기
    for pattern in negation_patterns:
        matches = sum(1 for case in cases if re.search(pattern, case))
        if matches >= len(cases) * 0.1:  # 10% 이상 매칭
            patterns.append(pattern)
    
    return patterns


def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("키워드 및 패턴 추출")
    print("=" * 70)
    
    input_file = "data/contracts/violation_cases/generated_violation_cases.json"
    output_file = "data/contracts/violation_cases/extracted_keywords_patterns.json"
    
    if not os.path.exists(input_file):
        print(f"❌ 입력 파일 없음: {input_file}")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = {}
    total = len(data)
    current = 0
    
    for article_id, item in data.items():
        current += 1
        cases = item.get('violation_cases', [])
        article_group = item.get('article_group', '')
        
        print(f"\n[{current}/{total}] {article_id}...", end=" ", flush=True)
        
        # LLM으로 키워드 추출 시도
        keywords = extract_keywords_llm(article_id, cases)
        if not keywords or len(keywords) < 5:
            # 폴백: 통계적 방법
            keywords = extract_keywords_statistical(cases, top_n=20)
            print(f"통계적 방법 사용", end=" ")
        
        # LLM으로 패턴 추출 시도
        patterns = extract_patterns_llm(article_id, cases)
        if not patterns or len(patterns) < 3:
            # 폴백: 통계적 방법
            patterns = extract_patterns_statistical(cases)
        
        results[article_id] = {
            "article_group": article_group,
            "content": item.get('content', ''),
            "keywords": keywords,
            "patterns": patterns,
            "case_count": len(cases)
        }
        
        print(f"✅ 키워드 {len(keywords)}개, 패턴 {len(patterns)}개")
    
    # 결과 저장
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ 완료!")
    print(f"   저장 위치: {output_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

