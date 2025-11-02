"""
LLM을 사용한 고급 패턴 추출
실패한 케이스를 분석해서 더 정확한 패턴 생성
"""
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from utils.llm_client import get_llm_client
    LLM_AVAILABLE = True
except:
    LLM_AVAILABLE = False


def extract_discriminative_keywords(expected_article: str, failed_texts: List[str], 
                                    confused_with: str, all_cases: Dict[str, List[str]]) -> List[str]:
    """실패한 케이스와 혼동된 조항의 차이를 분석하여 구분 키워드 추출"""
    if not LLM_AVAILABLE or not failed_texts:
        return []
    
    try:
        llm = get_llm_client()
        if not llm:
            return []
        
        # 예상 조항의 정답 케이스 일부
        correct_examples = all_cases.get(expected_article, [])[:5]
        confused_examples = all_cases.get(confused_with, [])[:5]
        
        prompt = f"""
다음은 약관법 조항 판단에서 혼동되는 케이스입니다.

**예상 조항: {expected_article}** (정답 케이스)
{chr(10).join(f"- {ex}" for ex in correct_examples)}

**혼동된 조항: {confused_with}** (혼동 케이스)
{chr(10).join(f"- {ex}" for ex in confused_examples)}

**실패한 텍스트** ({expected_article}이어야 하는데 {confused_with}로 잘못 판단됨):
{chr(10).join(f"- {text}" for text in failed_texts[:3])}

요구사항:
1. {expected_article}와 {confused_with}를 구분할 수 있는 핵심 키워드/구문 추출
2. {expected_article}에만 특유한 표현 찾기
3. 5-10개의 구체적인 키워드/구문 제안

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
        print(f"   ⚠️ LLM 키워드 추출 실패: {e}")
        return []


def improve_with_llm_analysis(failed_cases: List[Dict], cases_by_article: Dict[str, List[str]], 
                             current_patterns: Dict) -> Dict:
    """LLM을 사용하여 실패 케이스를 분석하고 패턴 개선"""
    if not LLM_AVAILABLE:
        return current_patterns, {}
    
    improvements_made = {}
    
    # 혼동 패턴 분석
    confusion_groups = {}
    for case in failed_cases:
        expected = case['expected']
        predicted = case.get('predicted', '제6조')
        key = f"{expected}→{predicted}"
        if key not in confusion_groups:
            confusion_groups[key] = []
        confusion_groups[key].append(case['text'])
    
    # 각 혼동 패턴별로 LLM 분석
    for confusion_key, texts in confusion_groups.items():
        expected, predicted = confusion_key.split('→')
        
        # LLM으로 구분 키워드 추출
        discriminative_keywords = extract_discriminative_keywords(
            expected, texts, predicted, cases_by_article
        )
        
        if discriminative_keywords and expected in current_patterns:
            improved_keywords = current_patterns[expected].get('keywords', [])
            added = []
            
            for kw in discriminative_keywords[:5]:  # 상위 5개만
                if kw not in improved_keywords and len(kw) >= 4:
                    improved_keywords.append(kw)
                    added.append(kw)
            
            if added:
                current_patterns[expected]['keywords'] = improved_keywords[:40]  # 최대 40개
                if expected not in improvements_made:
                    improvements_made[expected] = []
                improvements_made[expected].extend([
                    f"LLM 구분 키워드 추가: {kw}" for kw in added
                ])
    
    return current_patterns, improvements_made

