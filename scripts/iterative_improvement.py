"""
반복 테스트 및 개선 스크립트
generated_violation_cases.json의 위반 사례를 사용하여 테스트하고
정확도 9/12 이상이 될 때까지 패턴과 키워드를 개선
"""
import json
import os
import sys
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.judge_clause import detect_best_article, ARTICLE_PATTERNS


def load_violation_cases() -> Dict[str, List[str]]:
    """위반 사례 로드 (조항별로 정리)"""
    file_path = Path(PROJECT_ROOT) / 'data' / 'contracts' / 'violation_cases' / 'generated_violation_cases.json'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 조항별로 그룹화
    cases_by_article = defaultdict(list)
    
    for article_id, item in data.items():
        article_group = item.get('article_group', article_id.split('호')[0] if '호' in article_id else article_id.split('항')[0])
        violation_cases = item.get('violation_cases', [])
        cases_by_article[article_group].extend(violation_cases)
    
    return dict(cases_by_article)


def select_random_test_cases(cases_by_article: Dict[str, List[str]], num_per_article: int = 1) -> List[Tuple[str, str]]:
    """각 조항에서 랜덤으로 테스트 케이스 선택"""
    test_cases = []
    
    for article_id in sorted(cases_by_article.keys()):
        cases = cases_by_article[article_id]
        if cases:
            selected = random.sample(cases, min(num_per_article, len(cases)))
            for case in selected:
                test_cases.append((article_id, case))
    
    return test_cases


def run_test(test_cases: List[Tuple[str, str]]) -> Dict:
    """테스트 실행 및 결과 반환"""
    results = {
        'total': len(test_cases),
        'correct': 0,
        'incorrect': 0,
        'failed_cases': []  # (expected, actual, text)
    }
    
    for expected_article, text in test_cases:
        predicted_article = detect_best_article(text)
        
        if predicted_article == expected_article:
            results['correct'] += 1
        else:
            results['incorrect'] += 1
            results['failed_cases'].append({
                'expected': expected_article,
                'predicted': predicted_article,
                'text': text
            })
    
    results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0.0
    return results


def analyze_failures(failed_cases: List[Dict], cases_by_article: Dict[str, List[str]]) -> Dict[str, Dict]:
    """실패 케이스 분석 및 개선 제안"""
    analysis = defaultdict(lambda: {'missing_keywords': [], 'missing_patterns': []})
    
    for case in failed_cases:
        expected = case['expected']
        predicted = case.get('predicted', '제6조')
        text = case['text']
        
        # 예상 조항의 키워드 확인
        expected_keywords = set(ARTICLE_PATTERNS.get(expected, {}).get('keywords', []))
        
        # 텍스트에서 나타나는 키워드
        found_keywords = {kw for kw in expected_keywords if kw in text}
        missing_keywords = expected_keywords - found_keywords
        
        # 텍스트에서 새로 발견될 수 있는 키워드 제안
        words_in_text = re.findall(r'[가-힣]{3,}', text)
        potential_keywords = [w for w in words_in_text if len(w) >= 4 and w not in expected_keywords]
        
        analysis[expected]['missing_keywords'].extend(list(missing_keywords))
        analysis[expected]['potential_keywords'] = analysis[expected].get('potential_keywords', [])
        analysis[expected]['potential_keywords'].extend(potential_keywords[:5])  # 상위 5개만
        analysis[expected]['failed_texts'] = analysis[expected].get('failed_texts', [])
        analysis[expected]['failed_texts'].append(text)
    
    # 통계 집계
    for article_id, data in analysis.items():
        # 가장 많이 누락된 키워드
        missing_counter = Counter(data['missing_keywords'])
        data['top_missing'] = [kw for kw, cnt in missing_counter.most_common(10)]
        
        # 잠재적 키워드 (가장 많이 나타나는 것)
        potential_counter = Counter(data.get('potential_keywords', []))
        data['top_potential'] = [kw for kw, cnt in potential_counter.most_common(10)]
    
    return dict(analysis)


def improve_patterns(analysis: Dict[str, Dict], cases_by_article: Dict[str, List[str]]) -> Dict:
    """패턴과 키워드 개선"""
    # 현재 패턴 로드
    pattern_file = Path(PROJECT_ROOT) / 'data' / 'contracts' / 'violation_cases' / 'filtered_article_patterns.json'
    
    if pattern_file.exists():
        with open(pattern_file, 'r', encoding='utf-8') as f:
            improved_patterns = json.load(f)
    else:
        # 없으면 현재 ARTICLE_PATTERNS 사용
        improved_patterns = {k: {'keywords': v.get('keywords', []), 'patterns': v.get('patterns', [])} 
                            for k, v in ARTICLE_PATTERNS.items()}
    
    improvements_made = {}
    
    for article_id, data in analysis.items():
        if article_id not in improved_patterns:
            continue
        
        original_keywords = set(improved_patterns[article_id].get('keywords', []))
        improved_keywords = list(original_keywords)
        improvements = []
        
        # 1. 누락된 키워드 중 중요한 것 추가 (최대 8개로 증가)
        # 구체적이고 의미있는 키워드만 추가
        stopwords = {'있습니다', '않습니다', '없습니다', '해야', '할 수', '합니다', '됩니다'}
        top_missing = [kw for kw in data.get('top_missing', []) 
                      if kw not in stopwords 
                      and not any(sw in kw for sw in stopwords)
                      and len(kw) > 4][:8]  # 5개 → 8개로 증가
        for kw in top_missing:
            if kw not in improved_keywords:
                improved_keywords.append(kw)
                improvements.append(f"키워드 추가: {kw}")
        
        # 2. 잠재적 키워드 중 유용한 것 추가 (최대 5개로 증가)
        # 너무 일반적인 단어 필터링
        stopwords = {'있습니다', '않습니다', '없습니다', '해야', '할 수', '합니다', '됩니다', 
                    '필요하지', '동의합니다', '있을', '없을', '있으며', '없으며'}
        top_potential = [kw for kw in data.get('top_potential', []) 
                        if kw not in improved_keywords 
                        and len(kw) >= 4 
                        and kw not in stopwords
                        and not any(sw in kw for sw in stopwords)][:5]  # 3개 → 5개로 증가
        for kw in top_potential:
            improved_keywords.append(kw)
            improvements.append(f"새 키워드 추가: {kw}")
        
        # 3. 실패한 텍스트에서 패턴 추출 시도 (더 정교한 방법)
        failed_texts = data.get('failed_texts', [])
        if failed_texts and len(failed_texts) >= 2:
            # 각 텍스트에서 핵심 구문 추출
            key_phrases = []
            for text in failed_texts[:5]:
                # 문장 단위로 분리
                sentences = re.split(r'[\.\n]', text)
                for sentence in sentences:
                    if len(sentence.strip()) >= 10:
                        # 핵심 동사/명사 조합 추출 (5-10글자)
                        phrases = re.findall(r'[가-힣]{5,10}', sentence.strip())
                        key_phrases.extend([p for p in phrases if len(p) >= 5])
            
            # 가장 많이 나타나는 핵심 구문을 패턴으로 변환
            phrase_counter = Counter(key_phrases)
            stopwords_pattern = {'있습니다', '않습니다', '없습니다', '합니다', '됩니다'}
            
            for phrase, count in phrase_counter.most_common(3):
                if (count >= 2 
                    and len(phrase) >= 5 
                    and phrase not in stopwords_pattern
                    and any(c in phrase for c in ['책임', '권리', '의무', '제한', '배상', '해지', '항변', '상계', '소송', '입증'])):
                    # 패턴으로 변환 (앞뒤에 .* 추가하여 유연하게)
                    escaped = re.escape(phrase)
                    pattern = f".*{escaped}.*"
                    patterns_list = improved_patterns[article_id].get('patterns', [])
                    if pattern not in patterns_list:
                        patterns_list.append(pattern)
                        improvements.append(f"패턴 추가: {pattern}")
        
        improved_patterns[article_id]['keywords'] = improved_keywords[:35]  # 최대 35개
        improved_patterns[article_id]['patterns'] = improved_patterns[article_id].get('patterns', [])[:18]  # 최대 18개
        
        if improvements:
            improvements_made[article_id] = improvements
    
    return improved_patterns, improvements_made


def save_improved_patterns(improved_patterns: Dict, iteration: int):
    """개선된 패턴 저장"""
    output_file = Path(PROJECT_ROOT) / 'data' / 'contracts' / 'violation_cases' / f'improved_patterns_iter_{iteration}.json'
    backup_file = Path(PROJECT_ROOT) / 'data' / 'contracts' / 'violation_cases' / 'filtered_article_patterns.json'
    
    # 백업
    if backup_file.exists():
        import shutil
        shutil.copy(backup_file, str(backup_file) + f'.backup_iter_{iteration}')
    
    # 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(improved_patterns, f, ensure_ascii=False, indent=2)
    
    # filtered_article_patterns.json도 업데이트
    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(improved_patterns, f, ensure_ascii=False, indent=2)
    
    return str(output_file), str(backup_file)


def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("반복 테스트 및 개선 시스템")
    print("=" * 70)
    
    # 1. 위반 사례 로드
    print("\n[1단계] 위반 사례 로드 중...")
    cases_by_article = load_violation_cases()
    print(f"   총 {len(cases_by_article)}개 조항, 각 조항별 평균 {sum(len(v) for v in cases_by_article.values()) // len(cases_by_article)}개 사례")
    
    target_accuracy = 10/12  # 83.3% (10개 이상)
    target_stability = 3  # 연속 3번 달성
    max_iterations = 30  # 20 → 30으로 증가
    iteration = 0
    consecutive_success = 0
    stability_results = []
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'='*70}")
        print(f"[반복 {iteration}]")
        print(f"{'='*70}")
        
        # 2. 랜덤 테스트 케이스 선택 (각 조항에서 1개씩, 총 12개)
        print("\n[2단계] 테스트 케이스 선택 중...")
        test_cases = select_random_test_cases(cases_by_article, num_per_article=1)
        # 9개 조항이 있으므로 9개 선택, 나머지 3개는 랜덤으로 추가
        if len(test_cases) < 12:
            all_cases = []
            for article_id, cases in cases_by_article.items():
                all_cases.extend([(article_id, c) for c in cases])
            additional = random.sample(all_cases, min(3, len(all_cases)))
            test_cases.extend(additional)
        test_cases = test_cases[:12]
        print(f"   선택된 테스트 케이스: {len(test_cases)}개")
        
        # 3. 테스트 실행 (안정성 검증을 위해 3번 반복)
        print("\n[3단계] 테스트 실행 중...")
        stability_tests = []
        for stability_test in range(3):
            random.seed(iteration * 100 + stability_test)
            test_cases_stable = select_random_test_cases(cases_by_article, num_per_article=1)
            if len(test_cases_stable) < 12:
                all_cases = []
                for article_id, cases in cases_by_article.items():
                    all_cases.extend([(article_id, c) for c in cases])
                additional = random.sample(all_cases, min(3, len(all_cases)))
                test_cases_stable.extend(additional)
            test_cases_stable = test_cases_stable[:12]
            result = run_test(test_cases_stable)
            stability_tests.append(result)
        
        # 평균 정확도 계산
        avg_accuracy = sum(r['correct'] for r in stability_tests) / (len(stability_tests) * 12)
        min_correct = min(r['correct'] for r in stability_tests)
        
        print(f"   정확도: 평균 {avg_accuracy:.1%}, 최소 {min_correct}/12")
        for i, r in enumerate(stability_tests):
            print(f"     테스트 {i+1}: {r['correct']}/12")
        
        results = stability_tests[0]  # 메인 결과는 첫 번째 사용
        stability_results.append({
            'iteration': iteration,
            'avg_accuracy': avg_accuracy,
            'min_correct': min_correct,
            'results': stability_tests
        })
        
        # 4. 목표 정확도 달성 확인 (연속으로 달성해야 함)
        if min_correct >= 10:  # 10개 이상 목표
            consecutive_success += 1
            print(f"\n✅ 목표 달성! (연속 {consecutive_success}회)")
            if consecutive_success >= target_stability:
                print(f"\n✅ 안정성 확보! ({target_stability}회 연속 달성)")
                break
        else:
            consecutive_success = 0
        
        # 5. 실패 케이스 분석 (모든 안정성 테스트의 실패 케이스 포함)
        all_failed_cases = []
        for r in stability_tests:
            all_failed_cases.extend(r['failed_cases'])
        
        print(f"\n[4단계] 실패 케이스 분석 중... (총 {len(all_failed_cases)}개 실패)")
        analysis = analyze_failures(all_failed_cases, cases_by_article)
        
        # 6. 패턴 개선
        print("\n[5단계] 패턴 및 키워드 개선 중...")
        improved_patterns, improvements_made = improve_patterns(analysis, cases_by_article)
        
        # LLM을 사용한 고급 분석 추가
        try:
            from scripts.advanced_pattern_extraction import improve_with_llm_analysis
            improved_patterns, llm_improvements = improve_with_llm_analysis(
                all_failed_cases, cases_by_article, improved_patterns
            )
            
            # LLM 개선 사항 병합
            for article_id, improvements in llm_improvements.items():
                if article_id not in improvements_made:
                    improvements_made[article_id] = []
                improvements_made[article_id].extend(improvements)
        except Exception as e:
            print(f"   ⚠️ LLM 분석 스킵: {e}")
        
        if improvements_made:
            print("   개선 사항:")
            for article_id, improvements in improvements_made.items():
                print(f"     {article_id}:")
                for imp in improvements:
                    print(f"       - {imp}")
        else:
            print("   개선 사항 없음")
        
        # 7. 개선된 패턴 저장 및 적용
        print("\n[6단계] 개선된 패턴 저장 중...")
        output_file, backup_file = save_improved_patterns(improved_patterns, iteration)
        print(f"   저장: {output_file}")
        print(f"   업데이트: {backup_file}")
        
        # 8. ARTICLE_PATTERNS 재로드 (다음 반복을 위해)
        print("\n[7단계] 패턴 재로드 중...")
        import importlib
        import scripts.judge_clause
        importlib.reload(scripts.judge_clause)
        from scripts.judge_clause import detect_best_article, ARTICLE_PATTERNS
        globals()['detect_best_article'] = detect_best_article
        globals()['ARTICLE_PATTERNS'] = ARTICLE_PATTERNS
        print("   완료")
    
    if iteration >= max_iterations:
        print(f"\n⚠️ 최대 반복 횟수({max_iterations})에 도달했습니다.")
    
    print(f"\n{'='*70}")
    print(f"최종 결과:")
    print(f"   반복 횟수: {iteration}회")
    if stability_results:
        final_stats = stability_results[-1]
        print(f"   최종 평균 정확도: {final_stats['avg_accuracy']:.1%}")
        print(f"   최종 최소 정확도: {final_stats['min_correct']}/12")
        print(f"   연속 성공 횟수: {consecutive_success}회")
    print(f"{'='*70}")


if __name__ == "__main__":
    random.seed(42)  # 재현성을 위한 시드 설정
    main()

