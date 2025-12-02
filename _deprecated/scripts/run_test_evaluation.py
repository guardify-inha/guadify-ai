"""
테스트 케이스 평가 스크립트

ArticleViolationScorer로 테스트 케이스를 평가하고 상세 결과 분석
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
from judge.article_violation_scorer import ArticleViolationScorer


def load_test_cases(filepath: str = "data/test_cases/test_cases.json"):
    """테스트 케이스 로드"""
    path = project_root / filepath
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_test_cases(scorer, test_data):
    """테스트 케이스 평가"""
    results = []

    for case in test_data['test_cases']:
        # 점수 계산
        scores = scorer.calculate_article_scores(case['text'])
        primary = scorer.get_primary_violation(scores)

        # 결과 기록
        result = {
            'id': case['id'],
            'text': case['text'],
            'expected_violation': case['expected_violation'],
            'expected_article': case['expected_primary_article'],
            'actual_scores': {article: data['score'] for article, data in scores.items()},
            'actual_primary_article': primary['article'],
            'actual_primary_score': primary['score'],
            'details': primary.get('details', {}),
            'severity': case['severity'],
            'violation_type': case['violation_type'],
            'notes': case.get('notes', '')
        }

        # 판정 (임계값 0.3)
        threshold = 0.3
        result['predicted_violation'] = result['actual_primary_score'] >= threshold

        # 정답 여부
        if case['expected_violation']:
            # 위반 케이스
            result['correct'] = (
                result['predicted_violation'] and
                result['actual_primary_article'] == case['expected_primary_article']
            )
            if result['predicted_violation'] and result['actual_primary_article'] == case['expected_primary_article']:
                result['classification'] = 'TP'  # True Positive
            elif not result['predicted_violation']:
                result['classification'] = 'FN'  # False Negative (놓침!)
            else:
                result['classification'] = 'FP_WRONG_ARTICLE'  # 다른 조항으로 잘못 판정
        else:
            # 비위반 케이스
            result['correct'] = not result['predicted_violation']
            if not result['predicted_violation']:
                result['classification'] = 'TN'  # True Negative
            else:
                result['classification'] = 'FP'  # False Positive (오탐!)

        results.append(result)

    return results


def analyze_results(results):
    """결과 분석"""
    total = len(results)
    correct = sum(1 for r in results if r['correct'])

    # 분류별 통계
    tp = [r for r in results if r['classification'] == 'TP']
    tn = [r for r in results if r['classification'] == 'TN']
    fp = [r for r in results if r['classification'] == 'FP']
    fn = [r for r in results if r['classification'] == 'FN']
    fp_wrong = [r for r in results if r['classification'] == 'FP_WRONG_ARTICLE']

    # 성능 지표
    precision = len(tp) / (len(tp) + len(fp) + len(fp_wrong)) if (len(tp) + len(fp) + len(fp_wrong)) > 0 else 0
    recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    analysis = {
        'total_cases': total,
        'correct': correct,
        'accuracy': correct / total,
        'tp': len(tp),
        'tn': len(tn),
        'fp': len(fp),
        'fn': len(fn),
        'fp_wrong_article': len(fp_wrong),
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'tp_cases': tp,
        'tn_cases': tn,
        'fp_cases': fp,
        'fn_cases': fn,
        'fp_wrong_cases': fp_wrong
    }

    return analysis


def print_detailed_report(results, analysis):
    """상세 리포트 출력"""
    print("\n" + "="*80)
    print("테스트 평가 결과 리포트")
    print("="*80)

    # 전체 통계
    print(f"\n📊 전체 통계:")
    print(f"   총 테스트 케이스: {analysis['total_cases']}개")
    print(f"   정답: {analysis['correct']}개 ({analysis['accuracy']:.1%})")
    print(f"   오답: {analysis['total_cases'] - analysis['correct']}개")

    print(f"\n📈 성능 지표:")
    print(f"   Precision (정밀도): {analysis['precision']:.3f}")
    print(f"   Recall (재현율):    {analysis['recall']:.3f}")
    print(f"   F1 Score:           {analysis['f1_score']:.3f}")

    print(f"\n🔍 분류 결과:")
    print(f"   ✅ True Positive (정상 탐지):  {analysis['tp']}개")
    print(f"   ✅ True Negative (정상 통과):  {analysis['tn']}개")
    print(f"   ❌ False Positive (오탐지):    {analysis['fp']}개")
    print(f"   ❌ False Negative (미탐지):    {analysis['fn']}개")
    print(f"   ⚠️  Wrong Article (조항 오판): {analysis['fp_wrong_article']}개")

    # False Negative 상세 (가장 중요!)
    if analysis['fn_cases']:
        print(f"\n" + "="*80)
        print(f"🚨 False Negative (미탐지) 상세 분석 - 개선 필요!")
        print("="*80)
        for case in analysis['fn_cases']:
            print(f"\n[{case['id']}]")
            print(f"텍스트: {case['text']}")
            print(f"예상: 제7조 위반 ({case['violation_type']})")
            print(f"실제 점수: {case['actual_primary_score']:.3f} (임계값 0.3 미만)")
            print(f"매칭된 패턴: {case['details']}")
            print(f"노트: {case['notes']}")
            print(f"➡️ 분석: 이 문장에서 놓친 키워드/패턴을 찾아 추가 필요")

    # False Positive 상세
    if analysis['fp_cases']:
        print(f"\n" + "="*80)
        print(f"⚠️ False Positive (오탐지) 상세 분석")
        print("="*80)
        for case in analysis['fp_cases']:
            print(f"\n[{case['id']}]")
            print(f"텍스트: {case['text']}")
            print(f"예상: 위반 아님")
            print(f"실제: {case['actual_primary_article']} 위반으로 오탐 (점수: {case['actual_primary_score']:.3f})")
            print(f"매칭된 패턴: {case['details']}")
            print(f"➡️ 분석: 이 패턴이 너무 광범위하거나 부정확함")

    # True Positive 샘플
    if analysis['tp_cases']:
        print(f"\n" + "="*80)
        print(f"✅ True Positive 샘플 (잘 작동하는 케이스)")
        print("="*80)
        for case in analysis['tp_cases'][:3]:  # 상위 3개만
            print(f"\n[{case['id']}] 점수: {case['actual_primary_score']:.3f}")
            print(f"텍스트: {case['text'][:80]}...")
            print(f"매칭: {case['details']['matched_high_risk'][:2] if case['details'].get('matched_high_risk') else '키워드 매칭'}")

    print("\n" + "="*80)


def save_results(results, analysis, filepath="data/test_cases/evaluation_results.json"):
    """결과 저장"""
    output = {
        'metadata': {
            'total_cases': analysis['total_cases'],
            'accuracy': analysis['accuracy'],
            'precision': analysis['precision'],
            'recall': analysis['recall'],
            'f1_score': analysis['f1_score']
        },
        'summary': {
            'tp': analysis['tp'],
            'tn': analysis['tn'],
            'fp': analysis['fp'],
            'fn': analysis['fn'],
            'fp_wrong_article': analysis['fp_wrong_article']
        },
        'detailed_results': results
    }

    path = project_root / filepath
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n💾 결과 저장: {path}")


def main():
    """메인 실행"""
    print("\n🚀 테스트 평가 시작...\n")

    # 1. 스코어러 초기화
    scorer = ArticleViolationScorer()

    # 2. 테스트 케이스 로드
    test_data = load_test_cases()
    print(f"📂 테스트 케이스 로드: {test_data['metadata']['total_cases']}개")
    print(f"   대상 조항: {', '.join(test_data['metadata']['articles_included'])}")

    # 3. 평가 실행
    print(f"\n⚙️  평가 실행 중...")
    results = evaluate_test_cases(scorer, test_data)

    # 4. 결과 분석
    analysis = analyze_results(results)

    # 5. 리포트 출력
    print_detailed_report(results, analysis)

    # 6. 결과 저장
    save_results(results, analysis)

    print("\n✅ 평가 완료!")


if __name__ == "__main__":
    main()
