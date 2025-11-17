"""
test_input.csv 평가 스크립트

⚠️ 주의: 이 스크립트는 TEST 전용입니다.
test_input.csv 데이터는 절대 학습/패턴 개선에 사용하지 마세요!

목표:
- 1-80번: 약관법 준수 문장 (임계값 검증용)
- 81-240번: 약관법 위반 문장 (정확도 검증용)
"""

import pandas as pd
import sys
from pathlib import Path
import json

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from judge.article_violation_scorer import ArticleViolationScorer


def extract_article_from_label(label: str) -> str:
    """
    '제7조' → '제7조'
    '없음' → None
    """
    if not label or label == '없음':
        return None

    # "제N조" 추출
    import re
    match = re.search(r'제(\d+)조', label)
    if match:
        return f'제{match.group(1)}조'

    return None


def test_compliant_cases(df_compliant, scorer):
    """1-80번: 준수 문장 테스트 (모든 조항 점수 출력)"""
    print("\n" + "="*80)
    print("📋 Part 1: 약관법 준수 문장 테스트 (1-80번)")
    print("="*80)
    print("목적: 임계값(0.3) 적정성 검증 - 준수 문장은 0.3 미만이어야 함")
    print()

    results = []
    threshold = 0.3
    false_positives = []

    for idx, row in df_compliant.iterrows():
        text = row['입력 문장']

        # 점수 계산
        article_scores = scorer.calculate_article_scores(text)
        primary = scorer.get_primary_violation(article_scores)

        # 모든 조항 점수
        all_scores = {art: data['score'] for art, data in article_scores.items()}

        # 위반으로 판정되는지 확인
        is_false_positive = primary['score'] >= threshold

        result = {
            'no': row['No.'],
            'text': text,
            'category': row['구분'],
            'all_article_scores': all_scores,
            'highest_article': primary['article'],
            'highest_score': primary['score'],
            'is_false_positive': is_false_positive,
            'details': primary.get('details', {})
        }

        results.append(result)

        if is_false_positive:
            false_positives.append(result)
            print(f"⚠️  오탐지 발견! [{result['no']}]")
            print(f"   텍스트: {text[:60]}...")
            print(f"   탐지: {primary['article']} (점수: {primary['score']:.3f})")
            print()

    # 통계
    total = len(results)
    fp_count = len(false_positives)
    tn_count = total - fp_count

    print(f"📊 준수 문장 테스트 결과:")
    print(f"   총 케이스: {total}개")
    print(f"   정상 통과 (TN): {tn_count}개 ({tn_count/total:.1%})")
    print(f"   오탐지 (FP): {fp_count}개 ({fp_count/total:.1%})")

    if fp_count > 0:
        print(f"\n⚠️  임계값 조정 권장: {fp_count}개 준수 문장이 위반으로 오탐지됨")
    else:
        print(f"\n✅ 임계값 0.3 적정: 모든 준수 문장이 정상 통과")

    return results, {'total': total, 'tn': tn_count, 'fp': fp_count, 'false_positives': false_positives}


def test_violation_cases(df_violation, scorer):
    """81-240번: 위반 문장 테스트 (예측 vs 실제 비교)"""
    print("\n" + "="*80)
    print("📋 Part 2: 약관법 위반 문장 테스트 (81-240번)")
    print("="*80)
    print("목적: 조항 분류 정확도 검증")
    print()

    results = []
    threshold = 0.3

    for idx, row in df_violation.iterrows():
        text = row['입력 문장']
        expected_article = extract_article_from_label(row['위반 조항'])

        # 점수 계산
        article_scores = scorer.calculate_article_scores(text)
        primary = scorer.get_primary_violation(article_scores)

        # 예측
        predicted_article = primary['article'] if primary['score'] >= threshold else None

        # 정답 여부
        is_correct = (predicted_article == expected_article)

        # 분류
        if is_correct and predicted_article is not None:
            classification = 'TP'  # True Positive
        elif predicted_article is None:
            classification = 'FN'  # False Negative (미탐지)
        else:
            classification = 'FP_WRONG_ARTICLE'  # 다른 조항으로 오판

        result = {
            'no': row['No.'],
            'text': text,
            'category': row['구분'],
            'expected_article': expected_article,
            'predicted_article': predicted_article,
            'predicted_score': primary['score'],
            'all_scores': {art: data['score'] for art, data in article_scores.items()},
            'correct': is_correct,
            'classification': classification,
            'details': primary.get('details', {})
        }

        results.append(result)

        # 진행률 표시
        if (idx + 1) % 40 == 0:
            current_accuracy = sum(1 for r in results if r['correct']) / len(results)
            print(f"  진행: {idx + 1}/{len(df_violation)} | 현재 정확도: {current_accuracy:.1%}")

    print()

    # 통계
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    tp = [r for r in results if r['classification'] == 'TP']
    fn = [r for r in results if r['classification'] == 'FN']
    fp_wrong = [r for r in results if r['classification'] == 'FP_WRONG_ARTICLE']

    precision = len(tp) / (len(tp) + len(fp_wrong)) if (len(tp) + len(fp_wrong)) > 0 else 0
    recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"📊 위반 문장 테스트 결과:")
    print(f"   총 케이스: {total}개")
    print(f"   정답: {correct}개 ({correct/total:.1%})")
    print(f"   오답: {total - correct}개")
    print()
    print(f"📈 성능 지표:")
    print(f"   True Positive (정확히 탐지): {len(tp)}개")
    print(f"   False Negative (미탐지): {len(fn)}개")
    print(f"   Wrong Article (조항 오판): {len(fp_wrong)}개")
    print(f"   Precision: {precision:.1%}")
    print(f"   Recall: {recall:.1%}")
    print(f"   F1 Score: {f1:.3f}")

    # 조항별 통계
    print("\n📋 조항별 성능:")
    print("-" * 80)

    article_stats = {}
    for r in results:
        exp = r['expected_article']
        if exp not in article_stats:
            article_stats[exp] = {'correct': 0, 'total': 0}

        article_stats[exp]['total'] += 1
        if r['correct']:
            article_stats[exp]['correct'] += 1

    for article in sorted(article_stats.keys()):
        stats = article_stats[article]
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        status = "✅" if acc >= 0.9 else "⚠️" if acc >= 0.7 else "🔴"
        print(f"  {status} {article}: {acc:.1%} ({stats['correct']}/{stats['total']})")

    # 오답 샘플
    errors = [r for r in results if not r['correct']]
    if errors:
        print(f"\n🚨 오답 샘플 (상위 5개):")
        print("-" * 80)

        for i, err in enumerate(errors[:5], 1):
            print(f"\n{i}. [{err['no']}]")
            print(f"   텍스트: {err['text'][:70]}...")
            print(f"   예상: {err['expected_article']} | 실제: {err['predicted_article']} (점수: {err['predicted_score']:.3f})")

            # 상위 3개 조항 점수
            top_scores = sorted(err['all_scores'].items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"   상위 점수: {', '.join([f'{art}({score:.3f})' for art, score in top_scores])}")

    return results, {
        'total': total,
        'correct': correct,
        'accuracy': correct / total,
        'tp': len(tp),
        'fn': len(fn),
        'fp_wrong': len(fp_wrong),
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'article_stats': article_stats,
        'errors': errors[:10]  # 상위 10개만
    }


def main():
    """메인 실행"""
    print("=" * 80)
    print("🚀 test_input.csv 테스트 시작")
    print("=" * 80)
    print()
    print("⚠️  중요: 이 데이터는 TEST 전용입니다!")
    print("   패턴 개선이나 학습에 절대 사용하지 마세요!")
    print()

    # 1. 데이터 로드
    csv_path = project_root / 'data' / 'test' / 'test_input.csv'
    df = pd.read_csv(csv_path)

    print(f"📂 데이터 로드: {len(df)}개")
    print(f"   경로: {csv_path}")
    print()

    # BOM 제거 (첫 번째 컬럼명에 BOM이 있을 수 있음)
    df.columns = df.columns.str.replace('\ufeff', '')

    # 2. 데이터 분리
    df_compliant = df[df['위반여부'] == 'X'].reset_index(drop=True)
    df_violation = df[df['위반여부'] == 'O'].reset_index(drop=True)

    print(f"   준수 문장: {len(df_compliant)}개 (1-80번)")
    print(f"   위반 문장: {len(df_violation)}개 (81-240번)")
    print()

    # 3. 스코어러 초기화
    print("⚙️  ArticleViolationScorer 초기화...")
    scorer = ArticleViolationScorer()
    print()

    # 4. Part 1: 준수 문장 테스트
    compliant_results, compliant_stats = test_compliant_cases(df_compliant, scorer)

    # 5. Part 2: 위반 문장 테스트
    violation_results, violation_stats = test_violation_cases(df_violation, scorer)

    # 6. 전체 요약
    print("\n" + "=" * 80)
    print("📝 전체 테스트 요약")
    print("=" * 80)
    print()
    print(f"Part 1 - 준수 문장 (1-80번):")
    print(f"   정상 통과: {compliant_stats['tn']}/{compliant_stats['total']} ({compliant_stats['tn']/compliant_stats['total']:.1%})")
    print(f"   오탐지: {compliant_stats['fp']}/{compliant_stats['total']} ({compliant_stats['fp']/compliant_stats['total']:.1%})")
    print()
    print(f"Part 2 - 위반 문장 (81-240번):")
    print(f"   정확도: {violation_stats['accuracy']:.1%} ({violation_stats['correct']}/{violation_stats['total']})")
    print(f"   Precision: {violation_stats['precision']:.1%}")
    print(f"   Recall: {violation_stats['recall']:.1%}")
    print(f"   F1 Score: {violation_stats['f1_score']:.3f}")
    print()

    # 전체 성능
    total_cases = compliant_stats['total'] + violation_stats['total']
    total_correct = compliant_stats['tn'] + violation_stats['correct']
    overall_accuracy = total_correct / total_cases

    print(f"전체 성능:")
    print(f"   총 케이스: {total_cases}개")
    print(f"   정답: {total_correct}개")
    print(f"   전체 정확도: {overall_accuracy:.1%}")
    print()

    # 7. 결과 저장
    output_path = project_root / 'data' / 'test' / 'test_input_evaluation_results.json'

    output_data = {
        'metadata': {
            'source': 'test_input.csv',
            'warning': '⚠️ TEST 데이터 - 학습/패턴 개선에 사용 금지!',
            'total_cases': total_cases,
            'compliant_cases': compliant_stats['total'],
            'violation_cases': violation_stats['total'],
            'overall_accuracy': overall_accuracy,
            'threshold': 0.3
        },
        'compliant_test': {
            'total': compliant_stats['total'],
            'true_negative': compliant_stats['tn'],
            'false_positive': compliant_stats['fp'],
            'fp_rate': compliant_stats['fp'] / compliant_stats['total'],
            'false_positive_cases': [
                {
                    'no': fp['no'],
                    'text': fp['text'][:100],
                    'detected_as': fp['highest_article'],
                    'score': fp['highest_score']
                }
                for fp in compliant_stats['false_positives']
            ],
            'detailed_results': [
                {
                    'no': r['no'],
                    'text': r['text'][:100],
                    'category': r['category'],
                    'all_scores': r['all_article_scores'],
                    'highest_article': r['highest_article'],
                    'highest_score': r['highest_score']
                }
                for r in compliant_results
            ]
        },
        'violation_test': {
            'total': violation_stats['total'],
            'correct': violation_stats['correct'],
            'accuracy': violation_stats['accuracy'],
            'true_positive': violation_stats['tp'],
            'false_negative': violation_stats['fn'],
            'wrong_article': violation_stats['fp_wrong'],
            'precision': violation_stats['precision'],
            'recall': violation_stats['recall'],
            'f1_score': violation_stats['f1_score'],
            'article_stats': {
                article: {
                    'total': stats['total'],
                    'correct': stats['correct'],
                    'accuracy': stats['correct'] / stats['total']
                }
                for article, stats in violation_stats['article_stats'].items()
            },
            'error_samples': [
                {
                    'no': err['no'],
                    'text': err['text'][:100],
                    'expected': err['expected_article'],
                    'predicted': err['predicted_article'],
                    'score': err['predicted_score'],
                    'classification': err['classification']
                }
                for err in violation_stats['errors']
            ],
            'detailed_results': [
                {
                    'no': r['no'],
                    'text': r['text'][:100],
                    'category': r['category'],
                    'expected': r['expected_article'],
                    'predicted': r['predicted_article'],
                    'score': r['predicted_score'],
                    'correct': r['correct'],
                    'classification': r['classification']
                }
                for r in violation_results
            ]
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"💾 결과 저장: {output_path}")
    print("=" * 80)
    print()

    # 8. 최종 평가
    print("📝 최종 평가:")
    if overall_accuracy >= 0.95:
        print("   🎉 탁월한 성능! (95% 이상)")
    elif overall_accuracy >= 0.90:
        print("   ✅ 우수한 성능 (90% 이상)")
    elif overall_accuracy >= 0.80:
        print("   ⚠️  양호한 성능 (80% 이상) - 개선 여지 있음")
    else:
        print("   🔴 개선 필요 (80% 미만)")

    if compliant_stats['fp'] > 0:
        print(f"   ⚠️  임계값 조정 검토 필요 (준수 문장 {compliant_stats['fp']}개 오탐지)")

    if violation_stats['recall'] < 0.9:
        print(f"   ⚠️  Recall 개선 필요 ({violation_stats['fn']}개 미탐지)")

    print()


if __name__ == "__main__":
    main()
