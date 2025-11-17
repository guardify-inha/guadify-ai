"""
ai.csv 실전 데이터로 ArticleViolationScorer 테스트

목표:
- 1,004개 실제 불공정 약관 데이터로 검증
- 조항별 성능 측정
- 실패 케이스 분석 및 패턴 개선 방향 도출
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
    '약관법 제7조 제1호' → '제7조'
    '약관법 제8조' → '제8조'
    '약관법 제6조 제2항 제1호' → '제6조'
    """
    if not label or not isinstance(label, str):
        return None

    # "제N조" 추출
    import re
    match = re.search(r'제(\d+)조', label)
    if match:
        return f'제{match.group(1)}조'

    return None


def test_with_ai_csv():
    """실전 테스트 메인 함수"""
    print("=" * 80)
    print("🚀 ai.csv 실전 데이터 테스트 시작")
    print("=" * 80)
    print()

    # 1. 데이터 로드
    csv_path = project_root / 'data' / 'contracts' / 'reference' / 'ai.csv'
    df = pd.read_csv(csv_path)

    print(f"📂 데이터 로드: {len(df)}개")
    print(f"   경로: {csv_path}")
    print()

    # 2. 스코어러 초기화
    print("⚙️  ArticleViolationScorer 초기화...")
    scorer = ArticleViolationScorer()
    print()

    # 3. 평가 실행
    print("🔍 평가 실행 중...")
    print("-" * 80)

    results = []
    skipped = 0

    for idx, row in df.iterrows():
        text = row['불공정 약관 원문']
        expected_label = row['근거 조항(약관법)']

        # 예상 조항 추출
        expected_article = extract_article_from_label(expected_label)

        if not expected_article:
            skipped += 1
            continue

        # 점수 계산
        article_scores = scorer.calculate_article_scores(text)
        primary = scorer.get_primary_violation(article_scores)

        # 정답 여부 (임계값 0.3)
        threshold = 0.3
        predicted_article = primary['article'] if primary['score'] >= threshold else None

        is_correct = (predicted_article == expected_article)

        results.append({
            'id': row['ID'],
            'text': text,
            'expected': expected_article,
            'predicted': predicted_article,
            'score': primary['score'],
            'correct': is_correct,
            'all_scores': {art: data['score'] for art, data in article_scores.items()},
            'details': primary.get('details', {})
        })

        # 진행률 표시
        if (idx + 1) % 100 == 0:
            current_accuracy = sum(1 for r in results if r['correct']) / len(results)
            print(f"  진행: {idx + 1}/{len(df)} | 현재 정확도: {current_accuracy:.1%}")

    print()
    print(f"✅ 평가 완료!")
    print(f"   처리: {len(results)}개")
    print(f"   스킵: {skipped}개 (조항 추출 실패)")
    print()

    # 4. 결과 분석
    print("=" * 80)
    print("📊 결과 분석")
    print("=" * 80)
    print()

    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    accuracy = correct / total if total > 0 else 0

    # 전체 성능
    print("🎯 전체 성능:")
    print(f"   총 케이스: {total}개")
    print(f"   정답: {correct}개")
    print(f"   오답: {total - correct}개")
    print(f"   정확도: {accuracy:.1%}")
    print()

    # True Positive, False Negative 분석
    tp = sum(1 for r in results if r['correct'] and r['predicted'] is not None)
    fn = sum(1 for r in results if not r['correct'] and r['predicted'] is None)
    fp = sum(1 for r in results if not r['correct'] and r['predicted'] is not None)

    print("📈 분류 성능:")
    print(f"   True Positive (정확히 탐지): {tp}개")
    print(f"   False Negative (미탐지): {fn}개")
    print(f"   False Positive/Wrong Article (오탐/조항 오류): {fp}개")

    if tp + fp > 0:
        precision = tp / (tp + fp)
        print(f"   Precision: {precision:.1%}")

    if tp + fn > 0:
        recall = tp / (tp + fn)
        print(f"   Recall: {recall:.1%}")

    print()

    # 5. 조항별 성능
    print("📋 조항별 성능:")
    print("-" * 80)

    article_stats = {}
    for r in results:
        exp = r['expected']
        if exp not in article_stats:
            article_stats[exp] = {'correct': 0, 'total': 0, 'avg_score': []}

        article_stats[exp]['total'] += 1
        if r['correct']:
            article_stats[exp]['correct'] += 1
        article_stats[exp]['avg_score'].append(r['score'])

    # 정렬 및 출력
    sorted_articles = sorted(article_stats.items(), key=lambda x: x[0])

    for article, stats in sorted_articles:
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        avg_score = sum(stats['avg_score']) / len(stats['avg_score']) if stats['avg_score'] else 0

        status = "✅" if acc >= 0.9 else "⚠️" if acc >= 0.7 else "🔴"
        print(f"  {status} {article}: {acc:.1%} ({stats['correct']}/{stats['total']}) | 평균 점수: {avg_score:.3f}")

    print()

    # 6. 오답 케이스 분석
    errors = [r for r in results if not r['correct']]

    if errors:
        print("=" * 80)
        print(f"🚨 오답 케이스 분석 ({len(errors)}개)")
        print("=" * 80)
        print()

        # 조항별 오답
        error_by_article = {}
        for err in errors:
            exp = err['expected']
            if exp not in error_by_article:
                error_by_article[exp] = []
            error_by_article[exp].append(err)

        print("조항별 오답 분포:")
        for article, errs in sorted(error_by_article.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"  {article}: {len(errs)}개")
        print()

        # 상위 10개 오답 상세
        print("상위 10개 오답 케이스:")
        print("-" * 80)

        for i, err in enumerate(errors[:10], 1):
            print(f"\n{i}. [{err['id']}]")
            print(f"   텍스트: {err['text'][:80]}...")
            print(f"   예상: {err['expected']} | 실제: {err['predicted']} (점수: {err['score']:.3f})")

            # 상위 3개 조항 점수
            top_scores = sorted(err['all_scores'].items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"   상위 점수: {', '.join([f'{art}({score:.3f})' for art, score in top_scores])}")

            # 매칭 상세
            details = err['details']
            if details.get('matched_high_risk'):
                print(f"   고위험 키워드: {', '.join(details['matched_high_risk'][:3])}")
            if details.get('matched_regex'):
                print(f"   매칭 패턴: {', '.join(details['matched_regex'][:3])}")

    print()

    # 7. 결과 저장
    output_path = project_root / 'data' / 'test_cases' / 'ai_csv_evaluation_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'total_cases': total,
            'correct': correct,
            'accuracy': accuracy,
            'skipped': skipped,
            'threshold': 0.3
        },
        'article_stats': {
            article: {
                'total': stats['total'],
                'correct': stats['correct'],
                'accuracy': stats['correct'] / stats['total'],
                'avg_score': sum(stats['avg_score']) / len(stats['avg_score'])
            }
            for article, stats in article_stats.items()
        },
        'detailed_results': [
            {
                'id': r['id'],
                'text': r['text'][:200],  # 처음 200자만
                'expected': r['expected'],
                'predicted': r['predicted'],
                'score': r['score'],
                'correct': r['correct']
            }
            for r in results
        ]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"💾 결과 저장: {output_path}")
    print("=" * 80)
    print()

    # 8. 최종 요약
    print("📝 최종 요약:")
    print(f"   ✅ 전체 정확도: {accuracy:.1%}")

    low_performance = [art for art, stats in article_stats.items()
                      if stats['correct'] / stats['total'] < 0.7]

    if low_performance:
        print(f"   ⚠️  개선 필요 조항: {', '.join(low_performance)}")
    else:
        print(f"   🎉 모든 조항 70% 이상 달성!")

    print()
    print("💡 다음 단계:")
    if accuracy < 0.9:
        print("   1. 오답 케이스 분석")
        print("   2. 부족한 패턴 추가")
        print("   3. 재테스트")
    else:
        print("   1. Streamlit UI 테스트")
        print("   2. GraphRAG 통합 검증")
        print("   3. 프로덕션 배포 준비")

    print()


if __name__ == "__main__":
    test_with_ai_csv()
