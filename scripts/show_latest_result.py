"""
최신 테스트 결과 조회 스크립트

evaluation_results.json의 최신 결과를 읽어서 요약 출력
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json


def load_latest_result():
    """최신 결과 로드"""
    result_path = project_root / "data/test_cases/evaluation_results.json"

    if not result_path.exists():
        print("❌ 결과 파일이 없습니다. 먼저 테스트를 실행하세요.")
        print(f"   python scripts/run_test_evaluation.py")
        return None

    with open(result_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_summary(data):
    """요약 출력"""
    meta = data['metadata']
    summary = data['summary']

    print("\n" + "="*80)
    print("📊 최신 테스트 결과 요약")
    print("="*80)

    print(f"\n📈 성능 지표:")
    print(f"   총 테스트: {meta['total_cases']}개")
    print(f"   정확도:    {meta['accuracy']:.1%}")
    print(f"   Precision: {meta['precision']:.3f}")
    print(f"   Recall:    {meta['recall']:.3f}")
    print(f"   F1 Score:  {meta['f1_score']:.3f}")

    print(f"\n🔍 분류 결과:")
    print(f"   ✅ True Positive:   {summary['tp']}개")
    print(f"   ✅ True Negative:   {summary['tn']}개")
    print(f"   ❌ False Positive:  {summary['fp']}개")
    print(f"   ❌ False Negative:  {summary['fn']}개")
    print(f"   ⚠️  Wrong Article:  {summary['fp_wrong_article']}개")

    # 문제 케이스 요약
    if summary['fn'] > 0:
        print(f"\n🚨 미탐지 {summary['fn']}건 - 개선 필요!")

    if summary['fp'] > 0:
        print(f"\n⚠️  오탐지 {summary['fp']}건")

    print("\n" + "="*80)


def print_detailed(data, show_all=False):
    """상세 결과 출력"""
    results = data['detailed_results']

    # False Negative
    fn_cases = [r for r in results if r['classification'] == 'FN']
    if fn_cases:
        print(f"\n🚨 False Negative (미탐지) - {len(fn_cases)}건")
        print("-" * 80)
        for case in fn_cases:
            print(f"\n[{case['id']}]")
            print(f"텍스트: {case['text'][:80]}...")
            print(f"예상: {case['expected_article']} 위반")
            print(f"실제 점수: {case['actual_primary_score']:.3f}")

    # False Positive
    fp_cases = [r for r in results if r['classification'] == 'FP']
    if fp_cases:
        print(f"\n⚠️  False Positive (오탐지) - {len(fp_cases)}건")
        print("-" * 80)
        for case in fp_cases:
            print(f"\n[{case['id']}]")
            print(f"텍스트: {case['text'][:80]}...")
            print(f"예상: 위반 아님")
            print(f"실제: {case['actual_primary_article']} 위반으로 오탐 (점수: {case['actual_primary_score']:.3f})")

    # True Positive 샘플
    if show_all:
        tp_cases = [r for r in results if r['classification'] == 'TP']
        if tp_cases:
            print(f"\n✅ True Positive 샘플 - {min(3, len(tp_cases))}건")
            print("-" * 80)
            for case in tp_cases[:3]:
                print(f"\n[{case['id']}] 점수: {case['actual_primary_score']:.3f}")
                print(f"텍스트: {case['text'][:80]}...")


def main():
    """메인 실행"""
    # 결과 로드
    data = load_latest_result()
    if not data:
        return

    # 요약 출력
    print_summary(data)

    # 상세 출력 (문제 케이스만)
    print_detailed(data, show_all=False)

    print("\n💡 전체 상세 결과 보기:")
    print("   python scripts/show_latest_result.py --all")
    print("\n💾 결과 파일:")
    print(f"   {project_root / 'data/test_cases/evaluation_results.json'}")
    print()


if __name__ == "__main__":
    import sys
    show_all = "--all" in sys.argv

    data = load_latest_result()
    if data:
        print_summary(data)
        print_detailed(data, show_all=show_all)

        if not show_all:
            print("\n💡 전체 상세 결과 보기: python scripts/show_latest_result.py --all")
        print()
