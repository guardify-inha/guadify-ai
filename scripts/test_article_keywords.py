"""약관법 조항별 키워드 매칭 테스트 스크립트"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.article_keywords import analyze_clause_keywords, get_top_matching_articles


def load_test_cases() -> Dict[str, List[Dict]]:
    """테스트 케이스 파일 로드"""
    test_cases_path = project_root / "data" / "test_cases" / "article_test_cases.json"
    with open(test_cases_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 조항별로 그룹화
    test_cases_by_article = {}
    for item in data:
        article = item["article"]
        test_cases_by_article[article] = item["test_cases"]
    
    return test_cases_by_article


def test_keyword_matching() -> Dict[str, Dict]:
    """키워드 매칭 테스트 실행"""
    test_cases = load_test_cases()
    results = {}
    
    for article, cases in test_cases.items():
        article_results = {
            "total": len(cases),
            "correct": 0,
            "incorrect": 0,
            "top1_correct": 0,
            "top3_correct": 0,
            "errors": []
        }
        
        for case in cases:
            input_clause = case["input_clause"]
            expected_article = case["expected_article"]
            
            # 키워드 분석
            scores = analyze_clause_keywords(input_clause)
            top_articles = get_top_matching_articles(input_clause, top_k=3)
            
            # 정확도 계산
            if scores and scores[0][0] == expected_article:
                article_results["top1_correct"] += 1
                article_results["correct"] += 1
            elif expected_article in top_articles:
                article_results["top3_correct"] += 1
                article_results["correct"] += 1
            else:
                article_results["incorrect"] += 1
                article_results["errors"].append({
                    "input": input_clause,
                    "expected": expected_article,
                    "predicted": scores[0][0] if scores else "N/A",
                    "top3": top_articles,
                    "scores": scores[:5] if scores else []
                })
        
        # 정확도 계산
        article_results["accuracy_top1"] = (
            article_results["top1_correct"] / article_results["total"] * 100
            if article_results["total"] > 0 else 0
        )
        article_results["accuracy_top3"] = (
            article_results["top3_correct"] / article_results["total"] * 100
            if article_results["total"] > 0 else 0
        )
        article_results["accuracy"] = (
            article_results["correct"] / article_results["total"] * 100
            if article_results["total"] > 0 else 0
        )
        
        results[article] = article_results
    
    return results


def generate_confusion_matrix(results: Dict[str, Dict]) -> Dict[str, Dict[str, int]]:
    """혼동 매트릭스 생성"""
    test_cases = load_test_cases()
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    for article, cases in test_cases.items():
        for case in cases:
            input_clause = case["input_clause"]
            expected_article = case["expected_article"]
            
            scores = analyze_clause_keywords(input_clause)
            predicted_article = scores[0][0] if scores else "N/A"
            
            confusion_matrix[expected_article][predicted_article] += 1
    
    return dict(confusion_matrix)


def print_results(results: Dict[str, Dict], confusion_matrix: Dict[str, Dict[str, int]]):
    """결과 출력"""
    print("\n" + "="*80)
    print("약관법 조항별 키워드 매칭 테스트 결과")
    print("="*80)
    
    # 조항별 정확도 리포트
    print("\n[조항별 정확도 리포트]")
    print("-" * 80)
    print(f"{'조항':<10} {'전체':<8} {'정확':<8} {'오류':<8} {'Top-1':<10} {'Top-3':<10} {'정확도(Top-1)':<15} {'정확도(Top-3)':<15}")
    print("-" * 80)
    
    total_cases = 0
    total_correct = 0
    total_top1_correct = 0
    total_top3_correct = 0
    
    for article in sorted(results.keys()):
        r = results[article]
        total_cases += r["total"]
        total_correct += r["correct"]
        total_top1_correct += r["top1_correct"]
        total_top3_correct += r["top3_correct"]
        
        print(f"{article:<10} {r['total']:<8} {r['correct']:<8} {r['incorrect']:<8} "
              f"{r['top1_correct']:<10} {r['top3_correct']:<10} "
              f"{r['accuracy_top1']:.2f}%{'':<10} {r['accuracy_top3']:.2f}%")
    
    print("-" * 80)
    overall_accuracy_top1 = (total_top1_correct / total_cases * 100) if total_cases > 0 else 0
    overall_accuracy_top3 = (total_top3_correct / total_cases * 100) if total_cases > 0 else 0
    print(f"{'전체':<10} {total_cases:<8} {total_correct:<8} {total_cases - total_correct:<8} "
          f"{total_top1_correct:<10} {total_top3_correct:<10} "
          f"{overall_accuracy_top1:.2f}%{'':<10} {overall_accuracy_top3:.2f}%")
    
    # 혼동 매트릭스
    print("\n[혼동 매트릭스]")
    print("-" * 80)
    articles = sorted(set(list(confusion_matrix.keys()) + 
                       [k for v in confusion_matrix.values() for k in v.keys()]))
    
    # 헤더
    header_label = "예상\\예측"
    print(f"{header_label:<12}", end="")
    for pred_article in articles:
        print(f"{pred_article:<10}", end="")
    print()
    print("-" * (12 + len(articles) * 10))
    
    # 행
    for expected_article in articles:
        print(f"{expected_article:<12}", end="")
        for pred_article in articles:
            count = confusion_matrix.get(expected_article, {}).get(pred_article, 0)
            print(f"{count:<10}", end="")
        print()
    
    # 오류 케이스 상세 리포트
    print("\n[오류 케이스 상세 리포트]")
    print("-" * 80)
    
    for article in sorted(results.keys()):
        r = results[article]
        if r["errors"]:
            print(f"\n{article} - 오류 케이스 ({len(r['errors'])}개):")
            for i, error in enumerate(r["errors"][:5], 1):  # 최대 5개만 표시
                print(f"  {i}. 입력: {error['input'][:60]}...")
                print(f"     예상: {error['expected']}, 예측: {error['predicted']}, Top-3: {error['top3']}")
                if error['scores']:
                    print(f"     점수: {error['scores'][:3]}")
            if len(r["errors"]) > 5:
                print(f"  ... 외 {len(r['errors']) - 5}개 오류 케이스 생략")


def main():
    """메인 함수"""
    print("약관법 조항별 키워드 매칭 테스트를 시작합니다...")
    
    # 테스트 실행
    results = test_keyword_matching()
    confusion_matrix = generate_confusion_matrix(results)
    
    # 결과 출력
    print_results(results, confusion_matrix)
    
    # 요약
    print("\n" + "="*80)
    print("[테스트 요약]")
    print("="*80)
    
    total_cases = sum(r["total"] for r in results.values())
    total_top1_correct = sum(r["top1_correct"] for r in results.values())
    total_top3_correct = sum(r["top3_correct"] for r in results.values())
    
    overall_accuracy_top1 = (total_top1_correct / total_cases * 100) if total_cases > 0 else 0
    overall_accuracy_top3 = (total_top3_correct / total_cases * 100) if total_cases > 0 else 0
    
    print(f"전체 테스트 케이스: {total_cases}개")
    print(f"Top-1 정확도: {overall_accuracy_top1:.2f}%")
    print(f"Top-3 정확도: {overall_accuracy_top3:.2f}%")
    
    # 목표 달성 여부
    print("\n[목표 달성 여부]")
    print(f"전체 정확도 90% 이상: {'✓' if overall_accuracy_top1 >= 90 else '✗'} ({overall_accuracy_top1:.2f}%)")
    
    articles_below_85 = [
        article for article, r in results.items()
        if r["accuracy_top1"] < 85
    ]
    print(f"각 조항별 85% 이상: {'✓' if not articles_below_85 else '✗'}")
    if articles_below_85:
        print(f"  - 85% 미만 조항: {', '.join(articles_below_85)}")
    
    confusion_rate = ((total_cases - total_top1_correct) / total_cases * 100) if total_cases > 0 else 0
    print(f"혼동 케이스 5% 이하: {'✓' if confusion_rate <= 5 else '✗'} ({confusion_rate:.2f}%)")


if __name__ == "__main__":
    main()

