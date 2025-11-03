"""
종합 테스트 실행 및 시각화
"""
import sys
from pathlib import Path
import json
from collections import defaultdict, Counter
from datetime import datetime

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

from database.neo4j_connector import Neo4jConnector
from retriever.enhanced_retriever import EnhancedRetriever

def run_comprehensive_test():
    """종합 테스트 실행 및 결과 분석"""
    print("=" * 80)
    print("📊 종합 테스트 실행 (120개 케이스)")
    print("=" * 80)
    
    # 테스트 케이스 로드
    test_cases_path = Path(__file__).parent / "comprehensive_test_cases.json"
    with open(test_cases_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    
    conn = Neo4jConnector()
    retriever = EnhancedRetriever(conn)
    
    results = []
    article_stats = defaultdict(lambda: {"correct": 0, "total": 0, "wrong": []})
    
    print(f"\n총 {len(test_cases)}개 테스트 케이스 실행 중...\n")
    
    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        expected = test_case["expected"]
        
        if i % 20 == 0:
            print(f"진행률: {i}/{len(test_cases)} ({i/len(test_cases)*100:.1f}%)")
        
        # 검색 실행
        results_list = retriever.retrieve(query, top_k=1, use_hyde=True, use_llm_judge=True)
        
        if results_list:
            top_result = results_list[0]
            top_article = top_result.get('article_id', '')
            
            # article_id에서 조 번호 추출
            if top_article:
                match = top_article.split('_')[0] if '_' in top_article else top_article
            else:
                match = top_result.get('id', '').split('_')[0] if '_' in top_result.get('id', '') else top_result.get('id', '')
            
            is_correct = expected in match or match == expected
            
            integrated_score = top_result.get('integrated_score', top_result.get('final_score', 0))
            llm_prob = top_result.get('llm_violation_probability', 0)
            
            results.append({
                "query": query,
                "expected": expected,
                "actual": match,
                "correct": is_correct,
                "integrated_score": integrated_score,
                "llm_probability": llm_prob
            })
            
            # 통계 업데이트
            article_stats[expected]["total"] += 1
            if is_correct:
                article_stats[expected]["correct"] += 1
            else:
                article_stats[expected]["wrong"].append({
                    "query": query,
                    "expected": expected,
                    "actual": match
                })
        else:
            results.append({
                "query": query,
                "expected": expected,
                "actual": None,
                "correct": False,
                "integrated_score": 0,
                "llm_probability": 0
            })
            article_stats[expected]["total"] += 1
            article_stats[expected]["wrong"].append({
                "query": query,
                "expected": expected,
                "actual": None
            })
    
    # 결과 분석
    total_correct = sum(1 for r in results if r.get("correct", False))
    total_count = len(results)
    overall_accuracy = (total_correct / total_count * 100) if total_count > 0 else 0
    
    print("\n" + "=" * 80)
    print("📈 테스트 결과 요약")
    print("=" * 80)
    
    # 전체 정확도
    print(f"\n🎯 전체 정확도: {total_correct}/{total_count} ({overall_accuracy:.1f}%)")
    
    if overall_accuracy >= 82.0:
        print("✅ 목표 정확도(82%) 달성!")
    else:
        print(f"⚠️ 목표 정확도(82%) 미달 (부족: {82.0 - overall_accuracy:.1f}%)")
    
    # 조항별 통계
    print("\n" + "-" * 80)
    print("📊 조항별 정확도")
    print("-" * 80)
    print(f"{'조항':<8} {'정확도':<12} {'정확/전체':<15} {'오분류 예시'}")
    print("-" * 80)
    
    for article in sorted(article_stats.keys()):
        stats = article_stats[article]
        accuracy = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        correct_count = stats["correct"]
        total_count = stats["total"]
        
        # 오분류 예시 (최대 2개)
        wrong_examples = stats["wrong"][:2]
        wrong_str = ", ".join([f"→{w['actual']}" for w in wrong_examples]) if wrong_examples else "-"
        
        status = "✓" if accuracy >= 80 else "✗"
        print(f"{status} {article:<6} {accuracy:>6.1f}%      {correct_count:>3}/{total_count:<3}      {wrong_str}")
    
    # 상세 결과 저장
    output_path = Path(__file__).parent / "comprehensive_test_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_cases": total_count,
            "total_correct": total_correct,
            "overall_accuracy": overall_accuracy,
            "article_stats": dict(article_stats),
            "detailed_results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 상세 결과 저장: {output_path}")
    
    # 시각화 데이터 생성
    visualize_results(article_stats, overall_accuracy, total_correct, total_count)
    
    conn.close()
    return results, article_stats, overall_accuracy

def visualize_results(article_stats, overall_accuracy, total_correct, total_count):
    """결과 시각화"""
    print("\n" + "=" * 80)
    print("📊 시각화 데이터")
    print("=" * 80)
    
    # 조항별 정확도 차트
    print("\n[조항별 정확도 차트]")
    print("=" * 80)
    
    for article in sorted(article_stats.keys()):
        stats = article_stats[article]
        accuracy = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        
        # ASCII 바 차트 (0-100%)
        bar_length = int(accuracy / 2)  # 50칸 = 100%
        bar = "█" * bar_length + "░" * (50 - bar_length)
        
        print(f"{article:<8} │{bar}│ {accuracy:>5.1f}% ({stats['correct']}/{stats['total']})")
    
    # 전체 통계
    print("\n" + "=" * 80)
    print("[전체 통계]")
    print("=" * 80)
    
    overall_bar_length = int(overall_accuracy / 2)
    overall_bar = "█" * overall_bar_length + "░" * (50 - overall_bar_length)
    
    print(f"전체 정확도 │{overall_bar}│ {overall_accuracy:>5.1f}% ({total_correct}/{total_count})")
    
    # 목표선 표시
    target_bar_length = int(82 / 2)
    target_marker = " " * target_bar_length + "┃ 목표(82%)"
    print(f"            {target_marker}")
    
    # 오분류 분석
    print("\n" + "=" * 80)
    print("[주요 오분류 패턴]")
    print("=" * 80)
    
    wrong_patterns = Counter()
    for article, stats in article_stats.items():
        for wrong in stats["wrong"]:
            if wrong["actual"]:
                wrong_patterns[f"{article} → {wrong['actual']}"] += 1
    
    for pattern, count in wrong_patterns.most_common(5):
        print(f"  {pattern}: {count}회")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    run_comprehensive_test()

