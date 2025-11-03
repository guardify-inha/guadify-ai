"""
하이브리드 리트리버 테스트 스크립트

2단계: 하이브리드 검색 전략 수립 - 테스트
"""
import sys
from pathlib import Path
import json

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

from database.neo4j_connector import Neo4jConnector
from retriever.hybrid_retriever import HybridRetriever

def test_retriever():
    """하이브리드 리트리버 테스트"""
    print("=" * 60)
    print("🧪 하이브리드 리트리버 테스트")
    print("=" * 60)
    
    conn = Neo4jConnector()
    retriever = HybridRetriever(conn)
    
    # 테스트 케이스
    test_cases = [
        {
            "query": "회사는 어떠한 피해배상도 하지않는다",
            "expected": "제7조",  # 면책 조항
            "description": "면책 조항 테스트"
        },
        {
            "query": "계약을 해지할 수 없습니다",
            "expected": "제9조",  # 해제·해지 제한
            "description": "해지 권한 제한 테스트"
        },
        {
            "query": "고객에게 부당하게 불리한 조항",
            "expected": "제6조",  # 일반원칙
            "description": "일반원칙 테스트"
        },
        {
            "query": "과도한 손해배상금을 부담해야 합니다",
            "expected": "제8조",  # 손해배상액의 예정
            "description": "과중한 손해배상 테스트"
        },
        {
            "query": "회사가 일방적으로 계약을 변경할 수 있습니다",
            "expected": "제10조",  # 급부 내용 변경
            "description": "일방적 변경 테스트"
        }
    ]
    
    print(f"\n총 {len(test_cases)}개 테스트 케이스 실행\n")
    
    results = []
    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        expected = test_case["expected"]
        description = test_case["description"]
        
        print(f"[테스트 {i}] {description}")
        print(f"  질의: \"{query}\"")
        print(f"  기대 조항: {expected}")
        
        # 검색 실행
        results_list = retriever.retrieve(query, top_k=5)
        
        if results_list:
            top_result = results_list[0]
            top_article = top_result.get('article_id', '')
            
            # article_id에서 조 번호 추출
            if top_article:
                # "제6조" 형식에서 "제6조" 추출
                match = top_article.split('_')[0] if '_' in top_article else top_article
                if not match.startswith('제'):
                    # 노드 타입에서 추출 시도
                    if top_result.get('node_type') == '조':
                        match = top_result.get('id', '')
            else:
                match = top_result.get('id', '').split('_')[0] if '_' in top_result.get('id', '') else top_result.get('id', '')
            
            is_correct = expected in match or match == expected
            
            status = "✓" if is_correct else "✗"
            print(f"  결과: {status} {match} (점수: {top_result.get('final_score', 0):.3f})")
            
            if not is_correct:
                print(f"  ⚠️ 기대: {expected}, 실제: {match}")
            
            # 상위 3개 결과 출력
            print(f"  상위 3개 결과:")
            for j, result in enumerate(results_list[:3], 1):
                score = result.get('final_score', 0)
                semantic = result.get('semantic_score', 0)
                keyword = result.get('keyword_score', 0)
                graph = result.get('graph_score', 0)
                node_id = result.get('id', 'N/A')
                node_type = result.get('node_type', 'N/A')
                
                print(f"    {j}. {node_id} ({node_type}) - "
                      f"최종: {score:.3f} "
                      f"(의미: {semantic:.3f}, 키워드: {keyword:.3f}, 그래프: {graph:.3f})")
            
            results.append({
                "test": i,
                "query": query,
                "expected": expected,
                "actual": match,
                "correct": is_correct,
                "top_score": top_result.get('final_score', 0)
            })
        else:
            print(f"  ✗ 검색 결과 없음")
            results.append({
                "test": i,
                "query": query,
                "expected": expected,
                "actual": None,
                "correct": False,
                "top_score": 0
            })
        
        print()
    
    # 결과 요약
    print("=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)
    
    correct_count = sum(1 for r in results if r.get("correct", False))
    total_count = len(results)
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    
    print(f"정확도: {correct_count}/{total_count} ({accuracy:.1f}%)")
    print(f"\n상세 결과:")
    for r in results:
        status = "✓" if r.get("correct") else "✗"
        print(f"  {status} 테스트 {r['test']}: {r['expected']} (점수: {r['top_score']:.3f})")
    
    print("\n" + "=" * 60)
    if accuracy >= 80:
        print("✅ 테스트 통과! 리트리버가 잘 작동합니다.")
    elif accuracy >= 60:
        print("⚠️ 테스트 부분 통과. 개선이 필요합니다.")
    else:
        print("❌ 테스트 실패. 리트리버 개선이 필요합니다.")
    print("=" * 60)
    
    conn.close()
    return results

if __name__ == "__main__":
    test_retriever()

