"""
향상된 리트리버 테스트: 하이브리드 + HyDE + LLM-as-Judge

3단계: LLM-as-Judge 및 HyDE 기법 구현 - 테스트
"""
import sys
from pathlib import Path

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

def test_enhanced_retriever():
    """향상된 리트리버 테스트"""
    print("=" * 60)
    print("🧪 향상된 리트리버 테스트 (하이브리드 + HyDE + LLM-as-Judge)")
    print("=" * 60)
    
    conn = Neo4jConnector()
    retriever = EnhancedRetriever(conn)
    
    # 테스트 케이스
    test_cases = [
        {
            "query": "회사는 어떠한 피해배상도 하지않는다",
            "expected": "제7조",
            "description": "면책 조항 테스트"
        },
        {
            "query": "계약을 해지할 수 없습니다",
            "expected": "제9조",
            "description": "해지 권한 제한 테스트"
        },
        {
            "query": "고객에게 부당하게 불리한 조항",
            "expected": "제6조",
            "description": "일반원칙 테스트"
        },
        {
            "query": "과도한 손해배상금을 부담해야 합니다",
            "expected": "제8조",
            "description": "과중한 손해배상 테스트"
        },
        {
            "query": "회사가 일방적으로 계약을 변경할 수 있습니다",
            "expected": "제10조",
            "description": "일방적 변경 테스트"
        },
        {
            "query": "고객은 기한의 이익을 상실합니다",
            "expected": "제11조",
            "description": "제11조 테스트 (기한의 이익 박탈)"
        },
        {
            "query": "고객의 항변권과 상계권을 배제합니다",
            "expected": "제11조",
            "description": "제11조 테스트 (항변권/상계권 배제)"
        },
        {
            "query": "고객이 답변하지 않으면 동의한 것으로 간주합니다",
            "expected": "제12조",
            "description": "제12조 테스트 (의사표시 의제)"
        },
        {
            "query": "고객의 대리인이 의무를 이행할 책임을 집니다",
            "expected": "제13조",
            "description": "제13조 테스트 (대리인 책임 가중)"
        },
        {
            "query": "이 계약에 관한 소송은 회사 본사 소재지 관할법원으로 합니다",
            "expected": "제14조",
            "description": "제14조 테스트 (소송 제기/관할)"
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
        
        # 향상된 검색 실행
        results_list = retriever.retrieve(query, top_k=5, use_hyde=True, use_llm_judge=True)
        
        if results_list:
            top_result = results_list[0]
            top_article = top_result.get('article_id', '')
            
            # article_id에서 조 번호 추출
            if top_article:
                match = top_article.split('_')[0] if '_' in top_article else top_article
                if not match.startswith('제'):
                    match = top_result.get('id', '').split('_')[0] if '_' in top_result.get('id', '') else top_result.get('id', '')
            else:
                match = top_result.get('id', '').split('_')[0] if '_' in top_result.get('id', '') else top_result.get('id', '')
            
            is_correct = expected in match or match == expected
            
            status = "✓" if is_correct else "✗"
            integrated_score = top_result.get('integrated_score', top_result.get('final_score', 0))
            llm_prob = top_result.get('llm_violation_probability', 0)
            
            print(f"  결과: {status} {match} (통합 점수: {integrated_score:.3f}, LLM 위반 가능성: {llm_prob:.3f})")
            
            if not is_correct:
                print(f"  ⚠️ 기대: {expected}, 실제: {match}")
            
            # 상위 3개 결과 출력
            print(f"  상위 3개 결과:")
            for j, result in enumerate(results_list[:3], 1):
                integrated = result.get('integrated_score', result.get('final_score', 0))
                llm_prob = result.get('llm_violation_probability', 0)
                llm_reasoning = result.get('llm_reasoning', '')
                node_id = result.get('id', 'N/A')
                node_type = result.get('node_type', 'N/A')
                
                print(f"    {j}. {node_id} ({node_type}) - 통합: {integrated:.3f}, LLM 위반 가능성: {llm_prob:.3f}")
                if llm_reasoning:
                    print(f"       근거: {llm_reasoning[:80]}...")
            
            results.append({
                "test": i,
                "query": query,
                "expected": expected,
                "actual": match,
                "correct": is_correct,
                "integrated_score": integrated_score,
                "llm_probability": llm_prob
            })
        else:
            print(f"  ✗ 검색 결과 없음")
            results.append({
                "test": i,
                "query": query,
                "expected": expected,
                "actual": None,
                "correct": False,
                "integrated_score": 0,
                "llm_probability": 0
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
        print(f"  {status} 테스트 {r['test']}: {r['expected']} "
              f"(통합: {r['integrated_score']:.3f}, LLM 위반 가능성: {r['llm_probability']:.3f})")
    
    print("\n" + "=" * 60)
    if accuracy >= 80:
        print("✅ 테스트 통과! 향상된 리트리버가 잘 작동합니다.")
    elif accuracy >= 60:
        print("⚠️ 테스트 부분 통과. 추가 개선 여지가 있습니다.")
    else:
        print("❌ 테스트 실패. 리트리버 개선이 필요합니다.")
    print("=" * 60)
    
    conn.close()
    return results

if __name__ == "__main__":
    test_enhanced_retriever()

