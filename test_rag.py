#!/usr/bin/env python3
"""
RAG 테스트 스크립트
"""
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

from scripts.judge_clause import run
from config.settings import settings

def main():
    print("=" * 70)
    print("Graph RAG 테스트")
    print("=" * 70)
    print(f"\n현재 설정:")
    print(f"  - LLM 제공자: {settings.LLM_PROVIDER}")
    print(f"  - LLM 모델: {settings.LLM_MODEL}")
    print(f"  - Text2Cypher 사용: {settings.USE_TEXT2CYPHER}")
    print(f"  - 비교 모드: {settings.COMPARE_METHODS}")
    
    # LLM 클라이언트 확인
    from utils.llm_client import get_llm_client
    llm = get_llm_client()
    if not llm:
        print("\n⚠️  LLM 클라이언트 초기화 실패!")
        print("   .env 파일에 API 키를 설정했는지 확인하세요.")
        return
    
    print(f"\n✅ LLM 클라이언트 준비 완료\n")
    
    # 테스트 케이스
    test_cases = [
        "회사는 어떠한 경우에도 책임을 지지 않습니다",
        "고객의 계약 해지 권리를 제한합니다",
        "손해배상액을 과도하게 높게 설정합니다"
    ]
    
    for idx, test_text in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"테스트 {idx}: {test_text}")
        print('='*70)
        
        try:
            result = run(test_text)
            
            print(f"\n📊 결과:")
            print(f"  - 위반 여부: {'위반' if result['violation'] else '비위반/불명확'}")
            print(f"  - 불공정도 점수: {result['score']:.2f}")
            print(f"  - 심각도: {result['severity']}")
            print(f"  - 관련 조항: {result['article_id']}")
            print(f"  - 사용 방법: {result.get('method', 'unknown')}")
            
            print(f"\n📝 설명:")
            print(f"  {result['explanation'][:200]}...")
            
            print(f"\n💡 제안:")
            print(f"  {result['suggestion'][:200]}...")
            
            if result.get('comparison'):
                print(f"\n🔄 비교 결과:")
                comp = result['comparison']
                print(f"  표준 방식 - 점수: {comp['standard']['score']:.2f}, 사례 수: {comp['standard']['cases_found']}")
                print(f"  Text2Cypher - 점수: {comp['text2cypher']['score']:.2f}, 사례 수: {comp['text2cypher']['cases_found']}")
                print(f"  차이 - 점수 차이: {comp['differences']['score_diff']:.2f}, 위반 일치: {comp['differences']['violation_match']}")
            
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("테스트 완료!")
    print('='*70)

if __name__ == "__main__":
    main()

