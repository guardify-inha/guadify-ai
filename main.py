"""
약관 심사 시스템 - 법률 그래프 구축
"""
from database import Neo4jConnector, GraphBuilder


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("🏛️  약관 규제 법률 그래프 데이터베이스 구축")
    print("=" * 60)
    
    # Neo4j 연결
    connector = Neo4jConnector()
    
    try:
        # 기존 데이터 초기화 여부 확인
        response = input("\n⚠️  기존 데이터를 모두 삭제하고 새로 구축하시겠습니까? (y/n): ")
        if response.lower() == 'y':
            connector.clear_database()
        
        # 그래프 구축
        builder = GraphBuilder(connector)
        builder.build_law_graph()
        
        print("\n" + "=" * 60)
        print("🎉 작업이 완료되었습니다!")
        print("=" * 60)
        print("\nNeo4j Browser에서 확인하려면:")
        print("  1. 브라우저에서 http://localhost:7474 접속")
        print("  2. 다음 쿼리 실행:")
        print("     MATCH (n) RETURN n LIMIT 50")
        print("\n또는 전체 구조를 보려면:")
        print("     MATCH p=(:법률)-[*]->() RETURN p")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
    finally:
        connector.close()


if __name__ == "__main__":
    main()
