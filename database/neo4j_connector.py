"""
Neo4j 데이터베이스 연결 관리
"""
from neo4j import GraphDatabase
from config.settings import settings


class Neo4jConnector:
    """Neo4j 데이터베이스 연결 클래스"""
    
    def __init__(self):
        """Neo4j 연결 초기화"""
        self.driver = None
        self.connect()
    
    def connect(self):
        """데이터베이스 연결"""
        try:
            self.driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
            # 연결 테스트
            self.driver.verify_connectivity()
            print("✅ Neo4j 데이터베이스 연결 성공!")
        except Exception as e:
            print(f"❌ Neo4j 연결 실패: {e}")
            raise
    
    def close(self):
        """데이터베이스 연결 종료"""
        if self.driver:
            self.driver.close()
            print("Neo4j 연결 종료")
    
    def execute_query(self, query, parameters=None):
        """
        쿼리 실행
        
        Args:
            query: Cypher 쿼리
            parameters: 쿼리 파라미터
            
        Returns:
            쿼리 결과
        """
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    
    def clear_database(self):
        """데이터베이스 초기화 (모든 노드와 관계 삭제)"""
        query = "MATCH (n) DETACH DELETE n"
        self.execute_query(query)
        print("🗑️  데이터베이스 초기화 완료")
