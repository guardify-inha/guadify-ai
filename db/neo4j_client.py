"""
Neo4j 데이터베이스 클라이언트

법률 지식 그래프를 위한 Neo4j 연결 및 기본 CRUD 작업
"""

import os
import logging
from typing import Dict, List, Any, Optional
from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError

logger = logging.getLogger(__name__)

class Neo4jClient:
    """Neo4j 데이터베이스 클라이언트"""
    
    def __init__(self, uri: str = None, username: str = None, password: str = None):
        """
        Neo4j 클라이언트 초기화
        
        Args:
            uri: Neo4j 서버 URI (기본값: 환경변수에서 읽기)
            username: 사용자명 (기본값: 환경변수에서 읽기)
            password: 비밀번호 (기본값: 환경변수에서 읽기)
        """
        self.uri = uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.username = username or os.getenv('NEO4J_USERNAME', 'neo4j')
        self.password = password or os.getenv('NEO4J_PASSWORD', 'password')
        
        self.driver: Optional[Driver] = None
        self._connect()
    
    def _connect(self):
        """Neo4j 서버에 연결"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            # 연결 테스트
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info(f"Neo4j 연결 성공: {self.uri}")
        except ServiceUnavailable as e:
            logger.error(f"Neo4j 서버 연결 실패: {e}")
            raise
        except AuthError as e:
            logger.error(f"Neo4j 인증 실패: {e}")
            raise
        except Exception as e:
            logger.error(f"Neo4j 연결 오류: {e}")
            raise
    
    def close(self):
        """연결 종료"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j 연결 종료")
    
    def get_session(self) -> Session:
        """새로운 세션 반환"""
        if not self.driver:
            raise RuntimeError("Neo4j 드라이버가 초기화되지 않았습니다")
        return self.driver.session()
    
    def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict]:
        """
        Cypher 쿼리 실행
        
        Args:
            query: Cypher 쿼리
            parameters: 쿼리 매개변수
            
        Returns:
            쿼리 결과 리스트
        """
        with self.get_session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    
    def create_node(self, label: str, properties: Dict[str, Any]) -> str:
        """
        노드 생성
        
        Args:
            label: 노드 라벨
            properties: 노드 속성
            
        Returns:
            생성된 노드의 ID
        """
        query = f"CREATE (n:{label} $properties) RETURN id(n) as node_id"
        result = self.execute_query(query, {"properties": properties})
        return result[0]["node_id"] if result else None
    
    def create_relationship(self, from_id: str, to_id: str, rel_type: str, properties: Dict[str, Any] = None) -> bool:
        """
        관계 생성
        
        Args:
            from_id: 시작 노드 ID
            to_id: 끝 노드 ID
            rel_type: 관계 타입
            properties: 관계 속성
            
        Returns:
            성공 여부
        """
        query = """
        MATCH (a), (b) 
        WHERE id(a) = $from_id AND id(b) = $to_id
        CREATE (a)-[r:{rel_type} $properties]->(b)
        RETURN r
        """.format(rel_type=rel_type)
        
        try:
            result = self.execute_query(query, {
                "from_id": int(from_id),
                "to_id": int(to_id),
                "properties": properties or {}
            })
            return len(result) > 0
        except Exception as e:
            logger.error(f"관계 생성 실패: {e}")
            return False
    
    def find_node(self, label: str, properties: Dict[str, Any]) -> List[Dict]:
        """
        노드 검색
        
        Args:
            label: 노드 라벨
            properties: 검색 조건
            
        Returns:
            검색된 노드 리스트
        """
        where_clause = " AND ".join([f"n.{k} = ${k}" for k in properties.keys()])
        query = f"MATCH (n:{label}) WHERE {where_clause} RETURN n"
        return self.execute_query(query, properties)
    
    def get_node_by_id(self, node_id: str) -> Optional[Dict]:
        """
        ID로 노드 조회
        
        Args:
            node_id: 노드 ID
            
        Returns:
            노드 데이터 또는 None
        """
        query = "MATCH (n) WHERE id(n) = $node_id RETURN n"
        result = self.execute_query(query, {"node_id": int(node_id)})
        return result[0]["n"] if result else None
    
    def delete_all(self):
        """모든 노드와 관계 삭제 (개발/테스트용)"""
        query = "MATCH (n) DETACH DELETE n"
        self.execute_query(query)
        logger.info("모든 노드와 관계 삭제 완료")
    
    def get_graph_stats(self) -> Dict[str, int]:
        """그래프 통계 정보 반환"""
        stats = {}
        
        # 노드 수
        node_count_query = "MATCH (n) RETURN count(n) as count"
        result = self.execute_query(node_count_query)
        stats["total_nodes"] = result[0]["count"] if result else 0
        
        # 관계 수
        rel_count_query = "MATCH ()-[r]->() RETURN count(r) as count"
        result = self.execute_query(rel_count_query)
        stats["total_relationships"] = result[0]["count"] if result else 0
        
        # 라벨별 노드 수
        label_query = "CALL db.labels() YIELD label RETURN label"
        labels = self.execute_query(label_query)
        for label_result in labels:
            label = label_result["label"]
            count_query = f"MATCH (n:{label}) RETURN count(n) as count"
            count_result = self.execute_query(count_query)
            stats[f"{label}_nodes"] = count_result[0]["count"] if count_result else 0
        
        return stats
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()