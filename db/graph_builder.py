"""
그래프 빌더

법률 문서에서 추출된 엔티티와 관계를 Neo4j 그래프 데이터베이스에 구축
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from .neo4j_client import Neo4jClient
from .entity_extractor import LegalEntityExtractor, Entity, Relationship, EntityType

logger = logging.getLogger(__name__)

class GraphBuilder:
    """법률 지식 그래프 빌더"""
    
    def __init__(self, neo4j_client: Neo4jClient):
        """
        그래프 빌더 초기화
        
        Args:
            neo4j_client: Neo4j 클라이언트 인스턴스
        """
        self.neo4j_client = neo4j_client
        self.entity_extractor = LegalEntityExtractor()
        self.entity_cache: Dict[str, str] = {}  # 텍스트 -> 노드 ID 매핑
    
    def build_graph_from_chunks(self, chunks_file: str):
        """
        청크 파일에서 그래프 구축
        
        Args:
            chunks_file: 청크 JSONL 파일 경로
        """
        logger.info(f"청크 파일에서 그래프 구축 시작: {chunks_file}")
        
        # 기존 데이터 삭제 (개발/테스트용)
        self.neo4j_client.delete_all()
        
        # 청크 파일 읽기 및 처리
        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    chunk_data = json.loads(line.strip())
                    self._process_chunk(chunk_data)
                    
                    if line_num % 100 == 0:
                        logger.info(f"처리된 청크: {line_num}")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"JSON 파싱 오류 (라인 {line_num}): {e}")
                    continue
                except Exception as e:
                    logger.error(f"청크 처리 오류 (라인 {line_num}): {e}")
                    continue
        
        # 인덱스 생성
        self._create_indexes()
        
        # 통계 출력
        stats = self.neo4j_client.get_graph_stats()
        logger.info(f"그래프 구축 완료: {stats}")
    
    def _process_chunk(self, chunk_data: Dict[str, Any]):
        """
        개별 청크 처리
        
        Args:
            chunk_data: 청크 데이터
        """
        text = chunk_data.get('text', '')
        source_file = chunk_data.get('source_file', '')
        chunk_id = chunk_data.get('chunk_id', '')
        source_tag = chunk_data.get('source_tag', 'unknown')
        
        if not text.strip():
            return
        
        # 엔티티 추출
        entities = self.entity_extractor.extract_entities(text, source_file, chunk_id)
        
        # 관계 추출
        relationships = self.entity_extractor.extract_relationships(entities, text)
        
        # 노드 생성
        for entity in entities:
            self._create_entity_node(entity, chunk_data)
        
        # 관계 생성
        for relationship in relationships:
            self._create_relationship(relationship)
        
        # 청크 노드 생성 (문서 구조 유지)
        self._create_chunk_node(chunk_data, entities)
    
    def _create_entity_node(self, entity: Entity, chunk_data: Dict[str, Any]) -> str:
        """
        엔티티 노드 생성
        
        Args:
            entity: 엔티티 객체
            chunk_data: 청크 데이터
            
        Returns:
            생성된 노드 ID
        """
        # 중복 체크
        cache_key = f"{entity.text}_{entity.entity_type.value}"
        if cache_key in self.entity_cache:
            return self.entity_cache[cache_key]
        
        # 노드 속성 구성
        properties = {
            "text": entity.text,
            "entity_type": entity.entity_type.value,
            "source_file": entity.source_file,
            "chunk_id": entity.chunk_id,
            "source_tag": chunk_data.get('source_tag', 'unknown'),
            "page": chunk_data.get('page'),
            "chunk_idx": chunk_data.get('chunk_idx'),
            **entity.properties
        }
        
        # 노드 생성
        node_id = self.neo4j_client.create_node(
            label=entity.entity_type.value,
            properties=properties
        )
        
        # 캐시에 저장
        self.entity_cache[cache_key] = str(node_id)
        
        return str(node_id)
    
    def _create_relationship(self, relationship: Relationship):
        """
        관계 생성
        
        Args:
            relationship: 관계 객체
        """
        # 시작 노드 ID 찾기
        from_cache_key = f"{relationship.from_entity.text}_{relationship.from_entity.entity_type.value}"
        to_cache_key = f"{relationship.to_entity.text}_{relationship.to_entity.entity_type.value}"
        
        from_node_id = self.entity_cache.get(from_cache_key)
        to_node_id = self.entity_cache.get(to_cache_key)
        
        if not from_node_id or not to_node_id:
            logger.warning(f"관계 생성 실패: 노드를 찾을 수 없음 - {from_cache_key} -> {to_cache_key}")
            return
        
        # 관계 속성
        rel_properties = {
            "confidence": relationship.confidence,
            "context": relationship.context[:500],  # 컨텍스트 길이 제한
            "source_file": relationship.from_entity.source_file
        }
        
        # 관계 생성
        success = self.neo4j_client.create_relationship(
            from_id=from_node_id,
            to_id=to_node_id,
            rel_type=relationship.relation_type,
            properties=rel_properties
        )
        
        if not success:
            logger.warning(f"관계 생성 실패: {relationship.relation_type}")
    
    def _create_chunk_node(self, chunk_data: Dict[str, Any], entities: List[Entity]):
        """
        청크 노드 생성 (문서 구조 유지용)
        
        Args:
            chunk_data: 청크 데이터
            entities: 추출된 엔티티 리스트
        """
        properties = {
            "chunk_id": chunk_data.get('chunk_id'),
            "source_file": chunk_data.get('source_file'),
            "source_tag": chunk_data.get('source_tag', 'unknown'),
            "page": chunk_data.get('page'),
            "chunk_idx": chunk_data.get('chunk_idx'),
            "text": chunk_data.get('text', '')[:1000],  # 텍스트 길이 제한
            "entity_count": len(entities),
            "has_article": any(e.entity_type == EntityType.ARTICLE for e in entities),
            "has_law": any(e.entity_type == EntityType.LAW for e in entities),
            "has_reference": any(e.entity_type == EntityType.REFERENCE for e in entities)
        }
        
        chunk_node_id = self.neo4j_client.create_node(
            label="Chunk",
            properties=properties
        )
        
        # 청크와 엔티티 간 관계 생성
        for entity in entities:
            entity_cache_key = f"{entity.text}_{entity.entity_type.value}"
            entity_node_id = self.entity_cache.get(entity_cache_key)
            
            if entity_node_id:
                self.neo4j_client.create_relationship(
                    from_id=str(chunk_node_id),
                    to_id=entity_node_id,
                    rel_type="CONTAINS_ENTITY",
                    properties={"position": entity.start_pos}
                )
    
    def _create_indexes(self):
        """성능 향상을 위한 인덱스 생성"""
        indexes = [
            "CREATE INDEX entity_text_index IF NOT EXISTS FOR (n:Article) ON (n.text)",
            "CREATE INDEX entity_text_index IF NOT EXISTS FOR (n:Law) ON (n.text)",
            "CREATE INDEX entity_text_index IF NOT EXISTS FOR (n:Reference) ON (n.text)",
            "CREATE INDEX entity_text_index IF NOT EXISTS FOR (n:Penalty) ON (n.text)",
            "CREATE INDEX entity_text_index IF NOT EXISTS FOR (n:Procedure) ON (n.text)",
            "CREATE INDEX source_file_index IF NOT EXISTS FOR (n) ON (n.source_file)",
            "CREATE INDEX chunk_id_index IF NOT EXISTS FOR (n:Chunk) ON (n.chunk_id)"
        ]
        
        for index_query in indexes:
            try:
                self.neo4j_client.execute_query(index_query)
            except Exception as e:
                logger.warning(f"인덱스 생성 실패: {e}")
    
    def search_entities(self, query: str, entity_type: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """
        엔티티 검색
        
        Args:
            query: 검색 쿼리
            entity_type: 엔티티 타입 필터
            limit: 결과 제한 수
            
        Returns:
            검색 결과 리스트
        """
        if entity_type:
            cypher_query = """
            MATCH (n:{entity_type})
            WHERE n.text CONTAINS $query
            RETURN n, labels(n) as labels
            ORDER BY n.text
            LIMIT $limit
            """.format(entity_type=entity_type)
        else:
            cypher_query = """
            MATCH (n)
            WHERE n.text CONTAINS $query AND NOT 'Chunk' IN labels(n)
            RETURN n, labels(n) as labels
            ORDER BY n.text
            LIMIT $limit
            """
        
        return self.neo4j_client.execute_query(cypher_query, {
            "query": query,
            "limit": limit
        })
    
    def find_related_entities(self, entity_text: str, max_depth: int = 2) -> List[Dict]:
        """
        관련 엔티티 찾기 (그래프 탐색)
        
        Args:
            entity_text: 기준 엔티티 텍스트
            max_depth: 최대 탐색 깊이
            
        Returns:
            관련 엔티티 리스트
        """
        cypher_query = """
        MATCH (start {text: $entity_text})
        MATCH path = (start)-[*1..{max_depth}]-(related)
        WHERE NOT 'Chunk' IN labels(related)
        RETURN DISTINCT related, labels(related) as labels, length(path) as distance
        ORDER BY distance, related.text
        LIMIT 50
        """.format(max_depth=max_depth)
        
        return self.neo4j_client.execute_query(cypher_query, {
            "entity_text": entity_text
        })
    
    def get_entity_network(self, entity_text: str, depth: int = 1) -> Dict[str, Any]:
        """
        엔티티 네트워크 조회
        
        Args:
            entity_text: 기준 엔티티 텍스트
            depth: 탐색 깊이
            
        Returns:
            네트워크 데이터
        """
        cypher_query = """
        MATCH (start {text: $entity_text})
        MATCH path = (start)-[*1..{depth}]-(related)
        WHERE NOT 'Chunk' IN labels(related)
        RETURN 
            collect(DISTINCT start) as center,
            collect(DISTINCT related) as nodes,
            collect(DISTINCT relationships(path)) as relationships
        """.format(depth=depth)
        
        result = self.neo4j_client.execute_query(cypher_query, {
            "entity_text": entity_text
        })
        
        return result[0] if result else {"center": [], "nodes": [], "relationships": []}
