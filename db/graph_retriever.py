"""
Graph RAG 검색 컴포넌트

Neo4j 그래프 데이터베이스에서 법률 지식 검색 및 관련 정보 추출
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .neo4j_client import Neo4jClient
from .entity_extractor import LegalEntityExtractor

logger = logging.getLogger(__name__)

@dataclass
class GraphSearchResult:
    """그래프 검색 결과"""
    entity_text: str
    entity_type: str
    confidence: float
    context: str
    related_entities: List[str]
    source_file: str
    chunk_id: str

class GraphRetriever:
    """Graph RAG 검색기"""
    
    def __init__(self, neo4j_client: Neo4jClient):
        """
        그래프 검색기 초기화
        
        Args:
            neo4j_client: Neo4j 클라이언트 인스턴스
        """
        self.neo4j_client = neo4j_client
        self.entity_extractor = LegalEntityExtractor()
    
    def search(self, query: str, max_results: int = 10) -> List[GraphSearchResult]:
        """
        그래프 기반 검색
        
        Args:
            query: 검색 쿼리
            max_results: 최대 결과 수
            
        Returns:
            검색 결과 리스트
        """
        logger.info(f"그래프 검색 시작: {query}")
        
        # 쿼리에서 엔티티 추출
        query_entities = self.entity_extractor.extract_entities(query, "query", "query")
        
        results = []
        
        # 1. 직접 매칭 검색
        direct_results = self._direct_search(query, max_results // 2)
        results.extend(direct_results)
        
        # 2. 엔티티 기반 검색
        if query_entities:
            entity_results = self._entity_based_search(query_entities, max_results // 2)
            results.extend(entity_results)
        
        # 3. 관련 엔티티 확장 검색
        if len(results) < max_results:
            expansion_results = self._expansion_search(query_entities, max_results - len(results))
            results.extend(expansion_results)
        
        # 중복 제거 및 정렬
        results = self._deduplicate_and_rank(results, query)
        
        return results[:max_results]
    
    def _direct_search(self, query: str, limit: int) -> List[GraphSearchResult]:
        """직접 텍스트 매칭 검색"""
        cypher_query = """
        MATCH (n)
        WHERE n.text CONTAINS $query AND NOT 'Chunk' IN labels(n)
        OPTIONAL MATCH (n)-[r]-(related)
        WHERE NOT 'Chunk' IN labels(related)
        RETURN 
            n,
            labels(n) as labels,
            collect(DISTINCT related.text) as related_entities,
            n.source_file as source_file,
            n.chunk_id as chunk_id
        ORDER BY n.text
        LIMIT $limit
        """
        
        results = self.neo4j_client.execute_query(cypher_query, {
            "query": query,
            "limit": limit
        })
        
        search_results = []
        for result in results:
            node = result["n"]
            search_results.append(GraphSearchResult(
                entity_text=node.get("text", ""),
                entity_type=result["labels"][0] if result["labels"] else "Unknown",
                confidence=0.9,  # 직접 매칭은 높은 신뢰도
                context=node.get("text", ""),
                related_entities=result["related_entities"],
                source_file=result["source_file"],
                chunk_id=result["chunk_id"]
            ))
        
        return search_results
    
    def _entity_based_search(self, query_entities: List, limit: int) -> List[GraphSearchResult]:
        """엔티티 기반 검색"""
        if not query_entities:
            return []
        
        search_results = []
        
        for entity in query_entities:
            # 엔티티와 정확히 매칭되는 노드 찾기
            cypher_query = """
            MATCH (n)
            WHERE n.text = $entity_text AND NOT 'Chunk' IN labels(n)
            OPTIONAL MATCH (n)-[r]-(related)
            WHERE NOT 'Chunk' IN labels(related)
            RETURN 
                n,
                labels(n) as labels,
                collect(DISTINCT related.text) as related_entities,
                n.source_file as source_file,
                n.chunk_id as chunk_id
            LIMIT 5
            """
            
            results = self.neo4j_client.execute_query(cypher_query, {
                "entity_text": entity.text
            })
            
            for result in results:
                node = result["n"]
                search_results.append(GraphSearchResult(
                    entity_text=node.get("text", ""),
                    entity_type=result["labels"][0] if result["labels"] else "Unknown",
                    confidence=0.8,
                    context=node.get("text", ""),
                    related_entities=result["related_entities"],
                    source_file=result["source_file"],
                    chunk_id=result["chunk_id"]
                ))
        
        return search_results[:limit]
    
    def _expansion_search(self, query_entities: List, limit: int) -> List[GraphSearchResult]:
        """관련 엔티티 확장 검색"""
        if not query_entities:
            return []
        
        search_results = []
        
        for entity in query_entities:
            # 관련 엔티티 찾기 (2단계 확장)
            cypher_query = """
            MATCH (start {text: $entity_text})
            MATCH path = (start)-[*1..2]-(related)
            WHERE NOT 'Chunk' IN labels(related)
            RETURN DISTINCT 
                related,
                labels(related) as labels,
                length(path) as distance,
                related.source_file as source_file,
                related.chunk_id as chunk_id
            ORDER BY distance, related.text
            LIMIT 10
            """
            
            results = self.neo4j_client.execute_query(cypher_query, {
                "entity_text": entity.text
            })
            
            for result in results:
                node = result["related"]
                distance = result["distance"]
                confidence = max(0.3, 0.8 - (distance * 0.2))  # 거리에 따른 신뢰도 조정
                
                search_results.append(GraphSearchResult(
                    entity_text=node.get("text", ""),
                    entity_type=result["labels"][0] if result["labels"] else "Unknown",
                    confidence=confidence,
                    context=node.get("text", ""),
                    related_entities=[],  # 확장 검색에서는 관련 엔티티 생략
                    source_file=result["source_file"],
                    chunk_id=result["chunk_id"]
                ))
        
        return search_results[:limit]
    
    def _deduplicate_and_rank(self, results: List[GraphSearchResult], query: str) -> List[GraphSearchResult]:
        """중복 제거 및 랭킹"""
        # 중복 제거
        seen = set()
        unique_results = []
        
        for result in results:
            key = result.entity_text
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
        
        # 신뢰도 기반 정렬
        unique_results.sort(key=lambda x: x.confidence, reverse=True)
        
        return unique_results
    
    def get_legal_context(self, entity_text: str) -> Dict[str, Any]:
        """
        법률 엔티티의 상세 컨텍스트 조회
        
        Args:
            entity_text: 엔티티 텍스트
            
        Returns:
            상세 컨텍스트 정보
        """
        # 엔티티 정보 조회
        entity_query = """
        MATCH (n {text: $entity_text})
        WHERE NOT 'Chunk' IN labels(n)
        RETURN n, labels(n) as labels
        """
        
        entity_result = self.neo4j_client.execute_query(entity_query, {
            "entity_text": entity_text
        })
        
        if not entity_result:
            return {}
        
        entity_node = entity_result[0]["n"]
        entity_labels = entity_result[0]["labels"]
        
        # 관련 조항 조회
        related_articles_query = """
        MATCH (n {text: $entity_text})-[:REFERENCES|CONTAINS_ENTITY*1..2]-(article:Article)
        RETURN DISTINCT article
        ORDER BY article.article_number
        LIMIT 10
        """
        
        related_articles = self.neo4j_client.execute_query(related_articles_query, {
            "entity_text": entity_text
        })
        
        # 관련 법령 조회
        related_laws_query = """
        MATCH (n {text: $entity_text})-[:CONTAINS|CONTAINS_ENTITY*1..2]-(law:Law)
        RETURN DISTINCT law
        LIMIT 5
        """
        
        related_laws = self.neo4j_client.execute_query(related_laws_query, {
            "entity_text": entity_text
        })
        
        return {
            "entity": entity_node,
            "entity_type": entity_labels[0] if entity_labels else "Unknown",
            "related_articles": [article["article"] for article in related_articles],
            "related_laws": [law["law"] for law in related_laws],
            "context_summary": self._generate_context_summary(entity_node, related_articles, related_laws)
        }
    
    def _generate_context_summary(self, entity: Dict, articles: List, laws: List) -> str:
        """컨텍스트 요약 생성"""
        summary_parts = []
        
        if entity.get("text"):
            summary_parts.append(f"엔티티: {entity['text']}")
        
        if articles:
            article_texts = [article.get("text", "") for article in articles[:3]]
            summary_parts.append(f"관련 조항: {', '.join(article_texts)}")
        
        if laws:
            law_texts = [law.get("text", "") for law in laws[:2]]
            summary_parts.append(f"관련 법령: {', '.join(law_texts)}")
        
        return " | ".join(summary_parts)
    
    def find_similar_cases(self, query: str, case_type: str = "contract") -> List[Dict[str, Any]]:
        """
        유사 사례 찾기
        
        Args:
            query: 검색 쿼리
            case_type: 사례 타입 (contract, law, standard)
            
        Returns:
            유사 사례 리스트
        """
        cypher_query = """
        MATCH (chunk:Chunk)
        WHERE chunk.source_tag = $case_type
        AND chunk.text CONTAINS $query
        OPTIONAL MATCH (chunk)-[:CONTAINS_ENTITY]->(entity)
        WHERE NOT 'Chunk' IN labels(entity)
        RETURN 
            chunk,
            collect(DISTINCT entity.text) as entities,
            chunk.entity_count as entity_count
        ORDER BY chunk.entity_count DESC
        LIMIT 10
        """
        
        results = self.neo4j_client.execute_query(cypher_query, {
            "query": query,
            "case_type": case_type
        })
        
        return results
