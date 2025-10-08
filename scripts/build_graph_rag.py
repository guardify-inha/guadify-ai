"""
Graph RAG 파이프라인 구축 스크립트

법률 문서에서 그래프 데이터베이스를 구축하고 Graph RAG 시스템을 설정
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from db.neo4j_client import Neo4jClient
from db.graph_builder import GraphBuilder
from db.graph_retriever import GraphRetriever

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def build_graph_rag(chunks_file: str, neo4j_uri: str = None, neo4j_username: str = None, neo4j_password: str = None):
    """
    Graph RAG 시스템 구축
    
    Args:
        chunks_file: 청크 JSONL 파일 경로
        neo4j_uri: Neo4j 서버 URI
        neo4j_username: Neo4j 사용자명
        neo4j_password: Neo4j 비밀번호
    """
    logger.info("🚀 Graph RAG 시스템 구축 시작")
    
    # 파일 존재 확인
    if not os.path.exists(chunks_file):
        logger.error(f"청크 파일을 찾을 수 없습니다: {chunks_file}")
        return False
    
    try:
        # Neo4j 클라이언트 초기화
        logger.info("📡 Neo4j 연결 중...")
        neo4j_client = Neo4jClient(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password
        )
        
        # 그래프 빌더 초기화
        logger.info("🏗️ 그래프 빌더 초기화...")
        graph_builder = GraphBuilder(neo4j_client)
        
        # 그래프 구축
        logger.info("📊 법률 지식 그래프 구축 중...")
        graph_builder.build_graph_from_chunks(chunks_file)
        
        # Graph RAG 검색기 테스트
        logger.info("🔍 Graph RAG 검색기 테스트...")
        graph_retriever = GraphRetriever(neo4j_client)
        
        # 테스트 검색
        test_queries = [
            "제1조",
            "약관 규제",
            "손해배상",
            "계약 해지"
        ]
        
        for query in test_queries:
            logger.info(f"테스트 검색: '{query}'")
            results = graph_retriever.search(query, max_results=3)
            for i, result in enumerate(results, 1):
                logger.info(f"  {i}. {result.entity_text} ({result.entity_type}) - 신뢰도: {result.confidence:.2f}")
        
        # 통계 출력
        stats = neo4j_client.get_graph_stats()
        logger.info("📈 그래프 통계:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("✅ Graph RAG 시스템 구축 완료!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Graph RAG 구축 실패: {e}")
        return False
    
    finally:
        if 'neo4j_client' in locals():
            neo4j_client.close()

def test_graph_search(query: str, neo4j_uri: str = None, neo4j_username: str = None, neo4j_password: str = None):
    """
    Graph RAG 검색 테스트
    
    Args:
        query: 검색 쿼리
        neo4j_uri: Neo4j 서버 URI
        neo4j_username: Neo4j 사용자명
        neo4j_password: Neo4j 비밀번호
    """
    logger.info(f"🔍 Graph RAG 검색 테스트: '{query}'")
    
    try:
        # Neo4j 클라이언트 초기화
        neo4j_client = Neo4jClient(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password
        )
        
        # Graph RAG 검색기 초기화
        graph_retriever = GraphRetriever(neo4j_client)
        
        # 검색 실행
        results = graph_retriever.search(query, max_results=10)
        
        logger.info(f"검색 결과 ({len(results)}개):")
        for i, result in enumerate(results, 1):
            logger.info(f"  {i}. {result.entity_text}")
            logger.info(f"     타입: {result.entity_type}")
            logger.info(f"     신뢰도: {result.confidence:.2f}")
            logger.info(f"     출처: {result.source_file}")
            if result.related_entities:
                logger.info(f"     관련 엔티티: {', '.join(result.related_entities[:3])}")
            logger.info("")
        
        # 상세 컨텍스트 조회 (첫 번째 결과)
        if results:
            first_result = results[0]
            context = graph_retriever.get_legal_context(first_result.entity_text)
            if context:
                logger.info("📋 상세 컨텍스트:")
                logger.info(f"  엔티티 타입: {context.get('entity_type', 'Unknown')}")
                if context.get('related_articles'):
                    logger.info(f"  관련 조항: {len(context['related_articles'])}개")
                if context.get('related_laws'):
                    logger.info(f"  관련 법령: {len(context['related_laws'])}개")
                logger.info(f"  요약: {context.get('context_summary', 'N/A')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 검색 테스트 실패: {e}")
        return False
    
    finally:
        if 'neo4j_client' in locals():
            neo4j_client.close()

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Graph RAG 시스템 구축 및 테스트')
    parser.add_argument('--action', choices=['build', 'search'], required=True,
                       help='실행할 작업 (build: 그래프 구축, search: 검색 테스트)')
    parser.add_argument('--chunks-file', default='outputs/chunks.jsonl',
                       help='청크 JSONL 파일 경로 (기본값: outputs/chunks.jsonl)')
    parser.add_argument('--query', help='검색 쿼리 (search 액션 시 필수)')
    parser.add_argument('--neo4j-uri', help='Neo4j 서버 URI')
    parser.add_argument('--neo4j-username', help='Neo4j 사용자명')
    parser.add_argument('--neo4j-password', help='Neo4j 비밀번호')
    
    args = parser.parse_args()
    
    if args.action == 'build':
        success = build_graph_rag(
            chunks_file=args.chunks_file,
            neo4j_uri=args.neo4j_uri,
            neo4j_username=args.neo4j_username,
            neo4j_password=args.neo4j_password
        )
        sys.exit(0 if success else 1)
    
    elif args.action == 'search':
        if not args.query:
            logger.error("검색 쿼리를 입력해주세요 (--query 옵션)")
            sys.exit(1)
        
        success = test_graph_search(
            query=args.query,
            neo4j_uri=args.neo4j_uri,
            neo4j_username=args.neo4j_username,
            neo4j_password=args.neo4j_password
        )
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
