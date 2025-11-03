"""
법률 조항(조, 항, 호)의 content 필드를 임베딩하여 Neo4j에 저장

1단계: 그래프 모델링 및 시맨틱 보강 - 1.3
"""
import sys
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

from database.neo4j_connector import Neo4jConnector

# 임베딩 모델 로드 (폴백 처리)
try:
    from sentence_transformers import SentenceTransformer
    try:
        # 주요 모델 시도
        MODEL = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'
        print("✓ 모델 로드 완료: paraphrase-multilingual-mpnet-base-v2")
    except Exception as e:
        print(f"⚠️ 주요 모델 로드 실패, 폴백 모델 사용: {e}")
        MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
        print("✓ 폴백 모델 로드 완료: paraphrase-multilingual-MiniLM-L12-v2")
except Exception as e:
    print(f"❌ 임베딩 모델 로드 실패: {e}")
    sys.exit(1)

def embed_text(text: str) -> Optional[List[float]]:
    """텍스트를 임베딩 벡터로 변환"""
    if not text or not text.strip():
        return None
    try:
        embedding = MODEL.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    except Exception as e:
        print(f"⚠️ 임베딩 생성 실패: {e}")
        return None

def embed_nodes(conn: Neo4jConnector, node_type: str, label: str) -> int:
    """
    특정 타입의 노드들을 임베딩하여 저장
    
    Args:
        conn: Neo4j 연결 객체
        node_type: 노드 타입 ('조', '항', '호')
        label: 출력용 라벨
        
    Returns:
        처리된 노드 개수
    """
    print(f"\n📝 {label} 노드 임베딩 생성 중...")
    
    # 노드 조회 (content가 있는 것만)
    query = f"""
    MATCH (n:{node_type})
    WHERE n.content IS NOT NULL AND n.content <> ''
    RETURN n.id as id, n.content as content
    """
    
    nodes = conn.execute_query(query)
    
    if not nodes:
        print(f"  ⚠️ {label} 노드가 없거나 content가 비어있습니다.")
        return 0
    
    processed = 0
    for node in nodes:
        node_id = node['id']
        content = node['content']
        
        # 임베딩 생성
        embedding = embed_text(content)
        if embedding is None:
            continue
        
        # Neo4j에 저장
        update_query = f"""
        MATCH (n:{node_type} {{id: $node_id}})
        SET n.embedding = $embedding
        RETURN n.id as id
        """
        
        try:
            result = conn.execute_query(update_query, {
                "node_id": node_id,
                "embedding": embedding
            })
            processed += 1
            if processed % 5 == 0:
                print(f"  ... {processed}개 처리 완료")
        except Exception as e:
            print(f"  ⚠️ 노드 {node_id} 저장 실패: {e}")
    
    print(f"  ✓ {label} 노드 {processed}/{len(nodes)}개 임베딩 저장 완료")
    return processed

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("🔤 법률 조항 임베딩 생성 및 저장")
    print("=" * 60)
    print(f"사용 모델: {MODEL_NAME}")
    print(f"임베딩 차원: {MODEL.get_sentence_embedding_dimension()}")
    
    conn = Neo4jConnector()
    
    try:
        total_processed = 0
        
        # 조 노드 임베딩
        total_processed += embed_nodes(conn, '조', '조')
        
        # 항 노드 임베딩
        total_processed += embed_nodes(conn, '항', '항')
        
        # 호 노드 임베딩
        total_processed += embed_nodes(conn, '호', '호')
        
        print("\n" + "=" * 60)
        print(f"🎉 전체 완료: 총 {total_processed}개 노드 임베딩 저장")
        print("=" * 60)
        
        # 통계 출력
        stats_query = """
        MATCH (n)
        WHERE n.embedding IS NOT NULL
        RETURN labels(n)[0] as label, count(n) as count
        ORDER BY label
        """
        stats = conn.execute_query(stats_query)
        print("\n📊 임베딩 저장 통계:")
        for stat in stats:
            print(f"  • {stat['label']}: {stat['count']}개")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

if __name__ == "__main__":
    main()

