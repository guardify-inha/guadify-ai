"""
불공정 조항 및 수정본에 임베딩 벡터 추가
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

try:
    from sentence_transformers import SentenceTransformer
    MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("✓ 임베딩 모델 로드 완료")
except Exception as e:
    print(f"✗ 임베딩 모델 로드 실패: {e}")
    sys.exit(1)


def add_embeddings_to_violations(conn: Neo4jConnector):
    """불공정 조항 원문에 임베딩 추가"""
    print("\n📊 불공정 조항 임베딩 생성 중...")
    
    # 모든 불공정 조항 조회
    query = """
    MATCH (v:불공정조항원문)
    WHERE v.text IS NOT NULL
    RETURN v.id as id, v.text as text
    """
    
    violations = conn.execute_query(query)
    
    if not violations:
        print("  ⚠️  불공정 조항이 없습니다.")
        return
    
    print(f"  → {len(violations)}개 조항 처리 중...")
    
    for idx, v in enumerate(violations, 1):
        # 임베딩 생성
        embedding = MODEL.encode(v['text'])
        embedding_list = embedding.tolist()
        
        # Neo4j에 저장
        update_query = """
        MATCH (v:불공정조항원문 {id: $id})
        SET v.embedding = $embedding
        """
        conn.execute_query(update_query, {
            "id": v['id'],
            "embedding": embedding_list
        })
        
        if idx % 10 == 0:
            print(f"  ... {idx}/{len(violations)} 완료")
    
    print(f"  ✓ {len(violations)}개 불공정 조항 임베딩 완료")


def add_embeddings_to_corrections(conn: Neo4jConnector):
    """수정 후 약관에 임베딩 추가"""
    print("\n📊 수정 후 약관 임베딩 생성 중...")
    
    # 모든 수정본 조회
    query = """
    MATCH (c:수정후약관)
    WHERE c.text IS NOT NULL
    RETURN c.id as id, c.text as text
    """
    
    corrections = conn.execute_query(query)
    
    if not corrections:
        print("  ⚠️  수정 후 약관이 없습니다.")
        return
    
    print(f"  → {len(corrections)}개 수정본 처리 중...")
    
    for idx, c in enumerate(corrections, 1):
        # 임베딩 생성
        embedding = MODEL.encode(c['text'])
        embedding_list = embedding.tolist()
        
        # Neo4j에 저장
        update_query = """
        MATCH (c:수정후약관 {id: $id})
        SET c.embedding = $embedding
        """
        conn.execute_query(update_query, {
            "id": c['id'],
            "embedding": embedding_list
        })
        
        if idx % 10 == 0:
            print(f"  ... {idx}/{len(corrections)} 완료")
    
    print(f"  ✓ {len(corrections)}개 수정본 임베딩 완료")


def verify_embeddings(conn: Neo4jConnector):
    """임베딩 추가 확인"""
    print("\n🔍 임베딩 검증 중...")
    
    query_violations = """
    MATCH (v:불공정조항원문)
    WHERE v.embedding IS NOT NULL
    RETURN count(v) as count
    """
    
    query_corrections = """
    MATCH (c:수정후약관)
    WHERE c.embedding IS NOT NULL
    RETURN count(c) as count
    """
    
    violation_count = conn.execute_query(query_violations)[0]['count']
    correction_count = conn.execute_query(query_corrections)[0]['count']
    
    print(f"  • 불공정 조항 (임베딩 있음): {violation_count}개")
    print(f"  • 수정 후 약관 (임베딩 있음): {correction_count}개")


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("불공정 조항 임베딩 생성 스크립트")
    print("=" * 60)
    
    conn = Neo4jConnector()
    
    try:
        # 1. 불공정 조항 임베딩
        add_embeddings_to_violations(conn)
        
        # 2. 수정본 임베딩
        add_embeddings_to_corrections(conn)
        
        # 3. 검증
        verify_embeddings(conn)
        
        print("\n✅ 모든 임베딩 생성 완료!")
        
    except Exception as e:
        print(f"\n✗ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()


if __name__ == "__main__":
    main()