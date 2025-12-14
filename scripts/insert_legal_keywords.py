"""
법률용어 키워드를 Neo4j에 LegalKeyword 노드로 삽입하는 스크립트
"""
import sys
from pathlib import Path
import json

# 프로젝트 루트 추가
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from database.neo4j_connector import Neo4jConnector


def insert_legal_keywords():
    """legal_keywords_nodes.json의 키워드들을 Neo4j에 삽입"""
    
    # JSON 파일 경로
    keywords_file = Path(PROJECT_ROOT) / "data" / "legal_terms" / "legal_keywords_nodes.json"
    
    if not keywords_file.exists():
        print(f"❌ 키워드 파일을 찾을 수 없습니다: {keywords_file}")
        return
    
    # JSON 파일 읽기
    print(f"📖 키워드 파일 읽기: {keywords_file}")
    with open(keywords_file, 'r', encoding='utf-8') as f:
        keywords_data = json.load(f)
    
    print(f"✅ {len(keywords_data)}개의 키워드 로드 완료\n")
    
    # Neo4j 연결
    print("🔌 Neo4j 연결 중...")
    conn = Neo4jConnector()
    
    try:
        # 인덱스 생성 (성능 향상)
        print("📑 인덱스 생성 중...")
        index_query = """
        CREATE INDEX legal_keyword_keyword IF NOT EXISTS
        FOR (k:LegalKeyword) ON (k.keyword)
        """
        try:
            conn.execute_query(index_query)
            print("  ✓ 인덱스 생성 완료")
        except Exception as e:
            # 인덱스가 이미 존재하거나 다른 오류
            print(f"  ⚠️ 인덱스 생성 스킵: {e}")
        
        # 키워드 삽입
        print("\n📦 LegalKeyword 노드 삽입 중...")
        insert_query = """
        MERGE (k:LegalKeyword {keyword: $keyword})
        SET k.description = $description
        RETURN k.keyword as keyword
        """
        
        inserted_count = 0
        for idx, keyword_data in enumerate(keywords_data, 1):
            keyword = keyword_data.get('keyword', '')
            description = keyword_data.get('description', '')
            
            if not keyword:
                print(f"  ⚠️ {idx}번째 항목: keyword가 없어 스킵")
                continue
            
            try:
                result = conn.execute_query(insert_query, {
                    'keyword': keyword,
                    'description': description
                })
                inserted_count += 1
                
                if idx % 10 == 0:
                    print(f"  ✓ {idx}/{len(keywords_data)} 진행 중...")
            except Exception as e:
                print(f"  ❌ '{keyword}' 삽입 실패: {e}")
        
        print(f"\n✅ 총 {inserted_count}개의 키워드 삽입 완료!")
        
        # 통계 출력
        print("\n📊 삽입된 노드 통계:")
        stats_query = "MATCH (k:LegalKeyword) RETURN count(k) as count"
        result = conn.execute_query(stats_query)
        total_count = result[0]['count']
        print(f"  • LegalKeyword 노드: {total_count}개")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()
        print("\n🔌 Neo4j 연결 종료")


if __name__ == "__main__":
    print("=" * 70)
    print("법률용어 키워드 삽입 스크립트")
    print("=" * 70)
    print()
    
    insert_legal_keywords()
    
    print("\n" + "=" * 70)
    print("완료!")
    print("=" * 70)

