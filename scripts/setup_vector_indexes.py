"""
Neo4j 벡터 인덱스 설정

Neo4j 5.x에서 벡터 검색을 위한 인덱스 생성
"""

from neo4j import GraphDatabase


class Neo4jConnector:
    """Neo4j 연결 및 쿼리 실행 클래스"""

    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="testpassword123"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def execute_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def close(self):
        self.driver.close()


def create_vector_indexes(conn: Neo4jConnector):
    """벡터 인덱스 생성"""

    print("🔧 Neo4j 벡터 인덱스 생성 중...\n")

    # 1. ViolationCase 벡터 인덱스
    print("1. ViolationCase 인덱스 생성...")
    try:
        conn.execute_query("""
        CREATE VECTOR INDEX violation_embeddings IF NOT EXISTS
        FOR (v:ViolationCase)
        ON v.embedding
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: 768,
                `vector.similarity_function`: 'cosine'
            }
        }
        """)
        print("   ✅ ViolationCase 인덱스 생성 완료")
    except Exception as e:
        print(f"   ⚠️ ViolationCase 인덱스: {e}")

    # 2. LawArticle 벡터 인덱스
    print("\n2. LawArticle 인덱스 생성...")
    try:
        conn.execute_query("""
        CREATE VECTOR INDEX law_embeddings IF NOT EXISTS
        FOR (l:LawArticle)
        ON l.embedding
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: 768,
                `vector.similarity_function`: 'cosine'
            }
        }
        """)
        print("   ✅ LawArticle 인덱스 생성 완료")
    except Exception as e:
        print(f"   ⚠️ LawArticle 인덱스: {e}")

    # 3. ViolationType 벡터 인덱스
    print("\n3. ViolationType 인덱스 생성...")
    try:
        conn.execute_query("""
        CREATE VECTOR INDEX violation_type_embeddings IF NOT EXISTS
        FOR (t:ViolationType)
        ON t.embedding
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: 768,
                `vector.similarity_function`: 'cosine'
            }
        }
        """)
        print("   ✅ ViolationType 인덱스 생성 완료")
    except Exception as e:
        print(f"   ⚠️ ViolationType 인덱스: {e}")

    # 4. 속성 인덱스 (빠른 검색용)
    print("\n4. 속성 인덱스 생성...")
    property_indexes = [
        ("ViolationCase", "id"),
        ("ViolationCase", "severity"),
        ("ViolationCase", "year"),
        ("ViolationCase", "article_id"),
        ("LawArticle", "id"),
        ("LawArticle", "category"),
        ("ViolationType", "name"),
        ("Company", "name"),
        ("Keyword", "text"),
    ]

    for label, property_name in property_indexes:
        try:
            index_name = f"{label.lower()}_{property_name}_index"
            conn.execute_query(f"""
            CREATE INDEX {index_name} IF NOT EXISTS
            FOR (n:{label})
            ON (n.{property_name})
            """)
            print(f"   ✅ {label}.{property_name} 인덱스 생성")
        except Exception as e:
            print(f"   ⚠️ {label}.{property_name}: {e}")

    # 5. 복합 인덱스
    print("\n5. 복합 인덱스 생성...")
    try:
        conn.execute_query("""
        CREATE INDEX violation_severity_year IF NOT EXISTS
        FOR (v:ViolationCase)
        ON (v.severity, v.year)
        """)
        print("   ✅ ViolationCase(severity, year) 복합 인덱스 생성")
    except Exception as e:
        print(f"   ⚠️ 복합 인덱스: {e}")

    print("\n✅ 모든 인덱스 생성 완료!\n")


def verify_indexes(conn: Neo4jConnector):
    """생성된 인덱스 확인"""
    print("📊 생성된 인덱스 목록:\n")

    result = conn.execute_query("SHOW INDEXES")

    for idx in result:
        index_type = idx.get('type', 'N/A')
        name = idx.get('name', 'N/A')
        state = idx.get('state', 'N/A')

        status_emoji = "✅" if state == "ONLINE" else "⏳"
        print(f"{status_emoji} [{index_type}] {name} - {state}")

    print()


if __name__ == "__main__":
    conn = Neo4jConnector()

    # 인덱스 생성
    create_vector_indexes(conn)

    # 확인
    verify_indexes(conn)

    conn.close()
