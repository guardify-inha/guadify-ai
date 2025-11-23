"""
Neo4j 벡터 인덱스 설정 v2.0

변경사항:
1. 이중 임베딩 구조 지원 (embedding_violation + embedding_corrected)
2. BAAI/bge-m3 모델 차원 (1024) 지원
3. 불필요한 LawArticle, ViolationType 인덱스 제거
4. 기존 모델(384/768 차원) 호환성 유지
"""

from neo4j import GraphDatabase
import os
import sys
from pathlib import Path

# 프로젝트 루트
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()


class Neo4jVectorIndexManager:
    """Neo4j 벡터 인덱스 관리"""

    def __init__(self, uri=None, user=None, password=None):
        """
        Args:
            uri: Neo4j URI (기본값: 환경변수에서 로드)
            user: Neo4j 사용자명 (기본값: 환경변수에서 로드)
            password: Neo4j 비밀번호 (기본값: 환경변수에서 로드)
        """
        self.uri = uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.user = user or os.getenv('NEO4J_USER', 'neo4j')
        self.password = password or os.getenv('NEO4J_PASSWORD', 'testpassword123')

        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password)
        )

        print(f"✅ Neo4j 연결: {self.uri}")

    def execute_query(self, query, parameters=None):
        """쿼리 실행"""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def close(self):
        """연결 종료"""
        self.driver.close()

    def drop_all_vector_indexes(self):
        """모든 벡터 인덱스 삭제"""
        print("\n🗑️  기존 벡터 인덱스 삭제 중...")

        indexes_to_drop = [
            'violation_embeddings',
            'corrected_embeddings',
            'law_embeddings',
            'violation_type_embeddings'
        ]

        for index_name in indexes_to_drop:
            try:
                self.execute_query(f"DROP INDEX {index_name} IF EXISTS")
                print(f"   ✅ {index_name} 삭제 완료")
            except Exception as e:
                print(f"   ⚠️  {index_name} 삭제 실패: {e}")

        print()

    def create_dual_vector_indexes(
        self,
        embedding_dim: int = 1024,
        similarity_function: str = 'cosine'
    ):
        """
        이중 벡터 인덱스 생성

        Args:
            embedding_dim: 임베딩 차원 (기본: 1024, bge-m3)
            similarity_function: 유사도 함수 ('cosine', 'euclidean', 'dot_product')
        """
        print("="*80)
        print(f"🔧 이중 벡터 인덱스 생성 중... (차원: {embedding_dim})")
        print("="*80)

        # 1. violation_embeddings: 위반 문장 검색용
        print(f"\n1️⃣  violation_embeddings 인덱스 생성...")
        print(f"   - 속성: embedding_violation")
        print(f"   - 차원: {embedding_dim}")
        print(f"   - 유사도: {similarity_function}")

        try:
            self.execute_query(f"""
            CREATE VECTOR INDEX violation_embeddings IF NOT EXISTS
            FOR (v:ViolationCase)
            ON v.embedding_violation
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {embedding_dim},
                    `vector.similarity_function`: '{similarity_function}'
                }}
            }}
            """)
            print("   ✅ violation_embeddings 생성 완료")
        except Exception as e:
            print(f"   ❌ violation_embeddings 생성 실패: {e}")

        # 2. corrected_embeddings: 준수 문장 검색용 (신규!)
        print(f"\n2️⃣  corrected_embeddings 인덱스 생성... ⭐ 신규")
        print(f"   - 속성: embedding_corrected")
        print(f"   - 차원: {embedding_dim}")
        print(f"   - 유사도: {similarity_function}")

        try:
            self.execute_query(f"""
            CREATE VECTOR INDEX corrected_embeddings IF NOT EXISTS
            FOR (v:ViolationCase)
            ON v.embedding_corrected
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {embedding_dim},
                    `vector.similarity_function`: '{similarity_function}'
                }}
            }}
            """)
            print("   ✅ corrected_embeddings 생성 완료")
        except Exception as e:
            print(f"   ❌ corrected_embeddings 생성 실패: {e}")

        print("\n✅ 이중 벡터 인덱스 생성 완료!\n")

    def create_property_indexes(self):
        """속성 인덱스 생성 (빠른 검색용)"""
        print("="*80)
        print("🔧 속성 인덱스 생성 중...")
        print("="*80)

        property_indexes = [
            ("ViolationCase", "id"),
            ("ViolationCase", "article_id"),
            ("ViolationCase", "company"),
            ("ViolationCase", "year"),
            ("ViolationCase", "category"),
        ]

        print()
        for label, property_name in property_indexes:
            try:
                index_name = f"{label.lower()}_{property_name}_index"
                self.execute_query(f"""
                CREATE INDEX {index_name} IF NOT EXISTS
                FOR (n:{label})
                ON (n.{property_name})
                """)
                print(f"   ✅ {label}.{property_name} 인덱스 생성")
            except Exception as e:
                print(f"   ⚠️  {label}.{property_name} 실패: {e}")

        print("\n✅ 속성 인덱스 생성 완료!\n")

    def verify_indexes(self):
        """생성된 인덱스 확인"""
        print("="*80)
        print("📊 생성된 인덱스 목록")
        print("="*80)

        result = self.execute_query("SHOW INDEXES")

        # 벡터 인덱스와 일반 인덱스 분리
        vector_indexes = []
        property_indexes = []

        for idx in result:
            index_type = idx.get('type', 'N/A')
            if 'VECTOR' in index_type.upper():
                vector_indexes.append(idx)
            else:
                property_indexes.append(idx)

        # 벡터 인덱스 출력
        print("\n🔍 벡터 인덱스:")
        if vector_indexes:
            for idx in vector_indexes:
                name = idx.get('name', 'N/A')
                state = idx.get('state', 'N/A')
                status_emoji = "✅" if state == "ONLINE" else "⏳"
                print(f"   {status_emoji} {name} - {state}")
        else:
            print("   ⚠️  벡터 인덱스 없음")

        # 속성 인덱스 출력
        print("\n📑 속성 인덱스:")
        if property_indexes:
            for idx in property_indexes:
                name = idx.get('name', 'N/A')
                state = idx.get('state', 'N/A')
                status_emoji = "✅" if state == "ONLINE" else "⏳"
                print(f"   {status_emoji} {name} - {state}")
        else:
            print("   ⚠️  속성 인덱스 없음")

        print()

    def check_embedding_dimensions(self):
        """현재 DB의 임베딩 차원 확인"""
        print("="*80)
        print("🔍 임베딩 차원 확인")
        print("="*80)

        query = """
        MATCH (v:ViolationCase)
        WHERE v.embedding_violation IS NOT NULL
        RETURN size(v.embedding_violation) as violation_dim,
               size(v.embedding_corrected) as corrected_dim
        LIMIT 1
        """

        try:
            result = self.execute_query(query)
            if result:
                violation_dim = result[0].get('violation_dim', 'N/A')
                corrected_dim = result[0].get('corrected_dim', 'N/A')

                print(f"\n현재 임베딩 차원:")
                print(f"   - embedding_violation: {violation_dim}차원")
                print(f"   - embedding_corrected: {corrected_dim}차원")
                print()

                # 권장사항 출력
                if violation_dim == 384:
                    print("⚠️  384차원 (이전 모델)")
                    print("   권장: Fine-tuned bge-m3 (1024차원)으로 업그레이드")
                elif violation_dim == 1024:
                    print("✅ 1024차원 (bge-m3 모델)")
                else:
                    print(f"⚠️  예상치 못한 차원: {violation_dim}")

                return violation_dim
            else:
                print("\n⚠️  ViolationCase 노드가 없습니다.")
                return None
        except Exception as e:
            print(f"\n❌ 임베딩 차원 확인 실패: {e}")
            return None


def setup_all_indexes(
    embedding_dim: int = 1024,
    drop_existing: bool = False
):
    """
    모든 인덱스 설정

    Args:
        embedding_dim: 임베딩 차원 (기본: 1024)
        drop_existing: 기존 인덱스 삭제 여부
    """
    manager = Neo4jVectorIndexManager()

    try:
        # 임베딩 차원 확인
        current_dim = manager.check_embedding_dimensions()

        # 차원 불일치 경고
        if current_dim and current_dim != embedding_dim:
            print(f"\n⚠️  경고: 현재 DB 차원({current_dim})과 설정 차원({embedding_dim})이 다릅니다!")
            print(f"   인덱스를 {current_dim}차원으로 생성하는 것을 권장합니다.")
            print()

            user_input = input(f"   {embedding_dim}차원으로 계속 진행하시겠습니까? (y/n): ")
            if user_input.lower() != 'y':
                print("\n❌ 작업 취소")
                manager.close()
                return

            # 차원 업데이트
            embedding_dim = current_dim

        # 기존 인덱스 삭제 (옵션)
        if drop_existing:
            manager.drop_all_vector_indexes()

        # 벡터 인덱스 생성
        manager.create_dual_vector_indexes(embedding_dim=embedding_dim)

        # 속성 인덱스 생성
        manager.create_property_indexes()

        # 확인
        manager.verify_indexes()

        print("="*80)
        print("🎉 모든 인덱스 설정 완료!")
        print("="*80)

    finally:
        manager.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Neo4j 벡터 인덱스 설정')
    parser.add_argument(
        '--dim',
        type=int,
        default=1024,
        help='임베딩 차원 (기본: 1024, bge-m3)'
    )
    parser.add_argument(
        '--drop',
        action='store_true',
        help='기존 벡터 인덱스 삭제 후 재생성'
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("🚀 Neo4j 벡터 인덱스 설정 v2.0")
    print("="*80)
    print(f"   임베딩 차원: {args.dim}")
    print(f"   기존 인덱스 삭제: {'예' if args.drop else '아니오'}")
    print()

    setup_all_indexes(
        embedding_dim=args.dim,
        drop_existing=args.drop
    )
