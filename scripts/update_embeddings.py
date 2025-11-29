"""
파인튜닝된 모델로 기존 ViolationCase 노드의 임베딩만 업데이트

사용 시점:
- train_model.py로 파인튜닝 완료 후
- 기존 노드 구조는 유지하고 임베딩만 재계산할 때

주요 기능:
1. 기존 ViolationCase 노드에서 텍스트 읽기
2. 파인튜닝 모델로 임베딩 재계산
3. embedding_violation, embedding_corrected 업데이트
4. SIMILAR_TO 관계 재생성
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sys
from pathlib import Path
from tqdm import tqdm

# 프로젝트 루트
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from database.neo4j_connector import Neo4jConnector


class EmbeddingUpdater:
    """기존 노드의 임베딩만 업데이트"""

    def __init__(self, neo4j_connector, model_path='./my_fine_tuned_model'):
        """
        Args:
            neo4j_connector: Neo4j 커넥터
            model_path: 파인튜닝된 모델 경로
        """
        self.conn = neo4j_connector

        print(f"🧠 모델 로드 중: {model_path}")
        self.model = SentenceTransformer(model_path)

        embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ 모델 로드 완료 (임베딩 차원: {embedding_dim})")

        if embedding_dim != 1024:
            print(f"⚠️  경고: 임베딩 차원이 {embedding_dim}차원입니다. bge-m3는 1024차원이어야 합니다.")

        self.embedding_dim = embedding_dim

    def load_existing_nodes(self) -> pd.DataFrame:
        """기존 ViolationCase 노드에서 텍스트 데이터 로드"""
        print("\n" + "="*80)
        print("📊 기존 ViolationCase 노드 로드 중...")
        print("="*80)

        query = """
        MATCH (v:ViolationCase)
        WHERE v.original_text IS NOT NULL
          AND v.corrected_text IS NOT NULL
        RETURN 
            v.id as id,
            v.original_text as original_text,
            v.corrected_text as corrected_text,
            v.article_id as article_id
        ORDER BY v.id
        """

        results = self.conn.execute_query(query)

        if not results:
            print("❌ ViolationCase 노드가 없습니다.")
            print("   먼저 build_graph_base.py를 실행하세요.")
            return pd.DataFrame()

        # DataFrame 생성
        data = []
        for r in results:
            data.append({
                'id': r['id'],
                'original_text': r['original_text'],
                'corrected_text': r['corrected_text'],
                'article_id': r.get('article_id', '')
            })

        df = pd.DataFrame(data)
        print(f"✅ {len(df)}개 노드 로드 완료\n")

        return df

    def update_embeddings(self, df: pd.DataFrame):
        """임베딩 재계산 및 업데이트"""
        print("="*80)
        print("🧠 임베딩 재계산 중...")
        print("="*80)

        # 1. 위반 문장 임베딩
        print("\n1️⃣  위반 문장 임베딩 재계산...")
        violation_texts = df['original_text'].tolist()
        violation_embeddings = self.model.encode(
            violation_texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True
        )

        # 2. 준수 문장 임베딩
        print("\n2️⃣  준수 문장 임베딩 재계산...")
        corrected_texts = df['corrected_text'].tolist()
        corrected_embeddings = self.model.encode(
            corrected_texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True
        )

        print("\n✅ 임베딩 재계산 완료\n")

        # 3. Neo4j에 업데이트
        print("="*80)
        print("💾 Neo4j 임베딩 업데이트 중...")
        print("="*80)

        updated_count = 0
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="임베딩 업데이트"):
            case_id = row['id']
            embedding_violation = violation_embeddings[idx].tolist()
            embedding_corrected = corrected_embeddings[idx].tolist()

            query = """
            MATCH (v:ViolationCase {id: $id})
            SET v.embedding_violation = $embedding_violation,
                v.embedding_corrected = $embedding_corrected
            """

            self.conn.execute_query(query, {
                'id': case_id,
                'embedding_violation': embedding_violation,
                'embedding_corrected': embedding_corrected
            })

            updated_count += 1

        print(f"\n✅ {updated_count}개 노드 임베딩 업데이트 완료\n")

        return {
            'violation': violation_embeddings,
            'corrected': corrected_embeddings
        }

    def recreate_similarity_relationships(self, df: pd.DataFrame, embeddings: Dict):
        """SIMILAR_TO 관계 재생성"""
        print("="*80)
        print("🔗 SIMILAR_TO 관계 재생성 중...")
        print("="*80)

        # 기존 SIMILAR_TO 관계 삭제
        print("  - 기존 SIMILAR_TO 관계 삭제 중...")
        delete_query = "MATCH ()-[r:SIMILAR_TO]->() DELETE r"
        self.conn.execute_query(delete_query)
        print("  ✅ 기존 관계 삭제 완료")

        # 유사도 행렬 계산
        print("  - 유사도 행렬 계산 중...")
        violation_embeddings = embeddings['violation']
        similarity_matrix = cosine_similarity(violation_embeddings)

        # 관계 생성
        print("  - SIMILAR_TO 관계 생성 중...")
        threshold = 0.7
        top_k = 5

        created_count = 0
        for i in tqdm(range(len(df)), desc="SIMILAR_TO 생성"):
            case_id_1 = df.iloc[i]['id']

            # 자기 자신 제외하고 유사도 높은 top_k 찾기
            similarities = similarity_matrix[i].copy()
            similarities[i] = -1  # 자기 자신 제외

            top_indices = np.argsort(similarities)[::-1][:top_k]

            for j in top_indices:
                if similarities[j] < threshold:
                    continue

                case_id_2 = df.iloc[j]['id']

                query = """
                MATCH (v1:ViolationCase {id: $id1})
                MATCH (v2:ViolationCase {id: $id2})
                MERGE (v1)-[:SIMILAR_TO {
                    similarity: $similarity,
                    method: 'cosine'
                }]->(v2)
                """

                self.conn.execute_query(query, {
                    'id1': case_id_1,
                    'id2': case_id_2,
                    'similarity': float(similarities[j])
                })

                created_count += 1

        print(f"\n✅ {created_count}개 SIMILAR_TO 관계 생성 완료\n")

    def update_vector_indexes(self):
        """벡터 인덱스 재생성 (차원 변경 시 필요)"""
        print("="*80)
        print("📊 벡터 인덱스 확인 중...")
        print("="*80)

        # 기존 인덱스 삭제
        try:
            self.conn.execute_query("DROP INDEX violation_embeddings IF EXISTS")
            self.conn.execute_query("DROP INDEX corrected_embeddings IF EXISTS")
            print("  ✅ 기존 인덱스 삭제 완료")
        except Exception as e:
            print(f"  ⚠️  인덱스 삭제 실패 (무시): {e}")

        # 새 인덱스 생성
        print(f"\n1️⃣  violation_embeddings 인덱스 생성 ({self.embedding_dim}차원)")
        query_violation = f"""
        CREATE VECTOR INDEX violation_embeddings IF NOT EXISTS
        FOR (v:ViolationCase)
        ON v.embedding_violation
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {self.embedding_dim},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
        self.conn.execute_query(query_violation)
        print("   ✅ violation_embeddings 인덱스 생성 완료")

        print(f"\n2️⃣  corrected_embeddings 인덱스 생성 ({self.embedding_dim}차원)")
        query_corrected = f"""
        CREATE VECTOR INDEX corrected_embeddings IF NOT EXISTS
        FOR (v:ViolationCase)
        ON v.embedding_corrected
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {self.embedding_dim},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
        self.conn.execute_query(query_corrected)
        print("   ✅ corrected_embeddings 인덱스 생성 완료")

        print("\n✅ 모든 벡터 인덱스 생성 완료\n")

    def update_all(self):
        """전체 업데이트 프로세스 실행"""
        print("\n" + "="*80)
        print("🔄 파인튜닝 모델로 임베딩 업데이트 시작")
        print("="*80)
        print()

        # 1. 기존 노드 로드
        df = self.load_existing_nodes()
        if df.empty:
            return

        # 2. 벡터 인덱스 재생성
        self.update_vector_indexes()

        # 3. 임베딩 재계산 및 업데이트
        embeddings = self.update_embeddings(df)

        # 4. SIMILAR_TO 관계 재생성
        self.recreate_similarity_relationships(df, embeddings)

        # 5. 완료
        print("="*80)
        print("🎉 임베딩 업데이트 완료!")
        print("="*80)
        print(f"📊 업데이트된 노드: {len(df)}개")
        print(f"📊 임베딩 차원: {self.embedding_dim}차원")
        print("="*80)
        print()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 파인튜닝 모델로 임베딩 업데이트")
    print("="*80)
    print()

    # Neo4j 연결
    conn = Neo4jConnector()

    # Updater 생성 (기본: ./my_fine_tuned_model)
    updater = EmbeddingUpdater(
        neo4j_connector=conn,
        model_path='./my_fine_tuned_model'  # 파인튜닝된 모델
    )

    # 업데이트 실행
    updater.update_all()

    # 연결 종료
    conn.close()

    print("\n" + "="*80)
    print("✅ 모든 작업 완료!")
    print("="*80)
    print("\n다음 단계:")
    print("1. .env 파일 업데이트:")
    print("   EMBEDDING_MODEL=./my_fine_tuned_model")
    print("\n2. 테스트 실행:")
    print("   python scripts/test_ai_csv.py")
    print("   python scripts/test_test_input.py")
    print()
