"""
파인튜닝된 모델로 기존 ViolationCase 노드의 임베딩만 업데이트

사용 시점:
- train_model.py로 파인튜닝 완료 후
- 기존 노드 구조는 유지하고 임베딩만 재계산할 때

주요 기능:
1. 기존 ViolationCase 노드에서 텍스트 읽기
2. Base 모델과 Finetuned 모델로 4개 임베딩 재계산
3. embedding_violation_base, embedding_violation_finetuned, embedding_corrected_base, embedding_corrected_finetuned 업데이트
4. 레거시 필드(embedding_violation, embedding_corrected) 삭제
5. SIMILAR_TO 관계 재생성
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
    """기존 노드의 임베딩만 업데이트 (이중 임베딩 전략)"""

    def __init__(self, neo4j_connector, model_path_base=None, model_path_finetuned=None):
        """
        Args:
            neo4j_connector: Neo4j 커넥터
            model_path_base: Base 모델 경로 (기본: config.settings에서 읽음)
            model_path_finetuned: Finetuned 모델 경로 (기본: config.settings에서 읽음)
        """
        self.conn = neo4j_connector

        # config.settings에서 모델 경로 읽기
        from config.settings import settings
        model_path_base = model_path_base or settings.EMBEDDING_MODEL_BASE
        model_path_finetuned = model_path_finetuned or settings.EMBEDDING_MODEL_FINETUNED

        print(f"🧠 Base 모델 로드 중: {model_path_base}")
        self.base_model = SentenceTransformer(model_path_base)
        base_dim = self.base_model.get_sentence_embedding_dimension()
        print(f"✅ Base 모델 로드 완료 (임베딩 차원: {base_dim})")

        print(f"🧠 Finetuned 모델 로드 중: {model_path_finetuned}")
        self.finetuned_model = SentenceTransformer(model_path_finetuned)
        finetuned_dim = self.finetuned_model.get_sentence_embedding_dimension()
        print(f"✅ Finetuned 모델 로드 완료 (임베딩 차원: {finetuned_dim})")

        if base_dim != 1024 or finetuned_dim != 1024:
            print(f"⚠️  경고: 임베딩 차원이 일치하지 않습니다. bge-m3는 1024차원이어야 합니다.")
            print(f"   Base: {base_dim}차원, Finetuned: {finetuned_dim}차원")

        self.embedding_dim = base_dim  # 두 모델 모두 같은 차원이어야 함

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
        """4개 임베딩 재계산 및 업데이트 (Base + Finetuned 모델)"""
        print("="*80)
        print("🧠 이중 임베딩 재계산 중 (Base + Finetuned 모델)...")
        print("="*80)

        violation_texts = df['original_text'].tolist()
        corrected_texts = df['corrected_text'].tolist()

        # 1. 위반 문장 - Base 모델
        print("\n1️⃣  위반 문장 (original_text) - Base 모델 임베딩 재계산...")
        violation_embeddings_base = self.base_model.encode(
            violation_texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True
        )

        # 2. 위반 문장 - Finetuned 모델
        print("\n2️⃣  위반 문장 (original_text) - Finetuned 모델 임베딩 재계산...")
        violation_embeddings_finetuned = self.finetuned_model.encode(
            violation_texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True
        )

        # 3. 준수 문장 - Base 모델
        print("\n3️⃣  준수 문장 (corrected_text) - Base 모델 임베딩 재계산...")
        corrected_embeddings_base = self.base_model.encode(
            corrected_texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True
        )

        # 4. 준수 문장 - Finetuned 모델
        print("\n4️⃣  준수 문장 (corrected_text) - Finetuned 모델 임베딩 재계산...")
        corrected_embeddings_finetuned = self.finetuned_model.encode(
            corrected_texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True
        )

        print("\n✅ 4개 임베딩 재계산 완료\n")

        # 5. Neo4j에 업데이트 (4개 필드 모두)
        print("="*80)
        print("💾 Neo4j 임베딩 업데이트 중 (4개 필드)...")
        print("="*80)

        updated_count = 0
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="임베딩 업데이트"):
            case_id = row['id']
            embedding_violation_base = violation_embeddings_base[idx].tolist()
            embedding_violation_finetuned = violation_embeddings_finetuned[idx].tolist()
            embedding_corrected_base = corrected_embeddings_base[idx].tolist()
            embedding_corrected_finetuned = corrected_embeddings_finetuned[idx].tolist()

            query = """
            MATCH (v:ViolationCase {id: $id})
            SET v.embedding_violation_base = $embedding_violation_base,
                v.embedding_violation_finetuned = $embedding_violation_finetuned,
                v.embedding_corrected_base = $embedding_corrected_base,
                v.embedding_corrected_finetuned = $embedding_corrected_finetuned
            REMOVE v.embedding_violation, v.embedding_corrected
            """

            self.conn.execute_query(query, {
                'id': case_id,
                'embedding_violation_base': embedding_violation_base,
                'embedding_violation_finetuned': embedding_violation_finetuned,
                'embedding_corrected_base': embedding_corrected_base,
                'embedding_corrected_finetuned': embedding_corrected_finetuned
            })

            updated_count += 1

        print(f"\n✅ {updated_count}개 노드 임베딩 업데이트 완료 (4개 필드)\n")

        return {
            'violation_base': violation_embeddings_base,
            'violation_finetuned': violation_embeddings_finetuned,
            'corrected_base': corrected_embeddings_base,
            'corrected_finetuned': corrected_embeddings_finetuned
        }

    def recreate_similarity_relationships(self, df: pd.DataFrame, embeddings: Dict):
        """SIMILAR_TO 관계 재생성 (Base 모델 임베딩 사용)"""
        print("="*80)
        print("🔗 SIMILAR_TO 관계 재생성 중...")
        print("="*80)

        # 기존 SIMILAR_TO 관계 삭제
        print("  - 기존 SIMILAR_TO 관계 삭제 중...")
        delete_query = "MATCH ()-[r:SIMILAR_TO]->() DELETE r"
        self.conn.execute_query(delete_query)
        print("  ✅ 기존 관계 삭제 완료")

        # 유사도 행렬 계산 (Base 모델 임베딩 사용 - RAG 검색용)
        print("  - 유사도 행렬 계산 중 (Base 모델 임베딩 사용)...")
        violation_embeddings = embeddings['violation_base']
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
        """벡터 인덱스 재생성 (4개 인덱스: base + finetuned)"""
        print("="*80)
        print("📊 벡터 인덱스 확인 중 (4개 인덱스)...")
        print("="*80)

        # 기존 인덱스 삭제
        try:
            self.conn.execute_query("DROP INDEX violation_embeddings_base IF EXISTS")
            self.conn.execute_query("DROP INDEX violation_embeddings_finetuned IF EXISTS")
            self.conn.execute_query("DROP INDEX corrected_embeddings_base IF EXISTS")
            self.conn.execute_query("DROP INDEX corrected_embeddings_finetuned IF EXISTS")
            print("  ✅ 기존 인덱스 삭제 완료")
        except Exception as e:
            print(f"  ⚠️  인덱스 삭제 실패 (무시): {e}")

        # 새 인덱스 생성 (4개)
        print(f"\n1️⃣  violation_embeddings_base 인덱스 생성 ({self.embedding_dim}차원)")
        query_violation_base = f"""
        CREATE VECTOR INDEX violation_embeddings_base IF NOT EXISTS
        FOR (v:ViolationCase)
        ON v.embedding_violation_base
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {self.embedding_dim},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
        self.conn.execute_query(query_violation_base)
        print("   ✅ violation_embeddings_base 인덱스 생성 완료")

        print(f"\n2️⃣  violation_embeddings_finetuned 인덱스 생성 ({self.embedding_dim}차원)")
        query_violation_finetuned = f"""
        CREATE VECTOR INDEX violation_embeddings_finetuned IF NOT EXISTS
        FOR (v:ViolationCase)
        ON v.embedding_violation_finetuned
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {self.embedding_dim},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
        self.conn.execute_query(query_violation_finetuned)
        print("   ✅ violation_embeddings_finetuned 인덱스 생성 완료")

        print(f"\n3️⃣  corrected_embeddings_base 인덱스 생성 ({self.embedding_dim}차원)")
        query_corrected_base = f"""
        CREATE VECTOR INDEX corrected_embeddings_base IF NOT EXISTS
        FOR (v:ViolationCase)
        ON v.embedding_corrected_base
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {self.embedding_dim},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
        self.conn.execute_query(query_corrected_base)
        print("   ✅ corrected_embeddings_base 인덱스 생성 완료")

        print(f"\n4️⃣  corrected_embeddings_finetuned 인덱스 생성 ({self.embedding_dim}차원)")
        query_corrected_finetuned = f"""
        CREATE VECTOR INDEX corrected_embeddings_finetuned IF NOT EXISTS
        FOR (v:ViolationCase)
        ON v.embedding_corrected_finetuned
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {self.embedding_dim},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
        self.conn.execute_query(query_corrected_finetuned)
        print("   ✅ corrected_embeddings_finetuned 인덱스 생성 완료")

        print("\n✅ 모든 벡터 인덱스 생성 완료 (4개 인덱스)\n")

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
        print(f"📊 업데이트된 필드: 4개 (violation_base, violation_finetuned, corrected_base, corrected_finetuned)")
        print("="*80)
        print()


if __name__ == "__main__":
    import os    
    print("\n" + "="*80)
    print("🚀 파인튜닝 모델로 임베딩 업데이트")
    print("="*80)
    print()

    # config.settings에서 모델 경로 읽기
    from config.settings import settings
    model_path_base = settings.EMBEDDING_MODEL_BASE
    model_path_finetuned = settings.EMBEDDING_MODEL_FINETUNED
    print(f"📦 Base 모델: {model_path_base}")
    print(f"📦 Finetuned 모델: {model_path_finetuned}")
    print(f"   (.env 파일의 EMBEDDING_MODEL_BASE, EMBEDDING_MODEL_FINETUNED 환경변수에서 읽음)")
    print()

    # Neo4j 연결
    conn = Neo4jConnector()

    # Updater 생성 (두 모델 모두 사용)
    updater = EmbeddingUpdater(
        neo4j_connector=conn
        # model_path_base, model_path_finetuned는 config.settings에서 자동으로 읽음
    )

    # 업데이트 실행
    updater.update_all()

    # 연결 종료
    conn.close()

    print("\n" + "="*80)
    print("✅ 모든 작업 완료!")
    print("="*80)
    print(f"\n📦 사용된 모델:")
    print(f"   - Base: {model_path_base}")
    print(f"   - Finetuned: {model_path_finetuned}")
    print("\n다음 단계:")
    print("1. 벡터 인덱스 확인 (필요시):")
    print("   python scripts/setup_vector_indexes_v2.py --dim 1024")
    print("\n2. 테스트 실행:")
    print("   python scripts/test_ai_csv.py")
    print("   python scripts/test_test_input.py")
    print()
