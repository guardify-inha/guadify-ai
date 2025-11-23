"""
Fine-tuned BAAI/bge-m3 모델을 사용하여 Neo4j 그래프 재구성

주요 변경사항:
1. 임베딩 모델: paraphrase-multilingual-MiniLM-L12-v2 (384차원)
   → BAAI/bge-m3 fine-tuned (1024차원)

2. 이중 임베딩 구조:
   - embedding_violation: original_text (위반 문장) 임베딩
   - embedding_corrected: corrected_text (준수 문장) 임베딩

3. 이중 벡터 인덱스:
   - violation_embeddings: 위반 문장 검색용
   - corrected_embeddings: 준수 문장 검색용 (새로 추가!)
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sys
from pathlib import Path
from tqdm import tqdm
import json

# 프로젝트 루트
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from database.neo4j_connector import Neo4jConnector


class GraphRAGRebuilder:
    """Fine-tuned 모델로 그래프 재구성"""

    def __init__(
        self,
        neo4j_connector,
        model_path='./my_fine_tuned_model'
    ):
        """
        Args:
            neo4j_connector: Neo4j 커넥터
            model_path: Fine-tuned 모델 경로 (기본: ./my_fine_tuned_model)
        """
        self.conn = neo4j_connector

        print(f"🧠 모델 로드 중: {model_path}")
        self.model = SentenceTransformer(model_path)

        embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ 모델 로드 완료 (임베딩 차원: {embedding_dim})")

        if embedding_dim != 1024:
            print(f"⚠️  경고: 임베딩 차원이 {embedding_dim}차원입니다. bge-m3는 1024차원이어야 합니다.")

        self.embedding_dim = embedding_dim

        # 패턴 데이터 로드
        self._load_patterns()

    def _load_patterns(self):
        """patterns_by_article_v2.json 로드"""
        try:
            pattern_path = project_root / "data" / "contracts" / "reference" / "patterns_by_article_v2.json"

            if pattern_path.exists():
                with open(pattern_path, 'r', encoding='utf-8') as f:
                    self.patterns = json.load(f)
                print(f"✅ 패턴 데이터 로드: {pattern_path}")
            else:
                print(f"⚠️  패턴 파일 없음: {pattern_path}")
                self.patterns = {}
        except Exception as e:
            print(f"⚠️  패턴 로드 실패: {e}")
            self.patterns = {}

    def clear_database(self):
        """기존 데이터 삭제"""
        print("\n" + "="*80)
        print("🗑️  기존 데이터 삭제 중...")
        print("="*80)

        # 모든 노드와 관계 삭제
        self.conn.execute_query("MATCH (n) DETACH DELETE n")

        # 인덱스 삭제 (에러 무시)
        try:
            self.conn.execute_query("DROP INDEX violation_embeddings IF EXISTS")
            self.conn.execute_query("DROP INDEX corrected_embeddings IF EXISTS")
        except:
            pass

        print("✅ 기존 데이터 삭제 완료\n")

    def create_vector_indexes(self):
        """벡터 인덱스 생성 (이중 인덱스)"""
        print("\n" + "="*80)
        print("📊 벡터 인덱스 생성 중...")
        print("="*80)

        # 1. 위반 문장 인덱스
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

        # 2. 준수 문장 인덱스 (새로 추가!)
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

    def load_and_filter_csv(self, csv_path: str) -> pd.DataFrame:
        """CSV 로드 및 필터링"""
        print(f"\n📂 CSV 로드: {csv_path}")
        df = pd.read_csv(csv_path, encoding='utf-8-sig')

        # BOM 제거
        df.columns = df.columns.str.replace('\ufeff', '')

        # 유효한 행만 필터링
        df_valid = df[
            df['불공정 약관 원문'].notna() &
            (df['불공정 약관 원문'].str.strip() != '') &
            df['수정 후 약관 조항'].notna() &
            (df['수정 후 약관 조항'].str.strip() != '')
        ].reset_index(drop=True)

        print(f"   전체: {len(df)}개 → 유효: {len(df_valid)}개")

        return df_valid

    def generate_dual_embeddings(self, df: pd.DataFrame) -> Dict:
        """이중 임베딩 생성 (violation + corrected)"""
        print("\n" + "="*80)
        print("🧠 이중 임베딩 생성 중...")
        print("="*80)

        embeddings = {}

        # 1️⃣ 위반 문장 임베딩
        print("\n1️⃣  위반 문장 (original_text) 임베딩...")
        violation_texts = df['불공정 약관 원문'].tolist()
        embeddings['violation'] = self.model.encode(
            violation_texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True  # 코사인 유사도 최적화
        )
        print(f"   ✅ {len(violation_texts)}개 임베딩 완료")

        # 2️⃣ 준수 문장 임베딩 (새로 추가!)
        print("\n2️⃣  준수 문장 (corrected_text) 임베딩...")
        corrected_texts = df['수정 후 약관 조항'].tolist()
        embeddings['corrected'] = self.model.encode(
            corrected_texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True
        )
        print(f"   ✅ {len(corrected_texts)}개 임베딩 완료")

        print(f"\n✅ 이중 임베딩 생성 완료")
        print(f"   - violation: {embeddings['violation'].shape}")
        print(f"   - corrected: {embeddings['corrected'].shape}\n")

        return embeddings

    def create_nodes(self, df: pd.DataFrame, embeddings: Dict, id_offset: int = 0):
        """노드 생성 (이중 임베딩 포함)"""
        print("\n" + "="*80)
        print("📦 ViolationCase 노드 생성 중...")
        print("="*80)

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="노드 생성"):
            case_id = f"CASE_{id_offset + idx + 1}"

            # 이중 임베딩 추출
            embedding_violation = embeddings['violation'][idx].tolist()
            embedding_corrected = embeddings['corrected'][idx].tolist()

            query = """
            CREATE (v:ViolationCase {
                id: $id,
                original_text: $original_text,
                corrected_text: $corrected_text,
                violation_reason: $reason,
                company: $company,
                year: $year,
                category: $category,
                subcategory: $subcategory,
                article_id: $article_id,
                other_legal_basis: $other_legal_basis,
                embedding_violation: $embedding_violation,
                embedding_corrected: $embedding_corrected
            })
            """

            self.conn.execute_query(query, {
                'id': case_id,
                'original_text': row['불공정 약관 원문'],
                'corrected_text': row['수정 후 약관 조항'],
                'reason': row['시정 요청 사유'],
                'company': self._extract_company_name(row.get('파일명', '')),
                'year': self._extract_year(row.get('파일명', '')),
                'category': row.get('대주제', ''),
                'subcategory': row.get('소주제', ''),
                'article_id': self._extract_article_id(row.get('근거 조항(약관법)', '')),
                'other_legal_basis': row.get('근거 조항(약관법 외)', ''),
                'embedding_violation': embedding_violation,
                'embedding_corrected': embedding_corrected
            })

        print(f"✅ {len(df)}개 노드 생성 완료\n")

    def _extract_company_name(self, filename: str) -> str:
        """파일명에서 회사명 추출"""
        if not filename or pd.isna(filename):
            return "Unknown"
        # 예: "KB국민카드_2023_약관.pdf" → "KB국민카드"
        return filename.split('_')[0] if '_' in filename else filename.split('.')[0]

    def _extract_year(self, filename: str) -> int:
        """파일명에서 연도 추출"""
        if not filename or pd.isna(filename):
            return 0
        # 예: "KB국민카드_2023_약관.pdf" → 2023
        import re
        match = re.search(r'20\d{2}', filename)
        return int(match.group()) if match else 0

    def _extract_article_id(self, article_text: str) -> str:
        """조항 텍스트에서 조항 ID 추출"""
        if not article_text or pd.isna(article_text):
            return "Unknown"

        import re
        # "제6조", "제7조 제1호" 등 추출
        match = re.search(r'제\s*(\d+)\s*조', str(article_text))
        if match:
            return f"제{match.group(1)}조"
        return str(article_text).split()[0] if article_text else "Unknown"

    def rebuild_from_multiple_csv(self, csv_paths: List[str]):
        """여러 CSV에서 그래프 재구성"""
        print("\n" + "="*80)
        print("🔄 그래프 재구성 시작")
        print("="*80)
        print(f"📁 CSV 파일: {len(csv_paths)}개")
        for path in csv_paths:
            print(f"   - {path}")
        print()

        # 1. 기존 데이터 삭제
        self.clear_database()

        # 2. 벡터 인덱스 생성
        self.create_vector_indexes()

        # 3. CSV별로 처리
        id_offset = 0

        for csv_path in csv_paths:
            # 3-1. CSV 로드
            df = self.load_and_filter_csv(csv_path)

            if len(df) == 0:
                print(f"⚠️  {csv_path}: 유효한 데이터 없음, 스킵\n")
                continue

            # 3-2. 이중 임베딩 생성
            embeddings = self.generate_dual_embeddings(df)

            # 3-3. 노드 생성
            self.create_nodes(df, embeddings, id_offset=id_offset)

            id_offset += len(df)

            print(f"✅ {csv_path} 처리 완료 (누적: {id_offset}개 노드)\n")

        # 4. 완료
        print("\n" + "="*80)
        print("🎉 그래프 재구성 완료!")
        print("="*80)
        print(f"📊 총 {id_offset}개 노드 생성")
        print(f"📊 이중 임베딩 구조:")
        print(f"   - embedding_violation: {self.embedding_dim}차원 (위반 문장)")
        print(f"   - embedding_corrected: {self.embedding_dim}차원 (준수 문장)")
        print(f"\n🔍 벡터 인덱스:")
        print(f"   - violation_embeddings (위반 검색용)")
        print(f"   - corrected_embeddings (준수 검색용)")
        print()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 Fine-tuned 모델로 Neo4j 그래프 재구성")
    print("="*80)
    print()

    # Neo4j 연결
    conn = Neo4jConnector()

    # Rebuilder 생성 (fine-tuned 모델 사용)
    rebuilder = GraphRAGRebuilder(
        neo4j_connector=conn,
        model_path='./my_fine_tuned_model'  # train_model.py에서 생성한 모델
    )

    # 그래프 재구성
    rebuilder.rebuild_from_multiple_csv([
        "data/contracts/reference/보도자료_데이터_전처리_최종.csv",
        "data/contracts/reference/ai.csv"
    ])

    # 연결 종료
    conn.close()

    print("\n" + "="*80)
    print("✅ 모든 작업 완료!")
    print("="*80)
    print("\n다음 단계:")
    print("1. rag/hybrid_graphrag.py 수정:")
    print("   - 모델 경로를 './my_fine_tuned_model'로 변경")
    print("   - 임베딩 차원을 1024로 변경")
    print("   - 인덱스 이름을 'violation_embeddings', 'corrected_embeddings'로 변경")
    print("\n2. 테스트 실행:")
    print("   python3 scripts/test_input_csv.py")
    print()
