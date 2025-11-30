"""
Fine-tuned BAAI/bge-m3 모델을 사용하여 Neo4j 그래프 재구성

주요 변경사항:
1. 임베딩 모델: paraphrase-multilingual-MiniLM-L12-v2 (384차원)
   → BAAI/bge-m3 fine-tuned (1024차원)

2. 이중 임베딩 구조 (4개 필드):
   - embedding_violation_base: Base 모델로 계산한 위반 조항 임베딩
   - embedding_violation_finetuned: Finetuned 모델로 계산한 위반 조항 임베딩
   - embedding_corrected_base: Base 모델로 계산한 수정 조항 임베딩
   - embedding_corrected_finetuned: Finetuned 모델로 계산한 수정 조항 임베딩

3. 4개 벡터 인덱스:
   - violation_embeddings_base: 위반 문장 검색용 (Base 모델)
   - violation_embeddings_finetuned: 위반 문장 검색용 (Finetuned 모델)
   - corrected_embeddings_base: 준수 문장 검색용 (Base 모델)
   - corrected_embeddings_finetuned: 준수 문장 검색용 (Finetuned 모델)
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
import re  # 🆕 추가

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
        model_path_base=None,
        model_path_finetuned=None
    ):
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

        # 하위 호환성을 위해 self.model도 base 모델로 설정
        self.model = self.base_model

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
        """
        ⚠️ DEPRECATED: 전체 삭제는 법률 구조를 날립니다!
        대신 clear_violation_data()를 사용하세요.
        """
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
    
    def clear_violation_data(self):
        """
        🎯 ViolationCase 노드만 선택적 삭제 (법률 구조 보존!)
        
        삭제 대상:
        - ViolationCase 노드 및 관련 관계
        - 벡터 인덱스
        
        보존 대상:
        - 법률 노드 (법률, 조, 항, 호)
        - HAS_ARTICLE, HAS_HANG, HAS_HO 관계
        """
        print("\n" + "="*80)
        print("🗑️  ViolationCase 데이터만 선택적 삭제 (법률 구조 보존)")
        print("="*80)
        
        # ViolationCase 노드 삭제 전 개수 확인
        count_query = "MATCH (v:ViolationCase) RETURN count(v) as count"
        result = self.conn.execute_query(count_query)
        old_count = result[0]['count'] if result else 0
        
        print(f"  - 기존 ViolationCase: {old_count}개")
        
        # ViolationCase 노드 및 관련 관계 삭제
        delete_query = "MATCH (v:ViolationCase) DETACH DELETE v"
        self.conn.execute_query(delete_query)
        
        print(f"  ✅ {old_count}개 ViolationCase 삭제 완료")
        
        # 벡터 인덱스 삭제 (에러 무시)
        try:
            self.conn.execute_query("DROP INDEX violation_embeddings IF EXISTS")
            self.conn.execute_query("DROP INDEX corrected_embeddings IF EXISTS")
            print("  ✅ 벡터 인덱스 삭제 완료")
        except Exception as e:
            print(f"  ⚠️  인덱스 삭제 실패 (무시): {e}")
        
        # 법률 구조 확인 (보존되었는지 검증)
        law_check = """
        MATCH (law:법률) 
        OPTIONAL MATCH (law)-[:HAS_ARTICLE]->(article:조)
        OPTIONAL MATCH (article)-[:HAS_HANG]->(hang:항)
        OPTIONAL MATCH (article)-[:HAS_HO|HAS_HANG*]->(ho:호)
        RETURN 
            count(DISTINCT law) as law_count,
            count(DISTINCT article) as article_count,
            count(DISTINCT hang) as hang_count,
            count(DISTINCT ho) as ho_count
        """
        
        law_result = self.conn.execute_query(law_check)
        if law_result:
            print("\n  📊 보존된 법률 구조:")
            print(f"     - 법률: {law_result[0]['law_count']}개")
            print(f"     - 조: {law_result[0]['article_count']}개")
            print(f"     - 항: {law_result[0]['hang_count']}개")
            print(f"     - 호: {law_result[0]['ho_count']}개")
        
        print("\n✅ ViolationCase 데이터 삭제 완료 (법률 구조 보존됨)\n")

    def create_vector_indexes(self):
        """벡터 인덱스 생성 (4개 인덱스: base + finetuned)"""
        print("\n" + "="*80)
        print("📊 벡터 인덱스 생성 중 (4개 인덱스)")
        print("="*80)

        # 1. 위반 문장 - Base 모델 인덱스
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

        # 2. 위반 문장 - Finetuned 모델 인덱스
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

        # 3. 준수 문장 - Base 모델 인덱스
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

        # 4. 준수 문장 - Finetuned 모델 인덱스
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
        """4개 임베딩 생성 (violation_base, violation_finetuned, corrected_base, corrected_finetuned)"""
        print("\n" + "="*80)
        print("🧠 이중 임베딩 생성 중 (Base + Finetuned 모델)")
        print("="*80)

        embeddings = {}
        violation_texts = df['불공정 약관 원문'].tolist()
        corrected_texts = df['수정 후 약관 조항'].tolist()

        # 1️⃣ 위반 문장 - Base 모델
        print("\n1️⃣  위반 문장 (original_text) - Base 모델 임베딩...")
        embeddings['violation_base'] = self.base_model.encode(
            violation_texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True
        )
        print(f"   ✅ {len(violation_texts)}개 임베딩 완료")

        # 2️⃣ 위반 문장 - Finetuned 모델
        print("\n2️⃣  위반 문장 (original_text) - Finetuned 모델 임베딩...")
        embeddings['violation_finetuned'] = self.finetuned_model.encode(
            violation_texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True
        )
        print(f"   ✅ {len(violation_texts)}개 임베딩 완료")

        # 3️⃣ 준수 문장 - Base 모델
        print("\n3️⃣  준수 문장 (corrected_text) - Base 모델 임베딩...")
        embeddings['corrected_base'] = self.base_model.encode(
            corrected_texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True
        )
        print(f"   ✅ {len(corrected_texts)}개 임베딩 완료")

        # 4️⃣ 준수 문장 - Finetuned 모델
        print("\n4️⃣  준수 문장 (corrected_text) - Finetuned 모델 임베딩...")
        embeddings['corrected_finetuned'] = self.finetuned_model.encode(
            corrected_texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True
        )
        print(f"   ✅ {len(corrected_texts)}개 임베딩 완료")

        print(f"\n✅ 4개 임베딩 생성 완료")
        print(f"   - violation_base: {embeddings['violation_base'].shape}")
        print(f"   - violation_finetuned: {embeddings['violation_finetuned'].shape}")
        print(f"   - corrected_base: {embeddings['corrected_base'].shape}")
        print(f"   - corrected_finetuned: {embeddings['corrected_finetuned'].shape}\n")

        return embeddings

    def create_nodes(self, df: pd.DataFrame, embeddings: Dict, id_offset: int = 0):
        """노드 생성 (4개 임베딩 필드 포함)"""
        print("\n" + "="*80)
        print("📦 ViolationCase 노드 생성 중...")
        print("="*80)

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="노드 생성"):
            case_id = f"CASE_{id_offset + idx + 1}"

            # 4개 임베딩 추출
            embedding_violation_base = embeddings['violation_base'][idx].tolist()
            embedding_violation_finetuned = embeddings['violation_finetuned'][idx].tolist()
            embedding_corrected_base = embeddings['corrected_base'][idx].tolist()
            embedding_corrected_finetuned = embeddings['corrected_finetuned'][idx].tolist()

            # 4개 임베딩 필드만 생성 (레거시 필드 제거)
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
                embedding_violation_base: $embedding_violation_base,
                embedding_violation_finetuned: $embedding_violation_finetuned,
                embedding_corrected_base: $embedding_corrected_base,
                embedding_corrected_finetuned: $embedding_corrected_finetuned
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
                'embedding_violation_base': embedding_violation_base,
                'embedding_violation_finetuned': embedding_violation_finetuned,
                'embedding_corrected_base': embedding_corrected_base,
                'embedding_corrected_finetuned': embedding_corrected_finetuned
            })

        print(f"✅ {len(df)}개 노드 생성 완료 (4개 임베딩 필드 포함)\n")


    
    # =========================================================================
    # 🆕 관계 생성 메서드들
    # =========================================================================
    
    def create_law_relationships(self, df: pd.DataFrame):
        """
        ViolationCase → 법률 노드 (조/항/호) 관계 생성
        
        우선순위:
        1. 호 노드 (가장 구체적)
        2. 항 노드
        3. 조 노드
        """
        print("  - ViolationCase → 법률 노드 연결 중...")
        
        created_count = 0
        skipped_count = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="VIOLATES 관계"):
            case_id = f"CASE_{row.get('_new_id', row.get('ID', idx + 1))}"
            legal_basis = row.get('근거 조항(약관법)', '')
            
            if not legal_basis or pd.isna(legal_basis):
                skipped_count += 1
                continue
            
            # 조항 파싱
            parsed = self._parse_legal_article(legal_basis)
            
            if not parsed.get('article'):
                skipped_count += 1
                continue
            
            # 법률 노드 찾기
            target_node = self._find_law_target_node(
                parsed['article'],
                parsed.get('hang'),
                parsed.get('ho')
            )
            
            if target_node:
                self._create_violation_relationship(case_id, target_node)
                created_count += 1
            else:
                skipped_count += 1
        
        print(f"  ✅ 생성: {created_count}개, 스킵: {skipped_count}개")
    
    def _parse_legal_article(self, legal_basis: str) -> Dict:
        """법조항 문자열 파싱"""
        result = {
            'article': None,
            'hang': None,
            'ho': None
        }
        
        if not legal_basis or pd.isna(legal_basis):
            return result
        
        legal_basis = str(legal_basis)
        
        # 조 추출
        article_match = re.search(r'제(\d+)조', legal_basis)
        if article_match:
            result['article'] = f"제{article_match.group(1)}조"
        
        # 항 추출
        hang_match = re.search(r'제(\d+)항', legal_basis)
        if hang_match:
            result['hang'] = f"제{hang_match.group(1)}항"
        
        # 호 추출
        ho_match = re.search(r'제(\d+)호', legal_basis)
        if ho_match:
            result['ho'] = f"제{ho_match.group(1)}호"
        
        return result
    
    def _find_law_target_node(self, article_id: str, hang_id: str = None, ho_id: str = None) -> Dict:
        """법률 노드 찾기 (우선순위: 호 > 항 > 조)"""
        # 1. 호 노드 찾기
        if ho_id:
            if hang_id:
                ho_full_id = f"{article_id}_{hang_id}_{ho_id}"
            else:
                ho_full_id = f"{article_id}_{ho_id}"
            
            query = "MATCH (ho:호 {id: $ho_id}) RETURN ho.id as id LIMIT 1"
            result = self.conn.execute_query(query, {'ho_id': ho_full_id})
            
            if result:
                return {'type': '호', 'id': result[0]['id']}
        
        # 2. 항 노드 찾기
        if hang_id:
            hang_full_id = f"{article_id}_{hang_id}"
            query = "MATCH (hang:항 {id: $hang_id}) RETURN hang.id as id LIMIT 1"
            result = self.conn.execute_query(query, {'hang_id': hang_full_id})
            
            if result:
                return {'type': '항', 'id': result[0]['id']}
        
        # 3. 조 노드 찾기
        query = "MATCH (article:조 {id: $article_id}) RETURN article.id as id LIMIT 1"
        result = self.conn.execute_query(query, {'article_id': article_id})
        
        if result:
            return {'type': '조', 'id': result[0]['id']}
        
        return None
    
    def _create_violation_relationship(self, case_id: str, target_node: Dict):
        """ViolationCase → 법률 노드 관계 생성"""
        node_type = target_node['type']
        node_id = target_node['id']
        
        query = f"""
        MATCH (v:ViolationCase {{id: $case_id}})
        MATCH (law:{node_type} {{id: $node_id}})
        MERGE (v)-[:VIOLATES {{
            confidence: 1.0,
            method: 'law_centric'
        }}]->(law)
        """
        
        self.conn.execute_query(query, {
            'case_id': case_id,
            'node_id': node_id
        })
    
    def create_similarity_relationships(self, df: pd.DataFrame, embeddings: Dict):
        """
        유사도 기반 SIMILAR_TO 관계 생성
        
        Args:
            df: DataFrame
            embeddings: 임베딩 딕셔너리 (violation_base 키 사용)
        """
        print("  - 유사도 행렬 계산 중...")
        
        # violation_base 임베딩 사용 (RAG 검색용)
        violation_embeddings = embeddings['violation_base']
        similarity_matrix = cosine_similarity(violation_embeddings)
        
        print("  - SIMILAR_TO 관계 생성 중...")
        
        threshold = 0.7
        top_k = 5
        
        created_count = 0
        
        for i in tqdm(range(len(df)), desc="SIMILAR_TO 관계"):
            case_id_1 = f"CASE_{df.iloc[i].get('_new_id', df.iloc[i].get('ID', i + 1))}"
            
            # 자기 자신 제외
            similarities = similarity_matrix[i].copy()
            similarities[i] = -1
            
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            for j in top_indices:
                if similarities[j] < threshold:
                    continue
                
                case_id_2 = f"CASE_{df.iloc[j].get('_new_id', df.iloc[j].get('ID', j + 1))}"
                
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
        
        print(f"  ✅ {created_count}개 SIMILAR_TO 관계 생성")

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
        """
        여러 CSV에서 그래프 재구성 (개선됨!)
        
        ✅ 개선 사항:
        1. 임베딩 중복 계산 방지 (1회만 계산)
        2. 법률 구조 보존 (ViolationCase만 삭제)
        3. 메모리 효율적 처리
        """
        print("\n" + "="*80)
        print("🔄 그래프 재구성 시작 (개선 버전)")
        print("="*80)
        print(f"📁 CSV 파일: {len(csv_paths)}개")
        for path in csv_paths:
            print(f"   - {path}")
        print()

        # =====================================================================
        # Step 0: 기존 ViolationCase 데이터만 삭제 (법률 구조 보존!)
        # =====================================================================
        self.clear_violation_data()

        # =====================================================================
        # Step 1: 모든 CSV 로드 및 통합
        # =====================================================================
        print("="*80)
        print("📂 Step 1: CSV 데이터 통합")
        print("="*80)
        
        all_dfs = []
        for csv_path in csv_paths:
            print(f"\n  - {csv_path} 로드 중...")
            df = self.load_and_filter_csv(csv_path)
            if len(df) > 0:
                all_dfs.append(df)
        
        if not all_dfs:
            print("❌ 유효한 데이터가 없습니다.")
            return
        
        # 전체 데이터 통합
        df_combined = pd.concat(all_dfs, ignore_index=True)
        df_combined['_new_id'] = range(1, len(df_combined) + 1)
        
        print(f"\n✅ 총 {len(df_combined)}개 유효한 사례 로드 완료\n")

        # =====================================================================
        # Step 2: 벡터 인덱스 생성
        # =====================================================================
        self.create_vector_indexes()

        # =====================================================================
        # Step 3: 임베딩 생성 (⚡ 1회만!)
        # =====================================================================
        print("="*80)
        print("🧠 Step 3: 이중 임베딩 생성 (전체 데이터, 1회만)")
        print("="*80)
        
        embeddings = self.generate_dual_embeddings(df_combined)
        
        print(f"✅ 임베딩 생성 완료 (재사용 가능)\n")

        # =====================================================================
        # Step 4: ViolationCase 노드 생성 (임베딩 재사용)
        # =====================================================================
        print("="*80)
        print("📦 Step 4: ViolationCase 노드 생성")
        print("="*80)
        
        self.create_nodes(df_combined, embeddings, id_offset=0)
        
        print(f"✅ {len(df_combined)}개 노드 생성 완료\n")

        # =====================================================================
        # Step 5: VIOLATES 관계 생성 (ViolationCase → 법률 노드)
        # =====================================================================
        print("="*80)
        print("⚖️ Step 5: ViolationCase → 법률 노드 관계 생성 (VIOLATES)")
        print("="*80)
        
        self.create_law_relationships(df_combined)

        # =====================================================================
        # Step 6: SIMILAR_TO 관계 생성 (임베딩 재사용!)
        # =====================================================================
        print("\n" + "="*80)
        print("🔗 Step 6: 유사도 기반 관계 생성 (SIMILAR_TO)")
        print("="*80)
        print("  ⚡ 기존 임베딩 재사용 (추가 계산 없음!)")
        
        # ⚡ 임베딩 재사용! 다시 계산하지 않음
        self.create_similarity_relationships(df_combined, embeddings)

        # =====================================================================
        # Step 7: 완료 및 통계
        # =====================================================================
        print("\n" + "="*80)
        print("🎉 그래프 재구성 완료!")
        print("="*80)
        print(f"📊 총 {len(df_combined)}개 ViolationCase 노드 생성")
        print(f"📊 이중 임베딩 구조 (4개 필드):")
        print(f"   - embedding_violation_base: {self.embedding_dim}차원 (위반 문장, Base 모델)")
        print(f"   - embedding_violation_finetuned: {self.embedding_dim}차원 (위반 문장, Finetuned 모델)")
        print(f"   - embedding_corrected_base: {self.embedding_dim}차원 (준수 문장, Base 모델)")
        print(f"   - embedding_corrected_finetuned: {self.embedding_dim}차원 (준수 문장, Finetuned 모델)")
        print(f"\n🔍 벡터 인덱스 (4개):")
        print(f"   - violation_embeddings_base (위반 검색용, Base)")
        print(f"   - violation_embeddings_finetuned (위반 검색용, Finetuned)")
        print(f"   - corrected_embeddings_base (준수 검색용, Base)")
        print(f"   - corrected_embeddings_finetuned (준수 검색용, Finetuned)")
        
        # 관계 통계
        print(f"\n📊 그래프 관계:")
        try:
            violates_count_query = "MATCH ()-[r:VIOLATES]->() RETURN count(r) as count"
            violates_result = self.conn.execute_query(violates_count_query)
            violates_count = violates_result[0]['count'] if violates_result else 0
            print(f"   - VIOLATES: {violates_count}개")
            
            similar_count_query = "MATCH ()-[r:SIMILAR_TO]->() RETURN count(r) as count"
            similar_result = self.conn.execute_query(similar_count_query)
            similar_count = similar_result[0]['count'] if similar_result else 0
            print(f"   - SIMILAR_TO: {similar_count}개")
            
            print(f"   - 총 관계: {violates_count + similar_count}개")
        except Exception as e:
            print(f"   ⚠️  관계 통계 조회 실패: {e}")
        
        print()
        print("="*80)
        print("✅ 법률 구조(조/항/호)는 보존되었습니다.")
        print("✅ 임베딩은 1회만 계산되었습니다. (비용/시간 절약!)")
        print("="*80)
        print()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 Fine-tuned 모델로 Neo4j 그래프 재구성")
    print("="*80)
    print()

    # Neo4j 연결
    conn = Neo4jConnector()

    # Rebuilder 생성 (config.settings에서 두 모델 모두 읽음)
    rebuilder = GraphRAGRebuilder(
        neo4j_connector=conn
        # model_path_base, model_path_finetuned는 config.settings에서 자동으로 읽음
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