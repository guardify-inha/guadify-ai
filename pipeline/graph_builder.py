"""
CSV 데이터를 진짜 GraphRAG로 변환하는 파이프라인

✅ NaN 처리: 필수 컬럼에 NaN이 있는 행은 아예 제외
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from tqdm import tqdm
import json

class GraphRAGBuilder:
    """CSV → GraphRAG 변환 빌더"""
    
    def __init__(self, neo4j_connector, embedding_model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        self.conn = neo4j_connector
        self.model = SentenceTransformer(embedding_model_name)
        
    def build_from_csv(self, csv_path: str):
        """CSV에서 전체 그래프 구축"""
        print("📊 Step 1: CSV 데이터 로드 중...")
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        
        # ✅ NaN 필터링
        df = self._filter_valid_rows(df)
        print(f"✅ {len(df)}개 유효한 사례 로드 완료\n")
        
        # Step 2: 임베딩 생성
        print("🧠 Step 2: 텍스트 임베딩 생성 중...")
        embeddings = self._generate_embeddings(df)
        
        # Step 3: 노드 생성
        print("\n📦 Step 3: 노드 생성 중...")
        self._create_nodes(df, embeddings)
        
        # Step 4: 유사도 기반 관계 생성
        print("\n🔗 Step 4: 유사도 관계 생성 중...")
        self._create_similarity_relationships(df, embeddings)
        
        # Step 5: 법률 조항 관계 생성
        print("\n⚖️ Step 5: 법률 조항 관계 생성 중...")
        self._create_law_relationships(df)
        
        # Step 6: 키워드 추출 및 관계 생성
        print("\n🔑 Step 6: 키워드 추출 중...")
        self._extract_and_link_keywords(df)
        
        # Step 7: 위반 유형 분류
        print("\n📋 Step 7: 위반 유형 분류 중...")
        self._categorize_violations(df)
        
        # Step 8: 통계 출력
        print("\n📈 그래프 구축 완료!")
        self._print_statistics()
    
    def build_from_multiple_csv(self, csv_paths: List[str]):
        """여러 CSV에서 그래프 구축"""
        print("📊 Step 1: 여러 CSV 데이터 로드 중...")
        
        dfs = []
        for path in csv_paths:
            print(f" - {path} 로드 중...")
            df = pd.read_csv(path, encoding='utf-8-sig')
            dfs.append(df)
        
        df_total = pd.concat(dfs, ignore_index=True)
        
        # ✅ NaN 필터링
        df_total = self._filter_valid_rows(df_total)
        print(f"✅ 총 {len(df_total)}개 유효한 사례 로드 완료 (통합)\n")

        # 이후 동일
        print("🧠 Step 2: 텍스트 임베딩 생성 중...")
        embeddings = self._generate_embeddings(df_total)

        print("\n📦 Step 3: 노드 생성 중...")
        self._create_nodes(df_total, embeddings)

        print("\n🔗 Step 4: 유사도 관계 생성 중...")
        self._create_similarity_relationships(df_total, embeddings)

        print("\n⚖️ Step 5: 법률 조항 관계 생성 중...")
        self._create_law_relationships(df_total)

        print("\n🔑 Step 6: 키워드 추출 중...")
        self._extract_and_link_keywords(df_total)

        print("\n📋 Step 7: 위반 유형 분류 중...")
        self._categorize_violations(df_total)

        print("\n📈 그래프 구축 완료!")
        self._print_statistics()
    
    def _filter_valid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        필수 컬럼에 NaN이 있는 행 제거
        
        필수 컬럼:
        - 불공정 약관 원문
        - 시정 요청 사유
        - 파일명
        - 대주제
        """
        print("  🔍 NaN 행 필터링 중...")
        
        original_len = len(df)
        
        # 필수 컬럼
        required_columns = [
            '불공정 약관 원문',
            '시정 요청 사유', 
            '파일명',
            '대주제'
        ]
        
        # 각 필수 컬럼에서 NaN인 행 체크
        valid_mask = pd.Series([True] * len(df))
        
        for col in required_columns:
            if col in df.columns:
                col_valid = df[col].notna() & (df[col] != '') & (df[col].astype(str).str.strip() != '')
                valid_mask &= col_valid
                nan_count = (~col_valid).sum()
                if nan_count > 0:
                    print(f"    • '{col}': {nan_count}개 행 제외")
        
        # 유효한 행만 필터링
        df_filtered = df[valid_mask].copy()
        
        # 인덱스 리셋 (중요!)
        df_filtered.reset_index(drop=True, inplace=True)
        
        removed = original_len - len(df_filtered)
        print(f"    ✅ {removed}개 행 제외, {len(df_filtered)}개 행 유효\n")
        
        return df_filtered
    
    def _generate_embeddings(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """텍스트 임베딩 생성 (이미 NaN 제거된 DataFrame)"""
        embeddings = {}

        # 1️⃣ 위반 사례 원문 임베딩
        print("  - 위반 사례 임베딩...")
        texts = df['불공정 약관 원문'].tolist()
        embeddings['violation_original'] = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )

        # 2️⃣ 수정 후 약관 임베딩
        print("\n  - 수정 약관 임베딩...")
        corrected_texts = df['수정 후 약관 조항'].fillna('').tolist()
        embeddings['violation_corrected'] = self.model.encode(
            corrected_texts,
            show_progress_bar=True,
            batch_size=32
        )

        # 3️⃣ 시정 사유 임베딩
        print("\n  - 시정 사유 임베딩...")
        reason_texts = df['시정 요청 사유'].tolist()
        embeddings['violation_reason'] = self.model.encode(
            reason_texts,
            show_progress_bar=True,
            batch_size=32
        )

        return embeddings
    
    def _create_nodes(self, df: pd.DataFrame, embeddings: Dict):
        """노드 생성"""
        # 위반 사례 노드 생성
        print("  - ViolationCase 노드 생성...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            case_id = f"CASE_{row['ID']}"
            
            # 벡터를 리스트로 변환
            embedding_list = embeddings['violation_original'][idx].tolist()
            
            query = """
            CREATE (v:ViolationCase {
                id: $id,
                original_text: $original_text,
                corrected_text: $corrected_text,
                violation_reason: $reason,
                company: $company,
                year: $year,
                severity: $severity,
                category: $category,
                subcategory: $subcategory,
                article_id: $article_id,
                other_legal_basis: $other_legal_basis,
                embedding: $embedding
            })
            """
            
            self.conn.execute_query(query, {
                'id': case_id,
                'original_text': row['불공정 약관 원문'],
                'corrected_text': row.get('수정 후 약관 조항', ''),
                'reason': row['시정 요청 사유'],
                'company': self._extract_company_name(row['파일명']),
                'year': self._extract_year(row['파일명']),
                'severity': self._determine_severity(row['시정 요청 사유']),
                'category': row['대주제'],
                'subcategory': row.get('소주제', ''),
                'article_id': self._extract_article_id(row.get('근거 조항(약관법)', '')),
                'other_legal_basis': row.get('근거 조항(약관법 외)', ''),
                'embedding': embedding_list
            })
        
        # 법률 조항 노드 생성
        print("  - LawArticle 노드 생성...")
        self._create_law_article_nodes(df)
        
        # 위반 유형 노드 생성
        print("  - ViolationType 노드 생성...")
        self._create_violation_type_nodes(df)
        
        # 회사 노드 생성
        print("  - Company 노드 생성...")
        self._create_company_nodes(df)
    
    def _create_similarity_relationships(self, df: pd.DataFrame, embeddings: Dict):
        """유사도 기반 관계 생성"""
        print("  - 유사도 행렬 계산 중...")
        original_embeddings = embeddings['violation_original']
        
        # 코사인 유사도 계산
        similarity_matrix = cosine_similarity(original_embeddings)
        
        print("  - SIMILAR_TO 관계 생성 중...")
        threshold = 0.7
        top_k = 5
        
        relationships_created = 0
        for i in tqdm(range(len(similarity_matrix))):
            case_id_i = f"CASE_{df.iloc[i]['ID']}"
            
            # 유사도 정렬 (자기 자신 제외)
            similarities = similarity_matrix[i]
            similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
            
            for j in similar_indices:
                sim_score = similarities[j]
                if sim_score < threshold:
                    continue
                
                case_id_j = f"CASE_{df.iloc[j]['ID']}"
                
                query = """
                MATCH (v1:ViolationCase {id: $id1})
                MATCH (v2:ViolationCase {id: $id2})
                CREATE (v1)-[:SIMILAR_TO {
                    similarity_score: $score,
                    similarity_type: 'semantic'
                }]->(v2)
                """
                
                self.conn.execute_query(query, {
                    'id1': case_id_i,
                    'id2': case_id_j,
                    'score': float(sim_score)
                })
                relationships_created += 1
        
        print(f"  ✅ {relationships_created}개 유사도 관계 생성 완료")
    
    def _create_law_relationships(self, df: pd.DataFrame):
        """법률 조항 관계 생성 (약관법만)"""
        print("  - VIOLATES 관계 생성 중...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            case_id = f"CASE_{row['ID']}"
            
            # 근거 조항(약관법)만 사용
            legal_basis_col = row.get('근거 조항(약관법)', '')
            article_id = self._extract_article_id(legal_basis_col)
            
            if not article_id:
                continue
            
            query = """
            MATCH (v:ViolationCase {id: $case_id})
            MATCH (l:LawArticle {id: $article_id})
            CREATE (v)-[:VIOLATES {
                confidence: $confidence,
                reason: $reason
            }]->(l)
            """
            
            confidence = self._calculate_violation_confidence(row)
            
            self.conn.execute_query(query, {
                'case_id': case_id,
                'article_id': article_id,
                'confidence': confidence,
                'reason': row['시정 요청 사유'][:500]
            })
    
    def _extract_and_link_keywords(self, df: pd.DataFrame):
        """키워드 추출 및 연결"""
        keyword_patterns = {
            '면책': r'면책|책임.*지지.*않|책임.*없',
            '일방적_변경': r'일방적.*변경|임의.*변경',
            '손해배상': r'손해배상|배상금|위약금',
            '계약해지': r'해지|해제|취소',
            '추가담보': r'추가.*담보|담보.*제공',
            '서면제한': r'서면.*제한|서면으로.*만',
        }
        
        # 키워드 노드 생성
        for keyword, pattern in keyword_patterns.items():
            query = """
            CREATE (k:Keyword {
                text: $text,
                type: 'pattern',
                regex_pattern: $pattern,
                weight: 1.0
            })
            """
            self.conn.execute_query(query, {
                'text': keyword,
                'pattern': pattern
            })
        
        # 키워드-사례 연결
        print("  - CONTAINS 관계 생성 중...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            case_id = f"CASE_{row['ID']}"
            text = row['불공정 약관 원문']
            
            for keyword, pattern in keyword_patterns.items():
                matches = list(re.finditer(pattern, text))
                if matches:
                    query = """
                    MATCH (v:ViolationCase {id: $case_id})
                    MATCH (k:Keyword {text: $keyword})
                    CREATE (v)-[:CONTAINS {
                        count: $count,
                        positions: $positions
                    }]->(k)
                    """
                    
                    positions = [m.start() for m in matches]
                    self.conn.execute_query(query, {
                        'case_id': case_id,
                        'keyword': keyword,
                        'count': len(matches),
                        'positions': positions
                    })
    
    def _categorize_violations(self, df: pd.DataFrame):
        """위반 유형 자동 분류"""
        print("  - CATEGORIZED_AS 관계 생성 중...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            case_id = f"CASE_{row['ID']}"
            violation_type = row['대주제']
            
            query = """
            MATCH (v:ViolationCase {id: $case_id})
            MATCH (t:ViolationType {name: $type_name})
            CREATE (v)-[:CATEGORIZED_AS {
                confidence: 1.0
            }]->(t)
            """
            
            self.conn.execute_query(query, {
                'case_id': case_id,
                'type_name': violation_type
            })
    
    # =============================================================================
    # 헬퍼 메서드
    # =============================================================================
    
    def _extract_company_name(self, filename: str) -> str:
        """파일명에서 회사명 추출 (안전 처리)"""
        if pd.isna(filename) or not isinstance(filename, str):
            return "Unknown"
        
        parts = filename.split('_')
        if len(parts) > 1:
            return parts[1]
        return "Unknown"
    
    def _extract_year(self, filename: str) -> int:
        """파일명에서 연도 추출 (안전 처리)"""
        if pd.isna(filename) or not isinstance(filename, str):
            return 2020
        
        match = re.search(r'(\d{2})(\d{2})(\d{2})', filename)
        if match:
            year_short = match.group(1)
            year = int('20' + year_short) if int(year_short) < 50 else int('19' + year_short)
            return year
        return 2020
    
    def _determine_severity(self, reason: str) -> str:
        """시정 사유로부터 심각도 판단"""
        if pd.isna(reason):
            return 'low'
        
        high_keywords = ['부당하게 불리', '중대한', '일체', '어떠한 경우에도']
        medium_keywords = ['불공정', '불리', '제한']
        
        reason_lower = str(reason).lower()
        
        if any(kw in reason_lower for kw in high_keywords):
            return 'high'
        elif any(kw in reason_lower for kw in medium_keywords):
            return 'medium'
        else:
            return 'low'
    
    def _extract_article_id(self, legal_basis: str) -> str:
        """근거 조항에서 조 번호 추출"""
        if not legal_basis or pd.isna(legal_basis):
            return None
        
        match = re.search(r'제(\d+)조', str(legal_basis))
        if match:
            return f"제{match.group(1)}조"
        return None
    
    def _calculate_violation_confidence(self, row: pd.Series) -> float:
        """위반 확신도 계산"""
        reason = row['시정 요청 사유']
        if pd.isna(reason):
            return 0.5
        
        strong_words = ['명백', '분명', '확실', '부당하게']
        confidence = 0.7
        
        for word in strong_words:
            if word in str(reason):
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _create_law_article_nodes(self, df: pd.DataFrame):
        """법률 조항 노드 생성 (약관법만)"""
        articles = set()
        
        if '근거 조항(약관법)' in df.columns:
            for legal_basis in df['근거 조항(약관법)'].dropna():
                article_id = self._extract_article_id(legal_basis)
                if article_id:
                    articles.add(article_id)
        
        for article_id in articles:
            query = """
            MERGE (l:LawArticle {id: $id})
            ON CREATE SET 
                l.title = $id,
                l.category = $category
            """
            
            category = self._determine_article_category(article_id)
            self.conn.execute_query(query, {
                'id': article_id,
                'category': category
            })
    
    def _create_violation_type_nodes(self, df: pd.DataFrame):
        """위반 유형 노드 생성"""
        violation_types = df['대주제'].unique()
        
        for vtype in violation_types:
            if pd.isna(vtype):
                continue
            
            query = """
            MERGE (t:ViolationType {name: $name})
            ON CREATE SET
                t.id = $id,
                t.description = $description
            """
            
            self.conn.execute_query(query, {
                'name': vtype,
                'id': f"TYPE_{vtype.replace(' ', '_')}",
                'description': vtype
            })
    
    def _create_company_nodes(self, df: pd.DataFrame):
        """회사 노드 생성"""
        companies = df['파일명'].apply(self._extract_company_name).unique()
        
        for company in companies:
            if company == "Unknown" or pd.isna(company):
                continue
            
            count = len(df[df['파일명'].apply(self._extract_company_name) == company])
            
            query = """
            MERGE (c:Company {name: $name})
            ON CREATE SET
                c.violation_count = $count,
                c.industry = $industry
            """
            
            self.conn.execute_query(query, {
                'name': company,
                'count': count,
                'industry': self._determine_industry(company)
            })
            
            # 회사-사례 연결
            company_cases = df[df['파일명'].apply(self._extract_company_name) == company]
            for idx, row in company_cases.iterrows():
                case_id = f"CASE_{row['ID']}"
                
                rel_query = """
                MATCH (v:ViolationCase {id: $case_id})
                MATCH (c:Company {name: $company})
                CREATE (v)-[:COMMITTED_BY {
                    date: date($year + '-01-01')
                }]->(c)
                """
                
                self.conn.execute_query(rel_query, {
                    'case_id': case_id,
                    'company': company,
                    'year': str(self._extract_year(row['파일명']))
                })
    
    def _determine_article_category(self, article_id: str) -> str:
        """조항 ID로부터 카테고리 판단"""
        article_categories = {
            '제6조': '부당조항',
            '제7조': '면책조항',
            '제8조': '손해배상',
            '제9조': '계약해지',
            '제10조': '급부변경',
            '제11조': '항변권제한',
            '제12조': '의사표시',
            '제13조': '대리인책임',
            '제14조': '재판관할'
        }
        return article_categories.get(article_id, '기타')
    
    def _determine_industry(self, company: str) -> str:
        """회사명으로부터 업종 판단"""
        if pd.isna(company):
            return '기타금융'
        
        company_str = str(company)
        if '은행' in company_str:
            return '은행'
        elif '보험' in company_str:
            return '보험'
        elif '카드' in company_str or '여신' in company_str:
            return '여신금융'
        elif '저축' in company_str:
            return '저축은행'
        else:
            return '기타금융'
    
    def _print_statistics(self):
        """그래프 통계 출력"""
        queries = {
            "ViolationCase": "MATCH (n:ViolationCase) RETURN count(n) as count",
            "LawArticle": "MATCH (n:LawArticle) RETURN count(n) as count",
            "ViolationType": "MATCH (n:ViolationType) RETURN count(n) as count",
            "Company": "MATCH (n:Company) RETURN count(n) as count",
            "Keyword": "MATCH (n:Keyword) RETURN count(n) as count",
            "SIMILAR_TO": "MATCH ()-[r:SIMILAR_TO]->() RETURN count(r) as count",
            "VIOLATES": "MATCH ()-[r:VIOLATES]->() RETURN count(r) as count",
            "CONTAINS": "MATCH ()-[r:CONTAINS]->() RETURN count(r) as count",
        }
        
        print("\n" + "="*50)
        print("📊 그래프 통계")
        print("="*50)
        
        for name, query in queries.items():
            result = self.conn.execute_query(query)
            count = result[0]['count'] if result else 0
            print(f"  {name:20s}: {count:6d}")
        
        print("="*50)


# =============================================================================
# 실행 예시
# =============================================================================

if __name__ == "__main__":
    from database.neo4j_connector import Neo4jConnector
    
    conn = Neo4jConnector()
    builder = GraphRAGBuilder(conn)
    
    # 단일 CSV
    # builder.build_from_csv("data/contracts/reference/corrected_terms.csv")
    
    # 여러 CSV
    builder.build_from_multiple_csv([
        "data/contracts/reference/보도자료_데이터_전처리_최종.csv",
        "data/contracts/reference/ai.csv"
    ])
    
    conn.close()