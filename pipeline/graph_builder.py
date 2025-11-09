"""
CSV 데이터를 Law-Centric GraphRAG로 변환하는 파이프라인

핵심 변경사항:
1. Law-Centric 구조: 법률 조항(조-항-호)을 중심으로 위반사례 연결
2. ViolationCase → 호 노드로 직접 연결 (VIOLATES 관계)
3. LawArticle 노드 제거, 기존 law_structure의 호 노드 활용
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
    """CSV → Law-Centric GraphRAG 변환 빌더"""
    
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
        
        # Step 5: 법률 조항 관계 생성 (Law-Centric!)
        print("\n⚖️ Step 5: 법률 조항 관계 생성 (Law-Centric)...")
        self._create_law_relationships_centric(df)
        
        # Step 6: 키워드 추출 및 관계 생성 (빈도 기반 가중치)
        print("\n🔑 Step 6: 키워드 추출 (빈도 가중치)...")
        self._extract_and_link_keywords_with_frequency(df)
        
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

        print("\n⚖️ Step 5: 법률 조항 관계 생성 (Law-Centric)...")
        self._create_law_relationships_centric(df_total)

        print("\n🔑 Step 6: 키워드 추출 (빈도 가중치)...")
        self._extract_and_link_keywords_with_frequency(df_total)

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
        
        for i in tqdm(range(len(df))):
            case_id_1 = f"CASE_{df.iloc[i]['ID']}"
            
            # 자기 자신 제외하고 유사도 높은 top_k 찾기
            similarities = similarity_matrix[i].copy()
            similarities[i] = -1  # 자기 자신 제외
            
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            for j in top_indices:
                if similarities[j] < threshold:
                    continue
                
                case_id_2 = f"CASE_{df.iloc[j]['ID']}"
                
                query = """
                MATCH (v1:ViolationCase {id: $id1})
                MATCH (v2:ViolationCase {id: $id2})
                CREATE (v1)-[:SIMILAR_TO {
                    similarity: $similarity,
                    method: 'cosine'
                }]->(v2)
                """
                
                self.conn.execute_query(query, {
                    'id1': case_id_1,
                    'id2': case_id_2,
                    'similarity': float(similarities[j])
                })
    
    def _create_law_relationships_centric(self, df: pd.DataFrame):
        """
        [Law-Centric 구조]
        ViolationCase → 호(또는 항, 조) 노드로 직접 연결
        
        연결 우선순위:
        1. 호 노드 (가장 구체적)
        2. 항 노드 (호가 없는 경우)
        3. 조 노드 (항도 없는 경우)
        """
        print("  - ViolationCase → 법률 노드 연결 (Law-Centric)...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            case_id = f"CASE_{row['ID']}"
            legal_basis = row.get('근거 조항(약관법)', '')
            
            if not legal_basis or pd.isna(legal_basis):
                continue
            
            # 조항 파싱 (예: "제7조 제2호", "제6조 제2항 제1호")
            parsed = self._parse_legal_article(legal_basis)
            
            if not parsed['article']:
                continue
            
            # 연결 대상 노드 찾기 (우선순위: 호 > 항 > 조)
            target_node = self._find_law_target_node(
                parsed['article'],
                parsed.get('hang'),
                parsed.get('ho')
            )
            
            if target_node:
                self._create_violation_relationship(case_id, target_node)
    
    def _parse_legal_article(self, legal_basis: str) -> Dict:
        """
        법조항 문자열 파싱
        
        예시:
        - "제7조 제2호" → {article: "제7조", ho: "제2호"}
        - "제6조 제2항 제1호" → {article: "제6조", hang: "제2항", ho: "제1호"}
        - "제8조" → {article: "제8조"}
        """
        result = {}
        
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
        """
        법률 노드 찾기 (우선순위: 호 > 항 > 조)
        
        Returns:
            {'type': 'ho'|'항'|'조', 'id': node_id}
        """
        # 1순위: 호 노드 찾기
        if ho_id:
            if hang_id:
                # 제6조 제2항 제1호 형식
                ho_full_id = f"{article_id}_{hang_id}_{ho_id}"
            else:
                # 제7조 제2호 형식
                ho_full_id = f"{article_id}_{ho_id}"
            
            query = """
            MATCH (ho:호 {id: $ho_id})
            RETURN ho.id as id
            LIMIT 1
            """
            result = self.conn.execute_query(query, {'ho_id': ho_full_id})
            
            if result:
                return {'type': '호', 'id': result[0]['id']}
        
        # 2순위: 항 노드 찾기
        if hang_id:
            hang_full_id = f"{article_id}_{hang_id}"
            
            query = """
            MATCH (hang:항 {id: $hang_id})
            RETURN hang.id as id
            LIMIT 1
            """
            result = self.conn.execute_query(query, {'hang_id': hang_full_id})
            
            if result:
                return {'type': '항', 'id': result[0]['id']}
        
        # 3순위: 조 노드 찾기
        query = """
        MATCH (article:조 {id: $article_id})
        RETURN article.id as id
        LIMIT 1
        """
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
        CREATE (v)-[:VIOLATES {{
            confidence: 1.0,
            method: 'law_centric'
        }}]->(law)
        """
        
        self.conn.execute_query(query, {
            'case_id': case_id,
            'node_id': node_id
        })
    
    def _extract_and_link_keywords_with_frequency(self, df: pd.DataFrame):
        """
        키워드 추출 및 관계 생성 (빈도 기반 가중치)
        
        개선사항:
        - 키워드 빈도를 계산하여 가중치 부여
        - 자주 나타나는 키워드는 더 높은 위험도
        """
        print("  - 키워드 추출 및 빈도 계산...")
        
        # 전체 텍스트에서 키워드 빈도 계산
        keyword_frequency = {}
        all_keywords = set()
        
        for idx, row in df.iterrows():
            text = row['불공정 약관 원문']
            reason = row['시정 요청 사유']
            
            keywords = self._extract_keywords(text, reason)
            
            for kw in keywords:
                all_keywords.add(kw)
                keyword_frequency[kw] = keyword_frequency.get(kw, 0) + 1
        
        # 빈도 기준 상위 키워드만 노드로 생성
        print("  - Keyword 노드 생성 (빈도 기반)...")
        min_frequency = 2  # 최소 2회 이상 출현한 키워드만
        
        for kw, freq in keyword_frequency.items():
            if freq < min_frequency:
                continue
            
            # 빈도에 따른 중요도 점수
            importance = min(freq / 10.0, 1.0)
            
            query = """
            MERGE (k:Keyword {text: $keyword})
            ON CREATE SET
                k.id = $id,
                k.frequency = $frequency,
                k.importance = $importance,
                k.risk_level = $risk_level
            """
            
            self.conn.execute_query(query, {
                'keyword': kw,
                'id': f"KW_{kw}",
                'frequency': freq,
                'importance': importance,
                'risk_level': self._determine_keyword_risk(kw, freq)
            })
        
        # ViolationCase → Keyword 관계 생성 (빈도 가중치 포함)
        print("  - CONTAINS 관계 생성 (빈도 가중치)...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            case_id = f"CASE_{row['ID']}"
            text = row['불공정 약관 원문']
            reason = row['시정 요청 사유']
            
            keywords = self._extract_keywords(text, reason)
            
            for kw in keywords:
                if keyword_frequency.get(kw, 0) < min_frequency:
                    continue
                
                # 빈도에 따른 가중치
                weight = keyword_frequency[kw] / max(keyword_frequency.values())
                
                query = """
                MATCH (v:ViolationCase {id: $case_id})
                MATCH (k:Keyword {text: $keyword})
                CREATE (v)-[:CONTAINS {
                    weight: $weight,
                    frequency_score: $frequency
                }]->(k)
                """
                
                self.conn.execute_query(query, {
                    'case_id': case_id,
                    'keyword': kw,
                    'weight': weight,
                    'frequency': keyword_frequency[kw]
                })
    
    def _determine_keyword_risk(self, keyword: str, frequency: int) -> str:
        """
        키워드 위험도 판단 (빈도 고려)
        """
        high_risk_keywords = [
            '어떠한 경우에도', '일체', '책임지지 않', '면책', '부당하게',
            '과도한', '부담', '권리행사', '제한', '포기'
        ]
        
        medium_risk_keywords = [
            '책임', '손해배상', '불가항력', '해제', '해지', '변경'
        ]
        
        # 빈도가 높으면 위험도 상승
        if frequency >= 10:
            if any(hrk in keyword for hrk in high_risk_keywords):
                return 'critical'
            elif any(mrk in keyword for mrk in medium_risk_keywords):
                return 'high'
            else:
                return 'medium'
        elif frequency >= 5:
            if any(hrk in keyword for hrk in high_risk_keywords):
                return 'high'
            elif any(mrk in keyword for mrk in medium_risk_keywords):
                return 'medium'
            else:
                return 'low'
        else:
            if any(hrk in keyword for hrk in high_risk_keywords):
                return 'medium'
            else:
                return 'low'
    
    def _categorize_violations(self, df: pd.DataFrame):
        """위반 유형별 분류"""
        print("  - ViolationType 관계 생성...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            case_id = f"CASE_{row['ID']}"
            violation_type = row['대주제']
            
            if pd.isna(violation_type):
                continue
            
            query = """
            MATCH (v:ViolationCase {id: $case_id})
            MATCH (t:ViolationType {name: $type_name})
            CREATE (v)-[:CLASSIFIED_AS]->(t)
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
    
    def _extract_keywords(self, text: str, reason: str) -> List[str]:
        """키워드 추출"""
        keywords = set()
        
        # 위험 패턴
        risk_patterns = [
            '어떠한 경우에도', '일체', '책임지지 않', '면책',
            '부당하게', '과도한', '불가항력', '해제', '해지',
            '변경', '손해배상', '제한', '포기', '권리행사'
        ]
        
        combined_text = f"{text} {reason}"
        
        for pattern in risk_patterns:
            if pattern in combined_text:
                keywords.add(pattern)
        
        return list(keywords)
    
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
            "ViolationType": "MATCH (n:ViolationType) RETURN count(n) as count",
            "Company": "MATCH (n:Company) RETURN count(n) as count",
            "Keyword": "MATCH (n:Keyword) RETURN count(n) as count",
            "SIMILAR_TO": "MATCH ()-[r:SIMILAR_TO]->() RETURN count(r) as count",
            "VIOLATES": "MATCH ()-[r:VIOLATES]->() RETURN count(r) as count",
            "CONTAINS": "MATCH ()-[r:CONTAINS]->() RETURN count(r) as count",
        }
        
        print("\n" + "="*50)
        print("📊 그래프 통계 (Law-Centric)")
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
    
    # 여러 CSV
    builder.build_from_multiple_csv([
        "data/contracts/reference/보도자료_데이터_전처리_최종.csv",
        "data/contracts/reference/ai.csv"
    ])
    
    conn.close()