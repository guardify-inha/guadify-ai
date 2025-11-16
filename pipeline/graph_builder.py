"""
CSV 데이터를 Law-Centric GraphRAG로 변환하는 파이프라인

핵심 변경사항:
1. Law-Centric 구조: 법률 조항(조-항-호)을 중심으로 위반사례 연결
2. ViolationCase → 호 노드로 직접 연결 (VIOLATES 관계)
3. LawArticle 노드 제거, 기존 law_structure의 호 노드 활용
4. Company 노드 제거 (속성으로만 유지)
5. severity 제거 (Keyword 기반으로 실시간 계산)
6. frequency → case_count 변경 (사례 등장 수 기반)
7. 중주제/소주제 계층 구조 지원
8. ✨ 정규표현식 기반 키워드 추출 추가 (표현 변형 대응)
9. 🔧 KeyError 수정 (v1.1)
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
        
        # patterns_by_article_v2.json 로드
        self._load_patterns()
    
    def _load_patterns(self):
        """patterns_by_article_v2.json 로드"""
        try:
            from pathlib import Path
            
            # 현재 파일 기준 경로
            current_dir = Path(__file__).parent
            pattern_path = current_dir.parent / "data" / "contracts" / "reference" / "patterns_by_article_v2.json"
            
            # 대체 경로들
            if not pattern_path.exists():
                alternative_paths = [
                    Path("data/contracts/reference/patterns_by_article_v2.json"),
                    Path("../data/contracts/reference/patterns_by_article_v2.json"),
                    Path("../../data/contracts/reference/patterns_by_article_v2.json"),
                ]
                for alt_path in alternative_paths:
                    if alt_path.exists():
                        pattern_path = alt_path
                        break
            
            if pattern_path.exists():
                with open(pattern_path, 'r', encoding='utf-8') as f:
                    self.patterns = json.load(f)
                print(f"✅ 패턴 데이터 로드 완료: {pattern_path}")
            else:
                print(f"⚠️ 패턴 파일을 찾을 수 없습니다: {pattern_path}")
                print(f"   기본 하드코딩 패턴을 사용합니다.")
                self.patterns = {}
        except Exception as e:
            print(f"⚠️ 패턴 로드 실패: {e}")
            print(f"   기본 하드코딩 패턴을 사용합니다.")
            self.patterns = {}
        
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
        
        # Step 6: 키워드 추출 및 관계 생성 (사례 수 기반)
        print("\n🔑 Step 6: 키워드 추출 (사례 수 기반 + 정규표현식)...")
        self._extract_and_link_keywords_with_case_count(df)
        
        # Step 7: 위반 유형 분류 (중주제/소주제 지원)
        print("\n📋 Step 7: 위반 유형 분류 중...")
        self._categorize_violations_with_hierarchy(df)
        
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

        print("\n🔑 Step 6: 키워드 추출 (사례 수 기반 + 정규표현식)...")
        self._extract_and_link_keywords_with_case_count(df_total)

        print("\n📋 Step 7: 위반 유형 분류 중...")
        self._categorize_violations_with_hierarchy(df_total)

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
        """
        텍스트 임베딩 생성
        
        임베딩 대상:
        1. violation_original: 불공정 약관 원문 (유사도 검색용)
        2. violation_corrected: 수정 후 약관 (공정 사례 비교용)
        3. violation_reason: 시정 요청 사유 (추가 컨텍스트)
        """
        embeddings = {}

        # 1️⃣ 위반 사례 원문 임베딩 (가장 중요!)
        print("  - 위반 사례 임베딩...")
        texts = df['불공정 약관 원문'].tolist()
        embeddings['violation_original'] = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )

        # 2️⃣ 수정 후 약관 임베딩 (공정한 약관 비교용)
        print("\n  - 수정 약관 임베딩...")
        corrected_texts = df['수정 후 약관 조항'].fillna('').tolist()
        embeddings['violation_corrected'] = self.model.encode(
            corrected_texts,
            show_progress_bar=True,
            batch_size=32
        )

        # 3️⃣ 시정 사유 임베딩 (추가 컨텍스트)
        print("\n  - 시정 사유 임베딩...")
        reason_texts = df['시정 요청 사유'].tolist()
        embeddings['violation_reason'] = self.model.encode(
            reason_texts,
            show_progress_bar=True,
            batch_size=32
        )

        return embeddings
    
    def _create_nodes(self, df: pd.DataFrame, embeddings: Dict):
        """노드 생성 (severity 제거!)"""
        # 위반 사례 노드 생성
        print("  - ViolationCase 노드 생성...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            case_id = f"CASE_{row['ID']}"
            
            # 벡터를 리스트로 변환 (Neo4j 저장용)
            embedding_list = embeddings['violation_original'][idx].tolist()
            
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
                
                # ❌ severity 제거! (Keyword 기반으로 실시간 계산)
                
                'category': row['대주제'],
                'subcategory': row.get('소주제', ''),
                'article_id': self._extract_article_id(row.get('근거 조항(약관법)', '')),
                'other_legal_basis': row.get('근거 조항(약관법 외)', ''),
                'embedding': embedding_list
            })
        
        # 위반 유형 노드는 나중에 생성 (_categorize_violations_with_hierarchy에서)
    
    def _create_similarity_relationships(self, df: pd.DataFrame, embeddings: Dict):
        """
        유사도 기반 관계 생성
        
        ViolationCase 간 SIMILAR_TO 관계:
        - 코사인 유사도 0.7 이상인 사례끼리 연결
        - 각 사례당 최대 5개까지만 연결
        """
        print("  - 유사도 행렬 계산 중...")
        original_embeddings = embeddings['violation_original']
        
        # 코사인 유사도 계산
        similarity_matrix = cosine_similarity(original_embeddings)
        
        print("  - SIMILAR_TO 관계 생성 중...")
        threshold = 0.7  # 유사도 임계값
        top_k = 5        # 최대 연결 개수
        
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
        [Law-Centric 구조] ViolationCase → 호/항/조 노드 연결
        
        연결 우선순위:
        1. 호 노드 (가장 구체적) - 예: "제7조_제2호"
        2. 항 노드 (호가 없는 경우) - 예: "제6조_제1항"
        3. 조 노드 (항도 없는 경우) - 예: "제8조"
        """
        print("  - ViolationCase → 법률 노드 연결 (Law-Centric)...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            case_id = f"CASE_{row['ID']}"
            legal_basis = row.get('근거 조항(약관법)', '')
            
            if not legal_basis or pd.isna(legal_basis):
                continue
            
            # 조항 파싱 (예: "제7조 제2호", "제6조 제2항 제1호")
            parsed = self._parse_legal_article(legal_basis)
            
            # 🔧 수정: .get() 사용으로 KeyError 방지
            if not parsed.get('article'):
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
        - "약관법 제7조 제1호" → {article: "제7조", ho: "제1호"}
        - "제6조 제2항 제1호" → {article: "제6조", hang: "제2항", ho: "제1호"}
        - "제8조" → {article: "제8조"}
        """
        # 🔧 수정: 항상 모든 키를 포함하는 딕셔너리 반환 (KeyError 방지)
        result = {
            'article': None,
            'hang': None,
            'ho': None
        }
        
        # 빈 값 체크
        if not legal_basis or pd.isna(legal_basis):
            return result
        
        # 문자열로 변환
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
    
    def _extract_and_link_keywords_with_case_count(self, df: pd.DataFrame):
        """
        키워드 추출 및 관계 생성 (사례 수 기반 + 정규표현식)
        
        ✅ 개선: frequency → case_count
        ✅ 신규: 정규표현식으로 표현 변형 대응
        
        변경 사항:
        - frequency (텍스트에서 등장 횟수) → case_count (사례 등장 수)
        - importance → prevalence (일반성 지표)
        
        Keyword 노드 속성:
        - text: 키워드 텍스트
        - case_count: 이 키워드가 등장한 사례 수
        - prevalence: 사례 등장 비율 (case_count / 50)
        - risk_level: JSON의 risk_level
        """
        print("  - 키워드 추출 및 사례 수 계산 (정규표현식 포함)...")
        
        # 키워드가 등장한 사례 수 계산
        keyword_case_count = {}
        keyword_case_ids = {}  # 디버깅용: 어느 사례에 등장했는지
        
        for idx, row in df.iterrows():
            case_id = f"CASE_{row['ID']}"
            text = row['불공정 약관 원문']
            reason = row['시정 요청 사유']
            
            # 키워드 추출 (문자열 매칭 + 정규표현식)
            keywords = self._extract_keywords(text, reason)
            
            # 각 키워드마다 사례 1개 카운트 (중복 제거)
            for kw in set(keywords):
                keyword_case_count[kw] = keyword_case_count.get(kw, 0) + 1
                
                if kw not in keyword_case_ids:
                    keyword_case_ids[kw] = []
                keyword_case_ids[kw].append(case_id)
        
        # 사례 수 기준 상위 키워드만 노드로 생성
        print("  - Keyword 노드 생성 (사례 수 기반)...")
        min_case_count = 2  # 최소 2개 사례 이상에서 등장한 키워드만
        
        total_cases = len(df)
        
        for kw, case_count in keyword_case_count.items():
            if case_count < min_case_count:
                continue
            
            # 사례 등장 비율 (일반성 지표)
            # 50개 이상 사례에서 등장하면 1.0
            prevalence = min(case_count / 50.0, 1.0)
            
            # 퍼센트로도 계산
            percentage = (case_count / total_cases) * 100
            
            query = """
            MERGE (k:Keyword {text: $keyword})
            ON CREATE SET
                k.id = $id,
                k.case_count = $case_count,
                k.prevalence = $prevalence,
                k.percentage = $percentage,
                k.risk_level = $risk_level
            """
            
            self.conn.execute_query(query, {
                'keyword': kw,
                'id': f"KW_{kw}",
                'case_count': case_count,
                'prevalence': prevalence,
                'percentage': round(percentage, 2),
                'risk_level': self._get_keyword_risk_from_json(kw)
            })
        
        # ViolationCase → Keyword 관계 생성
        print("  - CONTAINS 관계 생성...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            case_id = f"CASE_{row['ID']}"
            text = row['불공정 약관 원문']
            reason = row['시정 요청 사유']
            
            keywords = self._extract_keywords(text, reason)
            
            for kw in set(keywords):  # 중복 제거
                # 사례 수 2개 미만 키워드는 노드가 없으므로 스킵
                if keyword_case_count.get(kw, 0) < min_case_count:
                    continue
                
                query = """
                MATCH (v:ViolationCase {id: $case_id})
                MATCH (k:Keyword {text: $keyword})
                CREATE (v)-[:CONTAINS]->(k)
                """
                
                self.conn.execute_query(query, {
                    'case_id': case_id,
                    'keyword': kw
                })
    
    def _get_keyword_risk_from_json(self, keyword: str) -> str:
        """
        patterns_by_article_v2.json에서 키워드의 위험도 조회
        """
        # 패턴이 없으면 기본값
        if not self.patterns:
            return self._determine_keyword_risk_fallback(keyword)
        
        # 1️⃣ universal_risk_keywords에서 찾기
        universal = self.patterns.get('universal_risk_keywords', {})
        if 'keywords' in universal:
            for kw_info in universal['keywords']:
                if kw_info['keyword'] == keyword:
                    return kw_info.get('risk_level', 'medium')
        
        # 2️⃣ 각 조항의 패턴에서 찾기
        for article_id in ['제6조', '제7조', '제8조', '제9조', '제10조', 
                          '제11조', '제12조', '제13조', '제14조']:
            
            if article_id not in self.patterns:
                continue
            
            article_data = self.patterns[article_id]
            
            for pattern in article_data.get('patterns', []):
                # high_risk_keywords에 있으면 해당 패턴의 risk_level
                if keyword in pattern.get('high_risk_keywords', []):
                    return pattern.get('risk_level', 'high')
                
                # 일반 keywords에 있으면 한 단계 낮춤
                if keyword in pattern.get('keywords', []):
                    risk = pattern.get('risk_level', 'medium')
                    if risk == 'critical':
                        return 'high'
                    elif risk == 'high':
                        return 'medium'
                    else:
                        return risk
        
        # 3️⃣ 복합 패턴에서 찾기
        combined = self.patterns.get('combined_pattern_risks', {})
        if 'patterns' in combined:
            for pattern in combined['patterns']:
                combination = pattern.get('combination', [])
                if '+' in keyword:
                    parts = keyword.split('+')
                    if all(part in combination for part in parts):
                        return pattern.get('risk_level', 'high')
        
        # 못 찾으면 기본값
        return 'medium'
    
    def _determine_keyword_risk_fallback(self, keyword: str) -> str:
        """패턴 JSON이 없을 때 사용할 기본 위험도 판단"""
        high_risk_keywords = [
            '어떠한 경우에도', '일체', '책임지지 않', '면책', '부당하게',
            '과도한', '무조건', '환불 불가', '환불이 불가', '즉시 채무 변제'
        ]
        
        medium_risk_keywords = [
            '책임', '손해배상', '불가항력', '해제', '해지', '변경',
            '별도 통지 없이', '자동', '즉시'
        ]
        
        if any(hrk in keyword for hrk in high_risk_keywords):
            return 'high'
        elif any(mrk in keyword for mrk in medium_risk_keywords):
            return 'medium'
        else:
            return 'low'
    
    def _categorize_violations_with_hierarchy(self, df: pd.DataFrame):
        """
        위반 유형 계층 구조 생성
        
        구조:
        - 최종본 CSV: 대주제 (조항명) > 중주제 > 소주제
        - ai.csv: 대주제만 (ViolationType)
        
        노드:
        - MajorTopic (중주제)
        - MinorTopic (소주제)
        - ViolationType (대주제, ai.csv용)
        """
        
        # 중주제/소주제 컬럼 존재 여부 확인
        has_major = '중주제' in df.columns
        has_minor = '소주제' in df.columns
        
        if has_major:
            print("  - 중주제/소주제 계층 구조 생성...")
            self._create_detailed_hierarchy(df, has_minor)
        else:
            print("  - 대주제 기반 분류 (ai.csv)...")
            self._create_simple_category(df)
    
    def _create_detailed_hierarchy(self, df: pd.DataFrame, has_minor: bool):
        """
        중주제/소주제 계층 구조 생성 (최종본 CSV)
        
        ViolationCase
          ├─[HAS_MAJOR_TOPIC]─> MajorTopic (중주제)
          └─[HAS_MINOR_TOPIC]─> MinorTopic (소주제)
        """
        # 1. MajorTopic 노드 생성
        major_topics = df['중주제'].dropna().unique()
        
        for major_topic in major_topics:
            query = """
            MERGE (m:MajorTopic {name: $name})
            ON CREATE SET
                m.id = $id,
                m.description = $name
            """
            self.conn.execute_query(query, {
                'name': major_topic,
                'id': f"MAJOR_{major_topic.replace(' ', '_')}"
            })
        
        # 2. MinorTopic 노드 생성 (있으면)
        if has_minor:
            minor_topics = df['소주제'].dropna().unique()
            
            for minor_topic in minor_topics:
                query = """
                MERGE (n:MinorTopic {name: $name})
                ON CREATE SET
                    n.id = $id,
                    n.description = $name
                """
                self.conn.execute_query(query, {
                    'name': minor_topic,
                    'id': f"MINOR_{minor_topic.replace(' ', '_')}"
                })
        
        # 3. 관계 생성
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            case_id = f"CASE_{row['ID']}"
            
            # ViolationCase → MajorTopic
            major_topic = row.get('중주제')
            if pd.notna(major_topic):
                query = """
                MATCH (v:ViolationCase {id: $case_id})
                MERGE (m:MajorTopic {name: $major_topic})
                MERGE (v)-[:HAS_MAJOR_TOPIC]->(m)
                """
                self.conn.execute_query(query, {
                    'case_id': case_id,
                    'major_topic': major_topic
                })
            
            # ViolationCase → MinorTopic
            if has_minor:
                minor_topic = row.get('소주제')
                if pd.notna(minor_topic):
                    query = """
                    MATCH (v:ViolationCase {id: $case_id})
                    MERGE (n:MinorTopic {name: $minor_topic})
                    MERGE (v)-[:HAS_MINOR_TOPIC]->(n)
                    """
                    self.conn.execute_query(query, {
                        'case_id': case_id,
                        'minor_topic': minor_topic
                    })
    
    def _create_simple_category(self, df: pd.DataFrame):
        """
        대주제 기반 분류 (ai.csv용)
        
        ViolationCase
          └─[CLASSIFIED_AS]─> ViolationType (대주제)
        """
        # ViolationType 노드 생성
        violation_types = df['대주제'].dropna().unique()
        
        for vtype in violation_types:
            query = """
            MERGE (t:ViolationType {name: $name})
            ON CREATE SET
                t.id = $id,
                t.description = $name
            """
            self.conn.execute_query(query, {
                'name': vtype,
                'id': f"TYPE_{vtype.replace(' ', '_')}"
            })
        
        # 관계 생성
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            case_id = f"CASE_{row['ID']}"
            violation_type = row.get('대주제')
            
            if pd.notna(violation_type):
                query = """
                MATCH (v:ViolationCase {id: $case_id})
                MERGE (t:ViolationType {name: $type_name})
                MERGE (v)-[:CLASSIFIED_AS]->(t)
                """
                self.conn.execute_query(query, {
                    'case_id': case_id,
                    'type_name': violation_type
                })
    
    # =============================================================================
    # 키워드 추출 (정규표현식 추가!)
    # =============================================================================
    
    def _extract_keywords(self, text: str, reason: str) -> List[str]:
        """
        키워드 추출 (문자열 매칭 + 정규표현식)
        
        추출 방식:
        1. 기존 문자열 매칭
        2. 정규표현식 패턴 매칭 (표현 변형 대응)
        3. 복합 패턴 체크
        """
        keywords = set()
        
        # 1️⃣ 기존 문자열 매칭
        keywords.update(self._extract_keywords_string_match(text, reason))
        
        # 2️⃣ 정규표현식 패턴 매칭
        keywords.update(self._extract_keywords_with_regex(text, reason))
        
        # 3️⃣ 복합 패턴 체크
        keywords.update(self._check_combined_patterns(text, reason))
        
        return list(keywords)
    
    def _extract_keywords_string_match(self, text: str, reason: str) -> List[str]:
        """
        기존 문자열 매칭 방식 (patterns_by_article_v2.json 사용)
        """
        keywords = set()
        combined_text = f"{text} {reason}"
        
        # 패턴이 로드되지 않은 경우 기본 패턴 사용
        if not self.patterns:
            return self._extract_keywords_fallback(text, reason)
        
        # 1️⃣ 범용 위험 키워드
        universal = self.patterns.get('universal_risk_keywords', {})
        if 'keywords' in universal:
            for kw_info in universal['keywords']:
                keyword = kw_info['keyword']
                if keyword in combined_text:
                    keywords.add(keyword)
        
        # 2️⃣ 조항별 패턴
        for article_id in ['제6조', '제7조', '제8조', '제9조', '제10조', 
                          '제11조', '제12조', '제13조', '제14조']:
            
            if article_id not in self.patterns:
                continue
            
            article_data = self.patterns[article_id]
            
            for pattern in article_data.get('patterns', []):
                # 일반 키워드
                for kw in pattern.get('keywords', []):
                    if kw in combined_text:
                        keywords.add(kw)
                
                # 고위험 키워드
                for kw in pattern.get('high_risk_keywords', []):
                    if kw in combined_text:
                        keywords.add(kw)
        
        return list(keywords)
    
    def _extract_keywords_with_regex(self, text: str, reason: str) -> List[str]:
        """
        정규표현식으로 유사 표현 처리
        
        예시:
        - "책임지지 않는다" / "책임 지지 않습니다" / "책임을 부담하지 않는다"
          → 모두 "책임지지 않음" 키워드로 통일
        """
        keywords = set()
        combined_text = f"{text} {reason}"
        
        # JSON에서 regex_patterns 사용
        if not self.patterns:
            return []
        
        # 1️⃣ 범용 정규표현식 패턴
        universal = self.patterns.get('universal_risk_keywords', {})
        if 'regex_patterns' in universal:
            for pattern_info in universal['regex_patterns']:
                if re.search(pattern_info['regex'], combined_text):
                    keywords.add(pattern_info['keyword'])
        
        # 2️⃣ 조항별 정규표현식 패턴
        for article_id in ['제6조', '제7조', '제8조', '제9조', '제10조', 
                          '제11조', '제12조', '제13조', '제14조']:
            
            if article_id not in self.patterns:
                continue
            
            article_data = self.patterns[article_id]
            
            if 'regex_patterns' in article_data:
                for pattern_info in article_data['regex_patterns']:
                    if re.search(pattern_info['regex'], combined_text):
                        keywords.add(pattern_info['keyword'])
        
        return list(keywords)
    
    def _check_combined_patterns(self, text: str, reason: str) -> List[str]:
        """
        복합 패턴 체크 (정규표현식 지원)
        
        예시:
        - "무조건" + "책임지지 않" → critical
        - "서면으로만" + "자동연장" → critical
        """
        combined_keywords = []
        combined_text = f"{text} {reason}"
        
        if not self.patterns:
            return []
        
        combined = self.patterns.get('combined_pattern_risks', {})
        if 'patterns' not in combined:
            return []
        
        for pattern in combined['patterns']:
            combination = pattern.get('combination', [])
            
            # 모든 키워드가 포함되어 있는지 체크 (정규식으로)
            all_matched = True
            for kw in combination:
                # 해당 키워드의 정규식 패턴 찾기
                regex_pattern = self._find_regex_for_keyword(kw)
                
                if regex_pattern:
                    # 정규식으로 매칭
                    if not re.search(regex_pattern, combined_text):
                        all_matched = False
                        break
                else:
                    # 문자열 매칭
                    if kw not in combined_text:
                        all_matched = False
                        break
            
            if all_matched:
                # 복합 키워드 생성 (예: "무조건+책임지지 않")
                combo_key = '+'.join(combination[:2])
                combined_keywords.append(combo_key)
        
        return combined_keywords
    
    def _find_regex_for_keyword(self, keyword: str) -> str:
        """
        키워드에 대응하는 정규표현식 패턴 찾기
        
        Returns:
            정규표현식 문자열 (없으면 None)
        """
        if not self.patterns:
            return None
        
        # 1️⃣ 범용 패턴에서 찾기
        universal = self.patterns.get('universal_risk_keywords', {})
        if 'regex_patterns' in universal:
            for pattern_info in universal['regex_patterns']:
                if pattern_info['keyword'] == keyword:
                    return pattern_info['regex']
        
        # 2️⃣ 조항별 패턴에서 찾기
        for article_id in ['제6조', '제7조', '제8조', '제9조', '제10조', 
                          '제11조', '제12조', '제13조', '제14조']:
            
            if article_id not in self.patterns:
                continue
            
            article_data = self.patterns[article_id]
            
            if 'regex_patterns' in article_data:
                for pattern_info in article_data['regex_patterns']:
                    if pattern_info['keyword'] == keyword:
                        return pattern_info['regex']
        
        return None
    
    def _extract_keywords_fallback(self, text: str, reason: str) -> List[str]:
        """패턴 로드 실패 시 기본 패턴"""
        keywords = set()
        
        risk_patterns = [
            '어떠한 경우에도', '일체', '책임지지 않', '면책',
            '부당하게', '과도한', '불가항력', '해제', '해지',
            '변경', '손해배상', '제한', '포기', '권리행사',
            '무조건', '별도 통지 없이', '즉시', '자동',
            '환불 불가', '환불이 불가', '서면으로만'
        ]
        
        combined_text = f"{text} {reason}"
        
        for pattern in risk_patterns:
            if pattern in combined_text:
                keywords.add(pattern)
        
        return list(keywords)
    
    # =============================================================================
    # 헬퍼 메서드
    # =============================================================================
    
    def _extract_company_name(self, filename: str) -> str:
        """파일명에서 회사명 추출"""
        if pd.isna(filename) or not isinstance(filename, str):
            return "Unknown"
        
        parts = filename.split('_')
        if len(parts) > 1:
            return parts[1]
        return "Unknown"
    
    def _extract_year(self, filename: str) -> int:
        """파일명에서 연도 추출"""
        if pd.isna(filename) or not isinstance(filename, str):
            return 2020
        
        match = re.search(r'(\d{2})(\d{2})(\d{2})', filename)
        if match:
            year_short = match.group(1)
            year = int('20' + year_short) if int(year_short) < 50 else int('19' + year_short)
            return year
        return 2020
    
    def _extract_article_id(self, legal_basis: str) -> str:
        """근거 조항에서 조 번호만 추출"""
        if not legal_basis or pd.isna(legal_basis):
            return None
        
        match = re.search(r'제(\d+)조', str(legal_basis))
        if match:
            return f"제{match.group(1)}조"
        return None
    
    def _print_statistics(self):
        """그래프 통계 출력"""
        queries = {
            "ViolationCase": "MATCH (n:ViolationCase) RETURN count(n) as count",
            "MajorTopic": "MATCH (n:MajorTopic) RETURN count(n) as count",
            "MinorTopic": "MATCH (n:MinorTopic) RETURN count(n) as count",
            "ViolationType": "MATCH (n:ViolationType) RETURN count(n) as count",
            "Keyword": "MATCH (n:Keyword) RETURN count(n) as count",
            "SIMILAR_TO": "MATCH ()-[r:SIMILAR_TO]->() RETURN count(r) as count",
            "VIOLATES": "MATCH ()-[r:VIOLATES]->() RETURN count(r) as count",
            "CONTAINS": "MATCH ()-[r:CONTAINS]->() RETURN count(r) as count",
            "HAS_MAJOR_TOPIC": "MATCH ()-[r:HAS_MAJOR_TOPIC]->() RETURN count(r) as count",
            "HAS_MINOR_TOPIC": "MATCH ()-[r:HAS_MINOR_TOPIC]->() RETURN count(r) as count",
        }
        
        print("\n" + "="*50)
        print("📊 그래프 통계 (Law-Centric v2 + Regex)")
        print("="*50)
        
        for name, query in queries.items():
            result = self.conn.execute_query(query)
            count = result[0]['count'] if result else 0
            if count > 0:  # 0이 아닌 것만 출력
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