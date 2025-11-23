# rag/hybrid_graphrag.py

"""
LangChain + Neo4j GraphRAG 통합
[수정됨: 2025-11-08] - LangChain 1.x 및 드라이버 주입 방식 적용
"""

from neo4j import GraphDatabase
from langchain_core.prompts import PromptTemplate # 👈 PromptTemplate 경로 수정
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# 👇 Neo4j 관련 모듈은 모두 'langchain_neo4j'에서 가져옵니다.
from langchain_neo4j import (
    GraphCypherQAChain,
    Neo4jGraph,
    Neo4jVector
)

from typing import List, Dict, Any
import numpy as np
# --- [수정] 여기까지 덮어쓰세요 ---


class HybridGraphRAG:
    """GraphRAG + VectorRAG 하이브리드 검색"""
    
    # 👇 [수정됨] 이 __init__ 부분은 이미 올바르게 수정되었습니다.
    
    def __init__(
        self,
        driver: GraphDatabase.driver, 
        openai_api_key: str,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None
    ):
        import os
        
        # Neo4j 드라이버 저장
        self.driver = driver
        
        # Neo4j 연결 정보 추출 (환경변수 또는 파라미터)
        self.neo4j_uri = neo4j_uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.neo4j_user = neo4j_user or os.getenv('NEO4J_USER', 'neo4j')
        self.neo4j_password = neo4j_password or os.getenv('NEO4J_PASSWORD', 'password')
        
        # Neo4jGraph 초기화 - URL/username/password 방식 사용
        self.graph = Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_user,
            password=self.neo4j_password
        )
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            openai_api_key=openai_api_key
        )
        
        # 임베딩 모델
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.local_embeddings = HuggingFaceEmbeddings(
            model_name='paraphrase-multilingual-MiniLM-L12-v2'
        )
        
        # 벡터 스토어 (Neo4j Vector Index)
        # retrieval_query로 id를 명시적으로 포함
        retrieval_query = """
        RETURN node.original_text + ' ' + coalesce(node.violation_reason, '') AS text,
               score,
               {
                   id: node.id,
                   article_id: node.article_id,
                   category: node.category,
                   subcategory: node.subcategory,
                   company: node.company,
                   corrected_text: node.corrected_text,
                   year: node.year,
                   other_legal_basis: node.other_legal_basis
               } AS metadata
        """

        try:
            self.vector_store = Neo4jVector.from_existing_graph(
                self.local_embeddings,
                url=self.neo4j_uri,
                username=self.neo4j_user,
                password=self.neo4j_password,
                index_name="violation_embeddings",
                node_label="ViolationCase",
                embedding_node_property="embedding",
                retrieval_query=retrieval_query
            )
        except Exception as e:
            print(f"⚠️ 벡터 스토어 초기화 실패: {e}")
            self.vector_store = None
        
        # Cypher QA Chain
        self.cypher_chain = self._create_cypher_chain()
    
    # ... (이하 _create_cypher_chain 부터 파일 끝까지는 수정할 필요 없음) ...
    # ... (기존 코드 그대로 두세요) ...

    def _create_cypher_chain(self) -> GraphCypherQAChain:
        """Cypher 쿼리 생성 체인"""
        cypher_prompt = PromptTemplate(
            template="""
당신은 Neo4j Cypher 쿼리 전문가입니다.
사용자 질문을 받아 적절한 Cypher 쿼리를 생성하세요.

그래프 스키마:
{schema}

질문: {question}

Cypher 쿼리만 반환하세요 (설명 없이):
            """,
            input_variables=["schema", "question"]
        )
        
        return GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=True,
            return_intermediate_steps=True,
            allow_dangerous_requests=True  # ✅ 보안 확인 - 신뢰할 수 있는 데이터베이스에서만 사용
        )
    
    # =========================================================================
    # 핵심 검색 메서드
    # =========================================================================
    
    def search_similar_cases(
        self,
        query_text: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict]:
        """
        벡터 유사도 검색
        
        Args:
            query_text: 검색할 약관 텍스트
            top_k: 반환할 상위 결과 수
            similarity_threshold: 유사도 임계값
        
        Returns:
            유사 사례 리스트
        """
        print(f"🔍 벡터 검색: {query_text[:50]}...")
        
        if not self.vector_store:
            # 벡터 스토어가 없으면 직접 검색
            return self._fallback_search(query_text, top_k)
        
        # 벡터 검색
        try:
            results = self.vector_store.similarity_search_with_score(
                query_text,
                k=top_k
            )
            
            similar_cases = []
            for doc, score in results:
                if score >= similarity_threshold:
                    similar_cases.append({
                        'document': doc,
                        'similarity_score': float(score),
                        'metadata': doc.metadata
                    })
            
            return similar_cases
        except Exception as e:
            print(f"⚠️ 벡터 검색 실패: {e}")
            return self._fallback_search(query_text, top_k)
    
    def _fallback_search(self, query_text: str, top_k: int) -> List[Dict]:
        """벡터 스토어 실패 시 폴백 검색"""
        # 로컬 임베딩으로 검색
        query_embedding = self.local_embeddings.embed_query(query_text)
        
        # Neo4j에서 모든 사례 가져오기
        query = """
        MATCH (v:ViolationCase)
        RETURN v.id as id, v.original_text as text, 
               v.article_id as article_id, v.embedding as embedding
        LIMIT 100
        """
        
        results = self.graph.query(query)
        
        # 유사도 계산
        similarities = []
        for r in results:
            if r['embedding']:
                emb = np.array(r['embedding'])
                sim = np.dot(query_embedding, emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(emb)
                )
                similarities.append({
                    'id': r['id'],
                    'text': r['text'],
                    'article_id': r['article_id'],
                    'similarity_score': float(sim)
                })
        
        # 정렬 후 상위 k개 반환
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return [
            {
                'document': type('obj', (object,), {
                    'page_content': s['text'],
                    'metadata': {'id': s['id'], 'article_id': s['article_id']}
                })(),
                'similarity_score': s['similarity_score'],
                'metadata': {'id': s['id'], 'article_id': s['article_id']}
            }
            for s in similarities[:top_k]
        ]
    
    def explore_graph_neighborhood(
        self,
        case_id: str,
        max_depth: int = 2
    ) -> Dict:
        """
        그래프 이웃 탐색 - GraphRAG 핵심!
        
        주어진 사례 주변의 연결된 노드들을 탐색
        """
        print(f"🕸️ 그래프 탐색: {case_id}, 깊이={max_depth}")
        
        query = f"""
        MATCH path = (v:ViolationCase {{id: $case_id}})-[*1..{max_depth}]-(connected)
        RETURN 
            v as source_case,
            connected,
            labels(connected) as node_type,
            relationships(path) as relationships,
            length(path) as depth
        ORDER BY depth
        LIMIT 50
        """
        
        results = self.graph.query(query, params={'case_id': case_id})
        
        # 결과 구조화
        neighborhood = {
            'source_case': None,
            'similar_cases': [],
            'related_laws': [],
            'keywords': [],
            'companies': [],
            'violation_types': []
        }
        
        for record in results:
            node_type = record['node_type'][0] if record['node_type'] else None
            
            if node_type == 'ViolationCase' and record['depth'] > 0:
                neighborhood['similar_cases'].append(record['connected'])
            elif node_type == 'LawArticle':
                neighborhood['related_laws'].append(record['connected'])
            elif node_type == 'Keyword':
                neighborhood['keywords'].append(record['connected'])
            elif node_type == 'Company':
                neighborhood['companies'].append(record['connected'])
            elif node_type == 'ViolationType':
                neighborhood['violation_types'].append(record['connected'])
        
        return neighborhood
    
    def find_violation_patterns(
        self,
        article_id: str,
        min_similarity: float = 0.8
    ) -> List[Dict]:
        """
        특정 조항 위반 패턴 발견
        
        GraphRAG의 강점: 유사한 사례들의 공통 패턴 추출
        """
        print(f"🔎 패턴 발견: {article_id}")
        
        query = """
        // 1. 해당 조항을 위반한 모든 사례 찾기
        MATCH (v:ViolationCase)-[r:VIOLATES]->(l:LawArticle {id: $article_id})
        
        // 2. 각 사례와 유사한 다른 사례들 찾기
        MATCH (v)-[s:SIMILAR_TO]->(similar:ViolationCase)
        WHERE s.similarity_score >= $min_similarity
        
        // 3. 공통 키워드 찾기
        MATCH (v)-[:CONTAINS]->(k:Keyword)
        MATCH (similar)-[:CONTAINS]->(k)
        
        // 4. 집계
        RETURN 
            l.id as article_id,
            k.text as common_keyword,
            count(DISTINCT v) as violation_count,
            avg(s.similarity_score) as avg_similarity,
            collect(DISTINCT v.id) as case_ids
        ORDER BY violation_count DESC, avg_similarity DESC
        LIMIT 10
        """
        
        results = self.graph.query(query, params={
            'article_id': article_id,
            'min_similarity': min_similarity
        })
        
        return results
    
    def multi_hop_reasoning(
        self,
        query_text: str,
        max_hops: int = 3
    ) -> Dict:
        """
        다단계 추론 검색
        
        예: "면책조항이 소비자 권리에 미치는 영향"
        → 면책조항 사례 → 연관 손해배상 조항 → 실제 피해 사례
        """
        print(f"🧠 다단계 추론: {query_text}")
        
        # Step 1: 초기 벡터 검색
        initial_cases = self.search_similar_cases(query_text, top_k=3)
        
        if not initial_cases:
            return {'error': '관련 사례를 찾을 수 없습니다.'}
        
        # Step 2: 그래프 탐색으로 확장
        all_paths = []
        for case in initial_cases:
            case_id = case['metadata'].get('id')
            
            # 다단계 경로 탐색
            query = f"""
            MATCH path = (start:ViolationCase {{id: $case_id}})
                         -[*1..{max_hops}]-
                         (end:ViolationCase)
            WHERE start <> end
            RETURN 
                path,
                [node in nodes(path) | node.original_text] as texts,
                [rel in relationships(path) | type(rel)] as rel_types,
                length(path) as hop_count
            ORDER BY hop_count
            LIMIT 20
            """
            
            paths = self.graph.query(query, params={'case_id': case_id})
            all_paths.extend(paths)
        
        # Step 3: LLM으로 경로 분석 및 요약
        reasoning_result = self._analyze_paths_with_llm(query_text, all_paths)
        
        return {
            'query': query_text,
            'initial_cases': initial_cases,
            'reasoning_paths': all_paths,
            'analysis': reasoning_result
        }
    
    def hybrid_search(
        self,
        query_text: str,
        alpha: float = 0.5
    ) -> List[Dict]:
        """
        하이브리드 검색: 벡터 + 그래프 결합
        
        Args:
            query_text: 검색 쿼리
            alpha: 벡터 점수 가중치 (0~1)
                   alpha=1: 순수 벡터 검색
                   alpha=0: 순수 그래프 검색
        """
        print(f"🔀 하이브리드 검색: alpha={alpha}")
        
        # 1. 벡터 검색
        vector_results = self.search_similar_cases(query_text, top_k=10)
        
        # 2. 그래프 기반 점수 계산
        graph_scores = {}
        for result in vector_results:
            case_id = result['metadata'].get('id')
            
            # 그래프 중심성 점수 계산
            centrality_query = """
            MATCH (v:ViolationCase {id: $case_id})
            OPTIONAL MATCH (v)-[r]-()
            RETURN 
                count(r) as degree,
                size((v)-[:SIMILAR_TO]-()) as similar_count
            """
            
            centrality = self.graph.query(centrality_query, params={'case_id': case_id})
            
            if centrality:
                degree = centrality[0]['degree']
                similar_count = centrality[0]['similar_count']
                
                # 정규화 (0~1)
                graph_score = min((degree + similar_count) / 20.0, 1.0)
                graph_scores[case_id] = graph_score
        
        # 3. 점수 결합
        final_results = []
        for result in vector_results:
            case_id = result['metadata'].get('id')
            vector_score = result['similarity_score']
            graph_score = graph_scores.get(case_id, 0)
            
            # 가중 평균
            hybrid_score = alpha * vector_score + (1 - alpha) * graph_score
            
            final_results.append({
                **result,
                'graph_score': graph_score,
                'hybrid_score': hybrid_score
            })
        
        # 하이브리드 점수로 재정렬
        final_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return final_results
    
    # =========================================================================
    # LLM 기반 분석
    # =========================================================================
    
    def _analyze_paths_with_llm(self, query: str, paths: List[Dict]) -> str:
        """LLM을 사용한 경로 분석"""
        if not paths:
            return "분석할 경로가 없습니다."
        
        # 경로 요약
        path_summary = []
        for i, path in enumerate(paths[:5]):  # 상위 5개만
            path_summary.append(
                f"경로 {i+1}: " + 
                " → ".join(path['rel_types']) +
                f" (단계: {path['hop_count']})"
            )
        
        prompt = f"""
사용자 질문: {query}

발견된 연관 경로:
{chr(10).join(path_summary)}

위 경로들을 분석하여 다음을 설명하세요:
1. 질문과의 관련성
2. 발견된 패턴이나 인사이트
3. 주의해야 할 점

간결하게 3-5문장으로 요약:
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"분석 실패: {e}"
    
    def explain_violation(
        self,
        case_id: str
    ) -> Dict:
        """
        위반 사례 종합 설명
        
        GraphRAG의 강력함을 보여주는 예시
        """
        print(f"📝 위반 설명 생성: {case_id}")
        
        # 1. 사례 기본 정보
        case_query = """
        MATCH (v:ViolationCase {id: $case_id})
        RETURN v
        """
        case_data = self.graph.query(case_query, params={'case_id': case_id})
        
        if not case_data:
            return {'error': '사례를 찾을 수 없습니다.'}
        
        case_data = case_data[0]['v']
        
        # 2. 그래프 컨텍스트 수집
        context_query = """
        MATCH (v:ViolationCase {id: $case_id})
        
        // 위반 조항
        OPTIONAL MATCH (v)-[:VIOLATES]->(law:LawArticle)
        
        // 유사 사례 (상위 3개)
        OPTIONAL MATCH (v)-[s:SIMILAR_TO]->(similar:ViolationCase)
        WITH v, law, similar, s
        ORDER BY s.similarity_score DESC
        LIMIT 3
        
        // 키워드
        OPTIONAL MATCH (v)-[:CONTAINS]->(k:Keyword)
        
        // 회사
        OPTIONAL MATCH (v)-[:COMMITTED_BY]->(c:Company)
        
        RETURN 
            law.id as law_article,
            law.content as law_content,
            collect(DISTINCT similar.original_text)[..3] as similar_texts,
            collect(DISTINCT k.text) as keywords,
            c.name as company
        """
        
        context = self.graph.query(context_query, params={'case_id': case_id})
        
        if not context:
            context = [{}]
        
        context = context[0]
        
        # 3. LLM으로 종합 설명 생성
        prompt = f"""
다음 불공정 약관 사례를 분석하세요:

[원문]
{case_data.get('original_text', 'N/A')}

[위반 조항]
{context.get('law_article', 'N/A')}

[유사 사례들]
{chr(10).join(context.get('similar_texts', []) or ['없음'])}

[관련 키워드]
{', '.join(context.get('keywords', []) or ['없음'])}

다음 형식으로 설명하세요:

1. **위반 내용**: 무엇이 문제인가?
2. **법적 근거**: 어떤 법을 위반했는가?
3. **유사 패턴**: 비슷한 사례들의 공통점
4. **수정 방향**: 어떻게 고쳐야 하는가?

각 항목을 2-3문장으로 간결하게:
        """
        
        try:
            explanation = self.llm.invoke(prompt)
            
            return {
                'case_id': case_id,
                'original_text': case_data.get('original_text', ''),
                'context': context,
                'explanation': explanation.content
            }
        except Exception as e:
            return {
                'case_id': case_id,
                'original_text': case_data.get('original_text', ''),
                'context': context,
                'explanation': f"설명 생성 실패: {e}"
            }


# =============================================================================
# 실행 예시
# =============================================================================

if __name__ == "__main__":
    import os
    
    # [수정] 실행 예시도 driver 방식으로 변경
    try:
        from neo4j import GraphDatabase
        
        NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
        NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

        # 드라이버 생성
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        
        print("✅ Neo4j 드라이버 연결 성공")

        # 초기화 (드라이버 주입)
        rag = HybridGraphRAG(
            driver=driver,
            openai_api_key=OPENAI_API_KEY
        )
        
        # 예시 1: 벡터 검색
        print("\n" + "="*70)
        print("예시 1: 벡터 유사도 검색")
        print("="*70)
        results = rag.search_similar_cases(
            "회사는 어떠한 경우에도 책임을 지지 않습니다",
            top_k=3
        )
        for i, result in enumerate(results, 1):
            print(f"\n{i}. 유사도: {result['similarity_score']:.3f}")
            print(f"   내용: {result['document'].page_content[:100]}...")
            
        # 드라이버 종료
        driver.close()

    except Exception as e:
        print(f"❌ 실행 실패: {e}")