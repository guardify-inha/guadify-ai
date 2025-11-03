"""
HyDE (Hypothetical Document Embeddings): 가상 문서 생성 기반 검색

3단계: LLM-as-Judge 및 HyDE 기법 구현
"""
import sys
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

from database.neo4j_connector import Neo4jConnector
from utils.llm_client import get_llm_client

# 임베딩 모델
try:
    from sentence_transformers import SentenceTransformer
    MODEL = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
except Exception:
    try:
        MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    except Exception:
        MODEL = None


def cosine_similarity(v1, v2):
    """코사인 유사도 계산"""
    if v1 is None or v2 is None:
        return 0.0
    v1, v2 = np.array(v1), np.array(v2)
    if v1.size == 0 or v2.size == 0:
        return 0.0
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


class HyDERetriever:
    """
    HyDE 리트리버: 가상 문서 생성 기반 검색
    
    질의에 대한 가상의 답변 문서를 생성하고, 이를 임베딩하여 검색 수행
    """
    
    def __init__(self, conn: Neo4jConnector):
        self.conn = conn
        self.llm = get_llm_client()
        self.model = MODEL
    
    def generate_hypothetical_document(self, query: str) -> Optional[str]:
        """
        질의에 대한 가상 문서 생성
        
        예: 질의 "회사는 책임을 지지 않는다"
        -> 가상 문서 "이 약관 조항은 면책 조항으로, 제7조에 위반될 가능성이 높습니다..."
        """
        if not self.llm:
            return None
        
        system_prompt = """당신은 약관법 전문가입니다.
입력된 불공정 약관 조항에 대해, 이 조항이 어떤 법률 조항을 위반할 가능성이 있는지 설명하는 문서를 작성하세요."""

        user_prompt = f"""
다음 불공정 약관 조항에 대해, 이 조항이 위반할 가능성이 있는 법률 조항에 대해 설명하는 가상의 문서를 작성하세요.

**검사할 약관 조항:**
"{query}"

다음 형식으로 작성하세요:
- 이 약관 조항의 핵심 내용
- 어떤 법률 조항(제6조~제14조)을 위반할 가능성이 있는지
- 위반 이유 및 근거
- 관련 법률 조항의 내용 요약

예시:
"이 약관 조항은 사업자가 모든 책임을 면제하는 전면 면책 조항입니다. 
이러한 조항은 약관법 제7조 '면책조항의 금지'에 위반될 가능성이 매우 높습니다. 
제7조는 사업자의 고의 또는 중대한 과실로 인한 법률상의 책임을 배제하는 조항을 무효로 규정하고 있습니다..."

가상 문서를 작성하세요:
"""
        
        result = self.llm.generate(user_prompt, system_prompt, temperature=0.3, max_tokens=400)
        return result.strip() if result else None
    
    def search_with_hypothetical_doc(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        HyDE 기법을 사용한 검색
        
        1. 질의에 대한 가상 문서 생성
        2. 가상 문서를 임베딩
        3. 가상 문서 임베딩으로 법률 조항 검색
        """
        if not self.model:
            return []
        
        # 1. 가상 문서 생성
        hypothetical_doc = self.generate_hypothetical_document(query)
        
        if not hypothetical_doc:
            # 가상 문서 생성 실패 시 원본 질의로 검색
            search_text = query
        else:
            # 가상 문서와 원본 질의 결합
            search_text = f"{query}\n\n{hypothetical_doc}"
        
        # 2. 임베딩 생성
        search_embedding = self.model.encode(search_text, normalize_embeddings=True)
        
        # 3. 법률 조항 검색
        query_cypher = """
        MATCH (n)
        WHERE (n:조 OR n:항 OR n:호) 
          AND n.content IS NOT NULL 
          AND n.content <> ''
          AND n.embedding IS NOT NULL
        RETURN 
            n.id as id,
            labels(n)[0] as node_type,
            n.content as content,
            n.embedding as embedding,
            n.title as title,
            n.article_id as article_id
        """
        
        nodes = self.conn.execute_query(query_cypher)
        
        # 유사도 계산
        candidates = []
        for node in nodes:
            node_embedding = node.get('embedding')
            if not node_embedding:
                continue
            
            similarity = cosine_similarity(search_embedding, node_embedding)
            
            candidates.append({
                'id': node.get('id'),
                'node_type': node.get('node_type'),
                'content': node.get('content'),
                'title': node.get('title'),
                'article_id': node.get('article_id'),
                'hyde_score': similarity,
                'hypothetical_doc': hypothetical_doc  # 가상 문서 정보 저장
            })
        
        # 유사도 기준 정렬
        candidates.sort(key=lambda x: x['hyde_score'], reverse=True)
        return candidates[:top_k]
    
    def enhance_with_hyde(self, query: str, existing_candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        기존 후보 검색 결과를 HyDE로 보강
        
        기존 후보와 HyDE 검색 결과를 통합하여 더 나은 후보 선별
        """
        # HyDE 검색 수행
        hyde_candidates = self.search_with_hypothetical_doc(query, top_k=top_k)
        
        # 기존 후보와 통합 (중복 제거)
        existing_ids = {c.get('id') for c in existing_candidates}
        
        # 새로운 후보 추가
        for hyde_candidate in hyde_candidates:
            if hyde_candidate.get('id') not in existing_ids:
                # HyDE 점수를 기존 후보 형식에 맞게 변환
                hyde_candidate['semantic_score'] = hyde_candidate.pop('hyde_score', 0.0)
                hyde_candidate['keyword_score'] = 0.0  # HyDE 후보는 키워드 점수 없음
                hyde_candidate['graph_score'] = 0.0
                hyde_candidate['final_score'] = hyde_candidate['semantic_score'] * 0.5  # 의미 점수만 반영
                existing_candidates.append(hyde_candidate)
        
        # 최종 점수 기준 재정렬
        existing_candidates.sort(key=lambda x: x.get('final_score', 0.0), reverse=True)
        return existing_candidates[:top_k]

