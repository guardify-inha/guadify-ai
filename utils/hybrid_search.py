"""하이브리드 검색 유틸리티"""
from typing import List, Dict, Any
from langchain.schema import Document
import re


def extract_keywords(text: str) -> List[str]:
    """
    텍스트에서 키워드 추출
    
    Args:
        text: 입력 텍스트
    
    Returns:
        키워드 리스트
    """
    # 불용어 제거 및 키워드 추출
    stopwords = {"은", "는", "이", "가", "을", "를", "에", "의", "와", "과", "도", "로", "으로", 
                 "에서", "에게", "한", "한다", "하다", "한다", "하는", "된", "되는", "될", "이다",
                 "그", "이것", "저것", "것", "수", "것이다", "것으로", "것을", "것에", "것이"}
    
    # 한글, 영문, 숫자만 추출
    words = re.findall(r'[가-힣a-zA-Z0-9]+', text)
    
    # 불용어 제거 및 길이 필터링
    keywords = [w for w in words if w not in stopwords and len(w) >= 2]
    
    return keywords


def keyword_search(query: str, documents: List[Document], top_k: int = 5) -> List[tuple[Document, float]]:
    """
    키워드 기반 검색 (간단한 TF-IDF 스타일)
    
    Args:
        query: 검색 쿼리
        documents: 검색 대상 문서 리스트
        top_k: 반환할 상위 문서 개수
    
    Returns:
        (문서, 점수) 튜플 리스트
    """
    query_keywords = set(extract_keywords(query))
    
    if not query_keywords:
        return [(doc, 0.0) for doc in documents[:top_k]]
    
    scored_docs = []
    
    for doc in documents:
        doc_keywords = set(extract_keywords(doc.page_content))
        
        # Jaccard 유사도 계산
        intersection = len(query_keywords & doc_keywords)
        union = len(query_keywords | doc_keywords)
        
        if union > 0:
            score = intersection / union
        else:
            score = 0.0
        
        # 키워드 매칭 개수도 고려
        match_count = intersection
        score = score * 0.7 + (match_count / len(query_keywords)) * 0.3
        
        scored_docs.append((doc, score))
    
    # 점수 순으로 정렬
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    return scored_docs[:top_k]


def hybrid_search(
    query: str,
    vector_results: List[Document],
    keyword_results: List[tuple[Document, float]],
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3
) -> List[Document]:
    """
    벡터 검색과 키워드 검색 결과를 결합
    
    Args:
        query: 검색 쿼리
        vector_results: 벡터 검색 결과
        keyword_results: 키워드 검색 결과 (문서, 점수) 튜플 리스트
        vector_weight: 벡터 검색 가중치
        keyword_weight: 키워드 검색 가중치
    
    Returns:
        결합된 검색 결과 문서 리스트
    """
    # 모든 문서 수집
    all_docs = {}
    
    # 벡터 검색 결과에 점수 부여 (순위 기반)
    for i, doc in enumerate(vector_results):
        doc_id = id(doc)  # 문서 고유 ID
        if doc_id not in all_docs:
            all_docs[doc_id] = {
                "doc": doc,
                "vector_score": 1.0 - (i / len(vector_results)) if vector_results else 0.0,
                "keyword_score": 0.0
            }
    
    # 키워드 검색 결과에 점수 부여
    for doc, score in keyword_results:
        doc_id = id(doc)
        if doc_id not in all_docs:
            all_docs[doc_id] = {
                "doc": doc,
                "vector_score": 0.0,
                "keyword_score": score
            }
        else:
            all_docs[doc_id]["keyword_score"] = score
    
    # 가중 평균 점수 계산
    scored_docs = []
    for doc_id, scores in all_docs.items():
        combined_score = (
            scores["vector_score"] * vector_weight +
            scores["keyword_score"] * keyword_weight
        )
        scored_docs.append((scores["doc"], combined_score))
    
    # 점수 순으로 정렬
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    return [doc for doc, score in scored_docs]


