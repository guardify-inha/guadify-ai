"""설정 관리 모듈"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # API Keys
    openai_api_key: str
    llm_model: str = "gpt-4"
    
    # Embedding Model Settings
    # 한국어 법률 문서에 최적화된 모델 선택
    # 옵션: "text-embedding-ada-002" (OpenAI), "sentence-transformers" (로컬 한국어 모델)
    embedding_provider: str = "sentence-transformers"  # "openai" 또는 "sentence-transformers"
    embedding_model: str = "jhgan/ko-sroberta-multitask"  # 한국어 법률 문서에 적합한 모델
    # OpenAI 사용 시: "text-embedding-ada-002" 또는 "text-embedding-3-small" (더 저렴하고 성능 좋음)
    # Sentence-transformers 사용 시: "jhgan/ko-sroberta-multitask" (한국어 최적화)
    
    # Vector Store Paths
    vector_store_path: str = "./vector_stores"
    legal_precedent_store_path: str = "./vector_stores/legal_precedent"
    legal_dictionary_store_path: str = "./vector_stores/legal_dictionary"
    
    # Data Paths
    data_path: str = "./data"
    legal_docs_path: str = "./data/legal_docs"
    dictionary_data_path: str = "./data/dictionary"
    
    # Chunking Settings
    # 법률 문서는 더 긴 맥락이 필요하므로 청크 크기 증가
    chunk_size: int = 1500  # 1000 -> 1500 (법률 조문의 완전한 맥락 보존)
    chunk_overlap: int = 300  # 200 -> 300 (더 많은 오버랩으로 문맥 유지)
    
    # Retrieval Settings
    # Reranking을 위해 초기 검색 개수 증가
    top_k_retrieval: int = 5  # 최종 반환 개수
    initial_search_k: int = 20  # Reranking 전 초기 검색 개수 (더 많이 검색 후 필터링)
    
    # Reranking Settings
    rerank_enabled: bool = True
    rerank_threshold: float = 0.0  # 관련성 점수 threshold
    rerank_model: str = "Dongjin-kr/ko-reranker"  # 한국어 reranking 모델
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()


