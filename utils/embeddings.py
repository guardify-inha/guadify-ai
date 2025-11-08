"""임베딩 모델 유틸리티"""
from typing import Union
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import settings


def get_embeddings():
    """
    설정에 따라 적절한 임베딩 모델 반환
    
    Returns:
        Embeddings 인스턴스 (OpenAIEmbeddings 또는 HuggingFaceEmbeddings)
    """
    if settings.embedding_provider.lower() == "openai":
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key
        )
    elif settings.embedding_provider.lower() == "sentence-transformers":
        # 한국어 법률 문서에 최적화된 모델
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={'device': 'cpu'},  # GPU 사용 시 'cuda'로 변경
            encode_kwargs={'normalize_embeddings': True}  # 정규화로 성능 향상
        )
    else:
        raise ValueError(
            f"지원하지 않는 embedding_provider: {settings.embedding_provider}. "
            "사용 가능한 옵션: 'openai', 'sentence-transformers'"
        )

