"""
프로젝트 설정
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """전체 시스템 설정"""

    # =========================================================================
    # Neo4j 설정
    # =========================================================================
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

    # =========================================================================
    # LLM 설정
    # =========================================================================
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # =========================================================================
    # 임베딩 모델 설정 (이중 임베딩 전략)
    # =========================================================================
    # 환경변수로 모델 전환:
    # - 기본: 'moksil/bge-m3-korean-contract-finetuned' (파인튜닝 모델)
    # - 베이스 모델: 'BAAI/bge-m3'
    # - 로컬 모델: './my_fine_tuned_model'
    
    # 이중 임베딩 전략: Base 모델 (RAG 검색용)과 Finetuned 모델 (판단용)
    EMBEDDING_MODEL_BASE = os.getenv("EMBEDDING_MODEL_BASE", "BAAI/bge-m3")
    EMBEDDING_MODEL_FINETUNED = os.getenv("EMBEDDING_MODEL_FINETUNED", "moksil/bge-m3-korean-contract-finetuned-v2")
    
    VECTOR_DIMENSION = 1024  # bge-m3 차원


    # =========================================================================
    # GraphRAG 설정
    # =========================================================================
    TEMPERATURE = 0.5  # Prototypical Networks temperature

    SCORE_WEIGHTS = {
        'unfair': 0.2,
        'relative': 0.6,
        'pattern_json': 0.2
    }

    THRESHOLDS = {
        'high_risk': 0.8,
        'medium_risk': 0.7,
        'low_risk': 0.6
    }

    # =========================================================================
    # 법률 설정
    # =========================================================================
    LAW_NAME = "약관의 규제에 관한 법률"

    # =========================================================================
    # 경로 설정
    # =========================================================================
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / 'data'
    CONTRACTS_DIR = DATA_DIR / 'contracts' / 'reference'
    TEST_DIR = DATA_DIR / 'test'


settings = Settings()
