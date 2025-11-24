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
    # 임베딩 모델 설정
    # =========================================================================
    # 환경변수로 모델 전환:
    # - 기본: 'moksil/bge-m3-korean-contract-finetuned' (파인튜닝 모델)
    # - 베이스 모델: 'BAAI/bge-m3'
    # - 로컬 모델: './my_fine_tuned_model'
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "moksil/bge-m3-korean-contract-finetuned")
    VECTOR_DIMENSION = 1024  # bge-m3 차원

    # =========================================================================
    # 벡터 인덱스 설정 (이중 임베딩)
    # =========================================================================
    VECTOR_INDEX_VIOLATION = "violation_embeddings"
    VECTOR_INDEX_CORRECTED = "corrected_embeddings"

    EMBEDDING_PROPERTY_VIOLATION = "embedding_violation"
    EMBEDDING_PROPERTY_CORRECTED = "embedding_corrected"

    # =========================================================================
    # GraphRAG 설정
    # =========================================================================
    TEMPERATURE = 0.5  # Prototypical Networks temperature

    SCORE_WEIGHTS = {
        'unfair': 0.35,
        'relative': 0.50,
        'pattern_json': 0.15
    }

    THRESHOLDS = {
        'high_risk': 0.85,
        'medium_risk': 0.80,
        'low_risk': 0.75
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
