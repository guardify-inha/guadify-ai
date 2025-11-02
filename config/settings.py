import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Settings:
    """프로젝트 설정"""
    
    # Neo4j 설정
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    
    # 법률 정보
    LAW_NAME = "약관 규제에 관한 법률"
    
    # LLM 설정
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai, anthropic, local
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")  # gpt-3.5-turbo, gpt-4, claude-3-sonnet 등
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
    
    # RAG 설정
    USE_TEXT2CYPHER = os.getenv("USE_TEXT2CYPHER", "false").lower() == "true"  # 기본값: false
    COMPARE_METHODS = os.getenv("COMPARE_METHODS", "false").lower() == "true"  # 두 방식 비교 모드
    
settings = Settings()
