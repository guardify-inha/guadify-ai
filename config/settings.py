"""
프로젝트 설정
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Neo4j
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    
    # LLM
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")
    
    # Embedding
    EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    VECTOR_DIMENSION = 768
    
    # 법률
    LAW_NAME = "약관의 규제에 관한 법률"

settings = Settings()