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
    
settings = Settings()
