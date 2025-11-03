# Setup 디렉토리

이 디렉토리는 새로운 환경에서 프로젝트를 설정하기 위한 가이드와 스크립트를 포함합니다.

## 파일 설명

- **SETUP_GUIDE.md**: Neo4j 그래프 데이터베이스 구축 단계별 가이드

## 빠른 시작

자세한 내용은 [SETUP_GUIDE.md](./SETUP_GUIDE.md)를 참고하세요.

```bash
# 1. Neo4j 시작
docker-compose up -d

# 2. 그래프 구조 생성
python main.py

# 3. 임베딩 생성
python scripts/embed_law_articles.py

# 4. 검증
python scripts/verify_graph_structure.py
```

