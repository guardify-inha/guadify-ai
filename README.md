# Guadify AI - 불공정 약관 판단 시스템

GraphRAG 기반 불공정 약관 자동 판단 및 수정 제안 시스템

## 초기 구축 순서

처음부터 시스템을 구축할 때 다음 순서대로 실행하세요:

1. `pip install -r requirements.txt`
2. `docker compose up -d`
3. `.env` 파일 설정
4. `python main.py`
5. `python scripts/rebuild_graph.py`
6. `streamlit run scripts/app.py`

## 환경 설정

### .env 파일 설정

```bash
# Neo4j 데이터베이스 연결 정보
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=testpassword123

# LLM 설정
LLM_MODEL=gpt-4
OPENAI_API_KEY=your_api_key_here

# 이중 임베딩 전략: Base 모델 (RAG 검색용)과 Finetuned 모델 (판단용)
EMBEDDING_MODEL_BASE=BAAI/bge-m3
EMBEDDING_MODEL_FINETUNED=moksil/bge-m3-korean-contract-finetuned-v2
```

## 이중 임베딩 전략

이 프로젝트는 **이중 임베딩 전략**을 사용합니다:

- **Base 모델 (BAAI/bge-m3)**: RAG 검색용 - 유사 위반 사례 검색
- **Finetuned 모델 (moksil/bge-m3-korean-contract-finetuned-v2)**: 판단용 - Prototypical Networks 기반 불공정도 계산

### Neo4j 구조

**4개 임베딩 필드**:

- `embedding_violation_base`: Base 모델로 계산한 위반 조항 임베딩
- `embedding_violation_finetuned`: Finetuned 모델로 계산한 위반 조항 임베딩
- `embedding_corrected_base`: Base 모델로 계산한 수정 조항 임베딩
- `embedding_corrected_finetuned`: Finetuned 모델로 계산한 수정 조항 임베딩

**4개 벡터 인덱스**:

- `violation_embeddings_base`
- `violation_embeddings_finetuned`
- `corrected_embeddings_base`
- `corrected_embeddings_finetuned`

## 주요 명령어

### 그래프 구축

```bash
# 법률 그래프 구축
python main.py

# ViolationCase 그래프 구축 (이중 임베딩)
python scripts/rebuild_graph.py
```

### 파인튜닝 (선택사항)

이미 파인튜닝된 모델을 사용 중이면 생략 가능합니다.

```bash
python scripts/train_model.py
```

## 주의사항

1. **순서 준수**: 그래프 구축은 반드시 순서대로 실행해야 합니다.
2. **환경변수 설정**: `.env` 파일에 `EMBEDDING_MODEL_BASE`와 `EMBEDDING_MODEL_FINETUNED`를 반드시 설정하세요.
3. **모델 일관성**: RAG 검색은 base 모델, 판단은 finetuned 모델을 사용합니다.
4. **CSV 파일 위치**: 기본 CSV 파일이 `data/contracts/reference/` 경로에 있어야 합니다.
