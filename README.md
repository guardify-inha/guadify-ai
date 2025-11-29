법률중심 그래프 사용

# Step 1: 환경 설정

pip install -r requirements.txt

# Step 2: Neo4j 실행

docker run -d --name neo4j-graphrag \
 -p 7474:7474 -p 7687:7687 \
 -e NEO4J_AUTH=neo4j/testpassword123 \
 -e NEO4J_PLUGINS='["apoc"]' \
 neo4j:5.14.0

# Step 3: 법률 그래프 먼저 구축 (Law-Centric의 핵심!)

python main.py

# Step 4: 벡터 인덱스 생성

python scripts/setup_vector_indexes.py

# Step 5: GraphRAG 그래프 구축 (이게 핵심!)

python -c '
from pipeline.graph*builder import GraphRAGBuilder
from database.neo4j_connector import Neo4jConnector
conn = Neo4jConnector()
builder = GraphRAGBuilder(conn)
builder.build_from_multiple_csv([
"data/contracts/reference/보도자료*데이터*전처리*최종.csv",
"data/contracts/reference/ai.csv"
])
conn.close()
print("✅ Law-Centric GraphRAG 구축 완료!")
'

# Step 6: 테스트

streamlit run scripts/app.py

---

## 🔄 파인튜닝 적용 버전 실행 순서

### 전체 워크플로우

```
1. 법률 그래프 구축
   ↓
2. 베이스 모델로 ViolationCase 그래프 구축
   ↓
3. 파인튜닝 실행
   ↓
4. 파인튜닝 모델로 임베딩 업데이트
   ↓
5. 테스트 및 검증
```

### 상세 실행 순서

#### Step 1: 법률 그래프 구축

```bash
python main.py
```

- 법률 구조(법률 → 조 → 항 → 호) 구축

#### Step 2: 베이스 모델로 ViolationCase 그래프 구축

```bash
python scripts/build_graph_base.py
```

- 베이스 모델(`BAAI/bge-m3`)로 이중 임베딩 생성
- ViolationCase 노드 생성
- 법률 관계(VIOLATES) 생성
- 유사도 관계(SIMILAR_TO) 생성

#### Step 3: 파인튜닝 실행

```bash
python scripts/train_model.py
```

- Neo4j에서 데이터 로드
- Contrastive Learning으로 베이스 모델 파인튜닝
- `./my_fine_tuned_model`에 저장

#### Step 4: 파인튜닝 모델로 임베딩 업데이트

```bash
python scripts/update_embeddings.py
```

- 기존 ViolationCase 노드의 임베딩만 재계산
- `embedding_violation`, `embedding_corrected` 업데이트
- `SIMILAR_TO` 관계 재생성
- 벡터 인덱스 재생성

#### Step 5: 환경 설정 업데이트

```bash
# .env 파일 수정
EMBEDDING_MODEL=./my_fine_tuned_model
```

#### Step 6: 테스트 및 검증

```bash
# 혼동행렬 테스트 (학습 데이터)
python scripts/test_ai_csv.py

# 혼동행렬 테스트 (테스트 데이터)
python scripts/test_test_input.py

# 조항별 테스트
python scripts/test_with_test_input.py
```

### ⚠️ 주의사항

1. **순서 준수**: 반드시 위 순서대로 실행해야 합니다.
2. **베이스 모델 구축**: `build_graph_base.py`는 처음 한 번만 실행하면 됩니다.
3. **임베딩 업데이트**: `update_embeddings.py`는 파인튜닝 후에만 실행합니다.
4. **데이터 백업**: 중요한 데이터는 백업 후 진행하세요.

### 🔄 재학습 시나리오

베이스 모델로 다시 시작하려면:

```bash
# 1. 베이스 모델로 그래프 재구축
python scripts/build_graph_base.py

# 2. 파인튜닝
python scripts/train_model.py

# 3. 임베딩 업데이트
python scripts/update_embeddings.py
```
