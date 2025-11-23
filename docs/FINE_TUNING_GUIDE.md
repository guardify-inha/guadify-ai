# Fine-tuning 및 그래프 재구성 가이드

## 📋 목차
1. [개요](#개요)
2. [사전 준비](#사전-준비)
3. [실행 순서](#실행-순서)
4. [주의사항](#주의사항)
5. [Troubleshooting](#troubleshooting)

---

## 개요

기존 `paraphrase-multilingual-MiniLM-L12-v2` (384차원) 임베딩 모델을 `BAAI/bge-m3` (1024차원)로 교체하고, Neo4j 데이터를 활용하여 파인튜닝을 수행하는 전체 파이프라인입니다.

### 주요 변경사항

#### AS-IS (기존)
- **모델**: `paraphrase-multilingual-MiniLM-L12-v2`
- **임베딩 차원**: 384
- **임베딩 구조**: 단일 (original_text만)
- **벡터 인덱스**: 1개 (violation_embeddings)

#### TO-BE (변경 후)
- **모델**: `BAAI/bge-m3` (Fine-tuned)
- **임베딩 차원**: 1024
- **임베딩 구조**: 이중 (original_text + corrected_text)
- **벡터 인덱스**: 2개 (violation_embeddings + corrected_embeddings)

---

## 사전 준비

### 1. 필수 라이브러리 설치

```bash
pip install sentence-transformers==2.7.0
pip install torch>=2.0.0
pip install transformers>=4.36.0
pip install pandas numpy tqdm
```

### 2. 시스템 요구사항

- **RAM**: 최소 16GB (32GB 권장)
- **GPU**: CUDA 지원 GPU 권장 (없으면 CPU로 학습, 느림)
- **디스크**: 최소 10GB 여유 공간

### 3. Neo4j 데이터 확인

Neo4j에 다음 데이터가 있어야 합니다:
- `ViolationCase` 노드
- 필수 속성: `original_text`, `corrected_text`, `article_id`

확인 쿼리:
```cypher
MATCH (v:ViolationCase)
WHERE v.original_text IS NOT NULL
  AND v.corrected_text IS NOT NULL
  AND trim(v.corrected_text) <> ''
RETURN count(v) as total_cases
```

---

## 실행 순서

### Step 1: 모델 파인튜닝 (약 30분~2시간)

```bash
cd /Users/sungmin/25-2_prj/guadify-ai
python3 scripts/train_model.py
```

**예상 출력:**
```
🚀 BAAI/bge-m3 Fine-tuning 시작
📦 베이스 모델: BAAI/bge-m3
💾 저장 경로: ./my_fine_tuned_model
...
✅ 학습 완료!
💾 모델 저장 위치: ./my_fine_tuned_model
```

**생성 파일:**
- `./my_fine_tuned_model/` (파인튜닝된 모델)

### Step 2: 그래프 재구성 (약 10분~30분)

```bash
python3 scripts/rebuild_graph.py
```

**예상 출력:**
```
🔄 그래프 재구성 시작
🗑️  기존 데이터 삭제 중...
📊 벡터 인덱스 생성 중...
🧠 이중 임베딩 생성 중...
...
🎉 그래프 재구성 완료!
📊 총 1146개 노드 생성
```

**Neo4j 변경사항:**
- 모든 기존 노드/관계 삭제
- 새로운 노드 생성 (이중 임베딩 포함)
- 벡터 인덱스 2개 생성

### Step 3: 코드 업데이트

`rag/hybrid_graphrag.py` 수정:

```python
# 변경 전
self.local_embeddings = HuggingFaceEmbeddings(
    model_name='paraphrase-multilingual-MiniLM-L12-v2'
)

# 변경 후
from sentence_transformers import SentenceTransformer

self.local_model = SentenceTransformer('./my_fine_tuned_model')

# embed_query 메서드 래핑
class EmbeddingWrapper:
    def __init__(self, model):
        self.model = model

    def embed_query(self, text):
        return self.model.encode(text, normalize_embeddings=True)

self.local_embeddings = EmbeddingWrapper(self.local_model)
```

벡터 인덱스 이름 변경:
```python
# 변경 전
index_name="violation_embeddings"

# 변경 후 (두 인덱스 모두 사용 가능)
# 1. 위반 문장 검색용
index_name="violation_embeddings"  # embedding_violation 사용

# 2. 준수 문장 검색용 (Prototypical Networks에서 활용)
index_name="corrected_embeddings"  # embedding_corrected 사용
```

### Step 4: 테스트

```bash
python3 scripts/test_input_csv.py
```

---

## 주의사항

### 1. 학습 시간

- **GPU 사용**: 약 30분~1시간
- **CPU 사용**: 약 2시간~4시간

### 2. 메모리 부족 시

`train_model.py`에서 배치 크기 조정:

```python
# 변경 전
batch_size=16

# 변경 후 (메모리 부족 시)
batch_size=8  # 또는 4
```

### 3. 기존 데이터 백업

`rebuild_graph.py`는 **모든 Neo4j 데이터를 삭제**합니다!

백업 방법:
```bash
# Neo4j Desktop에서:
# 1. Database 정지
# 2. 우클릭 → "Dump"
# 3. 파일 저장
```

### 4. 모델 크기

`BAAI/bge-m3` 모델은 약 **2.3GB**입니다. 첫 실행 시 자동 다운로드됩니다.

---

## Troubleshooting

### 1. CUDA Out of Memory

**문제:**
```
RuntimeError: CUDA out of memory
```

**해결:**
```python
# train_model.py 수정
batch_size=8  # 또는 4
```

### 2. Neo4j 연결 실패

**문제:**
```
❌ Neo4j 연결 실패
```

**해결:**
1. Neo4j Desktop에서 데이터베이스 실행 확인
2. `.env` 파일 확인:
   ```
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   ```

### 3. 임베딩 차원 불일치

**문제:**
```
⚠️  경고: 임베딩 차원이 768차원입니다. bge-m3는 1024차원이어야 합니다.
```

**원인:**
- 잘못된 모델 로드 (다른 모델 사용)

**해결:**
```python
# rebuild_graph.py 확인
model_path='./my_fine_tuned_model'  # 경로 확인
```

### 4. 벡터 인덱스 생성 실패

**문제:**
```
Neo4j.ClientError.Schema.ConstraintCreationFailed
```

**해결:**
```cypher
-- Neo4j Browser에서 기존 인덱스 삭제
DROP INDEX violation_embeddings IF EXISTS;
DROP INDEX corrected_embeddings IF EXISTS;
```

### 5. 학습 데이터 부족

**문제:**
```
⚠️  Positive를 찾을 수 없는 케이스가 많습니다
```

**원인:**
- 같은 `article_id`를 가진 케이스가 적음

**해결:**
- 정상 동작 (자동으로 차선책 사용)
- 데이터를 더 추가하면 성능 향상

---

## 성능 벤치마크

### 예상 결과

| 지표 | 기존 (384차원) | Fine-tuned (1024차원) |
|------|----------------|----------------------|
| 정확도 | ~75% | **~85%** (예상) |
| 재현율 | ~70% | **~80%** (예상) |
| F1 Score | ~0.72 | **~0.82** (예상) |

### 성능 향상 원인

1. **더 높은 표현력**: 1024차원 > 384차원
2. **도메인 특화**: 약관 데이터로 파인튜닝
3. **Contrastive Learning**: 위반/준수 구분 강화
4. **이중 임베딩**: 준수 문장도 벡터화하여 활용

---

## FAQ

**Q1. 기존 모델로 되돌리려면?**

A: `rag/hybrid_graphrag.py`에서 모델 경로만 변경:
```python
model_name='paraphrase-multilingual-MiniLM-L12-v2'
```

그리고 기존 graph_builder.py 실행:
```bash
python3 pipeline/graph_builder.py
```

**Q2. 학습을 중단하고 다시 시작하려면?**

A: `train_model.py`는 중단 후 재시작 시 처음부터 시작합니다.
학습 완료 후 모델만 저장되므로, 중단 시 다시 전체 학습 필요.

**Q3. 다른 베이스 모델을 사용하려면?**

A: `train_model.py`에서 `base_model` 변경:
```python
base_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
```

단, 임베딩 차원이 다를 수 있으므로 `rebuild_graph.py`도 확인 필요.

---

## 참고 자료

- **BAAI/bge-m3**: https://huggingface.co/BAAI/bge-m3
- **Sentence Transformers**: https://www.sbert.net/
- **Neo4j Vector Index**: https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/

---

**작성일**: 2025-01-23
**버전**: 1.0
