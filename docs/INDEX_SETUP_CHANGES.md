# 벡터 인덱스 설정 업데이트 가이드

## 📋 변경 사항 요약

### setup_vector_indexes.py → setup_vector_indexes_v2.py

| 구분 | 기존 (v1) | 신규 (v2) |
|------|----------|----------|
| **임베딩 속성** | `v.embedding` | `v.embedding_violation` ⭐ |
| **임베딩 차원** | 768 | 1024 (bge-m3) ⭐ |
| **벡터 인덱스 개수** | 1개 | 2개 ⭐ |
| **corrected 인덱스** | ❌ 없음 | ✅ 추가 |
| **LawArticle 인덱스** | ✅ 있음 | ❌ 제거 (불필요) |
| **ViolationType 인덱스** | ✅ 있음 | ❌ 제거 (불필요) |
| **차원 자동 감지** | ❌ 없음 | ✅ 추가 |
| **CLI 인자** | ❌ 없음 | ✅ 추가 |

---

## 🔄 마이그레이션 가이드

### Step 1: 기존 인덱스 확인

```bash
# 현재 인덱스 상태 확인
python3 scripts/setup_vector_indexes_v2.py
```

**출력 예시:**
```
🔍 임베딩 차원 확인
현재 임베딩 차원:
   - embedding_violation: 384차원
   - embedding_corrected: 384차원

⚠️  384차원 (이전 모델)
   권장: Fine-tuned bge-m3 (1024차원)으로 업그레이드
```

### Step 2: Fine-tuning 완료 후 그래프 재구성

```bash
# 1. 모델 파인튜닝
python3 scripts/train_model.py

# 2. 그래프 재구성 (1024차원 임베딩 생성)
python3 scripts/rebuild_graph.py
```

### Step 3: 벡터 인덱스 재생성

```bash
# 기존 인덱스 삭제하고 1024차원으로 재생성
python3 scripts/setup_vector_indexes_v2.py --dim 1024 --drop
```

---

## 📊 상세 비교

### 1. 벡터 인덱스

#### 기존 (v1)
```python
# 1개 인덱스만
CREATE VECTOR INDEX violation_embeddings
FOR (v:ViolationCase)
ON v.embedding  # ❌ 속성명 변경됨
OPTIONS {
    indexConfig: {
        `vector.dimensions`: 768,  # ❌ 차원 변경됨
        `vector.similarity_function`: 'cosine'
    }
}
```

#### 신규 (v2)
```python
# 인덱스 1: 위반 문장 검색용
CREATE VECTOR INDEX violation_embeddings
FOR (v:ViolationCase)
ON v.embedding_violation  # ✅ 새 속성명
OPTIONS {
    indexConfig: {
        `vector.dimensions`: 1024,  # ✅ bge-m3 차원
        `vector.similarity_function`: 'cosine'
    }
}

# 인덱스 2: 준수 문장 검색용 (신규!)
CREATE VECTOR INDEX corrected_embeddings
FOR (v:ViolationCase)
ON v.embedding_corrected  # ⭐ 새로 추가
OPTIONS {
    indexConfig: {
        `vector.dimensions`: 1024,
        `vector.similarity_function`: 'cosine'
    }
}
```

### 2. 차원 자동 감지

#### 신규 기능 (v2)
```python
def check_embedding_dimensions(self):
    """현재 DB의 임베딩 차원 자동 확인"""
    query = """
    MATCH (v:ViolationCase)
    WHERE v.embedding_violation IS NOT NULL
    RETURN size(v.embedding_violation) as violation_dim,
           size(v.embedding_corrected) as corrected_dim
    LIMIT 1
    """
    # 자동으로 차원 감지하고 경고 출력
```

**장점:**
- 차원 불일치 자동 감지
- 사용자에게 경고 및 권장사항 제공
- 실수로 잘못된 차원으로 인덱스 생성 방지

### 3. CLI 인터페이스

#### 기존 (v1)
```bash
# 고정된 설정만 사용
python3 scripts/setup_vector_indexes.py
```

#### 신규 (v2)
```bash
# 유연한 설정 가능
python3 scripts/setup_vector_indexes_v2.py --dim 1024 --drop

# 옵션:
#   --dim: 임베딩 차원 지정
#   --drop: 기존 인덱스 삭제 후 재생성
```

---

## 🚨 주의사항

### 1. 속성명 변경

| 기존 | 신규 |
|------|------|
| `v.embedding` | `v.embedding_violation` |
| (없음) | `v.embedding_corrected` ⭐ |

**영향받는 코드:**
- `rag/hybrid_graphrag.py`
- `judge/graphrag_judge.py`
- Cypher 쿼리를 사용하는 모든 스크립트

### 2. 차원 변경

| 모델 | 차원 |
|------|------|
| paraphrase-multilingual-MiniLM-L12-v2 | 384 |
| 이전 설정 (setup_vector_indexes.py) | 768 |
| **BAAI/bge-m3 (신규)** | **1024** ⭐ |

**주의:**
- 차원이 다르면 벡터 검색 불가능!
- 반드시 모델과 인덱스 차원을 일치시켜야 함

### 3. 인덱스 재생성 시 검색 불가

```bash
# 인덱스 삭제 → 재생성 중에는 벡터 검색 불가!
python3 scripts/setup_vector_indexes_v2.py --drop
```

**해결:**
- 서비스 중단 시간 계획
- 또는 블루-그린 배포 방식 사용

---

## ✅ 체크리스트

### 마이그레이션 전

- [ ] 현재 임베딩 차원 확인
- [ ] 기존 인덱스 목록 확인
- [ ] Neo4j 백업 완료

### 마이그레이션 중

- [ ] train_model.py 실행 완료
- [ ] rebuild_graph.py 실행 완료
- [ ] setup_vector_indexes_v2.py 실행 완료

### 마이그레이션 후

- [ ] 벡터 인덱스 2개 생성 확인 (violation_embeddings, corrected_embeddings)
- [ ] 인덱스 상태 ONLINE 확인
- [ ] 테스트 실행 (`python3 scripts/test_input_csv.py`)
- [ ] 성능 확인 (정확도, 속도)

---

## 📝 실행 예시

### 시나리오 1: 처음 설정 (1024차원)

```bash
# 1. 그래프 생성 (rebuild_graph.py 이미 실행 완료)
# 2. 벡터 인덱스 생성
python3 scripts/setup_vector_indexes_v2.py --dim 1024

# 출력:
# 🔍 임베딩 차원 확인
# 현재 임베딩 차원:
#    - embedding_violation: 1024차원
#    - embedding_corrected: 1024차원
# ✅ 1024차원 (bge-m3 모델)
#
# 1️⃣  violation_embeddings 인덱스 생성...
#    ✅ violation_embeddings 생성 완료
#
# 2️⃣  corrected_embeddings 인덱스 생성... ⭐ 신규
#    ✅ corrected_embeddings 생성 완료
```

### 시나리오 2: 기존 인덱스 삭제 후 재생성

```bash
python3 scripts/setup_vector_indexes_v2.py --dim 1024 --drop

# 출력:
# 🗑️  기존 벡터 인덱스 삭제 중...
#    ✅ violation_embeddings 삭제 완료
#    ✅ corrected_embeddings 삭제 완료
#
# 1️⃣  violation_embeddings 인덱스 생성...
#    ✅ violation_embeddings 생성 완료
# ...
```

### 시나리오 3: 차원 불일치 감지

```bash
python3 scripts/setup_vector_indexes_v2.py --dim 1024

# 출력:
# 🔍 임베딩 차원 확인
# 현재 임베딩 차원:
#    - embedding_violation: 384차원  ← DB 차원
#    - embedding_corrected: 384차원
#
# ⚠️  경고: 현재 DB 차원(384)과 설정 차원(1024)이 다릅니다!
#    인덱스를 384차원으로 생성하는 것을 권장합니다.
#
#    1024차원으로 계속 진행하시겠습니까? (y/n): n
# ❌ 작업 취소
```

---

## 🎯 다음 단계

1. **기존 setup_vector_indexes.py 확인**
   ```bash
   cat scripts/setup_vector_indexes.py
   ```

2. **v2로 교체**
   ```bash
   # 백업
   mv scripts/setup_vector_indexes.py scripts/setup_vector_indexes_v1_backup.py

   # 새 버전 사용
   mv scripts/setup_vector_indexes_v2.py scripts/setup_vector_indexes.py
   ```

3. **인덱스 재생성**
   ```bash
   python3 scripts/setup_vector_indexes.py --dim 1024 --drop
   ```

4. **테스트**
   ```bash
   python3 scripts/test_input_csv.py
   ```

---

**작성일**: 2025-01-23
**버전**: 2.0
