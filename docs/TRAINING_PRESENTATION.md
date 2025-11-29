# BAAI/bge-m3 파인튜닝 프로젝트 발표자료

## 📋 프로젝트 개요

### 목표

**한국어 불공정 약관 탐지 성능 향상**을 위한 BAAI/bge-m3 모델 파인튜닝

### 핵심 과제

- 도메인 특화 임베딩 모델 구축
- Fair/Unfair 간 적절한 거리 학습 (과적합 방지)
- Prototypical Networks 기반 판단 시스템 지원

---

## 🔍 문제점 분석

### 기존 방식의 한계

#### 1. Pair 기반 학습의 문제

```
기존: (Anchor, Positive, label=1.0) + (Anchor, Negative, label=0.0)
```

**문제점:**

- ❌ Negative를 완전히 다른 문장(유사도 0)으로 학습
- ❌ 문맥 파괴: 수정 문장과 위반 문장이 실제로는 유사한 맥락
- ❌ 과적합 발생: Fair/Unfair 거리가 비정상적으로 벌어짐
- ❌ 유사도 0.1 미만으로 비정상적으로 낮게 학습됨

#### 2. CosineSimilarityLoss의 한계

- 절대적 유사도 학습 (0 또는 1)
- 상대적 관계 학습 불가
- Hard Negative 활용 어려움

---

## 💡 개선 방안: Triplet 기반 Contrastive Learning

### 핵심 아이디어

**Triplet 구조로 상대적 거리 학습**

```
[Anchor, Positive, Hard Negative]
```

- **Anchor**: 위반 문장 (original_text)
- **Positive**: 같은 조항의 다른 위반 문장
- **Hard Negative**: 수정 문장 (corrected_text)

### MultipleNegativesRankingLoss (MNRL)

**특징:**

- ✅ 라벨 없이 상대적 거리 학습
- ✅ Anchor가 Negative보다 Positive에 더 가깝게 위치
- ✅ 문맥 보존: 실제 유사한 문장 간 관계 유지
- ✅ 과적합 방지: 극단적 분리 방지

---

## 🛠️ 구현 내용

### 1. 데이터셋 구조 변경

#### Before (Pair 방식)

```python
# Positive pair
InputExample(texts=[anchor, positive], label=1.0)
# Negative pair
InputExample(texts=[anchor, negative], label=0.0)
```

#### After (Triplet 방식)

```python
# Triplet 구조 - 라벨 없음
InputExample(texts=[anchor, positive, hard_negative])
```

**장점:**

- 3개 문장을 하나의 단위로 학습
- 상대적 관계를 함께 학습
- 문맥 보존

### 2. Loss Function 변경

```python
# Before
train_loss = losses.CosineSimilarityLoss(model)

# After
train_loss = losses.MultipleNegativesRankingLoss(model=model)
```

**MNRL의 작동 원리:**

- 배치 내에서 Anchor-Positive 쌍을 찾음
- 나머지를 Negative로 간주
- Anchor가 Positive에 가깝고 Negative에서 멀어지도록 학습

### 3. Train/Dev 분리 및 평가

```python
# 9:1 비율로 분리
split_idx = int(len(all_examples) * 0.9)
train_examples = all_examples[:split_idx]
dev_examples = all_examples[split_idx:]

# TripletEvaluator 추가
dev_evaluator = evaluation.TripletEvaluator.from_input_examples(
    dev_examples,
    name="contract-violation-dev"
)
```

**학습 설정:**

- ✅ Dev set으로 학습 중 성능 모니터링
- ✅ 500 steps마다 평가
- ✅ Best model 자동 저장

### 4. 학습 파이프라인

```python
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    evaluator=dev_evaluator,      # ✅ 추가
    evaluation_steps=500,         # ✅ 추가
    save_best_model=True,         # ✅ Best model 저장
    optimizer_params={'lr': learning_rate}
)
```

---

## 📊 학습 설정

### 하이퍼파라미터

| 항목                | 값          | 설명             |
| ------------------- | ----------- | ---------------- |
| **Base Model**      | BAAI/bge-m3 | 1024차원 임베딩  |
| **Epochs**          | 3           | 학습 에폭 수     |
| **Batch Size**      | 16          | 배치 크기        |
| **Learning Rate**   | 2e-5        | 학습률           |
| **Warmup Steps**    | 100         | Warmup 단계      |
| **Train/Dev Split** | 9:1         | 데이터 분할 비율 |

### 데이터 구성

- **데이터 소스**: Neo4j ViolationCase 노드
- **Triplet 생성**: 각 케이스당 3개 샘플
- **구조**: [anchor, positive, hard_negative]

---

## 🎯 기대 효과

### 1. 과적합 방지

**Before:**

- Fair/Unfair 거리가 과도하게 벌어짐
- 유사도 0.1 미만으로 비정상적 학습

**After:**

- 상대적 거리 학습으로 적절한 분리
- 문맥 보존으로 자연스러운 유사도

### 2. 일반화 성능 향상

- ✅ Dev set으로 검증 가능
- ✅ Best model 자동 선택
- ✅ 학습 과정 모니터링

### 3. 판단 시스템 개선

**Prototypical Networks 활용:**

- Fair/Unfair prototype 간 적절한 거리
- 더 정확한 불공정도 계산
- 극단적 확률 분포 방지

---

## 🔬 기술적 특징

### 1. Triplet 구조의 장점

```
[Anchor, Positive, Hard Negative]
```

- **문맥 보존**: 실제 유사한 문장 간 관계 유지
- **상대적 학습**: 절대적 라벨 대신 상대적 거리
- **Hard Negative**: 어려운 샘플로 더 강건한 학습

### 2. MultipleNegativesRankingLoss

**작동 방식:**

1. 배치 내에서 Anchor-Positive 쌍 식별
2. 나머지 샘플을 Negative로 간주
3. Anchor가 Positive에 가깝고 Negative에서 멀어지도록 학습

**수식:**

```
Loss = -log(exp(sim(anchor, positive)) / Σ exp(sim(anchor, negative_i)))
```

### 3. 평가 메커니즘

**TripletEvaluator:**

- Dev set에서 Triplet 정확도 측정
- Anchor-Positive 거리 < Anchor-Negative 거리 비율 계산
- 학습 중 성능 추적

---

## 📈 학습 프로세스

### 단계별 흐름

```
1. Neo4j에서 데이터 로드
   ↓
2. Triplet 생성
   - Anchor: original_text
   - Positive: 같은 조항의 다른 original_text
   - Hard Negative: corrected_text
   ↓
3. Train/Dev 분리 (9:1)
   ↓
4. 모델 로드 (BAAI/bge-m3)
   ↓
5. MultipleNegativesRankingLoss 설정
   ↓
6. TripletEvaluator 설정
   ↓
7. 학습 실행
   - 500 steps마다 평가
   - Best model 자동 저장
   ↓
8. 학습 완료
```

---

## 🎓 핵심 인사이트

### 1. 상대적 거리 학습의 중요성

- 절대적 라벨(0/1)보다 상대적 관계가 더 효과적
- 문맥을 보존하면서 학습 가능
- 과적합 위험 감소

### 2. Hard Negative의 활용

- Easy Negative: 완전히 다른 문장 (효과 낮음)
- Hard Negative: 수정 문장 (같은 맥락, 다른 의미)
- 더 강건한 모델 학습

### 3. 평가 기반 학습

- Dev set으로 실시간 성능 모니터링
- Best model 자동 선택
- 과적합 조기 감지

---

## 🚀 다음 단계

### 1. 학습 실행

```bash
python scripts/train_model.py
```

### 2. 학습 모니터링

- Dev set 평가 결과 확인
- Best model 저장 확인

### 3. 임베딩 업데이트

```bash
python scripts/update_embeddings.py
```

### 4. 성능 검증

- Judge 시스템에서 테스트
- Fair/Unfair 거리 확인
- 판단 정확도 평가

---

## 📝 요약

### 핵심 메시지

1. **Triplet 구조**: Pair 방식의 한계를 극복
2. **MNRL**: 상대적 거리 학습으로 과적합 방지
3. **평가 기반**: Dev set으로 학습 품질 보장
4. **문맥 보존**: 실제 유사도 관계 유지

### 기대 성과

- ✅ 과적합 방지
- ✅ 일반화 성능 향상
- ✅ 판단 시스템 개선
- ✅ 더 정확한 불공정도 계산

---

**작성일**: 2025-01-23
**버전**: 2.0 (Triplet 기반 MNRL)
