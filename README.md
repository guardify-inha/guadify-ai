# 계약서 불공정 약관 분석 챗봇

RAG/LangChain 기반 계약서 불공정 약관 분석 웹 서비스 백엔드

## 기능

- 계약서 텍스트/파일 업로드 및 분석
- 불공정 약관 자동 탐지
- 법률 용어 쉬운 풀이
- 위험도 평가 및 대안 제안

## 설치

### 1. 가상환경 생성 및 활성화

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (macOS/Linux)
source venv/bin/activate

# 가상환경 활성화 (Windows)
venv\Scripts\activate
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

## 환경 설정

`.env.example`을 참고하여 `.env` 파일을 생성하고 API 키를 설정하세요.

## 사용 방법

> **참고**: 모든 명령어는 가상환경이 활성화된 상태에서 실행해야 합니다.
> 가상환경이 활성화되지 않았다면 `source venv/bin/activate` (macOS/Linux) 또는 `venv\Scripts\activate` (Windows)를 먼저 실행하세요.

### 1. 지식 베이스 구축

```bash
# 법률 선례 벡터 스토어 생성
python scripts/build_legal_precedent_store.py

# 법률 용어 사전 벡터 스토어 생성
python scripts/build_legal_dictionary_store.py
```

### 2. 서버 실행

```bash
uvicorn main:app --reload
```

### 3. API 사용

```bash
# 텍스트 분석
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "계약서 텍스트..."}'

# 파일 업로드 분석
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@contract.pdf"
```

# 계약서 불공정 약관 분석 시스템 - 상세 설명서

## 1. 프로젝트 개요

이 프로젝트는 **RAG(Retrieval-Augmented Generation)**와 **LangChain**을 활용한 계약서 불공정 약관 분석 웹 서비스입니다. 계약서를 입력받아 불공정 조항을 탐지하고, 법률 용어를 쉬운 말로 풀이하며, 위험도를 평가합니다.

---

## 2. 핵심 개념 설명

### 2.1 RAG (Retrieval-Augmented Generation)

- **의미**: 외부 지식 베이스에서 관련 정보를 검색하여 LLM에 제공하는 방식
- **이 프로젝트에서의 활용**: 법률 선례와 용어 사전을 벡터 스토어에 저장하고, 계약서 조항과 유사한 법률 정보를 검색하여 분석에 활용

### 2.2 벡터 임베딩 (Vector Embedding)

- **의미**: 텍스트를 숫자 벡터로 변환하여 의미적 유사도를 계산
- **이 프로젝트에서의 활용**: 한국어 법률 문서에 최적화된 `jhgan/ko-sroberta-multitask` 모델을 사용하여 법률 문서를 벡터화

### 2.3 하이브리드 검색 (Hybrid Search)

- **의미**: 벡터 검색과 키워드 검색을 결합하여 검색 정확도 향상
- **이 프로젝트에서의 활용**: 벡터 유사도(70%) + 키워드 매칭(30%) 결합

### 2.4 Reranking

- **의미**: 검색 결과를 관련성 점수로 재정렬
- **이 프로젝트에서의 활용**: 한국어 reranking 모델(`Dongjin-kr/ko-reranker`)을 사용하여 최종 검색 결과의 관련성 재평가

---

## 3. 데이터 가공 프로세스

### 3.1 약관법 구조화 데이터 가공

**입력 데이터**: `data/legal_docs/약관법_구조화.json`

이 파일은 약관법 제6조~제14조를 구조화된 JSON 형식으로 저장합니다:

```json
[
  {
    "article": "제6조",
    "title": "일반원칙",
    "category": "일반원칙",
    "keywords": ["일반원칙", "신의성실", "공정성", "무효"],
    "full_content": "제6조 : (일반원칙)\n제6조 제1항 : 신의성실의 원칙을 위반하여 공정성을 잃은 약관 조항은 무효이다...",
    "sub_articles": [
      {
        "type": "항",
        "number": "제1항",
        "content": "신의성실의 원칙을 위반하여 공정성을 잃은 약관 조항은 무효이다."
      }
    ],
    "priority": 0
  }
]
```

**가공 프로세스** (`scripts/build_legal_precedent_store.py`):

1. **전체 조항 저장**: 각 조항의 전체 내용을 하나의 문서로 저장
2. **하위 호 분리**: 각 하위 호(제1호, 제2호 등)도 별도 문서로 저장하여 세밀한 검색 가능
3. **메타데이터 추가**:
   - `type: "terms_act"` (전체 조항)
   - `type: "terms_act_sub"` (하위 호)
   - `article`, `title`, `category`, `keywords`, `priority` 등

**가공 결과**:

- 각 조항을 전체 내용과 하위 호로 분리하여 저장
- 메타데이터에 `type: "terms_act"` 또는 `"terms_act_sub"` 표시
- 검색 시 약관법 조항을 우선적으로 식별 가능

### 3.2 CSV 불공정 약관 사례 데이터 가공

**입력 데이터**: `data/legal_docs/ai.csv` (예시)

CSV 파일 형식:

```
ID,대주제,불공정약관원문,시정요청사유,근거조항(약관법),수정 후 약관 조항
V-001,면책조항의 금지,회사는 어떠한 피해배상도 하지 않는다.,...
```

**가공 프로세스**:

1. **CSV 행 단위 처리**: 각 행을 하나의 문서로 변환
2. **정보 결합**: 다음 정보를 하나의 문서로 결합
   - 불공정 약관 원문
   - 시정 요청 사유
   - 근거 조항(약관법)
   - 수정 후 약관 조항
   - 대주제, 중주제, 소주제
3. **메타데이터 생성**:
   - `type: "case"`
   - `case_id`, `category`, `sub_category`, `violated_article` 등

**가공 결과**:

- 각 CSV 행을 하나의 문서로 변환
- 불공정 약관 원문, 시정 사유, 근거 조항, 수정 후 조항을 결합
- 메타데이터에 `type: "case"` 표시

### 3.3 벡터 스토어 구축

**핵심 가공 원칙**:

1. **약관법 조항과 CSV 사례는 분할하지 않음**: 문맥 보존을 위해 그대로 저장
2. **기타 문서는 청킹**: 1500자 단위로 청킹 (오버랩 300자)
3. **메타데이터 포함**: 모든 문서에 메타데이터 포함 (`type`, `category`, `article` 등)
4. **한국어 최적화**: 한국어 법률 문서에 최적화된 임베딩 모델 사용

---

## 4. 불공정 약관 판단 프로세스 (상세)

불공정 약관 판단은 `chains/unfair_term_detector.py`에서 수행됩니다. 단계별로 상세히 설명합니다.

### 4.1 전체 프로세스 개요

입력: `{"clause": "계약서 조항 텍스트"}`
출력: `{"clause": "...", "is_unfair": true/false, "reason": "...", "evidence_law": "..."}`

### 4.2 단계 1: 위험 키워드 체크

**목적**: 조항에 위험 키워드가 포함되어 있는지 빠르게 확인

**위험 키워드 리스트** (`utils/risk_keywords.py`):

- "일체의 책임을 지지 않는다"
- "어떠한 경우에도 이의를 제기할 수 없다"
- "계약 해지 불가"
- "일방적 해지"
- "손해배상 책임을 지지 않는다"
- 등 30개 이상의 위험 키워드

**동작 방식**:

1. 조항 텍스트를 소문자로 변환
2. `RISK_KEYWORDS` 집합의 각 키워드가 포함되어 있는지 확인
3. 매칭된 키워드 리스트 반환

**예시**:

- 입력: "회사는 어떠한 경우에도 손해배상 책임을 지지 않는다"
- 결과: `{"has_risk_keywords": true, "matched_keywords": ["어떠한 경우에도", "손해배상 책임을 지지 않는다"]}`

### 4.3 단계 2: 법률 선례 검색 (RAG)

**목적**: 조항과 유사한 법률 선례와 약관법 조항을 검색

#### 4.3.1 1단계: 벡터 유사도 검색

**목적**: 의미적 유사도로 관련 문서 검색

**동작 방식**:

1. 입력 조항을 임베딩 벡터로 변환
2. FAISS 벡터 스토어에서 코사인 유사도로 상위 20개 문서 검색
3. Reranking을 위해 더 많이 검색 (최종 5개가 아닌 20개)

#### 4.3.2 2단계: 키워드 검색

**목적**: 키워드 매칭으로 관련 문서 보완 검색

**키워드 추출**:

1. 조항과 문서에서 키워드 추출 (불용어 제거)
2. 한글, 영문, 숫자만 추출
3. 2자 이상의 단어만 키워드로 사용

**Jaccard 유사도 계산**:

- `교집합 크기 / 합집합 크기`
- 키워드 매칭 개수도 고려: `Jaccard 점수 * 0.7 + (매칭 개수 / 전체 키워드 수) * 0.3`

#### 4.3.3 3단계: 하이브리드 검색

**목적**: 벡터 검색과 키워드 검색 결과를 가중치로 결합

**동작 방식**:

1. 벡터 검색 결과에 순위 기반 점수 부여 (1위: 1.0, 2위: 0.95, ...)
2. 키워드 검색 결과에 키워드 매칭 점수 부여
3. 가중 평균 계산: `벡터 점수 * 0.7 + 키워드 점수 * 0.3`
4. 결합 점수로 정렬

#### 4.3.4 4단계: Reranking

**목적**: CrossEncoder를 사용하여 관련성 재평가

**동작 방식**:

1. 한국어 reranking 모델(`Dongjin-kr/ko-reranker`) 로드
2. 각 쿼리-문서 쌍에 대해 관련성 점수 계산
3. 점수 순으로 재정렬
4. threshold 이상인 문서만 필터링

#### 4.3.5 5단계: 약관법 조항 우선 분리

**목적**: 약관법 조항을 우선 배치하여 LLM이 먼저 참조하도록 함

**동작 방식**:

1. 메타데이터 `type`이 `"terms_act"` 또는 `"terms_act_sub"`인 문서를 약관법 조항으로 분리
2. 약관법 조항과 기타 선례를 별도로 reranking
3. 약관법 조항 상위 5개, 기타 선례 상위 3개 선택
4. 약관법 조항을 먼저 배치하여 컨텍스트 구성

**최종 컨텍스트 형식**:

```
=== 약관법 핵심조항 (우선 참조) ===
약관법 제7조 (면책조항의 금지)
...

=== 기타 법률 선례 및 판례 ===
불공정 약관 원문: 회사는 어떠한 피해배상도 하지 않는다.
시정 요청 사유: ...
```

### 4.4 단계 3: Few-shot 예시 선택

**목적**: 유사한 판단 사례를 LLM에 제공하여 판단 정확도 향상

**Few-shot 예시 데이터** (`data/few_shot_examples.json`):

```json
[
  {
    "input_clause": "회사는 어떠한 경우에도 손해배상 책임을 지지 않는다",
    "is_unfair": true,
    "reason": "이 조항은 사업자의 모든 손해배상 책임을 배제하는 것으로, 약관법 제7조 제2호에 위배됩니다.",
    "evidence_law": "약관법 제7조",
    "category": "손해배상"
  }
]
```

**동작 방식**:

1. 입력 조항과 예시에서 키워드 추출
2. Jaccard 유사도로 유사도 계산
3. 유사도 순으로 정렬하여 상위 3개 선택

### 4.5 단계 4: LLM 분석

**목적**: 수집한 모든 정보를 바탕으로 LLM이 불공정 여부 판단

**LLM이 받는 정보**:

1. 계약서 조항 텍스트
2. 검색된 법률 선례 및 약관법 조항 (약관법 우선 배치)
3. 가장 관련된 약관법 조항 전체 내용
4. 위험 키워드 정보 (매칭된 키워드 리스트)
5. Few-shot 예시 (유사한 판단 사례 3개)

**프롬프트 핵심 원칙**:

1. 약관법 핵심조항(제6조~제14조)을 가장 우선적으로 참조
2. 약관법 제6조는 매우 일반적이므로, 더 구체적인 조항이 있으면 그것을 우선
3. 약관법 제6조는 다른 구체적인 조항이 없을 때만 사용
4. 약관법 조항이 없을 경우에만 다른 법률(민법, 상법 등) 참조

**LLM 출력 형식**:

```json
{
  "is_unfair": true,
  "reason": "이 조항은 사업자의 모든 손해배상 책임을 배제하는 것으로, 약관법 제7조 제2호에 위배됩니다.",
  "evidence_law": "약관법 제7조"
}
```

### 4.6 단계 5: JSON 파싱 및 결과 반환

**목적**: LLM 출력을 JSON으로 파싱하고 최종 결과 구성

**동작 방식**:

1. LLM 출력에서 JSON 코드 블록 제거
2. JSON 파싱
3. 원본 조항과 근거 법률 전체 내용 추가
4. 파싱 실패 시 기본값 반환

---

## 5. 전체 프로세스 요약

### 5.1 데이터 가공 단계

1. 약관법 구조화 JSON 로드 → 전체 조항 + 하위 호로 분리
2. CSV 불공정 약관 사례 로드 → 각 행을 문서로 변환
3. 벡터 임베딩 생성 → 한국어 법률 문서 최적화 모델 사용
4. FAISS 벡터 스토어 저장 → 메타데이터 포함

### 5.2 불공정 약관 판단 단계

1. **위험 키워드 체크** → 빠른 1차 필터링
2. **법률 선례 검색 (RAG)**:
   - 벡터 유사도 검색 (20개)
   - 키워드 검색 (10개)
   - 하이브리드 결합 (가중치 0.7:0.3)
   - Reranking (한국어 모델)
   - 약관법 조항 우선 분리
3. **Few-shot 예시 선택** → 유사한 판단 사례 3개
4. **LLM 분석** → GPT-4로 최종 판단
5. **JSON 파싱** → 구조화된 결과 반환

---

## 6. 핵심 기술 요약

1. **하이브리드 검색**: 벡터 검색(70%) + 키워드 검색(30%) 결합
2. **Reranking**: 한국어 CrossEncoder를 사용한 관련성 재평가
3. **약관법 우선 배치**: 메타데이터를 활용한 약관법 조항 식별 및 우선 배치
4. **Few-shot Learning**: 유사한 판단 사례를 LLM에 제공
5. **구조화된 프롬프트**: 구체적인 조항 우선 참조 원칙 명시

이러한 프로세스를 통해 계약서 조항의 불공정 여부를 정확하게 판단합니다.

---

## 7. 주요 파일 구조

### 7.1 핵심 파일

- `chains/unfair_term_detector.py`: 불공정 약관 탐지 메인 로직
- `chains/full_analysis.py`: 전체 분석 파이프라인
- `chains/legal_term_translator.py`: 법률 용어 번역
- `scripts/build_legal_precedent_store.py`: 벡터 스토어 구축
- `utils/hybrid_search.py`: 하이브리드 검색 구현
- `utils/risk_keywords.py`: 위험 키워드 정의

### 7.2 데이터 파일

- `data/legal_docs/약관법_구조화.json`: 구조화된 약관법 조항
- `data/legal_docs/*.csv`: 불공정 약관 시정 사례
- `data/few_shot_examples.json`: Few-shot 학습 예시
- `data/dictionary/legal_terms.json`: 법률 용어 사전

### 7.3 설정 파일

- `config.py`: 애플리케이션 설정
- `.env`: 환경 변수 (API 키 등)

---

## 8. 사용 방법

### 8.1 초기 설정

```bash
# 1. 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 또는 venv\Scripts\activate  # Windows

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 환경 변수 설정 (.env 파일 생성)
OPENAI_API_KEY=your_api_key_here

# 4. 벡터 스토어 구축
python scripts/build_legal_precedent_store.py
python scripts/build_legal_dictionary_store.py
```

### 8.2 서버 실행

```bash
uvicorn main:app --reload
```

### 8.3 API 사용

```bash
# 텍스트 분석
curl -X POST "http://localhost:8000/analyze/text" \
  -H "Content-Type: application/json" \
  -d '{"text": "계약서 텍스트..."}'

# 파일 업로드
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@contract.pdf"
```

---

## 9. 기술 스택

- **백엔드**: FastAPI, LangChain
- **LLM**: OpenAI GPT-4
- **임베딩**: jhgan/ko-sroberta-multitask (한국어 최적화)
- **Reranking**: Dongjin-kr/ko-reranker (한국어 최적화)
- **벡터 스토어**: FAISS
- **파일 처리**: pypdf, python-docx

---

## 10. 참고 사항

- 이 시스템은 한국어 법률 문서에 최적화되어 있습니다
- 약관법 제6조~제14조를 우선적으로 참조합니다
- 벡터 스토어는 사전에 구축되어 있어야 합니다
- Few-shot 예시를 추가하면 판단 정확도가 향상됩니다
