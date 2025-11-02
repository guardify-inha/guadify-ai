# Graph RAG 구현 가이드

이 문서는 Neo4j Graph RAG 구현 사용법을 설명합니다.

## 구현 완료 내용

### 1. Retrieval (검색)

- ✅ Neo4j 그래프 기반 검색
- ✅ 벡터 유사도 검색 (Embedding)
- ✅ 두 가지 검색 방식:
  - **표준 방식**: 규칙 기반 조항 판별 + 고정 Cypher 쿼리
  - **Text2Cypher 방식**: LLM 기반 조항 분석 + 동적 Cypher 쿼리 생성

### 2. Augmentation (증강)

- ✅ 검색된 정보를 구조화하여 RAG 컨텍스트 구성
- ✅ 법률 조문, 위반 사례, 수정본 통합

### 3. Generation (생성)

- ✅ LLM 기반 설명 생성
- ✅ LLM 기반 수정 제안 생성

## 환경 설정

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 환경변수 설정 (.env 파일)

```bash
# Neo4j 설정
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# LLM 설정 (OpenAI 사용 시)
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo
OPENAI_API_KEY=your_openai_api_key

# 또는 Anthropic 사용 시
# LLM_PROVIDER=anthropic
# LLM_MODEL=claude-3-sonnet-20240229
# ANTHROPIC_API_KEY=your_anthropic_api_key

# RAG 설정
USE_TEXT2CYPHER=false  # true로 설정하면 Text2Cypher 방식 사용
COMPARE_METHODS=false  # true로 설정하면 두 방식 모두 실행하여 비교
```

## 사용 방법

### 기본 사용 (표준 방식)

```python
from scripts.judge_clause import run

result = run("회사는 어떠한 경우에도 책임을 지지 않습니다")
print(result["explanation"])
print(result["suggestion"])
```

### Text2Cypher 방식 사용

환경변수에서 설정:

```bash
USE_TEXT2CYPHER=true
```

또는 코드에서:

```python
from scripts.judge_clause import comprehensive_judgment
from database.neo4j_connector import Neo4jConnector

conn = Neo4jConnector()
result = comprehensive_judgment(
    "회사는 어떠한 경우에도 책임을 지지 않습니다",
    conn,
    use_text2cypher=True
)
```

### 두 방식 비교 모드

환경변수에서 설정:

```bash
COMPARE_METHODS=true
```

비교 모드에서는 두 방식 모두 실행하여 결과를 비교합니다:

```json
{
  "comparison": {
    "standard": {
      "violation": true,
      "score": 0.85,
      "explanation": "...",
      "cases_found": 10
    },
    "text2cypher": {
      "violation": true,
      "score": 0.87,
      "explanation": "...",
      "cases_found": 12,
      "llm_confidence": 0.9
    },
    "differences": {
      "score_diff": 0.02,
      "violation_match": true,
      "cases_diff": 2
    }
  }
}
```

## 결과 구조

```python
{
    "violation": bool,           # 위반 여부
    "score": float,              # 불공정도 점수 (0.0-1.0)
    "severity": str,             # 심각도 ("높음", "중간", "낮음")
    "article_id": str,           # 관련 조항 ID (예: "제7조")
    "explanation": str,          # LLM 생성 설명
    "suggestion": str,           # LLM 생성 수정 제안
    "top_reasons": [            # 근거 목록
        {
            "level": str,
            "id": str,
            "article_id": str,
            "snippet": str,
            "score": float
        }
    ],
    "method": str,               # "standard" 또는 "text2cypher" 또는 "compare"
    "debug": {                   # 디버그 정보
        "base_similarity": float,
        "contrastive": float,
        "unfairness": float,
        "cases_found": int,
        ...
    }
}
```

## 두 방식의 차이점

### 표준 방식 (USE_TEXT2CYPHER=false)

- **장점**: 빠름, 안정적, 비용 낮음
- **단점**: 유연성 낮음, 복잡한 쿼리 불가
- **사용 시기**: 대부분의 경우

### Text2Cypher 방식 (USE_TEXT2CYPHER=true)

- **장점**: 유연함, 복잡한 검색 조건 처리 가능, LLM 기반 조항 분석
- **단점**: 느림, LLM 비용, 쿼리 오류 가능성
- **사용 시기**: 복잡한 검색이 필요하거나 더 정확한 조항 판별이 필요한 경우

## LLM 실패 시 폴백

모든 LLM 호출은 실패 시 자동으로 기본 방식(규칙 기반)으로 폴백됩니다:

- Text2Cypher 실패 → 표준 방식으로 폴백
- 설명 생성 실패 → 기본 설명 반환
- 제안 생성 실패 → 기본 제안 반환

## 테스트

```bash
# 기본 테스트
python scripts/judge_clause.py "회사는 어떠한 경우에도 책임을 지지 않습니다"

# Streamlit 앱으로 테스트
streamlit run scripts/app.py
```

## 주의사항

1. LLM API 키가 설정되어 있어야 합니다
2. Neo4j가 실행 중이어야 합니다
3. 데이터베이스에 위반 사례가 로드되어 있어야 합니다
4. Text2Cypher는 추가 LLM 호출이 필요하므로 비용이 더 듭니다
