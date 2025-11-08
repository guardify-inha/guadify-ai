# 사용 가이드

## 초기 설정

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 다음 내용을 입력하세요:

```env
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-4
EMBEDDING_MODEL=text-embedding-ada-002
```

### 3. 지식 베이스 구축

#### 법률 선례 벡터 스토어 생성

```bash
python scripts/build_legal_precedent_store.py
```

이 스크립트는:

- `data/legal_docs/` 디렉토리의 법률 문서를 로드
- 텍스트를 청크로 분할
- 임베딩 생성 및 FAISS 벡터 스토어에 저장
- `vector_stores/legal_precedent/`에 저장

**참고**: 실제 사용 시에는 다음 파일들을 `data/legal_docs/`에 준비하세요:

- 약관법 전문
- 관련 민법/상법 조항
- 공정거래위원회 표준약관
- 불공정 약관 시정 사례
- 관련 판례

#### 법률 용어 사전 벡터 스토어 생성

```bash
python scripts/build_legal_dictionary_store.py
```

이 스크립트는:

- `data/dictionary/legal_terms.json` 파일을 로드
- 각 용어-설명 쌍을 임베딩하여 벡터 스토어에 저장
- `vector_stores/legal_dictionary/`에 저장

**참고**: `data/dictionary/legal_terms.json` 파일 형식:

```json
[
  {
    "term": "용어",
    "explanation": "쉬운 설명"
  },
  ...
]
```

## 서버 실행

```bash
uvicorn main:app --reload
```

서버가 `http://localhost:8000`에서 실행됩니다.

## API 사용

### 1. 텍스트 직접 전송

**방법 1: 한 줄로 작성 (zsh/bash 모두 호환)**

```bash
curl -X POST "http://localhost:8000/analyze/text" \
  -H "Content-Type: application/json" \
  -d '{"text": "제1조 (목적) 본 계약은 서비스 이용에 관한 사항을 정함을 목적으로 한다. 제2조 회사는 어떠한 경우에도 손해배상 책임을 지지 않는다."}'
```

**방법 2: 큰따옴표 사용 (zsh 권장)**

```bash
curl -X POST "http://localhost:8000/analyze/text" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"제1조 (목적) 본 계약은 서비스 이용에 관한 사항을 정함을 목적으로 한다. 제2조 회사는 어떠한 경우에도 손해배상 책임을 지지 않는다.\"}"
```

**방법 3: JSON 파일 사용 (가장 안전)**

```bash
# request.json 파일 생성
echo '{"text": "제1조 (목적) 본 계약은 서비스 이용에 관한 사항을 정함을 목적으로 한다. 제2조 회사는 어떠한 경우에도 손해배상 책임을 지지 않는다."}' > request.json

# curl 실행
curl -X POST "http://localhost:8000/analyze/text" \
  -H "Content-Type: application/json" \
  -d @request.json
```

### 2. 파일 업로드

```bash
# PDF 파일
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@contract.pdf"

# TXT 파일
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@contract.txt"

# DOCX 파일
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@contract.docx"
```

### 3. Python 클라이언트 예제

```python
import requests

# 텍스트 분석
response = requests.post(
    "http://localhost:8000/analyze/text",
    json={"text": "계약서 텍스트..."}
)
result = response.json()

# 파일 업로드
with open("contract.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze",
        files={"file": f}
    )
result = response.json()
```

## 응답 형식

```json
{
  "overall_risk_assessment": "높음",
  "summary": "총 3개의 불공정 소지가 있는 조항이 발견되었습니다...",
  "clauses": [
    {
      "original_clause": "제2조 회사는 어떠한 경우에도 손해배상 책임을 지지 않는다.",
      "analysis": {
        "is_unfair": true,
        "reason": "고객에게 부당하게 불리한 조항이며, 약관법 제6조 위반 소지가 있습니다.",
        "evidence_law": "약관법 제6조 (일반원칙)"
      },
      "easy_translation": "이 조항은 회사가 어떤 경우에도 손해를 배상하지 않겠다는 뜻입니다...",
      "suggestion": "대안: '회사의 고의 또는 중과실로 인한 손해에 대해서는 배상 책임을 집니다.'"
    }
  ]
}
```

## 프로젝트 구조

```
EEC3100_new/
├── main.py                          # FastAPI 애플리케이션
├── config.py                        # 설정 관리
├── chains/                          # LangChain 체인
│   ├── unfair_term_detector.py     # 불공정 약관 탐지
│   ├── legal_term_translator.py    # 법률 용어 번역
│   └── full_analysis.py            # 전체 분석 파이프라인
├── utils/                           # 유틸리티
│   ├── text_splitter.py            # 텍스트 분할
│   ├── file_processor.py           # 파일 처리
│   └── risk_keywords.py            # 위험 키워드
├── scripts/                         # 구축 스크립트
│   ├── build_legal_precedent_store.py
│   └── build_legal_dictionary_store.py
├── vector_stores/                   # 벡터 스토어 (생성됨)
│   ├── legal_precedent/
│   └── legal_dictionary/
└── data/                            # 원본 데이터
    ├── legal_docs/
    └── dictionary/
```

## 문제 해결

### 벡터 스토어 로드 실패

벡터 스토어가 생성되지 않았다면:

1. `scripts/build_legal_precedent_store.py` 실행
2. `scripts/build_legal_dictionary_store.py` 실행
3. `vector_stores/` 디렉토리가 생성되었는지 확인

### API 키 오류

`.env` 파일에 올바른 `OPENAI_API_KEY`가 설정되어 있는지 확인하세요.

### 파일 처리 오류

지원하는 파일 형식: `.txt`, `.pdf`, `.docx`

- PDF: `pypdf` 라이브러리 사용
- DOCX: `python-docx` 라이브러리 사용

## 성능 최적화

1. **청크 크기 조정**: `config.py`에서 `chunk_size`, `chunk_overlap` 조정
2. **검색 개수 조정**: `config.py`에서 `top_k_retrieval` 조정
3. **LLM 모델 변경**: 더 빠른 모델 사용 (예: `gpt-3.5-turbo`)

## 확장 가능성

1. **추가 벡터 스토어**: 다른 법률 도메인 추가 가능
2. **멀티모달 지원**: 이미지 기반 계약서 분석
3. **실시간 스트리밍**: 분석 결과를 실시간으로 스트리밍
4. **사용자 피드백**: 분석 결과에 대한 사용자 피드백 수집 및 학습
