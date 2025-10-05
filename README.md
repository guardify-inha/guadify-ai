# 🛡️ Guardify - AI 기반 계약서 불공정약관 탐지 서비스

## 📋 프로젝트 개요

Vector RAG + Graph RAG + Langchain 기반 시스템으로, 계약서에서 불공정한 약관을 자동으로 탐지하고 법적 근거를 제시하는 고급 AI 모델 개발 프로젝트입니다.

## 🏗️ 시스템 아키텍처

### 현재 시스템 (Vector RAG)

```
PDF 문서들 → 텍스트 청킹 → 벡터 임베딩 → FAISS 인덱싱 → 검색 & 분석
```

### 목표 시스템 (하이브리드 RAG + Langchain)

```
PDF 문서 → 다중 임베딩 → Vector RAG + Graph RAG → Langchain 통합 → 지능형 에이전트
```

### 핵심 컴포넌트

- **extract_and_chunk.py**: PDF → 텍스트 청킹
- **embed_and_index.py**: 텍스트 → 벡터 임베딩
- **query_and_extract.py**: 불공정 조항 후보 추출
- **query_pipeline.py**: 최종 위험도 평가
- **db/**: Neo4j 그래프 데이터베이스 연결 (개발 중)

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate

# 모든 의존성 설치 (requirements.txt 기반)
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
# OpenAI API 키 설정
export OPENAI_API_KEY=your_openai_api_key_here

```

### 3. 시스템 실행

```bash
# 전체 파이프라인 실행
./run_demo.sh

# 또는 단계별 실행
python scripts/extract_and_chunk.py      # 1단계: 문서 전처리
python scripts/embed_and_index.py        # 2단계: 벡터 임베딩
python scripts/query_and_extract.py --file data/contracts/user/test.txt --openai  # 3단계: 조항 추출
python scripts/query_pipeline.py outputs/query_results.json  # 4단계: 위험도 평가
```

## 📊 데이터 구성

- **Contract**: 511개 청크 (불공정 사례)
- **Standard**: 611개 청크 (표준 약관)
- **Law**: 224개 청크 (법령 문서)

## 📁 출력 파일

```
outputs/
├── analysis_YYYYMMDD_HHMMSS/     # 타임스탬프별 분석 결과
│   ├── query_results.json        # 불공정 조항 후보
│   ├── all_results.json          # 최종 위험도 평가
│   └── summary.txt               # 사람이 읽기 좋은 요약
├── chunks.jsonl                  # 청킹된 텍스트 데이터
├── faiss.index                   # FAISS 벡터 인덱스
└── faiss_meta.pkl                # FAISS 메타데이터
```

## 📦 의존성 (requirements.txt)

### 핵심 AI/ML 라이브러리

- **sentence-transformers==2.2.2**: 다중 임베딩 모델 시스템
- **faiss-cpu==1.7.4**: 벡터 검색 및 인덱싱
- **numpy==1.24.3**: 수치 계산
- **pandas==2.0.3**: 데이터 처리
- **scikit-learn==1.3.0**: 머신러닝 유틸리티

### 문서 처리

- **PyMuPDF==1.23.8**: PDF 텍스트 추출
- **pdfplumber==0.9.0**: PDF 구조 분석

### LLM 및 AI 서비스

- **openai==1.3.0**: OpenAI GPT 모델 연동
- **anthropic==0.7.0**: Claude 모델 연동

### 데이터베이스

- **neo4j==5.14.0**: Graph RAG용 그래프 데이터베이스

### 유틸리티

- **python-dotenv==1.0.0**: 환경 변수 관리
- **tqdm==4.66.1**: 진행률 표시

## 🔧 설정 옵션

### 모델 설정

- **임베딩 모델**: `all-MiniLM-L6-v2` (기본), 다중 모델 지원 예정
- **LLM 모델**: `gpt-4o-mini`
- **유사도 임계값**: `0.4`

### 청킹 설정

- **청크 크기**: 1200자
- **오버랩**: 200자
- **문서 타입별 차별화**: contract/law/standard

### Graph RAG 설정 (개발 중)

- **Neo4j URI**: `bolt://localhost:7687`
- **노드 타입**: Law, Article, LegalConcept, Precedent
- **관계 타입**: REFERENCES, HIERARCHY_OF, CONFLICTS_WITH

## 🎯 개발 로드맵

### Phase 1: 기존 시스템 고도화 ✅

1. **다중 임베딩 모델 도입** (진행 중)
2. **적응형 청킹 전략 적용** (진행 중)
3. **쿼리 확장 시스템 구축** (진행 중)

### Phase 2: Graph RAG 통합 (진행 중)

1. **Neo4j 그래프 데이터베이스 구축** ← **현재 단계**
2. **법률 엔티티 추출 모델 개발**
3. **하이브리드 검색 파이프라인 구현**

### Phase 3: Langchain 통합 (예정)

1. **Langchain 기반 통합 프레임워크 구축**
2. **지능형 에이전트 시스템 개발**
3. **CLI 애플리케이션 구축**

### Phase 4: 고도화 및 최적화 (예정)

1. **성능 평가 시스템 구축**
2. **A/B 테스트 프레임워크 구현**
3. **확장성 및 안정성 개선**
