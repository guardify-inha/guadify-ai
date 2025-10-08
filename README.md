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
- **db/**: Neo4j 그래프 데이터베이스 (Graph RAG)
  - **neo4j_client.py**: Neo4j 연결 및 CRUD 작업
  - **entity_extractor.py**: 법률 엔티티 추출
  - **graph_builder.py**: 법률 지식 그래프 구축
  - **graph_retriever.py**: Graph RAG 검색

## 🚀 빠른 시작

### 🆕 새로운 환경에서 처음 실행하기

```bash
# 1. 프로젝트 클론
git clone <repository-url>
cd guadify-ai

# 2. 가상환경 설정
python3 -m venv venv
source venv/bin/activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 환경 변수 설정
cp env.example .env
# .env 파일에서 OPENAI_API_KEY 설정

# 5. Docker 설치 및 Neo4j 실행
# macOS
brew install --cask docker
open -a Docker

# Windows (PowerShell)
# Docker Desktop 다운로드: https://www.docker.com/products/docker-desktop/
# 설치 후 Docker Desktop 실행

# Linux (Ubuntu/Debian)
# sudo apt update && sudo apt install docker.io
# sudo systemctl start docker

# Neo4j 컨테이너 실행 (모든 OS 공통)
docker run -d --name neo4j-guardify -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/password -e NEO4J_PLUGINS='["apoc"]' neo4j:latest

# 6. AI 탐지 시스템 실행
# macOS/Linux
./run_ai_detection.sh

# Windows (PowerShell)
# .\run_ai_detection.sh
# 또는
# bash run_ai_detection.sh
```

## 🚀 상세 설정 가이드

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
# 환경 변수 파일 복사
cp env.example .env

# .env 파일 편집하여 API 키 설정
# NEO4J_URI=bolt://localhost:7687
# NEO4J_USERNAME=neo4j
# NEO4J_PASSWORD=password
# OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Docker 및 Neo4j 데이터베이스 설정

#### Docker 설치 (필요한 경우)

```bash
# macOS (Homebrew 사용)
brew install --cask docker
open -a Docker

# Windows
# 1. Docker Desktop 다운로드: https://www.docker.com/products/docker-desktop/
# 2. 설치 후 Docker Desktop 실행
# 3. PowerShell에서 확인
docker --version

# Linux (Ubuntu/Debian)
sudo apt update && sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker

# Docker 실행 확인 (모든 OS 공통)
docker --version
```

#### Neo4j 컨테이너 실행

```bash
# Neo4j Docker 컨테이너 실행 (백그라운드)
docker run -d \
  --name neo4j-guardify \
  -p 7687:7687 \
  -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/password \
  -e NEO4J_PLUGINS='["apoc"]' \
  neo4j:latest

# 컨테이너 상태 확인
docker ps

# Neo4j 웹 인터페이스 접속
# http://localhost:7474 (neo4j/password)
```

#### Neo4j 컨테이너 관리

```bash
# 컨테이너 중지
docker stop neo4j-guardify

# 컨테이너 재시작
docker start neo4j-guardify

# 컨테이너 삭제 (데이터 손실 주의)
docker rm neo4j-guardify
```

### 4. 시스템 실행

#### AI 기반 불공정 조항 탐지 (추천)

```bash
# AI 탐지 시스템 실행 (가장 최신 기능)
./run_ai_detection.sh

# 다른 계약서로 테스트
./run_ai_detection.sh your_contract.txt

# 결과 확인
# results/analysis_YYYYMMDD_HHMMSS/ 폴더에서
# - ai_detection_result.json (원본 JSON)
# - analysis_report.md (마크다운 보고서)
```

#### 전체 파이프라인 실행 (기존 Vector RAG)

```bash
# 전체 파이프라인 실행
./run_demo.sh

# 또는 단계별 실행
python scripts/extract_and_chunk.py      # 1단계: 문서 전처리
python scripts/embed_and_index.py        # 2단계: 벡터 임베딩
python scripts/build_graph_rag.py --action build  # 3단계: Graph RAG 구축
python scripts/query_and_extract.py --file test_inputs/sample_contract.txt --openai  # 4단계: 조항 추출
python scripts/query_pipeline.py outputs/query_results.json  # 5단계: 위험도 평가
```

### 5. Graph RAG 테스트

```bash
# Graph RAG 시스템 테스트
python test_graph_rag.py

# Graph RAG 검색 테스트
python scripts/build_graph_rag.py --action search --query "제1조"
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
