# 약관 불공정성 판단 시스템

GraphRAG 기반으로 약관의 불공정성을 판단하는 시스템입니다.

## 📋 목차
- [시스템 요구사항](#시스템-요구사항)
- [설치 방법](#설치-방법)
- [실행 방법](#실행-방법)
- [그래프 구조](#그래프-구조)
- [다음 단계](#다음-단계)

## 🔧 시스템 요구사항

### 필수 설치 항목
1. **Python 3.8 이상**
2. **Docker Desktop** (추천!)
   - 다운로드: https://www.docker.com/products/docker-desktop/
   - Neo4j를 Docker로 실행하면 간편합니다

또는

2. **Neo4j Desktop 또는 Neo4j Server**
   - 다운로드: https://neo4j.com/download/

## 📦 설치 방법

### 방법 A: Docker 사용 (추천! 🐳)

가장 간단하고 팀원들과 공유하기 좋은 방법입니다.

```bash
# 1. Docker 실행 (프로젝트 폴더에서)
docker-compose up -d

# 2. 브라우저에서 확인
# http://localhost:7474
# Username: neo4j
# Password: testpassword123
```

**상세 가이드:** [DOCKER_GUIDE.md](DOCKER_GUIDE.md) 참고!

---

### 방법 B: Neo4j Desktop 사용

#### Neo4j Desktop 사용
1. Neo4j Desktop 다운로드 및 설치
2. 새 프로젝트 생성
3. 새 데이터베이스 생성 (예: "unfair-terms-db")
4. 데이터베이스 시작
5. 비밀번호 설정 후 `.env` 파일 수정

---

### 1. Neo4j 설치 및 설정

#### Neo4j Desktop 사용 (추천)
---

### Python 환경 설정

```bash
# 1. 프로젝트 디렉토리로 이동
cd unfair-terms-detector

# 2. 가상환경 생성 (선택사항이지만 권장)
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 3. 필요한 패키지 설치
pip install -r requirements.txt
```

---

### 환경 변수 확인

**Docker 사용시:** `.env` 파일 그대로 사용하면 됩니다!

**Neo4j Desktop 사용시:** `.env` 파일에서 비밀번호만 수정하세요:

```env
NEO4J_PASSWORD=testpassword123  # 여기에 실제 비밀번호 입력
```

## 🚀 실행 방법

### 법률 그래프 구축

```bash
python main.py
```

실행하면:
1. Neo4j 데이터베이스에 연결
2. 기존 데이터 삭제 여부 확인
3. 약관 규제 법률(제6조~제14조) 구조를 그래프로 생성

### Neo4j Browser에서 확인

1. 브라우저에서 http://localhost:7474 접속
2. 연결 정보 입력 (bolt://localhost:7687, neo4j, your_password)
3. 다음 쿼리를 실행하여 그래프 확인:

```cypher
// 모든 노드 보기 (최대 50개)
MATCH (n) RETURN n LIMIT 50

// 전체 법률 구조 보기
MATCH p=(:법률)-[*]->() RETURN p

// 특정 조의 구조 보기 (예: 제7조)
MATCH p=(:조 {id: "제7조"})-[*]->() RETURN p

// 호까지 포함된 전체 경로 보기
MATCH p=(:법률)-[:HAS_ARTICLE]->(:조)-[:HAS_HANG]->(:항)-[:HAS_HO]->(:호)
RETURN p
```

## 📊 그래프 구조

현재 구현된 구조:

```
[법률: 약관 규제에 관한 법률]
  ↓ HAS_ARTICLE
[조: 제6조 ~ 제14조]
  ↓ HAS_HANG
[항: 제1항, 제2항, ...]
  ↓ HAS_HO
[호: 제1호, 제2호, ...]
```

### 노드 유형
- **법률 노드**: 법률명 포함
- **조 노드**: 조 번호, 제목, 내용 포함
- **항 노드**: 항 번호, 내용 포함
- **호 노드**: 호 번호, 내용 포함

### 관계 유형
- `HAS_ARTICLE`: 법률 → 조
- `HAS_HANG`: 조 → 항
- `HAS_HO`: 항 → 호

## 🔍 예제 쿼리

### 1. 제7조의 모든 호 조회
```cypher
MATCH (article:조 {id: "제7조"})-[:HAS_HANG]->(hang:항)-[:HAS_HO]->(ho:호)
RETURN article.title as 조, hang.hang_num as 항, ho.ho_num as 호, ho.content as 내용
```

### 2. 호가 있는 모든 조 찾기
```cypher
MATCH (article:조)-[:HAS_HANG]->(:항)-[:HAS_HO]->(ho:호)
RETURN article.id, article.title, count(ho) as 호_개수
ORDER BY 호_개수 DESC
```

### 3. 특정 키워드를 포함하는 호 찾기
```cypher
MATCH (ho:호)
WHERE ho.content CONTAINS "고의"
RETURN ho.id, ho.content
```

## 📁 프로젝트 구조

```
unfair-terms-detector/
├── config/
│   └── settings.py          # 설정 관리
├── data/
│   └── law_structure.py     # 법률 구조 데이터
├── database/
│   ├── __init__.py
│   ├── neo4j_connector.py   # Neo4j 연결 관리
│   └── graph_builder.py     # 그래프 구축 로직
├── main.py                  # 메인 실행 파일
├── requirements.txt         # Python 패키지
├── .env                     # 환경 변수 (수정 필요!)
└── README.md
```

## 🔮 다음 단계

### Phase 1: 위반 사례 추가 (다음 작업)
- [ ] 위반 사례 노드 생성
- [ ] 사례와 법조문 연결
- [ ] 임베딩 추가

### Phase 2: 수정본 추가
- [ ] 수정본 노드 생성
- [ ] 위반 사례와 수정본 연결

### Phase 3: 판단 로직 구현
- [ ] 키워드 기반 판단
- [ ] 유사도 기반 판단
- [ ] 통합 판단 시스템

## ⚠️ 문제 해결

### Neo4j 연결 실패
```
❌ Neo4j 연결 실패: ServiceUnavailable
```
**해결방법:**
1. Neo4j가 실행 중인지 확인
2. `.env` 파일의 연결 정보가 정확한지 확인
3. 방화벽이 7687 포트를 차단하지 않는지 확인

### 패키지 설치 오류
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 📞 문의

문제가 발생하거나 질문이 있으면 이슈를 등록해주세요.
