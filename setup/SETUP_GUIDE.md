# Neo4j 그래프 데이터베이스 구축 가이드

이 가이드는 새로운 환경에서 Graph RAG 시스템을 재현하기 위한 단계별 실행 가이드입니다.

## 사전 요구사항

1. **Neo4j 데이터베이스 실행**
   ```bash
   docker-compose up -d
   ```
   - Neo4j가 `localhost:7687`에서 실행되어야 합니다
   - 인증 정보: `neo4j/testpassword123` (docker-compose.yml 참고)

2. **Python 패키지 설치**
   ```bash
   pip install -r requirements.txt
   ```

3. **환경 변수 설정**
   - `.env` 파일에 Neo4j 연결 정보 설정 (필요시)

## 실행 순서

### 1단계: 그래프 구조 생성

법률 구조(조, 항, 호)를 Neo4j 그래프로 구축합니다.

```bash
python main.py
```

**실행 시:**
- 기존 데이터 초기화 여부 확인 (y/n 입력)
- `y` 입력 시 모든 노드와 관계 삭제 후 재구축
- 법률 노드, 조/항/호 노드 및 관계 생성

**확인 방법:**
```cypher
// Neo4j Browser에서 실행
MATCH (n) RETURN count(n) as total_nodes
MATCH ()-[r]->() RETURN count(r) as total_relationships
```

### 2단계: 법률 조항 임베딩 생성

조/항/호 노드의 `content` 필드를 임베딩하여 저장합니다.

```bash
python scripts/embed_law_articles.py
```

**실행 결과:**
- 모든 조/항/호 노드에 `embedding` 프로퍼티 추가
- 임베딩 차원: 768차원 (paraphrase-multilingual-mpnet-base-v2)
- 진행 상황 및 통계 출력

**확인 방법:**
```bash
python scripts/verify_graph_structure.py
```

### 3단계: 그래프 구조 검증

생성된 그래프 구조와 임베딩이 올바르게 저장되었는지 검증합니다.

```bash
python scripts/verify_graph_structure.py
```

**확인 항목:**
- 노드 및 관계 개수
- 위반사례/수정본 노드 속성
- 임베딩 저장 여부
- 개선 제안사항

## 예상 결과

정상적으로 실행되면:
- 법률 노드: 1개
- 조 노드: 9개 (제6조~제14조)
- 항 노드: 2개 (제6조만)
- 호 노드: 25개
- 모든 조/항/호 노드에 임베딩 저장됨

## 문제 해결

### Neo4j 연결 실패
- Docker 컨테이너가 실행 중인지 확인: `docker ps`
- 포트 충돌 확인: 7474, 7687
- 인증 정보 확인

### 임베딩 생성 실패
- 임베딩 모델 다운로드 시간 필요 (최초 실행 시)
- 인터넷 연결 확인
- 메모리 부족 시 폴백 모델 사용

### 데이터 초기화
전체 재구축이 필요한 경우:
```bash
python main.py
# y 입력 후 전체 재구축
```

