# 🐳 Docker로 Neo4j 실행하기

## 📋 사전 준비

1. **Docker Desktop 설치 확인**
   ```bash
   docker --version
   docker-compose --version
   ```
   
   없으면 설치: https://www.docker.com/products/docker-desktop/

---

## 🚀 실행 방법

### 1️⃣ Neo4j 시작

프로젝트 폴더에서:

```bash
# Docker Compose로 Neo4j 실행
docker-compose up -d
```

**실행 확인:**
```bash
docker ps
```

다음과 같이 나오면 성공:
```
CONTAINER ID   IMAGE          STATUS         PORTS
xxxxx          neo4j:latest   Up 10 seconds  0.0.0.0:7474->7474/tcp, 0.0.0.0:7687->7687/tcp
```

### 2️⃣ Neo4j Browser 접속

브라우저에서: http://localhost:7474

**로그인 정보:**
- URL: `bolt://localhost:7687`
- Username: `neo4j`
- Password: `testpassword123`

### 3️⃣ Python 프로그램 실행

```bash
# 패키지 설치 (최초 1회만)
pip install -r requirements.txt

# 그래프 구축
python main.py
```

---

## 🛠️ Docker 관리 명령어

```bash
# Neo4j 중지
docker-compose stop

# Neo4j 재시작
docker-compose start

# 로그 확인
docker-compose logs -f

# 완전 종료 (컨테이너 삭제, 데이터는 유지)
docker-compose down

# 데이터까지 완전 삭제
docker-compose down -v
```

---

## ⚠️ 문제 해결

### 포트가 이미 사용 중일 때

```bash
# 7474나 7687 포트를 다른 프로그램이 사용 중이면
# docker-compose.yml에서 포트 번호 변경:

ports:
  - "17474:7474"  # 브라우저는 localhost:17474로 접속
  - "17687:7687"  # .env의 NEO4J_URI도 bolt://localhost:17687로 변경
```

### Docker가 안 켜져 있을 때

```
Error: Cannot connect to the Docker daemon
```

→ Docker Desktop을 실행해주세요!

---

## 📊 데이터 관리

### 데이터 백업
```bash
# 데이터 볼륨 확인
docker volume ls

# 백업 (선택사항)
docker run --rm \
  -v unfair-terms-detector_neo4j_data:/data \
  -v $(pwd):/backup \
  busybox tar czf /backup/neo4j-backup.tar.gz /data
```

### 데이터 초기화
```bash
# 1. 컨테이너 중지 및 삭제
docker-compose down -v

# 2. 다시 시작
docker-compose up -d

# 3. Python 프로그램 재실행
python main.py
```

---

## 👥 팀원과 공유

### 공유할 파일:
- ✅ `docker-compose.yml`
- ✅ `.env`
- ✅ 모든 Python 파일

### 팀원 실행 방법:
```bash
# 1. 프로젝트 클론
git clone [your-repo]

# 2. Docker 실행
docker-compose up -d

# 3. 패키지 설치
pip install -r requirements.txt

# 4. 실행
python main.py
```

---

## 🎯 다음 단계

Neo4j가 정상 실행되면:
1. http://localhost:7474 접속해서 연결 확인
2. `python main.py` 실행
3. Neo4j Browser에서 그래프 확인!
