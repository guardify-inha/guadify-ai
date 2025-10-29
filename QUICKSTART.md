# ⚡ 빠른 시작 가이드

## 1️⃣ Docker 실행 (30초)

```bash
# 프로젝트 폴더에서
docker-compose up -d
```

## 2️⃣ Python 패키지 설치 (1분)

```bash
pip install -r requirements.txt
```

## 3️⃣ 법률 그래프 구축 (10초)

```bash
python main.py
```

입력창에서 `y` 입력!

## 4️⃣ 확인하기

브라우저에서 http://localhost:7474 접속

**로그인:**
- Username: `neo4j`
- Password: `testpassword123`

**쿼리 실행:**
```cypher
MATCH (n) RETURN n LIMIT 50
```

---

## 🎉 완료!

이제 약관 규제 법률 그래프가 Neo4j에 저장되었습니다!

**다음 단계:**
- 위반 사례 추가
- 수정본 추가
- 판단 로직 구현

---

## ⚠️ 문제 발생시

### Docker 오류
```bash
# Docker가 실행 중인지 확인
docker ps

# 안되면 Docker Desktop을 켜주세요!
```

### Python 오류
```bash
# 가상환경 사용 권장
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 다시 설치
pip install -r requirements.txt
```

### Neo4j 연결 오류
```bash
# Neo4j 로그 확인
docker-compose logs

# 재시작
docker-compose restart
```

---

**더 자세한 내용:** [README.md](README.md) 또는 [DOCKER_GUIDE.md](DOCKER_GUIDE.md) 참고
