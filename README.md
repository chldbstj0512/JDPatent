# JDPatent EC2 실행 가이드

## 1) 서버 접속 후 프로젝트 이동

```bash
cd /path/to/JDPatent
```

## 2) 초기 1회 세팅 + 컨테이너 실행

```bash
chmod +x ec2_init.sh
./ec2_init.sh
```

- Docker / Docker Compose 설치
- `.env` 자동 준비(없으면 `.env.api.example` 기반 생성)
- `docker compose up -d --build` 실행

## 3) 설치만 하고 실행은 나중에

```bash
./ec2_init.sh --skip-up
```

그 다음 직접 실행:

```bash
docker compose up -d --build
```

## 4) 서비스 상태 확인

```bash
docker compose ps
curl http://localhost:8001/healthz
```

## 5) 작업 요청/조회 예시

### 작업 등록

```bash
curl -X POST "http://localhost:8001/api/v1/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "test-task-001",
    "raw_text": "여기에 추출된 특허 raw text",
    "user_id": "test-user"
  }'
```

### 결과 조회

```bash
curl "http://localhost:8001/api/v1/jobs/test-task-001"
```

## 6) 운영 중 자주 쓰는 명령어

```bash
docker compose logs -f jdpatent-api
docker compose logs -f jdpatent-worker
docker compose restart
docker compose down
```

