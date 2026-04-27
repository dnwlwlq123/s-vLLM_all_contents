# vLLM Serving Manager — FP8 14B Multi-LoRA 서빙 관리

S3/MinIO에서 모델을 다운로드하고, vLLM 프로세스를 API로 관리하는 FastAPI 서버.

## 아키텍처

```
운영자 / CI/CD                vLLM Serving Manager           vLLM 프로세스
                             (FastAPI, port 17810)
                                    │
POST /api/deploy ─────────→  1. S3 모델 다운로드
                             2. vLLM 프로세스 기동 ─────→  FP8 14B + Multi-LoRA
                             3. health check 대기           (port 8000)
                                    │
Gateway API ──────────────────────────────────────────→  추론 요청
(port 17801)                                            /v1/chat/completions
```

## 실행

```bash
cd src/vllm_serving

# .env 설정
cp .env.example .env
vi .env

# 서버 시작
python run.py
```

기본: `http://0.0.0.0:17810`

## API 엔드포인트

### 모델 관리

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/models` | 로컬에 다운로드된 모델/어댑터 목록 |
| GET | `/api/models/registry` | model_registry.yaml 내용 |
| POST | `/api/models/download` | S3/MinIO에서 모델 다운로드 |
| DELETE | `/api/models/{name}` | 로컬 모델/어댑터 삭제 |

### 서빙 관리

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/api/serving/start` | vLLM 프로세스 기동 |
| POST | `/api/serving/stop` | vLLM 프로세스 종료 |
| POST | `/api/serving/restart` | 종료 후 재기동 (adapter 업데이트 후) |
| GET | `/api/serving/status` | 프로세스 상태 + health check + 모델 목록 |

### 통합 배포

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/api/deploy` | 모든 모델 다운로드 + vLLM (재)시작 (CI/CD용) |
| GET | `/api/health` | 관리 서버 자체 상태 |

## 사용 예시

### 모델 다운로드

```bash
# intent adapter 다운로드 (registry 기반)
curl -X POST http://localhost:17810/api/models/download \
  -H "Content-Type: application/json" \
  -d '{"name": "intent"}'

# base 모델 다운로드
curl -X POST http://localhost:17810/api/models/download \
  -H "Content-Type: application/json" \
  -d '{"name": "base_model"}'

# S3 경로 직접 지정 (registry의 s3_path를 오버라이드)
curl -X POST http://localhost:17810/api/models/download \
  -H "Content-Type: application/json" \
  -d '{"name": "intent", "s3_path": "adapters/intent/v2/", "force": true}'
```

### 서빙 시작/종료

```bash
# vLLM 시작
curl -X POST http://localhost:17810/api/serving/start \
  -H "Content-Type: application/json" \
  -d '{"mode": "single"}'

# 상태 확인
curl http://localhost:17810/api/serving/status

# vLLM 재시작 (adapter 업데이트 후)
curl -X POST http://localhost:17810/api/serving/restart
```

### 원클릭 배포 (CI/CD)

```bash
# 모든 모델 다운로드 + vLLM 시작
curl -X POST http://localhost:17810/api/deploy \
  -H "Content-Type: application/json" \
  -d '{"force_download": false, "mode": "single"}'
```

## 설정

### 환경변수 (접두사: VS_)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `VS_S3_ENDPOINT_URL` | (빈 문자열) | MinIO: `http://minio:9000`, AWS S3: 빈 문자열 |
| `VS_S3_ACCESS_KEY` | | S3 access key |
| `VS_S3_SECRET_KEY` | | S3 secret key |
| `VS_S3_BUCKET` | `sllm-models` | S3 버킷 이름 |
| `VS_MODEL_CACHE_DIR` | `/opt/models` | 로컬 모델 캐시 경로 |
| `VS_VLLM_PORT` | `8000` | vLLM 서버 포트 |
| `VS_VLLM_GPU_MEMORY_UTILIZATION` | `0.85` | GPU 메모리 사용률 |
| `VS_VLLM_MAX_MODEL_LEN` | `4096` | 최대 시퀀스 길이 |
| `VS_VLLM_MAX_LORA_RANK` | `32` | LoRA 최대 rank |
| `VS_VLLM_STARTUP_TIMEOUT` | `300` | vLLM 시작 대기 시간(초) |

### model_registry.yaml

```yaml
base_model:
  name: "base_14b_fp8"
  s3_path: "models/base_14b_fp8/"
  local_path: "base_14b_fp8"

adapters:
  intent:
    s3_path: "adapters/intent/v1/"
    local_path: "adapters/intent"
    version: "v1"
```

- `s3_path`: S3 버킷 내 경로 (버킷명 제외)
- `local_path`: `MODEL_CACHE_DIR` 기준 상대 경로

## Gateway 연동

Gateway API는 vLLM 서버에 직접 연결합니다:

```
Gateway (port 17801) → vLLM (port 8000)
                       ↑
                 Serving Manager가 관리 (port 17810)
```

Gateway `.env`:
```
GW_VLLM_BASE_URL=http://localhost:8000
```

## CI/CD 배포 흐름

```bash
# 1. Serving Manager가 실행 중인 상태에서
# 2. 배포 트리거 (GitHub Actions 등)
curl -X POST http://서버IP:17810/api/deploy \
  -d '{"force_download": true, "mode": "single"}'

# 3. 결과: S3에서 최신 모델 다운로드 → vLLM 재시작
```
