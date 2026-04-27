# Gateway — FP8 14B Multi-LoRA 추론 프록시

외부 요청을 받아 vLLM 서버에 intent(LoRA) -> answer(바닐라) 2-Stage 추론을 수행하는 경량 Gateway API.

## 아키텍처

```
외부 클라이언트 → Gateway API (FastAPI, CPU) → vLLM 서버 (GPU, Multi-LoRA)
```

- Gateway는 GPU 불필요 (HTTP 프록시만 수행)
- 프롬프트는 클라이언트가 messages에 포함하여 전달
- vLLM 서버는 외부에서 독립적으로 관리 (`src/vllm_serving/`)

## 실행

```bash
cd src/gateway

# .env 설정
cp .env.example .env
vi .env

# 서버 시작
python run.py
```

기본: `http://0.0.0.0:17801`

## API 엔드포인트

### POST /api/completions — 2-Stage 통합 추론

```json
{
  "messages": [
    {"role": "system", "content": "당신은 상담사입니다..."},
    {"role": "user", "content": "적금 문의드려요"}
  ],
  "intent_messages": [
    {"role": "system", "content": "intent 분류기입니다..."},
    {"role": "user", "content": "고객: 적금 문의드려요"}
  ],
  "stream": false
}
```

응답:

```json
{
  "intent": {"name": "faq", "confidence": 0.0},
  "response": "안내해드리겠습니다...",
  "latency_ms": 1234.5
}
```

### POST /api/intent — intent 단독

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]}
```

### POST /api/answer — answer 단독

```json
{
  "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
  "stream": false
}
```

### GET /api/health

```json
{"gateway": "ok", "vllm": "ok", "vllm_url": "http://localhost:8000"}
```

## 설정 (환경변수)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `GW_VLLM_BASE_URL` | `http://localhost:8000` | vLLM 서버 URL |
| `GW_INTENT_MODEL_NAME` | `intent` | vLLM LoRA adapter 이름 |
| `GW_ANSWER_MODEL_NAME` | `base_14b_fp8` | vLLM base 모델 이름 |
| `GW_INTENT_MAX_TOKENS` | `128` | intent 최대 토큰 |
| `GW_ANSWER_MAX_TOKENS` | `1024` | answer 최대 토큰 |
| `GW_INTENT_TEMPERATURE` | `0.0` | intent 생성 온도 |
| `GW_ANSWER_TEMPERATURE` | `0.3` | answer 생성 온도 |
| `GW_PORT` | `17801` | Gateway 서버 포트 |
| `GW_HTTP_TIMEOUT` | `60.0` | vLLM 요청 타임아웃 (초) |

## vLLM 서버 연동

vLLM 서버는 `src/vllm_serving/`에서 별도로 실행합니다:

```bash
cd src/vllm_serving
bash run_single.sh   # 단일 서버 (GPU 1장)
```

Gateway는 `.env`의 `GW_VLLM_BASE_URL`로 연결합니다.
