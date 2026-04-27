# s-vLLM_all_contents

vLLM 콜봇 서빙 인프라 코드/스크립트 모음.

## 구조

```
vllm_pipeline/        — 5090 dev 환경 (게이트웨이 + 서빙 컨트롤러)
  vllm_gateway/       — OpenAI 호환 게이트웨이 (intent/answer 라우팅)
  vllm_serving/       — vLLM 프로세스 매니지먼트 + S3 모델 캐시

vllm_activate/        — RunPod / 실서빙 환경 스크립트
  gemma4/             — Gemma4-31B-it 띄우는 변종 5종
                         (FA3+FP8KV / FA3 v1.9.1 / FP8KV test / TurboQuant / 기본 FA4)
                       + Gemma4_server (가벼운 게이트웨이)
  qwen3.5/            — Qwen3.5-35B-A3B (MoE) 띄우는 변종
                         (기본 / v2 / TurboQuant / robust setup)
  bench/              — 콜봇 latency/burst 벤치마크 스크립트 (real/single/burst)
  trtllm_fixes/       — FlashInfer MoE FP8 cubin rename 버그 LD_PRELOAD shim
```

## 제외된 것

- 모델 weights (`/workspace/models/`)
- venv (`venv_gemma4`, `venv_gemma4_v19_1` 등)
- 벤치 결과 출력
- 컴파일 산출물 (.so)
- IP/포트/k8s manifest
