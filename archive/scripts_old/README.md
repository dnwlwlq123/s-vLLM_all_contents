# vLLM_server_dev

RunPod H100 상에서 Gemma4-31B-it / Qwen3.5-35B-A3B-FP8 를 vLLM 으로 서빙하기 위한 개발용 스크립트 + 패치 모음.

## 디렉토리

```
.
├── gemma4/               # Gemma4-31B-it 서빙 스크립트 + 벤치 + chatbot Gateway
│   ├── start_gemma4.sh               # 현재 운영 (FA4 + BF16 KV)
│   ├── start_gemma4_fp8kv_test.sh    # 신규: FA4 + FP8 KV (패치 필요)
│   ├── bench_real.py                 # Poisson arrival 6종 시나리오
│   ├── bench_burst_after.py          # burst (rate=inf) 벤치
│   ├── answer_prompt.md              # 실 chatbot system prompt (1675 tokens)
│   ├── run_burst.sh / wait_and_burst*.sh
│   └── Gateway/                      # AICC 호환 gateway (FastAPI)
│
├── qwen3.5/              # Qwen3.5-35B-A3B-FP8 MoE 서빙
│   ├── qwen35_setup_robust.sh        # FlashInfer CUTLASS 전체 setup
│   ├── start_qwen35{,_v2}.sh
│   ├── run_qwen.sh / start_vllm.sh
│   └── trtllm_watchdog.sh
│
├── vllm_patches/         # vLLM 소스 패치 (FA4 + FP8 KV gating 오픈)
│   ├── 01_fa4_fp8_kv_gating.patch
│   ├── 02_fa_supported_kv_cache_dtypes.patch
│   ├── apply_patches.sh
│   └── README.md
│
├── trtllm_fixes/         # FlashInfer MoE 구동 시 TRT-LLM cubin rename 버그 shim
│   ├── rename_fix.c
│   └── rename_fix.so
│
└── bench_results/        # 벤치 JSON / raw output
    └── 2026-04-21/
```

## 환경

- RunPod H100×1 (SM90, 80GB HBM3)
- CUDA 12.8
- vLLM 0.19.2rc1.dev96+g6ff8dea07 (nightly)
- Python 3.12 venv at `/workspace/venv_gemma4/`

## 운영 중 구성

**Gemma4-31B-it** (chatbot 메인):
- FA4 attention backend (SWA layers, head_dim=256)
- Triton fallback (Global layers, head_dim=512)
- FP8 weights + BF16 KV cache (현재. FA4+FP8 KV 패치 검증 중)
- TTFT ~54ms @ idle, 50채널 burst 수용

**Qwen3.5-35B-A3B-FP8** (MoE 실험):
- FlashInfer CUTLASS FP8 MoE backend (nvcc PATH fix 적용)
- FA3 + FP8 KV cache
- KV cache 991,408 tokens (Gemma4 대비 24배)

## Pod 재시작 후 복구 절차

RunPod 는 `/usr/local/` 과 `/tmp/` 가 ephemeral, `/workspace/` 만 persistent.

```bash
# 1. venv + vLLM 이미 /workspace/venv_gemma4/ 에 있음 → 재설치 불필요
source /workspace/venv_gemma4/bin/activate

# 2. vllm CLI wrapper transformers race fix 필요 (매 Pod 재시작마다)
sed -i '2a import transformers, torch' $(which vllm)

# 3. Gemma4 FA4+FP8 KV 테스트 하려면 패치 재적용
VENV=/workspace/venv_gemma4 bash vllm_patches/apply_patches.sh

# 4. 기동
bash gemma4/start_gemma4.sh
# 또는 FP8 KV 테스트
bash gemma4/start_gemma4_fp8kv_test.sh
```

## 기록된 이슈 / 해결책

- **TRT-LLM cubin rename 버그**: `trtllm_fixes/rename_fix.so` LD_PRELOAD
- **FlashInfer MoE 시 nvcc PATH 누락**: `export PATH=/usr/local/cuda/bin:$PATH`
- **vLLM transformers import race**: `/usr/local/bin/vllm` 래퍼에 transformers 명시 preload
- **FA4 + FP8 KV 거부**: `vllm_patches/` 두 파일 적용 (이 레포의 핵심 목적)
