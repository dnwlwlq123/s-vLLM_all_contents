#!/bin/bash
# ============================================================
# Gemma4-31B-it vLLM 기동 스크립트
# ============================================================
# 현재 설정: B200 (SM100) 기준
#
# GPU 별 설정 차이 (같은 Gemma4 모델, 이 중 attention/kv 부분만 바꿔 쓰면 됨)
#
# ┌──────────────┬──────────────────┬──────────────────────────────────────────┐
# │ GPU          │ attention backend│ kv-cache-dtype                           │
# ├──────────────┼──────────────────┼──────────────────────────────────────────┤
# │ B200 (SM100) │ TRITON_ATTN      │ auto (=BF16) / fp8 (Triton FP8 KV 지원)   │
# │ H100 (SM90)  │ FLASH_ATTN + v4  │ auto (=BF16)  ← FA4 는 FP8 KV 거부        │
# │              │ (SWA 레이어만,   │ (전 레이어 FP8 KV 하려면 vLLM 패치 필요) │
# │              │  Global 은 자동  │                                          │
# │              │  Triton fallback)│                                          │
# │ 5090/PRO6000 │ TRITON_ATTN      │ auto (=BF16) / fp8                       │
# │  (SM120)     │ (FA4 kernel 없음)│                                          │
# └──────────────┴──────────────────┴──────────────────────────────────────────┘
#
# H100 용 변경점 (아래 두 줄만 교체):
#   --attention-backend FLASH_ATTN \
#   --attention-config '{"flash_attn_version": 4}' \
# 5090/PRO6000 도 SM120 은 FA4 커널 없어서 현재 B200 설정 그대로 (TRITON_ATTN) 사용.
# ============================================================

export CUDA_VISIBLE_DEVICES=7
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_FLOAT32_MATMUL_PRECISION="high"
export VLLM_USE_DEEP_GEMM=0

MODEL=/workspace/gemma_server_for_poc/models/gemma-4-31B-it

vllm serve $MODEL \
  --served-model-name gemma-4-31B-it \
  --host 0.0.0.0 --port 8000 \
  --quantization fp8 \
  --kv-cache-dtype auto \
  --dtype auto \
  --max-model-len 16384 \
  --max-num-seqs 50 \
  --gpu-memory-utilization 0.92 \
  --async-scheduling \
  --attention-backend TRITON_ATTN \
  --trust-remote-code \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 32768 \
  --disable-log-stats \
  --limit-mm-per-prompt '{}' \
  --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}' \
  --compilation-config '{"custom_ops":["all"],"cudagraph_capture_sizes":[1,2,4,8,16,32,50]}'
