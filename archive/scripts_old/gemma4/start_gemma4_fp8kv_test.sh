#!/bin/bash
# FA4 + FP8 KV 조합 테스트 — gating 이 최신에서 풀렸는지 확인
source /workspace/venv_gemma4/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_USE_DEEP_GEMM=0

vllm serve /workspace/models/gemma-4-31B-it \
  --served-model-name gemma-4-31B-it \
  --host 0.0.0.0 --port 8000 \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --dtype auto \
  --max-model-len 8192 \
  --max-num-seqs 50 \
  --gpu-memory-utilization 0.92 \
  --attention-backend FLASH_ATTN \
  --attention-config '{"flash_attn_version": 4}' \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image":0,"audio":0,"video":0}' \
  --disable-log-stats
