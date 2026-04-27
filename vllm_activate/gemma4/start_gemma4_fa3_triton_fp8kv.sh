#!/bin/bash
# FA3 + Triton (hybrid) + FP8 KV — clean original vLLM (no patches)
#
# 대상 구성:
#   - SWA layers (head_dim=256) → FA3 + FP8 KV (vLLM 공식 지원, Qwen 에서 검증됨)
#   - Global layers (head_dim=512) → Triton + FP8 KV (head_dim > 256 이라 FA 불가, 자동 fallback)
#
# 주의: Gemma4 heterogeneous attention 때문에 일부 layer 가 config flash_attn_version=3
#       을 무시하고 FA4 로 dispatch 될 수 있음. 실제 기동해보고 결과 판단.

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
  --attention-config '{"flash_attn_version": 3}' \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image":0,"audio":0,"video":0}' \
  --disable-log-stats
