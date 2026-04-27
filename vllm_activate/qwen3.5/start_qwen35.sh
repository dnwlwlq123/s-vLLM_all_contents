#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_FLOAT32_MATMUL_PRECISION="high"
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_MOE_FP8=1
export FLASHINFER_MOE_BACKEND=throughput

vllm serve /workspace/models/Qwen3.5-35B-A3B-FP8 \
  --served-model-name Qwen3.5-35B-A3B-FP8 \
  --host 0.0.0.0 --port 8000 \
  --kv-cache-dtype fp8_e4m3 \
  --dtype auto \
  --max-model-len 16384 \
  --max-num-seqs 50 \
  --gpu-memory-utilization 0.96 \
  --max-num-batched-tokens 32768 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --async-scheduling \
  --attention-backend flash_attn \
  --attention-config '{"flash_attn_version":3}' \
  --trust-remote-code \
  --language-model-only \
  --reasoning-parser qwen3 \
  --gdn-prefill-backend triton \
  --disable-log-stats
