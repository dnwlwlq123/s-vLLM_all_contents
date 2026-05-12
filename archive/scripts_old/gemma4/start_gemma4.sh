#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_FLOAT32_MATMUL_PRECISION="high"
export VLLM_USE_DEEP_GEMM=0

vllm serve /workspace/models/gemma-4-31B-it \
  --served-model-name gemma-4-31B-it \
  --host 0.0.0.0 --port 8000 \
  --quantization fp8 \
  --kv-cache-dtype auto \
  --dtype auto \
  --max-model-len 8192 \
  --max-num-seqs 50 \
  --gpu-memory-utilization 0.95 \
  --async-scheduling \
  --attention-backend FLASH_ATTN \
  --attention-config '{"flash_attn_version": 4}' \
  --trust-remote-code \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 32768 \
  --disable-log-stats \
  --limit-mm-per-prompt '{}' \
  --compilation-config '{"custom_ops":["all"],"cudagraph_capture_sizes":[1,2,4,8,16,32,50]}'
