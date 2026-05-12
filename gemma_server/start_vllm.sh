#!/bin/bash
# B200 단일 GPU(index 7) Gemma4-31B-it FP8 + FA4 + FP8 KV
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_FLOAT32_MATMUL_PRECISION="high"
export VLLM_USE_DEEP_GEMM=0
export TRITON_CACHE_DIR=/workspace/gemma_server_for_poc/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/workspace/gemma_server_for_poc/.torchinductor_cache
export VLLM_CACHE_ROOT=/workspace/gemma_server_for_poc/.vllm_cache

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
