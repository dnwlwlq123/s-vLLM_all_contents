#!/bin/bash
# Gemma4-31B-it + TurboQuant (FP8 K + 4bit V) on RunPod H100
# Pure attention model, no GDN/Mamba hybrid → TQ KV 가능

source /workspace/venv_gemma4/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_FLOAT32_MATMUL_PRECISION="high"
export VLLM_USE_DEEP_GEMM=0
export FLASHINFER_DISABLE_VERSION_CHECK=1

vllm serve /workspace/models/gemma-4-31B-it \
  --served-model-name gemma-4-31B-it \
  --host 0.0.0.0 --port 8000 \
  --quantization fp8 \
  --kv-cache-dtype turboquant_k8v4 \
  --attention-backend turboquant \
  --dtype auto \
  --max-model-len 8192 \
  --max-num-seqs 50 \
  --gpu-memory-utilization 0.95 \
  --enforce-eager \
  --trust-remote-code \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 32768 \
  --limit-mm-per-prompt '{}' \
  --disable-log-stats
