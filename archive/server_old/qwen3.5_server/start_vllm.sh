#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_FLOAT32_MATMUL_PRECISION="high"
export VLLM_USE_DEEP_GEMM=0
unset MAX_JOBS NINJA_MAX_JOBS

MODEL_DIR=/workspace/models

case "$1" in
  qwen27b-fp8)
    MODEL_PATH="$MODEL_DIR/Qwen3.5-27B-FP8"
    MODEL_NAME="Qwen3.5-27B-FP8"
    ;;
  *)
    echo "사용법: bash start_vllm.sh [qwen27b-fp8]"; exit 1 ;;
esac

echo "=== $MODEL_NAME (Dense) 시작 — FLASHINFER (Blackwell SM120) ==="

vllm serve $MODEL_PATH \
  --served-model-name $MODEL_NAME \
  --host 0.0.0.0 --port 8000 \
  --quantization fp8 \
  \
  --dtype auto \
  --max-model-len 16384 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.95 \
  --max-cudagraph-capture-size 256 \
  --async-scheduling \
  --trust-remote-code \
  --enable-prefix-caching \
  --max-num-batched-tokens 32768 \
  --disable-log-stats \
  --language-model-only \
  --attention-backend TRITON_ATTN
