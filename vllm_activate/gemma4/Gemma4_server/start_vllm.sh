#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export FLASHINFER_DISABLE_VERSION_CHECK=1

MODEL_DIR=/workspace/models
IS_MOE=false
QUANT_ARGS=""

case "$1" in
  31b)
    MODEL_PATH="$MODEL_DIR/gemma-4-31B-it"
    MODEL_NAME="gemma-4-31B-it"
    QUANT_ARGS="--quantization fp8"
    echo "=== $MODEL_NAME (FP8 런타임 양자화, KV bf16 — FA4 호환) ==="
    ;;
  31b-nvfp4)
    MODEL_PATH="$MODEL_DIR/Gemma-4-31B-IT-NVFP4"
    MODEL_NAME="gemma-4-31B-it"
    QUANT_ARGS=""
    echo "=== Gemma-4-31B-IT-NVFP4 (사전양자화 4bit) ==="
    ;;
  31b-fp8)
    MODEL_PATH="$MODEL_DIR/gemma-4-31B-it-FP8-block"
    MODEL_NAME="gemma-4-31B-it"
    QUANT_ARGS=""
    echo "=== gemma-4-31B-it-FP8-block (사전양자화 FP8) ==="
    ;;
  26b)
    MODEL_PATH="$MODEL_DIR/gemma-4-26B-A4B-it"
    MODEL_NAME="gemma-4-26B-A4B-it"
    QUANT_ARGS="--quantization fp8"
    IS_MOE=true
    ;;
  *)
    echo "사용법: bash start_vllm.sh [31b|31b-fp8|31b-nvfp4|26b]"
    exit 1
    ;;
esac

if [ "$IS_MOE" = true ]; then
  export VLLM_MOE_USE_DEEP_GEMM=0
  export VLLM_USE_FLASHINFER_MOE_FP8=0
  echo "=== $MODEL_NAME (MoE) 시작 ==="
else
  export VLLM_FLOAT32_MATMUL_PRECISION="high"
  # CUTLASS FP8 쓰는 dense 런타임 양자화에서는 DeepGEMM warmup 불필요 — 외부 deep_gemm 패키지 미설치 시 vllm dev135 warmup 이 크래시함
  export VLLM_USE_DEEP_GEMM=0
fi

vllm serve $MODEL_PATH \
  --served-model-name $MODEL_NAME \
  --host 0.0.0.0 --port 8000 \
  $QUANT_ARGS \
  --dtype auto \
  --max-model-len 8192 \
  --max-num-seqs 10 \
  --gpu-memory-utilization 0.96 \
  --async-scheduling \
  --attention-backend FLASH_ATTN \
  --attention-config '{"flash_attn_version": 4}' \
  --trust-remote-code \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 32768 \
  --disable-log-stats \
  --limit-mm-per-prompt '{}' \
  --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}' \
  --compilation-config '{"custom_ops":["all"],"cudagraph_capture_sizes":[1,2,4,8,16,32]}'
