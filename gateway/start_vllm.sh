#!/bin/bash
# =============================================================
# vLLM 서버 시작 스크립트 (H100 최적화)
# 사용법: bash start_vllm.sh [27b|27b-fp8|35b|35b-fp8]
# =============================================================

source ~/vllm_env/bin/activate 2>/dev/null
export CUDA_HOME=${CUDA_HOME:-$HOME/cuda-12.6}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# =============================================================
# 공통 환경변수
# =============================================================
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# =============================================================
# 모델 선택 + MoE/Dense 분기
# =============================================================
MODEL_DIR=~/vLLM_server/qwen3.5/models
IS_MOE=false

case "$1" in
  27b)
    MODEL_PATH="$MODEL_DIR/Qwen3.5-27B"
    MODEL_NAME="Qwen3.5-27B"
    QUANT="--quantization fp8"
    ;;
  27b-fp8)
    MODEL_PATH="$MODEL_DIR/Qwen3.5-27B-FP8"
    MODEL_NAME="Qwen3.5-27B-FP8"
    QUANT=""
    ;;
  35b)
    MODEL_PATH="$MODEL_DIR/Qwen3.5-35B-A3B"
    MODEL_NAME="Qwen3.5-35B-A3B"
    QUANT="--quantization fp8"
    IS_MOE=true
    ;;
  35b-fp8)
    MODEL_PATH="$MODEL_DIR/Qwen3.5-35B-A3B-FP8"
    MODEL_NAME="Qwen3.5-35B-A3B-FP8"
    QUANT=""
    IS_MOE=true
    ;;
  *)
    echo "사용법: bash start_vllm.sh [27b|27b-fp8|35b|35b-fp8]"
    exit 1
    ;;
esac

# =============================================================
# MoE 전용 최적화 (35B-A3B만 적용)
# =============================================================
if [ "$IS_MOE" = true ]; then
  # --- MoE 전용 ---
  export VLLM_MOE_USE_DEEP_GEMM=0           # MoE grouped GEMM에서 DeepGEMM 비활성
  export VLLM_USE_FLASHINFER_MOE_FP8=1       # FP8 MoE FlashInfer 커널 활성화
  echo "=== $MODEL_NAME (MoE) 시작 — MoE 최적화 ON ==="
else
  # --- Dense 전용 ---
  export VLLM_FLOAT32_MATMUL_PRECISION="high" # TF32 텐서코어 활용 — 27B Dense에서 matmul 가속
  echo "=== $MODEL_NAME (Dense) 시작 — TF32 ON ==="
fi

# =============================================================
# vLLM 서빙 시작
# =============================================================
vllm serve $MODEL_PATH \
  --served-model-name $MODEL_NAME \
  --host 0.0.0.0 --port 8000 \
  $QUANT \
  --kv-cache-dtype fp8 \
  --dtype auto \
  --max-model-len 16384 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.95 \
  --max-cudagraph-capture-size 256 \
  --async-scheduling \
  --trust-remote-code \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 32768 \
  --disable-log-stats \
  --language-model-only \
  --attention-backend flash_attn \
  --attention-config '{"flash_attn_version":3,"use_prefill_query_quantization":true}'
  # MTP — Qwen3.5 GDN + CUDAGraph 충돌로 0.17.1에서 사용 불가. 0.18+ 패치 필요.
  # --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
