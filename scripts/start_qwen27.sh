#!/bin/bash
# Qwen3.6-27B-FP8 KB chatbot 셋업 (H100, FA3 + FP8 KV)
# Dense 27B (no MoE, no GDN)

mkdir -p /root/.cache/flashinfer ~/.tensorrt_llm/cache ~/.tensorrt_llm/tmp

VENV=/workspace/venv_qwen36_v20
source $VENV/bin/activate 2>/dev/null || true
export PATH=/usr/local/cuda/bin:$PATH:$VENV/bin

[ -f /workspace/trtllm_fixes/rename_fix.so ] && \
  export LD_PRELOAD=/workspace/trtllm_fixes/rename_fix.so

export TRITON_CACHE_DIR=/workspace/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/workspace/.torchinductor_cache
export VLLM_CACHE_ROOT=/workspace/.vllm_cache
export FLASHINFER_CACHE_DIR=/workspace/.flashinfer_cache
export HF_HOME=/workspace/.hf_cache

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_FLOAT32_MATMUL_PRECISION="high"
export FLASHINFER_DISABLE_VERSION_CHECK=1

$VENV/bin/vllm serve /workspace/models/Qwen3.6-27B-FP8 \
  --served-model-name Qwen3.6-27B-FP8 \
  --host 0.0.0.0 --port 8000 \
  --kv-cache-dtype fp8_e4m3 \
  --dtype auto \
  --max-model-len 5120 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.97 \
  --max-num-batched-tokens 196608 \
  --enable-chunked-prefill \
  --async-scheduling \
  --attention-backend flash_attn \
  --attention-config '{"flash_attn_version":3}' \
  --trust-remote-code \
  --compilation-config '{"custom_ops":["all"],"cudagraph_capture_sizes":[1,2,4,8,10,12,14,16,32,50]}' \
  --disable-log-stats
