#!/bin/bash
# Gemma-4-31B-it KB chatbot 측정용 (FA3 + FP8 KV, H100)
# Qwen3.6 셋업과 동일 최적화 + Gemma4 specific (limit-mm, language-model-only)

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

$VENV/bin/vllm serve /workspace/models/gemma-4-31B-it \
  --served-model-name gemma-4-31B-it \
  --host 0.0.0.0 --port 8000 \
  --quantization fp8 \
  --kv-cache-dtype auto \
  --dtype auto \
  --max-model-len 5120 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.97 \
  --max-num-batched-tokens 32768 \
  --enable-chunked-prefill \
  --async-scheduling \
  --attention-backend flash_attn \
  --attention-config '{"flash_attn_version":3}' \
  --trust-remote-code \
  --language-model-only \
  --limit-mm-per-prompt '{"image":0,"audio":0,"video":0}' \
  --compilation-config '{"custom_ops":["all"],"cudagraph_capture_sizes":[1,2,4,8,10,12,14,16,32,50]}' \
  --disable-log-stats
