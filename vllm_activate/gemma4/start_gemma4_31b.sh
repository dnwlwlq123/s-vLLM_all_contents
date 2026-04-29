#!/bin/bash
# Gemma4-31B-it on RunPod H100 — ECI start_vllm.sh 31b case 그대로 복제
mkdir -p /root/.cache/flashinfer ~/.tensorrt_llm/cache ~/.tensorrt_llm/tmp

VENV=/workspace/venv_qwen36_v20
export PATH=/usr/local/cuda/bin:$PATH:$VENV/bin

[ -f /workspace/trtllm_fixes/rename_fix.so ] && \
  export LD_PRELOAD=/workspace/trtllm_fixes/rename_fix.so

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_FLOAT32_MATMUL_PRECISION="high"
export VLLM_USE_DEEP_GEMM=0

$VENV/bin/vllm serve /workspace/models/gemma-4-31B-it \
  --served-model-name gemma-4-31B-it \
  --host 0.0.0.0 --port 8000 \
  --quantization fp8 \
  --dtype auto \
  --max-model-len 16384 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.95 \
  --async-scheduling \
  --attention-backend FLASH_ATTN \
  --attention-config '{"flash_attn_version": 4}' \
  --trust-remote-code \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 16384 \
  --disable-log-stats \
  --limit-mm-per-prompt '{}' \
  --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}' \
  --compilation-config '{"custom_ops":["all"],"cudagraph_capture_sizes":[1,2,4,8,16,32]}'
