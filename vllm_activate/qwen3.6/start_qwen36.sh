#!/bin/bash
# Qwen3.6-35B-A3B-FP8 on RunPod H100 (vLLM 0.20.0, FA3 + FlashInfer FP8 MoE + FP8 KV)
# 4/23 Qwen3.5 셋업과 동일 옵션 — 직접 비교 가능

mkdir -p /root/.cache/flashinfer ~/.tensorrt_llm/cache ~/.tensorrt_llm/tmp

VENV=/workspace/venv_qwen36_v20
source $VENV/bin/activate 2>/dev/null || true
export PATH=/usr/local/cuda/bin:$PATH:$VENV/bin

# rename shim (TRT-LLM cubin 버그 회피)
[ -f /workspace/trtllm_fixes/rename_fix.so ] && \
  export LD_PRELOAD=/workspace/trtllm_fixes/rename_fix.so

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_FLOAT32_MATMUL_PRECISION="high"
export VLLM_USE_DEEP_GEMM=1
export VLLM_MOE_USE_DEEP_GEMM=1
export VLLM_USE_FLASHINFER_MOE_FP8=1
export FLASHINFER_MOE_BACKEND=throughput
export FLASHINFER_DISABLE_VERSION_CHECK=1

$VENV/bin/vllm serve /workspace/models/Qwen3.6-35B-A3B-FP8 \
  --served-model-name Qwen3.6-35B-A3B-FP8 \
  --host 0.0.0.0 --port 8000 \
  --kv-cache-dtype fp8_e4m3 \
  --dtype auto \
  --max-model-len 5120 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.96 \
  --max-num-batched-tokens 32768 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --async-scheduling \
  --attention-backend flash_attn \
  --attention-config "{\"flash_attn_version\":3}" \
  --gdn-prefill-backend triton \
  --trust-remote-code \
  --language-model-only \
  --compilation-config '{"custom_ops":["all"],"cudagraph_capture_sizes":[1,2,4,8,16,32,50]}' \
  --disable-log-stats
