#!/bin/bash
# Qwen3.5-35B-A3B-FP8 MoE on RunPod H100 (vLLM 0.19.2rc1.dev96)
# FlashInfer MoE kernel 캐시 영구화 (Pod terminate 에도 살아남음)
mkdir -p /root/.cache
if [ ! -e /root/.cache/flashinfer ]; then
    ln -sfn /workspace/.cache/flashinfer /root/.cache/flashinfer
    echo "flashinfer symlink restored: /root/.cache/flashinfer -> /workspace/.cache/flashinfer"
fi

source /workspace/venv_gemma4/bin/activate
export PATH=/usr/local/cuda/bin:$PATH   # nvcc 필요 (FlashInfer MoE JIT)
export LD_PRELOAD=/workspace/trtllm_fixes/rename_fix.so   # TRT-LLM cubin rename shim
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_FLOAT32_MATMUL_PRECISION="high"
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_MOE_FP8=1
export FLASHINFER_MOE_BACKEND=throughput
export FLASHINFER_DISABLE_VERSION_CHECK=1

vllm serve /workspace/models/Qwen3.5-35B-A3B-FP8 \
  --served-model-name Qwen3.5-35B-A3B-FP8 \
  --host 0.0.0.0 --port 8000 \
  --kv-cache-dtype turboquant_k8v4 \
  --dtype auto \
  --max-model-len 16384 \
  --max-num-seqs 50 \
  --gpu-memory-utilization 0.96 \
  --max-num-batched-tokens 32768 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --enforce-eager \
  --trust-remote-code \
  --language-model-only \
   \
  --gdn-prefill-backend triton \
  --disable-log-stats
