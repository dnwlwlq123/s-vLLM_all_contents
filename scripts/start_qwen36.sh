#!/bin/bash
# Qwen3.6-35B-A3B-FP8 on RunPod H100 (vLLM 0.20.0, FA3 + FlashInfer FP8 MoE + FP8 KV)
# 4/23 Qwen3.5 셋업과 동일 옵션 — 직접 비교 가능

mkdir -p /root/.cache/flashinfer ~/.tensorrt_llm/cache ~/.tensorrt_llm/tmp

VENV=/workspace/venv_qwen36_v20
source $VENV/bin/activate 2>/dev/null || true
export CUDA_HOME=/workspace/cuda-13.0
export PATH=/workspace/cuda-13.0/bin:/home/timbel/.npm-global/bin:/home/timbel/.local/bin:/workspace/e2e_train_8k/kaldi/tools/openfst/bin:/workspace/e2e_train_8k/kaldi/tools/openfst/bin:/home/timbel/.local/bin:/home/timbel/.pyenv/plugins/pyenv-virtualenv/shims:/home/timbel/.pyenv/shims:/home/timbel/.pyenv/bin:/usr/local/cuda-12.8/bin:/usr/local/cuda-12.8/bin:/home/timbel/.vscode-server/data/User/globalStorage/github.copilot-chat/debugCommand:/home/timbel/.vscode-server/data/User/globalStorage/github.copilot-chat/copilotCli:/home/timbel/.vscode-server/cli/servers/Stable-10c8e557c8b9f9ed0a87f61f1c9a44bde731c409/server/bin/remote-cli:/home/timbel/.pyenv/bin:/home/timbel/.local/bin:/home/timbel/.npm-global/bin:/home/timbel/.local/bin:/workspace/e2e_train_8k/kaldi/tools/openfst/bin:/workspace/e2e_train_8k/kaldi/tools/openfst/bin:/home/timbel/.local/bin:/home/timbel/.local/bin:/home/timbel/.pyenv/plugins/pyenv-virtualenv/shims:/home/timbel/.pyenv/bin:/home/timbel/miniconda3/bin:/home/timbel/miniconda3/condabin:/usr/local/cuda-12.8/bin:/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/home/timbel/.claude/plugins/cache/claude-plugins-official/superpowers/5.0.7/bin
export PATH=$PATH:$VENV/bin

# rename shim (TRT-LLM cubin 버그 회피)
[ -f /workspace/trtllm_fixes/rename_fix.so ] && \
  export LD_PRELOAD=/workspace/trtllm_fixes/rename_fix.so

export TRITON_CACHE_DIR=/workspace/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/workspace/.torchinductor_cache
export VLLM_CACHE_ROOT=/workspace/.vllm_cache
export FLASHINFER_CACHE_DIR=/workspace/.flashinfer_cache
export HF_HOME=/workspace/.hf_cache
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_FLOAT32_MATMUL_PRECISION="high"
# export VLLM_USE_DEEP_GEMM=1  # sm_120 FP8 MoE 미지원
# export VLLM_MOE_USE_DEEP_GEMM=1  # sm_120 미지원
# export VLLM_USE_FLASHINFER_MOE_FP8=1  # sm_120 미지원
export FLASHINFER_MOE_BACKEND=throughput
export FLASHINFER_DISABLE_VERSION_CHECK=1

$VENV/bin/vllm serve /workspace/models/Qwen3.6-35B-A3B-FP8 \
  --served-model-name Qwen3.6-35B-A3B-FP8 \
  --host 0.0.0.0 --port 8000 \
  --kv-cache-dtype fp8_e4m3 \
  --dtype auto \
  --max-model-len 5120 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.97 \
  --max-num-batched-tokens 196608 \
  \
  --enable-chunked-prefill \
  --async-scheduling \
  --attention-backend triton_attn \
  --gdn-prefill-backend triton \
  --trust-remote-code \
  --language-model-only \
  --compilation-config '{"custom_ops":["all"],"cudagraph_capture_sizes":[1,2,4,8,10,12,14,16,32,50]}' \
  --disable-log-stats
