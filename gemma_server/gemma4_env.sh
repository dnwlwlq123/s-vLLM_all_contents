#!/bin/bash
# B200 Gemma4 example 환경
source /workspace/gemma_server_for_poc/venv_gemma4/bin/activate
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
echo "=== Gemma 4 환경 활성화 (B200, venv) ==="
