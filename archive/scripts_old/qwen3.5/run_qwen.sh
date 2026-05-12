#!/bin/bash
export PATH=/workspace/cuda-12.8/bin:$PATH
export CUDA_HOME=/workspace/cuda-12.8
export LD_LIBRARY_PATH=/workspace/cuda-12.8/lib64:$LD_LIBRARY_PATH
cd /workspace
bash start_vllm.sh qwen27b-fp8
