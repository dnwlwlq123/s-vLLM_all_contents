#!/bin/bash
# vLLM 서버 종료 스크립트
# 사용법: bash stop_vllm.sh

echo "=== vLLM 종료 ==="
pkill -9 -f "vllm serve" 2>/dev/null
pkill -9 -f "EngineCore" 2>/dev/null
sudo fuser -k 80/tcp 2>/dev/null

echo "GPU 메모리 해제 대기..."
sleep 5

echo "=== 완료 ==="
nvidia-smi --query-gpu=memory.used --format=csv,noheader
