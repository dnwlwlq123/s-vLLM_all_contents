#!/bin/bash
# Gateway 시작 스크립트
# 사용법: bash start_gateway.sh [모델명] [포트]
# 예시: bash start_gateway.sh gemma-4-31B-it 80

MODEL_NAME="${1:-gemma-4-31B-it}"
PORT="${2:-6006}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Gateway 시작 (모델: $MODEL_NAME, 포트: $PORT) ==="
cd $SCRIPT_DIR && \
GW_ANSWER_VLLM_URL=http://localhost:8000 \
GW_INTENT_VLLM_URL=http://localhost:8000 \
GW_BASE_MODEL_NAME=$MODEL_NAME \
GW_PORT=$PORT \
exec /workspace/gemma_server_for_poc/venv_gemma4/bin/python3 main.py
