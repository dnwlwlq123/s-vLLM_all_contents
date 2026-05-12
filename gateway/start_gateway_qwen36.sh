#!/bin/bash
# Qwen3.6-35B-A3B-FP8 게이트웨이 — RunPod vLLM (<host>:8000) 으로 라우팅
# 포트: 9720 (기본)
# 로그: tail -f /tmp/gateway_qwen36.log

PORT="${1:-9720}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR" && \
GW_HOST=0.0.0.0 \
GW_PORT=$PORT \
GW_ANSWER_VLLM_URL=http://<host>:8000 \
GW_INTENT_VLLM_URL=http://<host>:8000 \
GW_BASE_MODEL_NAME=Qwen3.6-35B-A3B-FP8 \
python3 main.py
