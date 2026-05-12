#!/bin/bash
pkill -9 -f 'vllm serve' 2>/dev/null
pkill -9 -f EngineCore 2>/dev/null
sleep 5
rm -f /tmp/vllm_qwen36.log
cd /workspace
nohup bash start_qwen36.sh > /tmp/vllm_qwen36.log 2>&1 < /dev/null &
disown
echo "started PID $!"
