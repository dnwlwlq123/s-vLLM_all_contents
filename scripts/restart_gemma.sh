#!/bin/bash
pkill -9 -f 'vllm serve' 2>/dev/null
pkill -9 -f EngineCore 2>/dev/null
sleep 5
rm -f /tmp/vllm_gemma4.log
cd /workspace
nohup bash start_gemma4.sh > /tmp/vllm_gemma4.log 2>&1 < /dev/null &
disown
echo "started PID $!"
