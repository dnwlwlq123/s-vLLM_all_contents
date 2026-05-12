#!/bin/bash
while pgrep -f "/workspace/bench_real.py" > /dev/null; do sleep 5; done
sleep 10
echo "=== 기존 6개 완료, burst 한국어 시작 $(date) ===" >> /tmp/real_bench.out
python3 -u /workspace/bench_burst_after.py >> /tmp/real_bench.out 2>&1
echo "=== burst done $(date) ===" >> /tmp/real_bench.out
