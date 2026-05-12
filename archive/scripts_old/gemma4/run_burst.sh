#!/bin/bash
# 기존 실 프롬프트 벤치 끝날 때까지 대기
while pgrep -f 'bench_real.py' > /dev/null; do sleep 5; done
R=/tmp/burst_results.txt
> $R
for SETUP in '10:0.5:peak_burst_50ms평균_실제몰림' '20:0.3:surge_극한폭주'; do
  RATE=$(echo $SETUP | cut -d: -f1)
  BURST=$(echo $SETUP | cut -d: -f2)
  LABEL=$(echo $SETUP | cut -d: -f3)
  echo "###############################################" | tee -a $R
  echo "### $LABEL (rate=$RATE, burstiness=$BURST)" | tee -a $R
  echo "###############################################" | tee -a $R
  vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
    --model gemma-4-31B-it --tokenizer /workspace/models/gemma-4-31B-it \
    --dataset-name random \
    --random-input-len 8000 --random-output-len 100 \
    --num-prompts 50 --max-concurrency 50 \
    --request-rate $RATE --burstiness $BURST \
    --metric-percentiles '50,95,99' \
    --host localhost --port 8000 2>&1 | grep -E 'Successful|Failed|Benchmark duration|TTFT|TPOT|ITL|Request throughput|Output token throughput' | tee -a $R
  echo '' | tee -a $R
  sleep 5
done
echo '===BURST DONE===' >> $R
