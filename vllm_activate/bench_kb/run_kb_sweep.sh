#!/bin/bash
# KB workflow sweep — 채널 × 패턴 × 모드 자동 측정
# 사용법: bash run_kb_sweep.sh [model_name] [tokenizer_path] [mode]
#   mode: sequential | parallel | both
MODEL="${1:-gemma-4-31B-it}"
TOK="${2:-/workspace/models/gemma-4-31B-it}"
MODE_ARG="${3:-both}"
URL="${URL:-http://localhost:8888/v1/chat/completions}"
PY=/workspace/venv_qwen36_v20/bin/python

OUT_DIR=/workspace/bench_kb_$(date +%Y%m%d_%H%M%S)_${MODEL//\//_}
mkdir -p $OUT_DIR
echo "결과 저장: $OUT_DIR"

CHANNELS=(1 5 10 15 20 25 30 35 40 45 50)
PATTERNS=(uniform poisson burst)
[ "$MODE_ARG" = "both" ] && MODES=(sequential parallel) || MODES=("$MODE_ARG")

# rate (uniform/poisson용): 채널수 / 5초로 하면 5초 안에 다 도착
# burst는 rate 무시
for MODE in "${MODES[@]}"; do
  for PAT in "${PATTERNS[@]}"; do
    for CH in "${CHANNELS[@]}"; do
      RATE=$(echo "scale=2; $CH / 5" | bc)
      [ "$PAT" = "burst" ] && RATE=999
      echo
      echo "=================================================="
      echo "  $MODE / $PAT / channels=$CH / rate=$RATE"
      echo "=================================================="
      $PY /workspace/bench_kb_workflow.py \
        --channels $CH --mode $MODE --pattern $PAT --rate $RATE \
        --url "$URL" --model "$MODEL" --tokenizer "$TOK" \
        --out "$OUT_DIR/${MODE}_${PAT}_ch${CH}.json" \
        2>&1 | tee -a $OUT_DIR/run.log
      sleep 3  # GPU cool-down
    done
  done
done

echo
echo "=== sweep 완료 ==="
echo "결과: $OUT_DIR"
ls $OUT_DIR | head -20
