#!/bin/bash
MODEL="${1:-Qwen3.6-35B-A3B-FP8}"
TOK="${2:-/workspace/models/Qwen3.6-35B-A3B-FP8}"
MODE_ARG="${3:-sequential}"
DURATION="${4:-120}"
GAP_MIN="${5:-1}"
GAP_MAX="${6:-5}"
URL="${URL:-http://localhost:9888/v1/chat/completions}"
PY=/workspace/venv_qwen36_v20/bin/python

OUT_DIR=/workspace/bench_kb_$(date +%Y%m%d_%H%M%S)_${MODEL//\//_}
mkdir -p $OUT_DIR
echo "결과 저장: $OUT_DIR"
echo "  duration=${DURATION}s gap=U(${GAP_MIN}~${GAP_MAX})s model=$MODEL url=$URL"

CHANNELS=(1 5 10 15 20 25 30 35 40 45 50)
PATTERNS=(uniform poisson burst)
[ "$MODE_ARG" = "both" ] && MODES=(sequential parallel) || MODES=("$MODE_ARG")

TOTAL=$((${#CHANNELS[@]} * ${#PATTERNS[@]} * ${#MODES[@]}))
echo "총 ${TOTAL} 시나리오 × ~$((DURATION+10))s = ~$((TOTAL * (DURATION+10) / 60))분"
echo

for MODE in "${MODES[@]}"; do
  for PAT in "${PATTERNS[@]}"; do
    for CH in "${CHANNELS[@]}"; do
      RATE=$(awk "BEGIN{print $CH/5}")
      [ "$PAT" = "burst" ] && RATE_ARG="" || RATE_ARG="--rate $RATE"
      echo "=================================================="
      echo "  $MODE / $PAT / channels=$CH"
      echo "=================================================="
      $PY /workspace/bench_kb_workflow.py \
        --channels $CH --mode $MODE --pattern $PAT $RATE_ARG \
        --duration $DURATION --gap-min $GAP_MIN --gap-max $GAP_MAX --warmup-turns 2 \
        --url "$URL" --model "$MODEL" --tokenizer "$TOK" \
        --out "$OUT_DIR/${MODE}_${PAT}_ch${CH}.json" \
        2>&1 | tee -a $OUT_DIR/run.log
      sleep 5
    done
  done
done

echo
echo "=== sweep 완료 ==="
echo "결과: $OUT_DIR"
ls $OUT_DIR | head -20
