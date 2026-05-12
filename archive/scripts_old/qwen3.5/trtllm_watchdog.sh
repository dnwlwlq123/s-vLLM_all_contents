#!/bin/bash
# Watch TRT-LLM tmp dir and pre-create cache subdir before rename
TMPDIR=/root/.tensorrt_llm/tmp
CACHE=/root/.tensorrt_llm/cache
mkdir -p $TMPDIR $CACHE
inotifywait -m -e create --format "%f" $TMPDIR 2>/dev/null | while read name; do
  # Strip _<timestamp>_<pid> suffix to get cache shape name
  shape=$(echo "$name" | sed -E "s/_[0-9]+_[0-9]+$//")
  if [ -n "$shape" ] && [ "$shape" != "$name" ]; then
    mkdir -p "$CACHE/$shape"
    echo "[watchdog] created $CACHE/$shape" >> /tmp/trtllm_watchdog.log
  fi
done
