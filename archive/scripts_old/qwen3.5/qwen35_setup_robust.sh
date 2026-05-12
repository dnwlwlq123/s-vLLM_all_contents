#!/bin/bash
# Qwen3.5-35B-A3B MoE 안정 기동 스크립트 — 알려진 모든 함정 회피
# Usage: bash qwen35_setup_robust.sh
set -e

MODEL_PATH=${MODEL_PATH:-/workspace/models/Qwen3.5-35B-A3B-FP8}
LOG=/tmp/vllm.log

echo "=== Pre-flight 체크 ==="

# 1. 드라이버 확인 (570+ 필요)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "$DRIVER" -lt "570" ]; then
  echo "❌ Driver $DRIVER < 570. FlashInfer MoE FP8 deadlock 발생. 드라이버 업데이트 필요."
  exit 1
fi
echo "✅ Driver $DRIVER OK"

# 2. CUDA toolkit 확인 (12.6+ 필요 — cuda::ptx::n32_t API)
NVCC=$(which nvcc 2>/dev/null || ls /usr/local/cuda*/bin/nvcc 2>/dev/null | head -1)
if [ -z "$NVCC" ]; then
  echo "⚠️  nvcc not in PATH — TRT-LLM JIT 컴파일 실패 가능"
fi

# 3. 모델 존재 확인
if [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "❌ Model not found at $MODEL_PATH"
  exit 1
fi
echo "✅ Model $MODEL_PATH"

# 4. vLLM 패키지 확인
pip show vllm >/dev/null 2>&1 || { echo "❌ vllm not installed"; exit 1; }
pip show pandas >/dev/null 2>&1 || { echo "⚠️  pandas missing — vllm v1 일부 경로에서 ImportError"; pip install --break-system-packages pandas; }
echo "✅ vLLM + pandas"

echo ""
echo "=== 캐시/환경 초기화 ==="

# 5. TensorRT-LLM cache subdir 사전 생성 (rename 버그 회피)
mkdir -p ~/.tensorrt_llm/cache
mkdir -p ~/.tensorrt_llm/tmp
mkdir -p ~/.cache/flashinfer
mkdir -p ~/.cache/vllm
chmod -R 777 ~/.tensorrt_llm 2>/dev/null
echo "✅ Cache 디렉토리 준비됨"

# 6. 깨진 부분 cubin (이전 실패 잔재) 청소 — 0byte 파일 / 빈 디렉토리만
find ~/.tensorrt_llm/cache -type f -size 0 -delete 2>/dev/null
find ~/.tensorrt_llm/cache -type d -empty -delete 2>/dev/null
echo "✅ 깨진 cache 청소"

# 7. FLA torch.compile 패치 (이미 됐으면 skip)
FLA_FILE=$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')/model_executor/layers/fla/ops/utils.py
if grep -q "torch.accelerator.device_index" "$FLA_FILE" 2>/dev/null; then
  sed -i "s|torch.accelerator.device_index(tensor.device.index)|torch.cuda.device(tensor.device.index)|" "$FLA_FILE"
  echo "✅ FLA torch.compile 패치 적용"
else
  echo "✅ FLA 패치 이미 적용됨"
fi

# 8. TRT-LLM rename 버그 우회용 watcher (백그라운드)
pkill -f trtllm_subdir_watcher 2>/dev/null
cat > /tmp/trtllm_subdir_watcher.sh <<'WATCH'
#!/bin/bash
TMPDIR=$HOME/.tensorrt_llm/tmp
CACHE=$HOME/.tensorrt_llm/cache
mkdir -p $TMPDIR $CACHE
while :; do
  for d in $(ls $TMPDIR 2>/dev/null); do
    shape=$(echo "$d" | sed -E "s/_[0-9]+_[0-9]+$//")
    [ -n "$shape" ] && [ "$shape" != "$d" ] && [ ! -d "$CACHE/$shape" ] && mkdir -p "$CACHE/$shape"
  done
  sleep 0.005
done
WATCH
chmod +x /tmp/trtllm_subdir_watcher.sh
nohup /tmp/trtllm_subdir_watcher.sh > /dev/null 2>&1 & disown
echo "✅ TRT-LLM subdir watcher 가동 (PID=$!)"

echo ""
echo "=== 환경변수 설정 ==="
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_FLOAT32_MATMUL_PRECISION="high"
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_MOE_FP8=1     # FlashInfer 시도, 실패 시 fallback
export FLASHINFER_MOE_BACKEND=throughput  # CUTLASS 경로 (latency=TRTLLM 은 SM100+ 전용)
unset MAX_JOBS NINJA_MAX_JOBS
env | grep -E "^VLLM_|^FLASHINFER_" | sort
echo ""

# 9. 기존 vLLM 종료
pgrep -f "vllm serve" | xargs -r kill -9 2>/dev/null
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
sleep 4

echo "=== vLLM 기동 ==="
nohup vllm serve "$MODEL_PATH" \
  --served-model-name Qwen3.5-35B-A3B-FP8 \
  --host 0.0.0.0 --port 8000 \
  --kv-cache-dtype fp8_e4m3 \
  --dtype auto \
  --max-model-len 16384 \
  --max-num-seqs 50 \
  --gpu-memory-utilization 0.96 \
  --max-num-batched-tokens 32768 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --async-scheduling \
  --attention-backend flash_attn \
  --attention-config '{"flash_attn_version":3}' \
  --gdn-prefill-backend triton \
  --reasoning-parser qwen3 \
  --trust-remote-code \
  --language-model-only \
  --disable-log-stats \
  > $LOG 2>&1 & disown

echo "vLLM PID=$!"
echo ""
echo "기동 모니터: tail -f $LOG"
echo "준비 확인:    curl http://localhost:8000/v1/models"
echo ""

# 10. Startup 대기 + FlashInfer 실패 시 자동 fallback
echo "=== 기동 대기 (최대 15분, FlashInfer 실패 시 Triton MoE 로 자동 fallback) ==="
for i in $(seq 1 180); do
  sleep 5
  if grep -q "Application startup complete" $LOG 2>/dev/null; then
    KV=$(grep -aoE "GPU KV cache size: [0-9,]+" $LOG | tail -1)
    echo "✅ 기동 완료 ($((i*5))s, $KV)"
    exit 0
  fi
  if grep -q "Assertion failed: !cubin.empty" $LOG 2>/dev/null; then
    echo "⚠️  FlashInfer MoE FP8 실패 (TRT-LLM cubin 버그). Triton MoE 로 fallback…"
    pgrep -f "vllm serve" | xargs -r kill -9
    sleep 4
    rm -rf ~/.tensorrt_llm/cache/* ~/.tensorrt_llm/tmp/*
    export VLLM_USE_FLASHINFER_MOE_FP8=0   # ← Triton MoE 강제
    nohup vllm serve "$MODEL_PATH" \
      --served-model-name Qwen3.5-35B-A3B-FP8 \
      --host 0.0.0.0 --port 8000 \
      --kv-cache-dtype fp8_e4m3 --dtype auto \
      --max-model-len 16384 --max-num-seqs 50 \
      --gpu-memory-utilization 0.96 --max-num-batched-tokens 32768 \
      --enable-prefix-caching --enable-chunked-prefill --async-scheduling \
      --attention-backend flash_attn --attention-config '{"flash_attn_version":3}' \
      --gdn-prefill-backend triton --reasoning-parser qwen3 \
      --trust-remote-code --language-model-only --disable-log-stats \
      > $LOG 2>&1 & disown
    echo "Triton MoE fallback PID=$!"
    continue
  fi
done
echo "❌ 기동 timeout"
exit 1
