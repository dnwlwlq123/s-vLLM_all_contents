#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════
# Qwen3.6-35B-A3B-FP8  /  H100 single GPU  /  vLLM 0.20.0
# ═════════════════════════════════════════════════════════════════════════════
# KB 콜봇 부하테스트 셋업 (2026-04-29 검증 완료, 모든 시도 옵션 반영)
#
# 모델 특성:
#   - Qwen3.6 35B-A3B (active 3B / total 36B), FP8 quantized
#   - Hybrid 40 layers: 30 GDN linear attention + 10 full attention
#   - GDN 레이어는 splitting_ops로 eager 실행 (CUDA graph 밖) ← 본질 한계
#
# 적용된 최적화:
#   - FlashAttention 3 (FA3) for full attention layers
#   - FlashInfer CUTLASS Fp8 MoE backend (auto-selected for H100)
#   - DeepGEMM FP8 weight kernels
#   - FP8 KV cache (fp8_e4m3) — KV 약 380K 토큰
#   - max_num_batched_tokens 192K (ch=40 burst 1 iter 처리)
#   - CUDA graph capture sizes: 1,2,4,8,10,12,14,16,32,50 (sweep 채널 영역)
#   - gpu_mem 0.97 (KV 최대 확보)
#
# 시도했지만 미적용 (이유):
#   - prefix-caching OFF: 운영이 재구성형(cache hit 0%)이라 의미 없음 + Mamba
#                         'align' 모드로 인한 spike 회피
#   - MTP (speculative): 효과 -1% 미미 (출력 토큰 짧아 메리트 작음)
#   - FLASHINFER_TRTLLM MoE: H100(Hopper) 미지원, Blackwell(B200) 전용
#   - enforce-eager: CUDA graph 효과 잃음 (사용 안 함)
#
# 환경:
#   - RunPod H100 SXM 80GB HBM3
#   - venv_qwen36_v20 (vLLM 0.20.0 + FlashInfer cubins + DeepGEMM)
#   - 모델 파일: /workspace/models/Qwen3.6-35B-A3B-FP8/
# ═════════════════════════════════════════════════════════════════════════════

mkdir -p /root/.cache/flashinfer ~/.tensorrt_llm/cache ~/.tensorrt_llm/tmp

VENV=/workspace/venv_qwen36_v20
source $VENV/bin/activate 2>/dev/null || true
export PATH=/usr/local/cuda/bin:$PATH:$VENV/bin

# TRT-LLM cubin rename shim (.flat→.so 버그 회피)
[ -f /workspace/trtllm_fixes/rename_fix.so ] && \
  export LD_PRELOAD=/workspace/trtllm_fixes/rename_fix.so

# Persistent caches on /workspace (Pod 재시작 시 컴파일 재사용)
export TRITON_CACHE_DIR=/workspace/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/workspace/.torchinductor_cache
export VLLM_CACHE_ROOT=/workspace/.vllm_cache
export FLASHINFER_CACHE_DIR=/workspace/.flashinfer_cache
export HF_HOME=/workspace/.hf_cache

# CUDA / vLLM perf env
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_FLOAT32_MATMUL_PRECISION="high"
export VLLM_USE_DEEP_GEMM=1              # FP8 weight GEMM
export VLLM_MOE_USE_DEEP_GEMM=1          # MoE expert GEMM
export VLLM_USE_FLASHINFER_MOE_FP8=1     # FlashInfer FP8 MoE 활성
export FLASHINFER_MOE_BACKEND=throughput # throughput 모드 (latency 모드보다 batch 효율 좋음)
export FLASHINFER_DISABLE_VERSION_CHECK=1

$VENV/bin/vllm serve /workspace/models/Qwen3.6-35B-A3B-FP8 \
  --served-model-name Qwen3.6-35B-A3B-FP8 \
  --host 0.0.0.0 --port 8000 \
  \
  --kv-cache-dtype fp8_e4m3 \             `# FP8 KV cache (fp8 e4m3 format)` \
  --dtype auto \                           `# bf16 baseline (FP8 weights 위에)` \
  --max-model-len 5120 \                   `# 5K context (KB 워크로드 max=4.4K + 여유)` \
  --max-num-seqs 64 \                      `# 동시 in-flight seq 64 (ch=50 여유)` \
  --gpu-memory-utilization 0.97 \          `# 80GB 중 77.6GB 사용 → KV 최대` \
  --max-num-batched-tokens 196608 \        `# 192K — ch=40 burst (40×4400=176K) 1 iter 처리` \
  \
  --enable-chunked-prefill \               `# prefill을 작은 chunk로 → decode와 interleave` \
  --async-scheduling \                     `# 비동기 스케줄러 (forward와 schedule 병렬)` \
  --attention-backend flash_attn \         `# FlashAttention` \
  --attention-config '{"flash_attn_version":3}' \  `# FA3 (Hopper 최적화)` \
  --gdn-prefill-backend triton \           `# GDN linear attn 전용 백엔드` \
  --trust-remote-code \                    `# Qwen 커스텀 코드 허용` \
  --language-model-only \                  `# vision encoder 비활성 (LLM 전용)` \
  \
  --compilation-config '{
    "custom_ops":["all"],
    "cudagraph_capture_sizes":[1,2,4,8,10,12,14,16,32,50]
  }' \
  \
  --disable-log-stats                      `# 로그 잡음 제거 (필요시 빼기)`

# ═════════════════════════════════════════════════════════════════════════════
# 자동 선택 결과 (런타임 로그에서 확인):
#   - FP8 MoE backend: FLASHINFER_CUTLASS (TRTLLM은 H100 미지원)
#   - FP8 Linear: FlashInferFp8DeepGEMMDynamicBlockScaledKernel
#   - GPU KV cache: ~379,376 tokens (Max concurrency 121x at 5K context)
# ═════════════════════════════════════════════════════════════════════════════

# 운영 모드로 prefix-caching 켜고 싶으면 (Mamba 'align' 모드 경고는 정상):
#   --enable-prefix-caching \
# 추가하면 됨. 단 cache hit이 실제로 발생하는 워크로드여야 의미 있음.

# 더 빠르게 하려면:
#   1) TP=2 (2장 H100): 인자에 --tensor-parallel-size 2 추가, prefill +50~80%
#   2) FP4 양자화 모델: 별도 모델 변환 필요
#   3) 모델 교체 (Qwen2.5-32B dense): GDN 없어 prefill 1.5x

# 기동 후 약 6분 (DeepGEMM warmup ~4분 + AutoTuner ~2분)
# 완료 신호: "Application startup complete"
