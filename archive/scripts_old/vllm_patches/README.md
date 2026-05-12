# vLLM 패치 — FA4 + FP8 KV cache on H100

## 배경

vLLM 0.19.2rc1.dev96 (및 main branch) 는 FlashAttention v4 backend 에서 FP8 KV cache 조합을 거부함:
- `fa_utils.py:flash_attn_supports_fp8()` 은 FA3 + SM90 만 True
- `FlashAttentionBackend.supported_kv_cache_dtypes` 리스트에 fp8 없음
- FlashAttentionImpl.__init__ 에서 quantized KV 만나면 `NotImplementedError` raise

## 실제 FA4 kernel 은?

Dao-AILab 의 flash-attn v4 kernel 자체는 FP8 KV cache 를 받도록 설계되어 있음. vLLM 쪽 gating 코드만 미수용 상태. 따라서 vLLM 쪽 두 지점을 열면 사용 가능 (kernel symbol 존재는 런타임 확인 필요).

## 변경점

**01_fa4_fp8_kv_gating.patch** — `vllm/v1/attention/backends/fa_utils.py`
- `flash_attn_supports_fp8()` 에서 `get_flash_attn_version() == 3` → `in (3, 4)`

**02_fa_supported_kv_cache_dtypes.patch** — `vllm/v1/attention/backends/flash_attn.py`
- `FlashAttentionBackend.supported_kv_cache_dtypes` 리스트에 `"fp8", "fp8_e4m3", "fp8_e5m2"` 추가

## 적용

```bash
VENV=/workspace/venv_gemma4 bash apply_patches.sh
```

## 대상 버전

`vllm==0.19.2rc1.dev96+g6ff8dea07` (main branch 기준, 2026-04-22 시점 nightly 최신)

## Gemma4 특수 주의

Gemma4 는 레이어마다 다른 head_dim 사용 (SWA=256, Global=512). FA kernel head_size 상한 256 때문에 Global 레이어는 Triton backend 자동 fallback.
`--kv-cache-dtype fp8` 지정 시:
- SWA (FA4) → 이 패치 덕에 수용
- Global (Triton) → Triton 이 자체 FP8 KV 지원 (공식)

즉 전 레이어 FP8 KV 가능. 단 FA4 kernel 의 FP8 KV symbol 이 wheel 에 실제 빌드됐는지는 기동 시점 확인.
