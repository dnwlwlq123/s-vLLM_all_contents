# vLLM serving experiments

Personal collection of vLLM serving / gateway / benchmark scripts and patches
collected while exploring large MoE + dense FP8 models on H100 / Blackwell GPUs.

## Layout

| Dir              | Purpose                                                                 |
|------------------|-------------------------------------------------------------------------|
| `gemma_server/`  | FastAPI gateway + vLLM startup for Gemma 4 31B-it (FA4 + TRITON + xgrammar)   |
| `scripts/`       | vLLM startup scripts for Qwen3.6 / Gemma 4 + benchmark harness          |
| `gateway/`       | OpenAI-compatible vLLM gateway (streaming TTFT/ITL logs, thinking off)  |
| `trtllm_fixes/`  | LD_PRELOAD shim for the TRT-LLM cubin rename bug                        |
| `vllm_patches/`  | FA4 + FP8 KV gating patches                                             |
| `results/`       | Sweep result JSONs (poisson + burst across channel counts)              |
| `docs/`          | Notes                                                                   |
| `archive/`       | Older setup snapshots                                                   |

## Notes

- Targets vLLM `0.20.0` (FA3 / FA4 + FP8 KV combinations).
- Gateway uses HTTP/2 + keepalive to a local vLLM (port 8000 by default).
- Bench harness drives a 3-step conversation workflow per turn to measure
  TTFT / ITL / first-punctuation latency under sustained load.
