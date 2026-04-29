#!/usr/bin/env python3
"""
KB 콜봇 PoC 부하테스트 v3 — TPOT 정확 계산 + per-call timeout
- 채널당 sustained loop (duration_s 동안 turn 반복, 매 turn 사이 gap_s)
- warmup turn 결과 stat 제외
- prompt 다변화 (RAG chunk 풀 + channel/turn-aware seed)
- TPOT = decode_ms / (completion_tokens - 1)  ← chunks 아닌 ct 사용
- per-call timeout 60s 명시 (한 호출 hang 방지)
- 에러율 추적 + 5% 초과 경고

사용법:
  python bench_kb_workflow.py \
    --channels 50 --mode sequential --pattern poisson \
    --duration 120 --gap 4 --warmup-turns 2 \
    --tokenizer /workspace/models/gemma-4-31B-it
"""
import asyncio, aiohttp, time, json, random, argparse, statistics
from pathlib import Path

STEPS = [
    {"name": "query_rewriting",  "in_tokens": 851,  "out_tokens": 20},
    {"name": "retrieval",        "in_tokens": 3147, "out_tokens": 12},
    {"name": "category_answer",  "in_tokens": 4403, "out_tokens": 48},
]

# 다양한 RAG chunk 풀 — 운영 환경 prefix cache hit율 시뮬
RAG_CHUNKS = [
    "[상품정보] 상품명: 군장병내일준비적금 / 금리 연 3.0~5.5% / 가입 6~24개월 / 최소 1만원. ",
    "[가입조건] 만 19세 이상 현역 복무 또는 사회복무요원. KB스타뱅킹 비대면 가입 가능. ",
    "[약관 제3조] 가입 7일내 해지 시 전액 환급. 중도해지 시 기본금리(연 0.1%) 적용. ",
    "[우대조건] 자동이체 +0.3%, 급여이체 +0.2%, 청약저축 보유 +0.2%. 최대 1.0%p 우대. ",
    "[Q&A] Q: 만기전 출금? A: 중도해지 가능, 기본금리 적용. Q: 비과세? A: 한도내 가능. ",
    "[고객사례 #A] 만 21세 사회복무요원, 월 30만원 24개월 가입, 만기 약 728만원 수령(세후). ",
    "[고객사례 #B] 만 23세 현역 병사, 월 50만원 18개월 가입, 만기 약 912만원 수령(세후). ",
    "[비교 상품] KB나라사랑우대통장: 입출금 자유, 우대금리 연 4.0%(특약). 별도 가입 가능. ",
    "[해지 절차] 영업점 방문 또는 KB스타뱅킹 앱 → 마이페이지 → 적금 → 해지. ",
    "[자주 문의] 1) 군 휴가시 이체 가능? Y / 2) 자동이체 변경? KB스타뱅킹에서 가능 / 3) 월 납입 한도? 50만원. ",
    "[수수료] 비대면 가입 무료. 대면 가입 시 인지세 면제 적용. ",
    "[문의처] KB국민은행 콜센터 1599-9999 / 영업시간 평일 9~17시 / 주말 미운영. ",
    "[관련 상품] KB아이꿈적금, KB청년다드림적금, KB ESG적금. 비교 시 만기금액 시뮬레이터 활용. ",
    "[중요 안내] 본 상품은 예금자보호법에 따라 보호되며 보호한도는 1인당 5천만원입니다. ",
    "[변경사항] 2026년 1월 금리 0.2%p 인하. 신규 가입자 우대조건 추가 (군간부 +0.3%). ",
]

# 다양한 query 풀 — 채널/턴 후반에도 같은 prompt 안 되게
QUERIES = [
    "군 장병 적금 연 금리 알려주세요",
    "군 장병 적금 가입 조건이 뭐죠",
    "군장병적금 만기 수령액 시뮬레이션",
    "장병내일준비적금 우대조건 추가 내용",
    "비대면 가입 절차 알려주세요",
    "현역 복무 중인데 가입 가능한가요",
    "사회복무요원도 가입할 수 있나요",
    "월 납입 한도가 어떻게 되나요",
    "중도해지 시 페널티가 있나요",
    "자동이체 우대 조건 알려주세요",
    "만기 후 자동 재가입 되나요",
    "비과세 한도는 어떻게 적용되나요",
    "급여이체 연계하면 우대금리 얼마인가요",
    "가입 후 금액 변경 가능한가요",
    "군 휴가 중에도 입금할 수 있나요",
    "휴직 시 적금 처리는 어떻게 되나요",
    "예금자 보호 범위가 어떻게 되나요",
    "KB 다른 적금 상품과 차이점 알려주세요",
    "온라인 가입 절차 알려주세요",
    "콜센터 운영 시간이 어떻게 되나요",
    "전역 후에도 우대금리 유지되나요",
    "결혼 자금으로 사용하려는데 추천 적금",
    "소득공제 받을 수 있는 적금인가요",
    "주택청약저축과 동시 가입 가능한가요",
    "신용등급에 영향이 있나요",
    "이자소득세는 어떻게 적용되나요",
    "온라인뱅킹 처음인데 가입 절차 알려주세요",
    "공무원 우대 조건이 별도로 있나요",
    "가입 시 필요한 서류가 무엇인가요",
    "재가입 우대 혜택이 있나요",
]


def build_prompt(tok, target_tokens: int, channel_id: int, turn: int, step_idx: int) -> str:
    """채널/턴/스텝별 다른 RAG 청크 조합 + 다양한 query."""
    rng = random.Random(channel_id * 10000 + turn * 100 + step_idx)
    parts = []
    pool = RAG_CHUNKS[:]
    rng.shuffle(pool)
    idx = 0
    while True:
        parts.append(pool[idx % len(pool)])
        idx += 1
        if len(tok.encode("\n".join(parts))) >= target_tokens - 30:
            break
    text = "\n".join(parts)
    enc = tok.encode(text)
    if len(enc) > target_tokens - 30:
        enc = enc[:target_tokens - 30]
        text = tok.decode(enc)
    q = QUERIES[(channel_id * 7 + turn * 3 + step_idx) % len(QUERIES)]
    text += f"\n\n[고객 #{channel_id} turn {turn}] {q}"
    return text


# 한 호출당 timeout — 한 호출 hang 시 sustained loop 안 멈추게
PER_CALL_TIMEOUT = aiohttp.ClientTimeout(total=60, connect=30)


async def call_llm(session, url, model, prompt, max_tokens, ch_id, turn, step_name):
    """1 LLM 호출 (streaming) — TTFT/TPOT/Total 측정."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    fb = last = None
    chunks = 0
    pt = ct = 0
    try:
        async with session.post(url, json=body, timeout=PER_CALL_TIMEOUT) as r:
            if r.status != 200:
                err = f"HTTP {r.status}: {(await r.text())[:200]}"
                return {"ch": ch_id, "turn": turn, "step": step_name, "error": err}
            async for line in r.content:
                if not line: continue
                if fb is None: fb = time.perf_counter()
                s = line.decode("utf-8", errors="ignore").strip()
                if s.startswith("data: ") and "[DONE]" not in s:
                    try:
                        j = json.loads(s[6:])
                        delta = j.get("choices", [{}])[0].get("delta", {})
                        d = delta.get("content", "") or delta.get("reasoning", "")
                        if d: chunks += 1
                        u = j.get("usage")
                        if u:
                            pt = u.get("prompt_tokens", 0)
                            ct = u.get("completion_tokens", 0)
                    except Exception:
                        pass
                last = time.perf_counter()
    except asyncio.TimeoutError:
        return {"ch": ch_id, "turn": turn, "step": step_name, "error": "TIMEOUT_60s"}
    except Exception as e:
        return {"ch": ch_id, "turn": turn, "step": step_name, "error": f"{type(e).__name__}: {e}"}

    total_ms = (last - t0) * 1000 if last else 0
    ttft_ms = (fb - t0) * 1000 if fb else total_ms
    decode_ms = total_ms - ttft_ms
    # TPOT (Time Per Output Token) — completion_tokens 기준 정확 계산
    tpot_ms = decode_ms / max(1, ct - 1) if ct > 1 else 0
    return {
        "ch": ch_id, "turn": turn, "step": step_name,
        "ttft_ms": ttft_ms, "tpot_ms": tpot_ms, "total_ms": total_ms,
        "chunks": chunks, "prompt_tok": pt, "completion_tok": ct,
    }


async def run_turn(session, url, model, tok, ch_id, turn, mode):
    """1 턴 = step1→step2→step3 (sequential) 또는 3 동시 (parallel)."""
    turn_t0 = time.perf_counter()
    if mode == "sequential":
        results = []
        for i, step in enumerate(STEPS):
            prompt = build_prompt(tok, step["in_tokens"], ch_id, turn, i)
            r = await call_llm(session, url, model, prompt, step["out_tokens"], ch_id, turn, step["name"])
            results.append(r)
            if "error" in r:
                break
    else:  # parallel
        tasks = [
            call_llm(session, url, model,
                     build_prompt(tok, step["in_tokens"], ch_id, turn, i),
                     step["out_tokens"], ch_id, turn, step["name"])
            for i, step in enumerate(STEPS)
        ]
        results = await asyncio.gather(*tasks)
    turn_total_ms = (time.perf_counter() - turn_t0) * 1000
    return {"turn": turn, "turn_total_ms": turn_total_ms, "steps": results}


async def channel_loop(session, url, model, tok, ch_id, t_start, duration_s, gap_s, mode, warmup_turns):
    """채널 1개 sustained loop."""
    await asyncio.sleep(t_start)
    ch_real_start = time.perf_counter()
    end_at = ch_real_start + duration_s
    turns = []
    t_idx = 0
    while time.perf_counter() < end_at:
        r = await run_turn(session, url, model, tok, ch_id, t_idx, mode)
        r["is_warmup"] = (t_idx < warmup_turns)
        turns.append(r)
        t_idx += 1
        if time.perf_counter() < end_at:
            await asyncio.sleep(gap_s)
    return {
        "ch": ch_id, "t_start": t_start,
        "real_start_s": ch_real_start, "real_end_s": time.perf_counter(),
        "turns": turns,
    }


def gen_arrivals(n: int, pattern: str, rate: float = 1.0):
    if pattern == "burst":
        return [0.0] * n
    if pattern == "uniform":
        return [i / max(rate, 1e-9) for i in range(n)]
    if pattern == "poisson":
        random.seed(42)
        times, t = [], 0.0
        for _ in range(n):
            times.append(t)
            t += random.expovariate(rate)
        return times
    raise ValueError(pattern)


def percentile(xs, p):
    if not xs: return 0
    xs = sorted(xs)
    return xs[min(int(len(xs) * p / 100), len(xs) - 1)]


def summarize(channel_results, mode, pattern, channels, rate, duration_s, warmup_turns):
    """warmup 제외 통계 + sustained 측정 구간 wall_s 계산."""
    all_steps_ok, all_steps_err, all_turns_ok = [], [], []
    for ch in channel_results:
        for turn in ch["turns"]:
            if turn["is_warmup"]:
                continue
            for s in turn["steps"]:
                if "error" in s:
                    all_steps_err.append(s)
                else:
                    all_steps_ok.append(s)
            if all("error" not in s for s in turn["steps"]):
                all_turns_ok.append(turn["turn_total_ms"])

    total_calls = len(all_steps_ok) + len(all_steps_err)
    err_rate = (len(all_steps_err) / total_calls * 100) if total_calls else 0

    # Effective sustained wall time:
    # 모든 채널이 "동시에" 측정 가능했던 기간
    # = min(real_end) - max(real_start) — 모두 active 였던 구간
    if channel_results:
        max_start = max(c["real_start_s"] for c in channel_results)
        min_end = min(c["real_end_s"] for c in channel_results)
        effective_wall_s = max(min_end - max_start, 0.001)
    else:
        effective_wall_s = duration_s

    print(f"\n=== {mode} / {pattern} / channels={channels} (rate={rate}/s, dur={duration_s}s) ===")
    print(f"  채널 = {channels}  warmup 제외 turn = {sum(len([t for t in c['turns'] if not t['is_warmup']]) for c in channel_results)}")
    print(f"  effective_wall = {effective_wall_s:.1f}s (모든 채널 active 구간)")
    print(f"  LLM 호출 OK={len(all_steps_ok)}  err={len(all_steps_err)}  err_rate={err_rate:.1f}%")
    if err_rate > 5:
        print(f"  ⚠️  에러율 {err_rate:.1f}% > 5% 임계치 — 결과 신뢰도 낮음")

    if not all_steps_ok:
        return {"err_rate": err_rate}

    summary = {"err_rate": err_rate, "effective_wall_s": effective_wall_s, "steps": {}}
    for step_def in STEPS:
        sname = step_def["name"]
        ss = [r for r in all_steps_ok if r["step"] == sname]
        if not ss: continue
        ttft = [r["ttft_ms"] for r in ss]
        tpot = [r["tpot_ms"] for r in ss if r["tpot_ms"] > 0]
        total = [r["total_ms"] for r in ss]
        s = {
            "n": len(ss),
            "ttft_mean": statistics.mean(ttft), "ttft_p95": percentile(ttft, 95), "ttft_p99": percentile(ttft, 99),
            "tpot_mean": statistics.mean(tpot) if tpot else 0,
            "tpot_p95": percentile(tpot, 95) if tpot else 0,
            "total_mean": statistics.mean(total), "total_p95": percentile(total, 95),
        }
        summary["steps"][sname] = s
        print(f"  [{sname:18s}] n={s['n']:4d}  TTFT μ={s['ttft_mean']:6.0f}/p95={s['ttft_p95']:6.0f}ms  "
              f"TPOT μ={s['tpot_mean']:5.1f}ms  Total μ={s['total_mean']:5.0f}ms")

    if all_turns_ok:
        s = {
            "n": len(all_turns_ok),
            "mean": statistics.mean(all_turns_ok),
            "p95": percentile(all_turns_ok, 95),
            "p99": percentile(all_turns_ok, 99),
        }
        summary["turn"] = s
        print(f"  [콜당 Total] n={s['n']:4d}  μ={s['mean']:.0f}ms  p95={s['p95']:.0f}  p99={s['p99']:.0f}")

    total_out = sum(r["completion_tok"] for r in all_steps_ok)
    total_in  = sum(r["prompt_tok"]    for r in all_steps_ok)
    tps_out = total_out / effective_wall_s if effective_wall_s > 0 else 0
    summary["throughput"] = {
        "input_tokens": total_in, "output_tokens": total_out,
        "tps_out": tps_out, "calls_per_s": total_calls / effective_wall_s if effective_wall_s else 0,
    }
    print(f"  [throughput] in={total_in}tok  out={total_out}tok  TPS_out={tps_out:.1f}tok/s  "
          f"calls/s={total_calls/effective_wall_s if effective_wall_s else 0:.2f}")
    return summary


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--channels", type=int, required=True)
    p.add_argument("--mode", choices=["sequential", "parallel"], required=True)
    p.add_argument("--pattern", choices=["uniform", "poisson", "burst"], required=True)
    p.add_argument("--rate", type=float, default=None,
                   help="채널 도착률 req/s. 미지정 시 channels/5초 (5초 안 모든 채널 진입)")
    p.add_argument("--duration", type=float, default=120.0, help="채널당 sustained 측정 시간(초)")
    p.add_argument("--gap", type=float, default=4.0, help="턴 사이 갭(초)")
    p.add_argument("--warmup-turns", type=int, default=2)
    p.add_argument("--url", default="http://localhost:8888/v1/chat/completions")
    p.add_argument("--model", default="gemma-4-31B-it")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.rate is None:
        args.rate = max(args.channels / 5.0, 0.2)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    arrivals = gen_arrivals(args.channels, args.pattern, args.rate)
    print(f"[setup] mode={args.mode} pattern={args.pattern} channels={args.channels} "
          f"rate={args.rate:.2f}/s dur={args.duration}s gap={args.gap}s warmup={args.warmup_turns}")
    print(f"[setup] arrival span: {min(arrivals):.1f}s ~ {max(arrivals):.1f}s")

    conn = aiohttp.TCPConnector(limit=0, limit_per_host=0, keepalive_timeout=600)

    async with aiohttp.ClientSession(connector=conn) as s:
        t0 = time.perf_counter()
        tasks = [
            asyncio.create_task(channel_loop(
                s, args.url, args.model, tok, i, arrivals[i],
                args.duration, args.gap, args.mode, args.warmup_turns
            ))
            for i in range(args.channels)
        ]
        channel_results = await asyncio.gather(*tasks)
        wall_s = time.perf_counter() - t0

    print(f"\n[wall] {wall_s:.1f}s (전체 task)")
    summary = summarize(channel_results, args.mode, args.pattern, args.channels,
                        args.rate, args.duration, args.warmup_turns)

    if args.out:
        Path(args.out).write_text(json.dumps({
            "config": {**vars(args)}, "wall_s": wall_s,
            "summary": summary, "raw": channel_results,
        }, ensure_ascii=False, indent=2))
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
