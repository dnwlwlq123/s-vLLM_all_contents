import asyncio, aiohttp, time, json, sys

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "gemma-4-31B-it"
SYS = open("/workspace/answer_prompt.md").read()

# ECS 은행 적금 상담 6턴 시나리오 (누적 대화)
TURNS = [
    ("인사/문의",       "안녕하세요, 목돈 모으려고 적금 들려고 하는데요"),
    ("상품 소개 요청",  "요즘 금리 높은 상품 있나요?"),
    ("금리 상세",       "연 4% 이상 되는 상품으로 추천해주세요"),
    ("가입조건",        "가입하려면 어떤 조건이 필요한가요?"),
    ("한도/기간",       "월 50만원씩 12개월 넣으면 만기에 얼마 받나요?"),
    ("최종확인",        "네 그걸로 가입할게요. 비대면으로 가능한가요?"),
]

async def call(sess, messages, label, tag):
    body = {"model":MODEL,"messages":messages,"max_tokens":300,"temperature":0.3,
            "stream":True,"stream_options":{"include_usage":True}}
    t0 = time.perf_counter()
    ttft=None; first_punct=None; last=None
    chunks=0; text=""
    prompt_tokens=0; completion_tokens=0
    async with sess.post(URL, json=body, timeout=aiohttp.ClientTimeout(total=300)) as resp:
        async for line in resp.content:
            if not line: continue
            s = line.decode("utf-8", errors="ignore").strip()
            if s.startswith("data: ") and "[DONE]" not in s:
                try:
                    j = json.loads(s[6:])
                    d = j.get("choices",[{}])[0].get("delta",{}).get("content","")
                    if d:
                        if ttft is None: ttft = time.perf_counter() - t0
                        chunks += 1
                        text += d
                        if first_punct is None and any(p in d for p in ".,?!。"):
                            first_punct = time.perf_counter() - t0
                    if "usage" in j and j["usage"]:
                        prompt_tokens = j["usage"].get("prompt_tokens",0)
                        completion_tokens = j["usage"].get("completion_tokens",0)
                except: pass
                last = time.perf_counter() - t0
    total = last
    itl = (total - ttft) / max(chunks-1, 1) * 1000 if ttft else 0
    print(f"[{tag}] {label}")
    print(f"  prompt_toks={prompt_tokens} completion_toks={completion_tokens}")
    print(f"  TTFT={ttft*1000:.0f}ms  ITL={itl:.1f}ms  first_punct={first_punct*1000 if first_punct else 0:.0f}ms  Total={total*1000:.0f}ms")
    print(f"  응답: {text[:200]}")
    print()
    return text

async def main():
    async with aiohttp.ClientSession() as sess:
        # Warmup (결과 무시)
        print("=== WARMUP ===")
        await call(sess, [{"role":"system","content":SYS},{"role":"user","content":"안녕"}], "warmup", "W")
        # Actual 6 turns (각 턴은 이전 history 포함)
        print("=== 6-TURN CALLBOT ===")
        history = [{"role":"system","content":SYS}]
        for i,(lbl,utt) in enumerate(TURNS, 1):
            history.append({"role":"user","content":utt})
            resp = await call(sess, history, lbl, f"T{i}")
            history.append({"role":"assistant","content":resp})

asyncio.run(main())
