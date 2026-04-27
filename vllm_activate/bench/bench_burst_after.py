import asyncio, aiohttp, time, random, json
from transformers import AutoTokenizer

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "gemma-4-31B-it"
TOK = AutoTokenizer.from_pretrained("/workspace/models/gemma-4-31B-it")
SYSTEM = open("/workspace/answer_prompt.md").read()
SYS_TOKS = len(TOK.encode(SYSTEM))
USER_PAD = 8000 - SYS_TOKS - 150   # 여유 150
N = 50
print(f"SYS={SYS_TOKS} USER_PAD={USER_PAD}", flush=True)

RAG = [
  "[상품 정보]\n상품명: {n}\n금리: 연 {r}%\n가입: {p}개월 / {a}만원\n세금: 이자소득세 15.4% 원천징수\n가입조건: 만 19세 이상 개인, 비대면 가입 가능\n" * 4,
  "[약관]\n제{i}조 (계약) ①가입 후 {d}일 이내 해지 시 환급. ②중도해지 시 기본금리 적용. ③자세한 내용은 약관 참조.\n" * 4,
  "[Q&A]\nQ: 만기전 출금 가능합니까?\nA: 중도해지 가능하며 기본금리가 적용됩니다.\nQ: 비과세 한도는?\nA: 상담원에게 문의 바랍니다.\n" * 5,
]
def user_msg(seed):
    random.seed(seed); parts=[]
    while True:
        t = random.choice(RAG).format(n=f"적금{random.randint(1,999)}",r=round(random.uniform(2,5),2),
            p=random.choice([12,24,36]),a=random.choice([50,100,500]),
            i=random.randint(1,30),d=random.choice([7,14,30]))
        parts.append(t)
        if len(TOK.encode("\n".join(parts))) >= USER_PAD: break
    return "\n".join(parts)+f"\n\n[고객 질문] 가입 조건 안내 부탁드립니다 (req#{seed})"

KR_PUNCT = ("다.","요.","까?","니다","어요","세요","죠.","세.")
async def send(sess, i, results):
    body = {"model":MODEL,"messages":[{"role":"system","content":SYSTEM},{"role":"user","content":user_msg(i)}],
            "max_tokens":100,"temperature":0.3,"stream":True,"stream_options":{"include_usage":True}}
    t0 = time.perf_counter()
    fb=None; fp=None; last=None; chunks=0; pt=0; ct=0; text_accum=""
    try:
        async with sess.post(URL, json=body, timeout=aiohttp.ClientTimeout(total=600)) as resp:
            if resp.status != 200:
                err = await resp.text()
                results.append({"i":i,"error":f"HTTP {resp.status}: {err[:200]}"})
                return
            async for line in resp.content:
                if not line: continue
                if fb is None: fb = time.perf_counter()
                s = line.decode("utf-8",errors="ignore").strip()
                if s.startswith("data: ") and "[DONE]" not in s:
                    try:
                        j = json.loads(s[6:])
                        d = j.get("choices",[{}])[0].get("delta",{}).get("content","")
                        if d:
                            chunks += 1
                            text_accum += d
                            if fp is None and any(p in text_accum for p in KR_PUNCT):
                                fp = time.perf_counter()
                        if "usage" in j and j["usage"]:
                            pt = j["usage"].get("prompt_tokens",0); ct = j["usage"].get("completion_tokens",0)
                    except: pass
                last = time.perf_counter()
    except Exception as e:
        results.append({"i":i,"error":str(e)}); return
    tot = (last-t0)*1000
    ttft = (fb-t0)*1000 if fb else tot
    itl = (last-fb)*1000/(chunks-1) if (fb and chunks>1) else 0
    results.append({"i":i,"ttft":ttft, "punct":(fp-t0)*1000 if fp else 0,
                    "itl":itl,"total":tot,"chunks":chunks,"prompt":pt,"completion":ct})

def stats(xs):
    if not xs: return (0,0,0,0)
    xs = sorted(xs); n = len(xs)
    return (sum(xs)/n, xs[n//2], xs[min(int(n*0.95),n-1)], xs[min(int(n*0.99),n-1)])

async def main():
    print("=== BURST 완전동시 50 (Korean 8K) ===", flush=True)
    results = []
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn) as sess:
        tasks = [asyncio.create_task(send(sess, i, results)) for i in range(N)]
        await asyncio.gather(*tasks)
    ok = [r for r in results if "error" not in r]
    errs = [r for r in results if "error" in r]
    ttft_m,_,ttft_p95,ttft_p99 = stats([r["ttft"] for r in ok])
    itl_m,_,itl_p95,itl_p99 = stats([r["itl"] for r in ok])
    punct_m,_,punct_p95,_ = stats([r["punct"] for r in ok if r["punct"]>0])
    tot_m,_,tot_p95,tot_p99 = stats([r["total"] for r in ok])
    print(f"  success={len(ok)}/{N} err={len(errs)}")
    for e in errs[:3]: print(f"  err sample: {e}")
    print(f"  TTFT  mean={ttft_m:.0f} p95={ttft_p95:.0f} p99={ttft_p99:.0f}")
    print(f"  ITL   mean={itl_m:.1f} p95={itl_p95:.1f} p99={itl_p99:.1f}")
    print(f"  punct mean={punct_m:.0f} p95={punct_p95:.0f}")
    print(f"  Total mean={tot_m:.0f} p95={tot_p95:.0f} p99={tot_p99:.0f}")
    out = {"burst_korean":dict(rate="inf",burstiness=0,success=len(ok),failed=len(errs),
        ttft_mean=ttft_m,ttft_p95=ttft_p95,ttft_p99=ttft_p99,
        itl_mean=itl_m,itl_p95=itl_p95,itl_p99=itl_p99,
        punct_mean=punct_m,punct_p95=punct_p95,
        total_mean=tot_m,total_p95=tot_p95,total_p99=tot_p99)}
    try: prev = json.load(open("/tmp/triton_fp8kv_results.json"))
    except: prev = {}
    prev.update(out)
    json.dump(prev, open("/tmp/triton_fp8kv_results.json","w"), indent=2, ensure_ascii=False)

asyncio.run(main())
print("ALL DONE", flush=True)
