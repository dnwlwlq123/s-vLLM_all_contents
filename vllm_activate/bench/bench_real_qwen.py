import asyncio, aiohttp, time, random, json, traceback
from transformers import AutoTokenizer

URL = 'http://localhost:8000/v1/chat/completions'
MODEL = 'Qwen3.5-35B-A3B-FP8'
TOK_PATH = '/workspace/models/Qwen3.5-35B-A3B-FP8'
SYS_PATH = '/workspace/answer_prompt.md'
TARGET_PROMPT_TOKS = 7800
OUTPUT_TOKS = 100
N_REQUESTS = 50

print('Loading tokenizer...', flush=True)
tok = AutoTokenizer.from_pretrained(TOK_PATH)
SYSTEM = open(SYS_PATH).read()
SYS_TOKS = len(tok.encode(SYSTEM))
USER_PAD_TOKS = TARGET_PROMPT_TOKS - SYS_TOKS - 30
print(f'System prompt: {SYS_TOKS} tokens', flush=True)
print(f'Target user msg: {USER_PAD_TOKS} tokens', flush=True)

RAG_TEMPLATES = [
    '[상품 정보]\n상품명: {name}\n금리: 연 {rate}% / 세후 {rate2}%\n가입기간: {period}개월\n최소금액: {amount}만원\n세금: 이자소득세 15.4% 원천징수\n중도해지: 중도해지 시 기본금리 적용\n가입조건: 만 {age}세 이상 개인, 비대면 가입 가능\n상세약관: 이 상품은 ECS은행 고객을 위한 특별 우대 상품으로 자동이체 연계 시 추가 금리 혜택이 있으며 청소년, 군인, 주택청약 연계 고객은 별도 우대금리가 적용됩니다.\n' * 3,
    '[약관 내용]\n제{n}조 (계약의 성립) ①본 상품은 고객이 {action}함으로써 효력이 발생합니다. ②가입 후 {days}일 이내 해지 시 전액 환급됩니다. ③중도해지 시 경과일수에 따라 차등 금리가 적용되며 자세한 사항은 약관을 참조하시기 바랍니다.\n' * 3,
    '[Q&A 자주 묻는 질문]\nQ: 만기 전 출금 가능한가요?\nA: 중도해지 가능하며 기본금리(연 0.1%)가 적용됩니다. 경과일수에 따라 차등 금리 적용되니 신중히 결정해주세요.\nQ: 세금 혜택이 있나요?\nA: 비과세 한도 내에서 가능하며 자세한 조건은 상담원에게 문의 바랍니다.\nQ: 가입 방법은?\nA: 영업점 방문 또는 ECS은행 앱을 통한 비대면 가입이 가능합니다.\n' * 3,
]

def gen_user_msg(seed):
    random.seed(seed)
    parts = []
    tgt = USER_PAD_TOKS
    while True:
        t = random.choice(RAG_TEMPLATES).format(
            name=f'대신{random.randint(1,999)}적금', rate=round(random.uniform(2.5,5.0),2),
            rate2=round(random.uniform(2.0,4.2),2), period=random.choice([6,12,24,36]),
            amount=random.choice([10,50,100,500]), age=random.randint(18,65),
            n=random.randint(1,30), action=random.choice(['청약','가입','신청']),
            days=random.choice([7,14,30]))
        parts.append(t)
        cur = len(tok.encode('\n'.join(parts)))
        if cur >= tgt: break
    return '\n'.join(parts) + f'\n\n[고객 질문] 이 상품 가입 조건 간단히 알려주세요 (요청#{seed})'

def gen_arrivals(n, rate, burstiness=1.0):
    # burstiness=1.0: Poisson (exponential). <1.0: clumpy (gamma)
    # inter-arrival ~ Gamma(shape=burstiness, scale=1/(rate*burstiness))
    times = []; t = 0
    for _ in range(n):
        times.append(t)
        inter = random.gammavariate(burstiness, 1.0/(rate*burstiness))
        t += inter
    return times

async def send(sess, i, delay, results):
    await asyncio.sleep(delay)
    user = gen_user_msg(i)
    body = {'model':MODEL,'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':user}],
            'max_tokens':OUTPUT_TOKS,'temperature':0.3,'stream':True,'chat_template_kwargs':{'enable_thinking':False},'stream_options':{'include_usage':True}}
    t0 = time.perf_counter()
    first_byte=None; first_punct=None; last=None; chunks=0
    prompt_tokens=0; completion_tokens=0
    try:
        async with sess.post(URL, json=body, timeout=aiohttp.ClientTimeout(total=300)) as resp:
            async for line in resp.content:
                if not line: continue
                if first_byte is None: first_byte = time.perf_counter()
                s = line.decode('utf-8', errors='ignore').strip()
                if s.startswith('data: ') and '[DONE]' not in s:
                    try:
                        j = json.loads(s[6:])
                        d = j.get('choices',[{}])[0].get('delta',{}).get('content','')
                        if d:
                            chunks += 1
                            if first_punct is None and any(p in d for p in '.,?!。'):
                                first_punct = time.perf_counter()
                        if 'usage' in j and j['usage']:
                            prompt_tokens = j['usage'].get('prompt_tokens',0)
                            completion_tokens = j['usage'].get('completion_tokens',0)
                    except: pass
                last = time.perf_counter()
    except Exception as e:
        results.append({'i':i,'error':str(e)})
        return
    total = (last-t0)*1000
    ttft = (first_byte-t0)*1000 if first_byte else total
    punct = (first_punct-t0)*1000 if first_punct else 0
    decode = total - ttft
    itl = decode/(chunks-1) if chunks > 1 else 0
    results.append({'i':i,'ttft':ttft,'itl':itl,'punct':punct,'total':total,'chunks':chunks,
                    'prompt':prompt_tokens,'completion':completion_tokens})

async def run(rate, burstiness, label):
    print(f'\n=== {label} (rate={rate}, burstiness={burstiness}) ===', flush=True)
    random.seed(42)  # reproducible arrivals
    arrivals = gen_arrivals(N_REQUESTS, rate, burstiness)
    print(f'arrival window: {max(arrivals):.1f}s', flush=True)
    results = []
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn) as sess:
        tasks = [asyncio.create_task(send(sess, i, arrivals[i], results)) for i in range(N_REQUESTS)]
        await asyncio.gather(*tasks)
    return results

def stats(xs):
    if not xs: return (0,0,0,0)
    xs = sorted(xs); n = len(xs)
    return (sum(xs)/n, xs[n//2], xs[min(int(n*0.95),n-1)], xs[min(int(n*0.99),n-1)])

async def main():
    CONFIGS = [
        ('0.5s_텀',       2.0,  1.0),
        ('1.0s_텀',       1.0,  1.0),
        ('1.5s_텀',       0.67, 1.0),
        ('2.0s_텀',       0.5,  1.0),
        ('peak_burst',    10,   0.5),
        ('surge_폭주',    20,   0.3),
    ]
    all_out = {}
    for label, rate, burst in CONFIGS:
        try:
            rs = await run(rate, burst, label)
            ok = [r for r in rs if 'error' not in r]
            err = len(rs) - len(ok)
            ttft_m, _, ttft_p95, ttft_p99 = stats([r['ttft'] for r in ok])
            itl_m, _, itl_p95, itl_p99 = stats([r['itl'] for r in ok])
            punct_m, _, punct_p95, _ = stats([r['punct'] for r in ok if r['punct']>0])
            tot_m, _, tot_p95, tot_p99 = stats([r['total'] for r in ok])
            prompt_avg = sum(r['prompt'] for r in ok)/max(len(ok),1)
            comp_avg = sum(r['completion'] for r in ok)/max(len(ok),1)
            all_out[label] = dict(rate=rate,burstiness=burst,success=len(ok),failed=err,
                ttft_mean=ttft_m,ttft_p95=ttft_p95,ttft_p99=ttft_p99,
                itl_mean=itl_m,itl_p95=itl_p95,itl_p99=itl_p99,
                punct_mean=punct_m,punct_p95=punct_p95,
                total_mean=tot_m,total_p95=tot_p95,total_p99=tot_p99,
                prompt_avg=prompt_avg, completion_avg=comp_avg)
            print(f'  success={len(ok)}/{N_REQUESTS} err={err}', flush=True)
            print(f'  TTFT  mean={ttft_m:.0f} p95={ttft_p95:.0f} p99={ttft_p99:.0f}', flush=True)
            print(f'  ITL   mean={itl_m:.1f} p95={itl_p95:.1f} p99={itl_p99:.1f}', flush=True)
            print(f'  punct mean={punct_m:.0f} p95={punct_p95:.0f}', flush=True)
            print(f'  Total mean={tot_m:.0f} p95={tot_p95:.0f} p99={tot_p99:.0f}', flush=True)
            print(f'  prompt~{prompt_avg:.0f}tok completion~{comp_avg:.0f}tok', flush=True)
        except Exception as e:
            print(f'  FAILED: {e}', flush=True)
            traceback.print_exc()
        with open('/tmp/triton_fp8kv_results.json','w') as f:
            json.dump(all_out, f, indent=2, ensure_ascii=False)
        await asyncio.sleep(5)

asyncio.run(main())
print('ALL DONE', flush=True)
