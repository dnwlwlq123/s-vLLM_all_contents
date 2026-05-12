# vLLM_server_prod

운영 배포용 vLLM 서빙 코드. Gemma4 / Qwen3.5.

## 구조

```
.
├── gemma_server_for_poc/     # Gemma4-31B-it Gateway + vLLM 서빙
│   ├── main.py, run.py       # FastAPI gateway 엔트리
│   ├── start_vllm.sh         # vLLM 기동 (GPU 별 설정 주석 포함)
│   ├── start_gateway.sh      # gateway 기동
│   ├── restart_gateway.sh, stop_vllm.sh
│   ├── gemma4_env.sh         # venv 활성화
│   ├── api/ clients/ config/ core/ services/
│   ├── requirements.txt, requirements_elice.txt
│   ├── Dockerfile, docker-compose.yml, .env.example  # Docker (옵션)
│   └── .dockerignore
│
└── qwen3.5_server/           # Qwen3.5-35B-A3B-FP8 MoE 서빙 스크립트
    ├── qwen35_setup_robust.sh
    ├── start_qwen35{,_v2}.sh
    ├── start_vllm.sh, run_qwen.sh
    └── trtllm_watchdog.sh
```

## 배포 대상

- NHN B200 (SM100) — 현재 Gemma4 운영
- RTX 5090 / PRO 6000 (SM120) — 동일 코드, `start_vllm.sh` 옵션만 조정
- H100 (SM90) — 동일 코드 + FA4 백엔드 사용 가능
- (미래) K8s 멀티 GPU 클러스터

---

## 새 서버 배포 절차 (체크리스트)

### 1. 호스트 전제 조건

| 항목 | 확인 명령 | 비고 |
|---|---|---|
| NVIDIA 드라이버 | `nvidia-smi` | 580+ 권장 (Blackwell), 550+ (H100) |
| CUDA 12.8+ | `nvcc --version` | Blackwell/Hopper 공통 |
| GCC | `gcc --version` | 패치 컴파일 / FlashInfer JIT 용 |
| Python 3.11+ | `python3 --version` | vLLM 0.19+ 요건 |
| Git LFS | `git lfs --version` | 필요 시 |
| 디스크 | `df -h /` | 모델당 60~100GB |
| GPU VRAM | `nvidia-smi --query-gpu=memory.total --format=csv` | 31B FP8 = 최소 40GB VRAM |

### 2. NVML / 드라이버 매칭 (Elice 스타일 호스트에서 필요)

일부 컨테이너/VM 호스트는 드라이버 버전이 NVML 라이브러리와 어긋나서 `nvidia-smi` 가 버전 mismatch 경고 냄. 이때:

```bash
DRIVER_VER=$(cat /proc/driver/nvidia/version | grep -oP 'Module\s+\K[0-9]+\.[0-9]+\.[0-9]+')
NVML_LIB="/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.${DRIVER_VER}"
echo "export LD_PRELOAD=$NVML_LIB" >> ~/.bashrc
source ~/.bashrc
```

RunPod/B200/대부분 정상 호스트는 불필요.

### 3. venv + 의존성

```bash
python3 -m venv ~/venv_gemma4
source ~/venv_gemma4/bin/activate

# vLLM (최신 nightly — FA4 + FP8 KV 는 dev96 이상)
pip install --pre vllm --extra-index-url https://wheels.vllm.ai/nightly

# Gateway deps
cd gemma_server_for_poc
pip install -r requirements.txt
```

### 4. 모델 weights 다운로드

```bash
export HF_HUB_DISABLE_XET=1   # xet backend 에러 회피
mkdir -p ~/models

# Gemma4
hf download google/gemma-4-31b-it --local-dir ~/models/gemma-4-31B-it

# Qwen3.5 (MoE)
hf download Qwen/Qwen3.5-35B-A3B-FP8 --local-dir ~/models/Qwen3.5-35B-A3B-FP8
```

### 5. `start_vllm.sh` 환경별 편집

- `MODEL=` 경로 수정 (다운로드 위치)
- `CUDA_VISIBLE_DEVICES=` GPU index 조정
- GPU 별 attention backend / kv-cache-dtype 선택 (스크립트 상단 주석 참조)

### 6. `.env` 생성 (Gateway)

```bash
cd gemma_server_for_poc
cp .env.example .env
# 포트, vLLM URL 등 서버에 맞게 수정
```

### 7. 네트워크 / 방화벽 (iptables)

**기본 포트**:
- `8000` — vLLM 내부 (localhost 만 열면 됨, 외부 노출 X)
- `17801` — Gateway (외부 오픈 필요)

**iptables 규칙 예 (Ubuntu)**:

```bash
# Gateway 만 외부 허용, vLLM 은 localhost 만
sudo iptables -A INPUT -p tcp --dport 17801 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 8000 -s 127.0.0.1 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 8000 -j DROP
sudo iptables-save | sudo tee /etc/iptables/rules.v4
```

**ufw 쓰는 경우**:

```bash
sudo ufw allow 17801/tcp
sudo ufw deny 8000
sudo ufw enable
```

**클라우드 보안그룹 (AWS/GCP/NHN)**: 인바운드에 17801 만 오픈, 8000 은 절대 오픈 X (vLLM 은 auth 없어서 외부 직접 접근 시 탈취 위험).

### 8. 기동

```bash
# vLLM (호스트 프로세스로 기동, GPU 점유)
cd gemma_server_for_poc
nohup bash start_vllm.sh > /tmp/vllm.log 2>&1 &
tail -f /tmp/vllm.log    # "Application startup complete" 대기 (~2-5분)

# Gateway (별도 터미널 또는 nohup)
bash start_gateway.sh
```

### 9. 정리 / 재기동

```bash
# 기동 중인 프로세스 정리 (프로젝트별 고유 경로로 매치)
pkill -f 'gemma_server_for_poc'   # vllm serve + EngineCore 둘 다 죽음
sleep 3
# 잔재 확인 (자동 kill 금지, 공유 GPU 에서 타 사용자 죽을 위험)
nvidia-smi --query-compute-apps=pid --format=csv,noheader
# 필요 시 수동 PID kill

# 다시 기동
nohup bash start_vllm.sh > /tmp/vllm.log 2>&1 &
```

---

## GPU 별 vLLM 설정 (요약)

자세한 내용은 `gemma_server_for_poc/start_vllm.sh` 상단 주석 참고.

| GPU | attention backend | kv-cache-dtype | 비고 |
|---|---|---|---|
| **B200** (SM100) | `TRITON_ATTN` | `auto` (BF16) 또는 `fp8` | Triton FP8 KV 지원 |
| **H100** (SM90) | `FLASH_ATTN` + `flash_attn_version: 4` | `auto` (BF16) | FA4 + FP8 KV 는 vLLM 패치 필요 (dev repo 참조) |
| **5090 / PRO 6000** (SM120) | `TRITON_ATTN` | `auto` (BF16) 또는 `fp8` | FA4 커널 없음 |

---

## Docker 사용 (옵션, 현재 미사용)

`gemma_server_for_poc/` 안에 Dockerfile / docker-compose.yml 있지만 **지금은 호스트 프로세스로 운영**. 필요 시:

```bash
cd gemma_server_for_poc
cp .env.example .env       # 편집
docker compose up -d       # gateway 만 containerize, vLLM 은 호스트 프로세스 유지
```

vLLM 도 컨테이너에 넣으려면 별도 `Dockerfile.vllm` + `nvidia/cuda` base + GPU device reservation 필요. 지금은 미구현.

---

## 자주 발생하는 이슈

| 이슈 | 해결 |
|---|---|
| `ImportError: cannot import name 'GenerationConfig' from 'transformers'` | vllm CLI 래퍼에 `import transformers, torch` 추가: `sed -i '2a import transformers, torch' $(which vllm)` |
| `TRT-LLM cubin rename error` (Qwen MoE) | `export PATH=/usr/local/cuda/bin:$PATH` 확인 (nvcc PATH 누락 시) + 필요 시 `LD_PRELOAD=rename_fix.so` |
| `Disk quota exceeded` (RunPod) | 모델 캐시 정리 또는 다른 볼륨 이동 |
| `port 8000 already in use` | `pkill -f 'gemma_server_for_poc'` 후 재기동 |
| `fp8 kv-cache not supported` | 현재 vLLM 은 FA4+FP8 KV 거부. Triton backend 쓰거나 BF16 KV 로 |

---

## 참고 repo

- **dev** (김태진 전용, RunPod H100 실험/벤치 포함): `github.com/example/repo`
- **5090 기존 gateway**: `github.com/example/gateway` (AICC 조직, 참고용)
