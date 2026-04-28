#!/bin/bash
# =============================================================================
# vLLM Setup Toolkit — RunPod / 일반 H100 서버에서 vLLM 환경 한 번에 구성
# =============================================================================
# 사용법:
#   bash vllm_toolkit.sh setup    <venv_name> [vllm_version=0.20.0]
#   bash vllm_toolkit.sh model    <hf_repo>   [target_dir]
#   bash vllm_toolkit.sh runpod-fixes
#   bash vllm_toolkit.sh status
#   bash vllm_toolkit.sh activate <venv_name>     # eval $(...) 용
#
# 예시:
#   bash vllm_toolkit.sh setup venv_qwen36_v20 0.20.0
#   bash vllm_toolkit.sh model Qwen/Qwen3.6-35B-A3B-FP8
#   bash vllm_toolkit.sh status
# =============================================================================

set -e

WORKSPACE="${WORKSPACE:-/workspace}"

# -----------------------------------------------------------------------------
# 알려진 이슈 회피 메모 (이 스크립트가 자동으로 처리하는 것):
#   1. PEP 668 (system Python externally-managed)
#      → 항상 venv 안의 python을 절대경로로 호출 ($VENV/bin/python -m pip)
#   2. venv 디렉터리 mv 시 shebang 깨짐
#      → mv 금지, 처음부터 원하는 이름으로 생성
#   3. system pip 폴백 방지
#      → source activate 안 씀, $PY 변수로 명시
#   4. RunPod /workspace network volume에 venv 생성 (/usr는 Pod 재시작 시 소실)
# -----------------------------------------------------------------------------

err() { echo "❌ $*" >&2; exit 1; }
ok()  { echo "✅ $*"; }
log() { echo "▶ $*"; }

# -----------------------------------------------------------------------------
# setup: venv 생성 + vLLM + 의존성 설치
# -----------------------------------------------------------------------------
cmd_setup() {
    local name="${1:?Usage: setup <venv_name> [vllm_version]}"
    local version="${2:-0.20.0}"
    local venv="$WORKSPACE/$name"

    [ -d "$venv" ] && err "venv 이미 존재: $venv (다른 이름 쓰거나 먼저 지우세요)"

    log "venv 생성: $venv"
    python3 -m venv "$venv"
    local PY="$venv/bin/python"

    log "pip / wheel / setuptools 업그레이드"
    "$PY" -m pip install --upgrade pip wheel setuptools >/dev/null

    log "vLLM $version 설치 (CUDA wheel 포함, 5~10분 소요)"
    "$PY" -m pip install "vllm==$version"

    log "추가 의존성 (huggingface_hub, transformers)"
    "$PY" -m pip install "huggingface_hub>=1.0.0" "transformers>=5.0.0"

    log "검증"
    "$PY" -c "import vllm; print('  vllm:', vllm.__version__)"
    "$PY" -c "import huggingface_hub; print('  hf_hub:', huggingface_hub.__version__)"
    "$PY" -c "import transformers; print('  transformers:', transformers.__version__)"

    ok "venv 준비 완료: $venv"
    echo "  Python: $PY"
    echo "  활성화: source $venv/bin/activate  (또는 PY=$PY 사용)"
}

# -----------------------------------------------------------------------------
# model: HuggingFace에서 모델 다운로드 (백그라운드)
# -----------------------------------------------------------------------------
cmd_model() {
    local repo="${1:?Usage: model <hf_repo> [target_dir]}"
    local target="${2:-$WORKSPACE/models/$(basename "$repo")}"

    # 가용한 venv 자동 탐색 (huggingface_hub 가진 첫 번째)
    local PY=""
    for v in "$WORKSPACE"/venv*/bin/python; do
        [ -x "$v" ] || continue
        if "$v" -c "import huggingface_hub" 2>/dev/null; then
            PY="$v"; break
        fi
    done
    [ -z "$PY" ] && err "huggingface_hub 가진 venv 없음. 먼저 'setup' 실행"

    [ -d "$target" ] && err "이미 존재: $target (덮어쓰려면 먼저 rm -rf)"

    mkdir -p "$(dirname "$target")"
    log "다운로드: $repo → $target  (using $PY)"

    local logfile="/tmp/hf_download_$(basename "$repo").log"
    nohup "$PY" -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$repo', local_dir='$target', max_workers=8)
print('DOWNLOAD_DONE')
" > "$logfile" 2>&1 & disown
    local pid=$!
    sleep 2
    if ps -p $pid >/dev/null; then
        ok "다운 시작됨 (PID $pid)"
        echo "  진행 모니터: tail -f $logfile"
        echo "  완료 확인:   grep DOWNLOAD_DONE $logfile"
    else
        err "다운로드 프로세스 시작 실패. 로그 확인: $logfile"
    fi
}

# -----------------------------------------------------------------------------
# runpod-fixes: 알려진 RunPod 이슈 우회 (캐시 디렉터리 + LD_PRELOAD shim)
# -----------------------------------------------------------------------------
cmd_runpod_fixes() {
    log "RunPod 알려진 이슈 패치"

    # 캐시 디렉터리 사전 생성
    mkdir -p ~/.tensorrt_llm/cache ~/.tensorrt_llm/tmp
    mkdir -p ~/.cache/flashinfer ~/.cache/vllm
    ok "캐시 디렉터리 준비됨 (~/.tensorrt_llm, ~/.cache)"

    # TRT-LLM cubin rename shim 소스 작성
    local fixdir="$WORKSPACE/trtllm_fixes"
    mkdir -p "$fixdir"
    cat > "$fixdir/rename_fix.c" <<'CSRC'
#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dlfcn.h>
#include <libgen.h>

static int mkdir_p(const char *path) {
    char tmp[4096]; snprintf(tmp, sizeof(tmp), "%s", path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') { *p = 0; mkdir(tmp, 0755); *p = '/'; }
    }
    return mkdir(tmp, 0755);
}
typedef int (*rename_fn_t)(const char *, const char *);
int rename(const char *oldpath, const char *newpath) {
    static rename_fn_t real = NULL;
    if (!real) real = (rename_fn_t)dlsym(RTLD_NEXT, "rename");
    if (newpath && strstr(newpath, "/.tensorrt_llm/cache/")) {
        char buf[4096]; snprintf(buf, sizeof(buf), "%s", newpath);
        mkdir_p(dirname(buf));
    }
    return real(oldpath, newpath);
}
typedef int (*renameat2_fn_t)(int, const char *, int, const char *, unsigned int);
int renameat2(int odf, const char *op, int ndf, const char *np, unsigned int flags) {
    static renameat2_fn_t real = NULL;
    if (!real) real = (renameat2_fn_t)dlsym(RTLD_NEXT, "renameat2");
    if (np && strstr(np, "/.tensorrt_llm/cache/")) {
        char buf[4096]; snprintf(buf, sizeof(buf), "%s", np);
        mkdir_p(dirname(buf));
    }
    return real(odf, op, ndf, np, flags);
}
CSRC

    # 컴파일 (gcc 있으면)
    if command -v gcc >/dev/null 2>&1; then
        gcc -shared -fPIC -o "$fixdir/rename_fix.so" "$fixdir/rename_fix.c" -ldl
        ok "rename shim 컴파일됨: $fixdir/rename_fix.so"
        echo "  사용: LD_PRELOAD=$fixdir/rename_fix.so vllm serve ..."
    else
        echo "⚠ gcc 없음 — 컴파일 수동 필요:"
        echo "  apt-get install gcc -y"
        echo "  gcc -shared -fPIC -o $fixdir/rename_fix.so $fixdir/rename_fix.c -ldl"
    fi
}

# -----------------------------------------------------------------------------
# status: 현재 환경 요약
# -----------------------------------------------------------------------------
cmd_status() {
    echo "=== /workspace 디스크 사용 ==="
    du -sh "$WORKSPACE" 2>/dev/null
    echo
    echo "=== venv 목록 ==="
    local found=0
    for v in "$WORKSPACE"/venv*; do
        [ -d "$v" ] || continue
        found=1
        local ver
        ver=$("$v/bin/python" -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "(vllm 미설치)")
        local size
        size=$(du -sh "$v" 2>/dev/null | cut -f1)
        echo "  $(basename "$v"): vllm=$ver  size=$size"
    done
    [ $found -eq 0 ] && echo "  (없음)"
    echo
    echo "=== 모델 목록 ==="
    if [ -d "$WORKSPACE/models" ]; then
        for m in "$WORKSPACE"/models/*/; do
            [ -d "$m" ] || continue
            local size
            size=$(du -sh "$m" 2>/dev/null | cut -f1)
            echo "  $(basename "$m"): $size"
        done
    else
        echo "  (없음)"
    fi
    echo
    echo "=== GPU ==="
    nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null
    echo
    echo "=== vLLM 프로세스 ==="
    pgrep -af "vllm serve" 2>/dev/null | head -3 || echo "  (없음)"
}

# -----------------------------------------------------------------------------
# activate: source 가능한 명령 출력
# -----------------------------------------------------------------------------
cmd_activate() {
    local name="${1:?Usage: activate <venv_name>}"
    local venv="$WORKSPACE/$name"
    [ -d "$venv" ] || err "venv 없음: $venv"
    echo "source $venv/bin/activate"
}

# -----------------------------------------------------------------------------
# main dispatcher
# -----------------------------------------------------------------------------
case "${1:-help}" in
    setup)         shift; cmd_setup "$@" ;;
    model)         shift; cmd_model "$@" ;;
    runpod-fixes)  cmd_runpod_fixes ;;
    status)        cmd_status ;;
    activate)      shift; cmd_activate "$@" ;;
    help|--help|-h|*)
        sed -n '/^# ===/,/^# ===/p; /^# 사용법:/,/^# ===/p' "$0" | head -20
        ;;
esac
