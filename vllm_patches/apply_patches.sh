#!/bin/bash
# vLLM 0.19.2rc1.dev96+ 패치 적용 스크립트
# 대상: FA4 + FP8 KV cache 조합 허용 (H100 SM90)
#
# 사용법:
#   VENV=/workspace/venv_gemma4 bash apply_patches.sh

set -e
VENV=${VENV:-/workspace/venv_gemma4}
SP=$(ls -d ${VENV}/lib/python*/site-packages | head -1)
BACKENDS=${SP}/vllm/v1/attention/backends

if [ ! -d "$BACKENDS" ]; then
    echo "ERROR: vLLM backends dir not found at $BACKENDS"
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

echo "Target: $BACKENDS"
echo "Backing up..."
cp -n ${BACKENDS}/fa_utils.py ${BACKENDS}/fa_utils.py.orig
cp -n ${BACKENDS}/flash_attn.py ${BACKENDS}/flash_attn.py.orig

cd $BACKENDS
patch -p3 < ${SCRIPT_DIR}/01_fa4_fp8_kv_gating.patch
patch -p3 < ${SCRIPT_DIR}/02_fa_supported_kv_cache_dtypes.patch

echo "=== Verification ==="
${VENV}/bin/python -c "
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
print('supported_kv_cache_dtypes:', FlashAttentionBackend.supported_kv_cache_dtypes)
"
