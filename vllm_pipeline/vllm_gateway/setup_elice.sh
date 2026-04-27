#!/bin/bash
# 엘리스 H100 서버 초기 세팅 스크립트
# 사용법: bash setup_elice.sh

echo "=== NVML 드라이버 미스매치 자동 매칭 ==="
# 기존 잘못된 LD_PRELOAD 제거 후 재감지
unset LD_PRELOAD
sed -i '/LD_PRELOAD.*libnvidia-ml/d' ~/.bashrc
DRIVER_VER=$(cat /proc/driver/nvidia/version | grep -oP 'Module\s+\K[0-9]+\.[0-9]+\.[0-9]+')
NVML_LIB="/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.${DRIVER_VER}"
if [ -f "$NVML_LIB" ]; then
    echo "드라이버 버전 $DRIVER_VER 감지, NVML 라이브러리 매칭"
    sed -i '/LD_PRELOAD.*libnvidia-ml/d' ~/.bashrc
    echo "export LD_PRELOAD=$NVML_LIB" >> ~/.bashrc
else
    echo "WARNING: $NVML_LIB 없음, LD_PRELOAD 설정 스킵"
fi

grep -q 'CUDA_HOME' ~/.bashrc || echo 'export CUDA_HOME=/usr' >> ~/.bashrc
grep -q 'VLLM_TARGET_DEVICE' ~/.bashrc || echo 'export VLLM_TARGET_DEVICE=cuda' >> ~/.bashrc
grep -q '/home/elicer/.local/bin' ~/.bashrc || echo 'export PATH=$PATH:/home/elicer/.local/bin' >> ~/.bashrc
grep -q 'vllm_env' ~/.bashrc || echo 'source ~/vllm_env/bin/activate' >> ~/.bashrc
source ~/.bashrc

echo "=== Python 의존성 설치 ==="
pip install -r requirements_elice.txt

echo "=== 모델 다운로드 (없으면) ==="
MODEL_DIR=~/vLLM_server/qwen3.5/models
mkdir -p $MODEL_DIR

if [ ! -d "$MODEL_DIR/Qwen3.5-27B" ]; then
    echo "27B 다운로드 중..."
    huggingface-cli download Qwen/Qwen3.5-27B --local-dir $MODEL_DIR/Qwen3.5-27B
fi

if [ ! -d "$MODEL_DIR/Qwen3.5-35B-A3B" ]; then
    echo "35B-A3B 다운로드 중..."
    huggingface-cli download Qwen/Qwen3.5-35B-A3B --local-dir $MODEL_DIR/Qwen3.5-35B-A3B
fi

echo "=== 완료 ==="
echo "vLLM 시작: bash start_vllm.sh 27b 또는 bash start_vllm.sh 35b"
echo "Gateway 시작: bash start_gateway.sh"
