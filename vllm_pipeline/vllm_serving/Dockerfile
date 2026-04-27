FROM vllm/vllm-openai:v0.17.1

WORKDIR /app

# 기본 의존성 설치
COPY requirements.txt ./vllm_serving/
RUN pip install --no-cache-dir -r vllm_serving/requirements.txt

# Qwen3.5 지원: transformers 5.x 강제 업그레이드 (vllm 의존성 무시)
RUN pip install --no-cache-dir --no-deps "transformers>=5.0.0" "huggingface_hub>=1.0.0"

# 코드 복사
COPY . ./vllm_serving/

ENTRYPOINT []
CMD ["python3", "vllm_serving/main.py"]
