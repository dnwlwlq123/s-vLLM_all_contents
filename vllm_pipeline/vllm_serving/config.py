"""
vllm_serving 설정 — Pydantic Settings 기반.

S3/MinIO 접속, 모델 경로, vLLM 프로세스 설정을 관리한다.
환경변수 접두사: VS_
"""

import os
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings


_MODULE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):

    # === 서버 ===
    HOST: str = "0.0.0.0"
    PORT: int = 17810
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # === S3/MinIO ===
    S3_ENDPOINT_URL: str = ""
    S3_ACCESS_KEY: str = ""
    S3_SECRET_KEY: str = ""
    S3_BUCKET: str = "sllm-models"
    S3_REGION: str = "us-east-1"

    # === 로컬 모델 캐시 ===
    MODEL_CACHE_DIR: str = str(_MODULE_DIR / "models")

    # === Registry ===
    REGISTRY_PATH: str = str(_MODULE_DIR / "model_registry.yaml")

    # === 로그 ===
    VLLM_LOG_DIR: str = str(_MODULE_DIR / "logs")

    # === vLLM 프로세스 ===
    VLLM_CONDA_ENV: str = ""
    VLLM_HOST: str = "0.0.0.0"
    VLLM_PORT: int = 8000
    VLLM_GPU_MEMORY_UTILIZATION: float = 0.85
    VLLM_MAX_MODEL_LEN: int = 4096
    VLLM_MAX_LORA_RANK: int = 32
    VLLM_SWAP_SPACE: int = 4
    VLLM_MAX_NUM_SEQS: int = 16
    VLLM_STARTUP_TIMEOUT: int = 300
    VLLM_DTYPE: str = "auto"

    @property
    def vllm_url(self) -> str:
        return f"http://localhost:{self.VLLM_PORT}"

    model_config = {
        "env_file": ".env",
        "env_prefix": "VS_",
        "case_sensitive": True,
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
