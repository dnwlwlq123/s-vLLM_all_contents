"""
Gateway 설정 — Pydantic Settings 기반.

vLLM 서버 연결 및 추론 파라미터를 관리한다.
환경변수 접두사: GW_
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):

    # === 서버 ===
    HOST: str = "0.0.0.0"
    PORT: int = 17801
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # === vLLM 연결 ===
    INTENT_VLLM_URL: str = "http://localhost:8000"
    ANSWER_VLLM_URL: str = "http://localhost:8000"
    INTENT_MODEL_NAME: str = "intent"
    ANSWER_MODEL_NAME: str = "answer"
    BASE_MODEL_NAME: str = "gemma-4-31B-it"

    # === vllm_serving 관리 서버 (서빙 시작/중지/상태 프록시용) ===
    INTENT_SERVING_URL: str = ""
    ANSWER_SERVING_URL: str = ""

    # === intent 추론 ===
    INTENT_MAX_TOKENS: int = 128
    INTENT_TEMPERATURE: float = 0.0

    # === answer 추론 ===
    ANSWER_MAX_TOKENS: int = 1024
    ANSWER_TEMPERATURE: float = 0.3

    # === 공통 ===
    DEFAULT_TOP_P: float = 0.9
    HTTP_TIMEOUT: float = 60.0

    model_config = {
        "env_file": ".env",
        "env_prefix": "GW_",
        "case_sensitive": True,
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
