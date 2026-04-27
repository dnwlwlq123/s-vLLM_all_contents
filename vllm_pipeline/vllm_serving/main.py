"""
vLLM 서빙 관리 서버 — FastAPI 앱.

모델 다운로드(S3/MinIO), vLLM 프로세스 관리, 상태 모니터링을 API로 제공한다.

실행:
    cd src/vllm_serving && python main.py
    cd src/vllm_serving && python run.py
    cd src && uvicorn vllm_serving.main:app --host 0.0.0.0 --port 17810
"""

import sys
import os

_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_this_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from vllm_serving.config import settings
from vllm_serving.api.endpoints.serving_endpoints import router
from vllm_serving.managers.vllm_process_manager import process_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"vLLM 서빙 관리 서버 시작: {settings.HOST}:{settings.PORT}")
    logger.info(f"  S3 endpoint: {settings.S3_ENDPOINT_URL or 'AWS S3'}")
    logger.info(f"  S3 bucket: {settings.S3_BUCKET}")
    logger.info(f"  모델 캐시: {settings.MODEL_CACHE_DIR}")
    logger.info(f"  vLLM URL: {settings.vllm_url}")

    yield

    logger.info("vLLM 서빙 관리 서버 종료")
    if process_manager.is_running:
        logger.info("vLLM 프로세스 정리 중...")
        process_manager.stop()


app = FastAPI(
    title="vLLM Serving Manager",
    description="FP8 14B Multi-LoRA vLLM 서빙 관리 — 모델 다운로드 + 프로세스 관리",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {
        "service": "vllm-serving-manager",
        "version": "1.0.0",
        "endpoints": {
            "models": ["/api/models", "/api/models/registry", "/api/models/download"],
            "serving": ["/api/serving/start", "/api/serving/stop", "/api/serving/restart", "/api/serving/status"],
            "deploy": "/api/deploy",
            "health": "/api/health",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "vllm_serving.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
