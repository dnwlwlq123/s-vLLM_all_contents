"""
Gateway API 서버 — FastAPI 앱.

Qwen3.5-9B FP8 Multi-LoRA vLLM 서버에 대한 경량 프록시.
intent(LoRA) -> answer(바닐라) 2-Stage 추론을 제공한다.

실행:
    cd src/vllm_gateway && python main.py
    cd src/vllm_gateway && python run.py
    cd src && uvicorn vllm_gateway.main:app --host 0.0.0.0 --port 17801
"""

import asyncio
import sys
import os

_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_this_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger

from core.config import settings
from api.endpoints.chat_endpoints import router as legacy_router, get_service
from api.endpoints.completions_endpoints import (
    router as completions_router,
    close_clients,
)
from api.endpoints.admin_endpoints import router as admin_router
from services.response_converter import response_converter
from services.metrics_collector import metrics_collector


async def _vllm_health_monitor(poll_interval: float = 15.0, fail_threshold: int = 3, http_timeout: float = 10.0):
    """주기적으로 vLLM /health 폴링. 순간 지연 무시, N회 연속 실패만 DOWN 으로 확정."""
    url = f"{settings.ANSWER_VLLM_URL}/health"
    last_state: str | None = None
    consecutive_fails = 0
    async with httpx.AsyncClient(timeout=http_timeout) as client:
        while True:
            try:
                resp = await client.get(url)
                ok = resp.status_code == 200
            except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError):
                ok = False
            except Exception as e:
                logger.warning(f"[vllm-health] 예상외 예외: {type(e).__name__}: {e}")
                ok = False

            if ok:
                consecutive_fails = 0
                state = "up"
            else:
                consecutive_fails += 1
                state = "down" if consecutive_fails >= fail_threshold else last_state or "up"

            if state != last_state:
                if last_state is None:
                    logger.info(f"[vllm-health] 초기 상태: vLLM {state.upper()} ({url})")
                elif state == "down":
                    logger.warning(f"[vllm-health] ⚠️ vLLM DOWN 확정 ({consecutive_fails}회 연속 실패, {url})")
                else:
                    logger.info(f"[vllm-health] ✅ vLLM UP 복구됨 ({url})")
                last_state = state

            await asyncio.sleep(poll_interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Gateway 시작: {settings.HOST}:{settings.PORT}")
    logger.info(f"  intent vLLM: {settings.INTENT_VLLM_URL}")
    logger.info(f"  answer vLLM: {settings.ANSWER_VLLM_URL}")
    logger.info(f"  intent model: {settings.INTENT_MODEL_NAME}")
    logger.info(f"  answer model: {settings.ANSWER_MODEL_NAME}")

    await metrics_collector.start()

    # vLLM up/down 모니터링 백그라운드 태스크
    health_task = asyncio.create_task(_vllm_health_monitor())

    yield

    logger.info("Gateway 종료")
    health_task.cancel()
    try:
        await health_task
    except asyncio.CancelledError:
        pass
    await metrics_collector.stop()
    await get_service().close()
    await close_clients()


app = FastAPI(
    title="SLLM Gateway",
    description="Qwen3.5-9B FP8 Multi-LoRA 추론 Gateway — OpenAI 호환 인터페이스",
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(completions_router, prefix="/v1")
app.include_router(legacy_router, prefix="/api")
app.include_router(admin_router, prefix="/admin")


# ── OpenAI 호환 에러 핸들러 ──

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTPException → OpenAI 호환 에러 형식으로 변환."""
    error_resp = response_converter.convert_error(
        status_code=exc.status_code,
        message=str(exc.detail),
        error_type="invalid_request_error" if exc.status_code < 500 else "server_error",
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_resp.model_dump(),
    )


@app.exception_handler(httpx.HTTPStatusError)
async def vllm_error_handler(request: Request, exc: httpx.HTTPStatusError):
    """vLLM HTTP 에러 → OpenAI 호환 에러 형식으로 변환."""
    status_map = {429: 429, 500: 502, 502: 502, 503: 503}
    status = status_map.get(exc.response.status_code, 502)
    error_resp = response_converter.convert_error(
        status_code=status,
        message=f"vLLM 서버 에러: {exc.response.status_code}",
        error_type="server_error",
    )
    return JSONResponse(
        status_code=status,
        content=error_resp.model_dump(),
    )


@app.get("/")
async def root():
    return {
        "service": "vllm-gateway",
        "version": "2.0.0",
        "endpoints": {
            "v1": ["/v1/chat/completions"],
            "legacy": ["/api/completions", "/api/intent", "/api/answer", "/api/health"],
            "admin": ["/admin/cache", "/admin/metrics", "/admin/metrics/reset", "/admin/health", "/admin/serving/status", "/admin/serving/start", "/admin/serving/stop", "/admin/serving/restart"],
        },
    }


@app.get("/health")
async def healthcheck():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        loop="uvloop",
        http="httptools",
        access_log=False,
    )
