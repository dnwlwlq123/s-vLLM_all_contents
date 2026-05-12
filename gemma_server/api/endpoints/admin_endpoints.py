"""
Gateway 관리 API — vllm_serving 프록시 + 통합 헬스체크.

추론 경로(/v1/chat/completions)와 완전히 분리된 관리 전용 엔드포인트.
vllm_serving 관리 서버의 서빙 시작/중지/상태를 Gateway에서 중계한다.

사용 조건:
  .env에 GW_INTENT_SERVING_URL, GW_ANSWER_SERVING_URL 설정 필요.
  미설정 시 해당 서버 관련 기능은 "not_configured" 반환.
"""

import json
from datetime import date
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from core.config import settings
from core.model_state import get_answer_model, toggle_answer_model
from services.metrics_collector import metrics_collector

router = APIRouter()

_PROXY_TIMEOUT = 30.0
# [advice from AI] 모델 다운로드, 서빙 시작 등 오래 걸리는 작업용 타임아웃
_LONG_PROXY_TIMEOUT = 600.0


class ServingActionRequest(BaseModel):
    mode: str = Field(default="single", description="서빙 모드")


class AdminTargetRequest(BaseModel):
    """특정 서버 대상 지정 (생략 시 전체)."""
    target: Optional[str] = Field(
        default=None,
        description="대상 서버: 'intent', 'answer', None(전체)",
    )


def _serving_urls() -> Dict[str, str]:
    """설정된 serving URL 목록."""
    urls = {}
    if settings.INTENT_SERVING_URL:
        urls["intent"] = settings.INTENT_SERVING_URL
    if settings.ANSWER_SERVING_URL:
        urls["answer"] = settings.ANSWER_SERVING_URL
    return urls


async def _proxy_get(url: str, path: str) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=_PROXY_TIMEOUT) as client:
            resp = await client.get(f"{url}{path}")
            return resp.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def _proxy_post(
    url: str, path: str, body: dict = None, timeout: float = _PROXY_TIMEOUT,
) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{url}{path}", json=body or {})
            if resp.status_code >= 400:
                return {"status": "error", "error": f"HTTP {resp.status_code}: {resp.text[:500]}"}
            return resp.json()
    except httpx.TimeoutException:
        return {"status": "error", "error": f"프록시 타임아웃 ({timeout}초): {url}{path}"}
    except httpx.ConnectError:
        return {"status": "error", "error": f"연결 실패: {url}{path}"}
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


def _vllm_urls() -> Dict[str, str]:
    """설정된 vLLM 추론 URL 목록."""
    return {
        "intent": settings.INTENT_VLLM_URL,
        "answer": settings.ANSWER_VLLM_URL,
    }


async def _parse_prefix_cache_metrics(raw_text: str) -> Dict[str, Any]:
    """Prometheus 형식의 메트릭 텍스트에서 prefix cache 히트율을 추출."""
    hits = 0.0
    queries = 0.0
    for line in raw_text.splitlines():
        if line.startswith("vllm:prefix_cache_hits_total"):
            hits = float(line.split()[-1])
        elif line.startswith("vllm:prefix_cache_queries_total"):
            queries = float(line.split()[-1])

    hit_rate = (hits / queries * 100) if queries > 0 else 0.0
    return {
        "queries_total": int(queries),
        "hits_total": int(hits),
        "hit_rate_pct": round(hit_rate, 1),
    }


# [advice from AI] vllm_serving 모델 다운로드 프록시
class ModelsDownloadRequest(BaseModel):
    """모델 다운로드 요청."""
    name: str = Field(..., description="다운로드 대상: 'base_model' 또는 adapter 이름")
    s3_path: Optional[str] = Field(default=None, description="S3 경로 오버라이드")
    force: bool = Field(default=False, description="True면 기존 파일 덮어쓰기")


@router.post("/models/download")
async def models_download(
    request: ModelsDownloadRequest,
    target: str = "intent",
) -> Dict[str, Any]:
    """vllm_serving에 모델 다운로드 요청. target: 'intent' 또는 'answer'."""
    serving_urls = _serving_urls()
    if target not in serving_urls:
        raise HTTPException(
            status_code=400,
            detail=f"대상 서버 '{target}' 미설정. 설정된 서버: {list(serving_urls.keys())}",
        )

    url = serving_urls[target]
    logger.info(f"[admin] {target} 모델 다운로드 요청 → {url} (name={request.name}, force={request.force})")
    result = await _proxy_post(
        url, "/api/models/download", request.model_dump(exclude_none=True),
        timeout=_LONG_PROXY_TIMEOUT,
    )

    if result.get("status") == "error":
        error_msg = result.get("error") or f"다운로드 실패 (응답: {result})"
        raise HTTPException(status_code=400, detail=error_msg)

    return {"target": target, **result}


# ──────────────────────────────────────────────
#  Answer 모델 런타임 스위칭
# ──────────────────────────────────────────────

@router.get("/answer-model")
async def answer_model_status() -> Dict[str, Any]:
    """현재 answer 추론에 사용 중인 모델명 조회."""
    current = get_answer_model()
    return {
        "answer_model": current,
        "is_base": current != settings.ANSWER_MODEL_NAME,
    }


@router.post("/answer-model/toggle")
async def answer_model_toggle() -> Dict[str, Any]:
    """answer ↔ base 모델 토글."""
    result = toggle_answer_model()
    logger.info(f"[admin] answer 모델 전환: {result['previous']} → {result['answer_model']}")
    return result


@router.get("/cache")
async def prefix_cache_stats(target: Optional[str] = None) -> Dict[str, Any]:
    """intent/answer vLLM의 prefix cache 히트율 조회."""
    vllm_urls = _vllm_urls()
    targets = {target: vllm_urls[target]} if target and target in vllm_urls else vllm_urls

    results = {}
    for name, url in targets.items():
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{url}/metrics")
                if resp.status_code == 200:
                    results[name] = await _parse_prefix_cache_metrics(resp.text)
                else:
                    results[name] = {"status": "error", "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            results[name] = {"status": "error", "error": str(e)}

    return results


@router.get("/metrics")
async def metrics_summary() -> Dict[str, Any]:
    """오늘 날짜 기준 메트릭 summary 반환."""
    metrics_collector._save_summary()

    today = date.today()
    summary_path = metrics_collector._log_dir / f"metrics_{today:%Y%m%d}_summary.json"

    if not summary_path.exists():
        return {"status": "no_data", "period": str(today)}

    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


@router.post("/metrics/reset")
async def metrics_reset() -> Dict[str, Any]:
    """메트릭 기록 초기화 (메모리 + 오늘 파일 삭제)."""
    today = date.today()
    deleted_files = []

    jsonl_path = metrics_collector._log_dir / f"metrics_{today:%Y%m%d}.jsonl"
    summary_path = metrics_collector._log_dir / f"metrics_{today:%Y%m%d}_summary.json"

    for path in [jsonl_path, summary_path]:
        if path.exists():
            path.unlink()
            deleted_files.append(path.name)

    metrics_collector._records.clear()
    metrics_collector._current_date = today

    logger.info(f"[admin] 메트릭 초기화 완료: {deleted_files}")
    return {
        "status": "reset",
        "period": str(today),
        "deleted_files": deleted_files,
    }


@router.get("/health")
async def admin_health() -> Dict[str, Any]:
    """Gateway + vllm_serving + vLLM 통합 헬스체크."""
    result: Dict[str, Any] = {"gateway": "ok"}

    serving_urls = _serving_urls()

    for name, url in serving_urls.items():
        serving_health = await _proxy_get(url, "/health")
        result[f"{name}_serving"] = "ok" if serving_health.get("status") != "error" else "unavailable"

        vllm_url = settings.INTENT_VLLM_URL if name == "intent" else settings.ANSWER_VLLM_URL
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{vllm_url}/health")
                result[f"{name}_vllm"] = "ok" if resp.status_code == 200 else "unavailable"
        except Exception:
            result[f"{name}_vllm"] = "unavailable"

    if not serving_urls:
        result["note"] = "serving URL 미설정 (GW_INTENT_SERVING_URL, GW_ANSWER_SERVING_URL)"

    return result


@router.get("/serving/status")
async def serving_status(target: Optional[str] = None) -> Dict[str, Any]:
    """vllm_serving 상태 조회. target: intent, answer, None(전체)."""
    serving_urls = _serving_urls()
    if not serving_urls:
        return {"status": "not_configured", "note": "serving URL 미설정"}

    results = {}
    targets = {target: serving_urls[target]} if target and target in serving_urls else serving_urls

    for name, url in targets.items():
        results[name] = await _proxy_get(url, "/api/serving/status")

    return results


@router.post("/serving/start")
async def serving_start(
    request: ServingActionRequest = ServingActionRequest(),
    target: Optional[str] = None,
) -> Dict[str, Any]:
    """vllm_serving 서빙 시작. target: intent, answer, None(전체)."""
    serving_urls = _serving_urls()
    if not serving_urls:
        return {"status": "not_configured"}

    results = {}
    targets = {target: serving_urls[target]} if target and target in serving_urls else serving_urls

    for name, url in targets.items():
        logger.info(f"[admin] {name} 서빙 시작 요청 → {url}")
        results[name] = await _proxy_post(
            url, "/api/serving/start", {"mode": request.mode},
            timeout=_LONG_PROXY_TIMEOUT,
        )

    return results


@router.post("/serving/stop")
async def serving_stop(target: Optional[str] = None) -> Dict[str, Any]:
    """vllm_serving 서빙 중지."""
    serving_urls = _serving_urls()
    if not serving_urls:
        return {"status": "not_configured"}

    results = {}
    targets = {target: serving_urls[target]} if target and target in serving_urls else serving_urls

    for name, url in targets.items():
        logger.info(f"[admin] {name} 서빙 중지 요청 → {url}")
        results[name] = await _proxy_post(url, "/api/serving/stop")

    return results


@router.post("/serving/restart")
async def serving_restart(
    request: ServingActionRequest = ServingActionRequest(),
    target: Optional[str] = None,
) -> Dict[str, Any]:
    """vllm_serving 서빙 재시작."""
    serving_urls = _serving_urls()
    if not serving_urls:
        return {"status": "not_configured"}

    results = {}
    targets = {target: serving_urls[target]} if target and target in serving_urls else serving_urls

    for name, url in targets.items():
        logger.info(f"[admin] {name} 서빙 재시작 요청 → {url}")
        results[name] = await _proxy_post(
            url, "/api/serving/restart", {"mode": request.mode},
            timeout=_LONG_PROXY_TIMEOUT,
        )

    return results
