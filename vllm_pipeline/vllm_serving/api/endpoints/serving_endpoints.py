"""
서빙 관리 API 엔드포인트 — 얇은 어댑터.

비즈니스 로직은 ServingService에 위임하고,
스키마 변환 + HTTPException 변환만 담당한다.
"""

from typing import Dict, Any, List

from fastapi import APIRouter, HTTPException

from vllm_serving.api.schemas.serving_schemas import (
    DownloadRequest, DownloadResponse,
    ServingStartRequest, ServingResponse, ServingStatusResponse,
    ModelInfo,
    DeployRequest, DeployResponse,
)
from vllm_serving.services.serving_service import serving_service

router = APIRouter()


# ──────────────────────────────────────────────
#  모델 관리
# ──────────────────────────────────────────────

@router.get("/models", response_model=List[ModelInfo])
def list_models():
    """로컬에 다운로드된 모델/어댑터 목록."""
    return serving_service.list_local_models()


@router.get("/models/registry")
def get_registry() -> Dict[str, Any]:
    """model_registry.yaml 내용."""
    return serving_service.load_registry()


@router.post("/models/download", response_model=DownloadResponse)
def download_model(request: DownloadRequest):
    """S3/MinIO에서 모델 다운로드."""
    result = serving_service.download_model(
        request.name, force=request.force, s3_path_override=request.s3_path,
    )

    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("error"))

    return DownloadResponse(**result)


@router.delete("/models/{name}")
def delete_model(name: str) -> Dict[str, Any]:
    """로컬 모델/어댑터 삭제."""
    result = serving_service.delete_local_model(name)
    if result.get("status") == "error":
        raise HTTPException(status_code=404, detail=result.get("error"))
    return result


# ──────────────────────────────────────────────
#  서빙 관리
# ──────────────────────────────────────────────

@router.post("/serving/start", response_model=ServingResponse)
def serving_start(request: ServingStartRequest):
    """vLLM 프로세스 기동."""
    result = serving_service.start_serving(mode=request.mode)

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error"))

    return ServingResponse(**result)


@router.post("/serving/stop", response_model=ServingResponse)
def serving_stop():
    """vLLM 프로세스 종료."""
    result = serving_service.stop_serving()
    return ServingResponse(**result)


@router.post("/serving/restart", response_model=ServingResponse)
def serving_restart(request: ServingStartRequest = ServingStartRequest()):
    """vLLM 프로세스 재시작."""
    result = serving_service.restart_serving(mode=request.mode)

    start_result = result.get("start", result)
    if start_result.get("status") == "error":
        raise HTTPException(status_code=500, detail=start_result.get("error"))

    return ServingResponse(**start_result)


@router.get("/serving/status", response_model=ServingStatusResponse)
def serving_status():
    """vLLM 프로세스 상태 + health check."""
    return ServingStatusResponse(**serving_service.get_serving_status())


# ──────────────────────────────────────────────
#  통합
# ──────────────────────────────────────────────

@router.post("/deploy", response_model=DeployResponse)
def deploy(request: DeployRequest):
    """배포: 모든 모델 다운로드 + vLLM 서빙 (재)시작."""
    result = serving_service.deploy(
        force_download=request.force_download,
        mode=request.mode,
    )

    return DeployResponse(**result)


@router.get("/health")
def health_check() -> Dict[str, Any]:
    """관리 서버 자체 상태."""
    return serving_service.get_health()
