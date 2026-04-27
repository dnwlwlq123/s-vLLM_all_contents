"""서빙 관리 API 요청/응답 스키마."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class DownloadRequest(BaseModel):
    """모델 다운로드 요청."""
    name: str = Field(..., description="다운로드 대상: 'base_model' 또는 adapter 이름")
    s3_path: Optional[str] = Field(default=None, description="S3 경로 오버라이드 (지정 시 registry 대신 이 경로에서 다운로드)")
    force: bool = Field(default=False, description="True면 기존 파일 덮어쓰기")


class DownloadResponse(BaseModel):
    """모델 다운로드 응답."""
    status: str
    local_path: Optional[str] = None
    s3_path: Optional[str] = None
    files: Optional[List[str]] = None
    reason: Optional[str] = None
    error: Optional[str] = None


class ServingStartRequest(BaseModel):
    """서빙 시작 요청."""
    mode: str = Field(default="single", description="서빙 모드: 'single'")


class ServingResponse(BaseModel):
    """서빙 관리 응답."""
    status: str
    pid: Optional[int] = None
    url: Optional[str] = None
    mode: Optional[str] = None
    error: Optional[str] = None


class ServingStatusResponse(BaseModel):
    """서빙 상태 응답."""
    running: bool
    mode: Optional[str] = None
    url: str
    pid: Optional[int] = None
    uptime_sec: Optional[float] = None
    vllm_health: bool = False
    models: List[str] = []


class ModelInfo(BaseModel):
    """로컬 모델 정보 — registry 기반 논리명 사용."""
    name: str = Field(..., description="논리명: base_model 또는 adapter 이름 (intent 등)")
    type: str = Field(..., description="모델 유형: base_model 또는 adapter")
    local_path: str = Field(..., description="로컬 저장 경로")
    downloaded: bool = Field(..., description="로컬에 다운로드 완료 여부")
    files: int = Field(default=0, description="파일 수")
    size_mb: float = Field(default=0.0, description="용량 (MB)")


class DeployRequest(BaseModel):
    """배포 요청 (다운로드 + 서빙 시작)."""
    force_download: bool = Field(default=False, description="모델 강제 재다운로드")
    mode: str = Field(default="single", description="서빙 모드")


class DeployResponse(BaseModel):
    """배포 응답."""
    status: str
    download_results: Optional[Dict[str, Any]] = None
    serving: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
