"""
서빙 관리 서비스 — 비즈니스 로직의 진입점.

S3Client(clients/)와 VLLMProcessManager(managers/)를 조합하여
모델 다운로드, 서빙 관리, 배포 유스케이스를 구현한다.
"""

import os
import shutil
from typing import Dict, Any, List, Optional

import yaml
from loguru import logger

from vllm_serving.config import settings
from vllm_serving.clients.s3_client import s3_client
from vllm_serving.managers.vllm_process_manager import process_manager


class ServingService:
    """모델 관리 + 서빙 관리 비즈니스 로직."""

    # ──────────────────────────────────────────────
    #  Registry
    # ──────────────────────────────────────────────
    def load_registry(self) -> Dict[str, Any]:
        with open(settings.REGISTRY_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    # ──────────────────────────────────────────────
    #  모델 다운로드
    # ──────────────────────────────────────────────
    def download_model(
        self, name: str, force: bool = False, s3_path_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        S3에서 모델을 다운로드.

        Args:
            name: "base_model" 또는 adapter 이름 (예: "intent")
            force: 강제 다운로드
            s3_path_override: 지정 시 registry의 s3_path 대신 이 경로 사용
        """
        registry = self.load_registry()
        s3_path, local_path = self._resolve_model_path(registry, name)

        if s3_path is None and s3_path_override is None:
            return {"status": "error", "error": f"registry에 '{name}' 없음"}

        if s3_path_override:
            s3_path = s3_path_override
            if local_path is None:
                local_path = name

        abs_local = self._to_abs_path(local_path)

        if os.path.exists(abs_local) and not force:
            existing = self._list_dir_files(abs_local)
            if existing:
                logger.info(f"[Service] 이미 존재: {abs_local} ({len(existing)}개 파일), 스킵")
                return {
                    "status": "skipped",
                    "reason": "already_exists",
                    "local_path": abs_local,
                    "files": existing,
                }

        logger.info(f"[Service] 다운로드 시작: s3://{settings.S3_BUCKET}/{s3_path} → {abs_local}")
        downloaded = s3_client.download_directory(s3_path, abs_local)

        logger.info(f"[Service] 다운로드 완료: {len(downloaded)}개 파일")
        return {
            "status": "downloaded",
            "local_path": abs_local,
            "s3_path": f"s3://{settings.S3_BUCKET}/{s3_path}",
            "files": downloaded,
        }

    def download_all(self, force: bool = False) -> Dict[str, Any]:
        """registry의 모든 모델/어댑터를 다운로드."""
        registry = self.load_registry()
        results = {}

        base = registry.get("base_model")
        if base:
            results["base_model"] = self.download_model("base_model", force)

        for name in registry.get("adapters", {}):
            results[name] = self.download_model(name, force)

        return results

    # ──────────────────────────────────────────────
    #  로컬 모델 관리
    # ──────────────────────────────────────────────
    def list_local_models(self) -> List[Dict[str, Any]]:
        """registry 기반 모델/어댑터 목록 + 로컬 존재 여부."""
        registry = self.load_registry()
        models = []

        base = registry.get("base_model", {})
        if base:
            abs_path = self._to_abs_path(base.get("local_path", ""))
            models.append(self._build_model_info(
                name="base_model", model_type="base_model", abs_path=abs_path,
            ))
        
        for adapter_name, info in registry.get("adapters", {}).items():
            abs_path = self._to_abs_path(info.get("local_path", ""))
            models.append(self._build_model_info(
                name=adapter_name, model_type="adapter", abs_path=abs_path,
            ))

        return models

    def delete_local_model(self, name: str) -> Dict[str, Any]:
        """논리명 기반 모델/어댑터 삭제."""
        registry = self.load_registry()
        _, local_path = self._resolve_model_path(registry, name)

        if local_path is None:
            return {"status": "error", "error": f"registry에 '{name}' 없음"}

        target = self._to_abs_path(local_path)
        if not os.path.exists(target):
            return {"status": "error", "error": f"로컬에 없음: {target}"}

        shutil.rmtree(target)
        logger.info(f"[Service] 삭제 완료: {name} → {target}")
        return {"status": "deleted", "name": name, "path": target}

    def _build_model_info(
        self, name: str, model_type: str, abs_path: str,
    ) -> Dict[str, Any]:
        """단일 모델/어댑터의 정보를 구성."""
        if os.path.isdir(abs_path):
            files = self._list_dir_files(abs_path)
            total_size = sum(
                os.path.getsize(os.path.join(abs_path, f))
                for f in files if os.path.isfile(os.path.join(abs_path, f))
            )
            return {
                "name": name,
                "type": model_type,
                "local_path": abs_path,
                "downloaded": True,
                "files": len(files),
                "size_mb": round(total_size / 1024 / 1024, 1),
            }

        return {
            "name": name,
            "type": model_type,
            "local_path": abs_path,
            "downloaded": False,
            "files": 0,
            "size_mb": 0.0,
        }

    # ──────────────────────────────────────────────
    #  서빙 관리 (manager 위임)
    # ──────────────────────────────────────────────
    def start_serving(self, mode: str = "single") -> Dict[str, Any]:
        return process_manager.start(mode=mode)

    def stop_serving(self) -> Dict[str, Any]:
        return process_manager.stop()

    def restart_serving(self, mode: Optional[str] = None) -> Dict[str, Any]:
        return process_manager.restart(mode=mode)

    def get_serving_status(self) -> Dict[str, Any]:
        return process_manager.status()

    # ──────────────────────────────────────────────
    #  통합 배포
    # ──────────────────────────────────────────────
    def deploy(self, force_download: bool = False, mode: str = "single") -> Dict[str, Any]:
        """
        원클릭 배포: 모든 모델 다운로드 + vLLM (재)시작.

        CI/CD에서 단일 호출로 전체 배포를 수행한다.
        """
        logger.info(f"[Service] 배포 시작: force={force_download}, mode={mode}")

        download_results = self.download_all(force=force_download)

        errors = [
            f"{k}: {v.get('error')}"
            for k, v in download_results.items()
            if v.get("status") == "error"
        ]
        if errors:
            return {
                "status": "error",
                "error": f"다운로드 실패: {'; '.join(errors)}",
                "download_results": download_results,
            }

        if process_manager.is_running:
            serving_result = self.restart_serving(mode=mode)
            start_result = serving_result.get("start", serving_result)
        else:
            start_result = self.start_serving(mode=mode)

        if start_result.get("status") == "error":
            return {
                "status": "error",
                "error": f"서빙 시작 실패: {start_result.get('error')}",
                "download_results": download_results,
                "serving": start_result,
            }

        logger.info("[Service] 배포 완료")
        return {
            "status": "deployed",
            "download_results": download_results,
            "serving": start_result,
        }

    def get_health(self) -> Dict[str, Any]:
        """서비스 상태."""
        status = process_manager.status()
        return {
            "manager": "ok",
            "vllm_running": status.get("running", False),
            "vllm_health": status.get("vllm_health", False),
        }

    # ──────────────────────────────────────────────
    #  헬퍼
    # ──────────────────────────────────────────────
    @staticmethod
    def _resolve_model_path(registry: Dict, name: str):
        if name == "base_model":
            info = registry.get("base_model", {})
            if not info:
                return None, None
            return info.get("s3_path"), info.get("local_path")

        adapters = registry.get("adapters", {})
        if name in adapters:
            info = adapters[name]
            return info.get("s3_path"), info.get("local_path")

        return None, None

    @staticmethod
    def _to_abs_path(path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(settings.MODEL_CACHE_DIR, path)

    @staticmethod
    def _list_dir_files(path: str) -> List[str]:
        files = []
        for root, _, filenames in os.walk(path):
            for fn in filenames:
                files.append(os.path.relpath(os.path.join(root, fn), path))
        return files


serving_service = ServingService()
