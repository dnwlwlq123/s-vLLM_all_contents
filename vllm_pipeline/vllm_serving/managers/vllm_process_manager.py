"""
vLLM 프로세스 매니저 — subprocess로 vLLM 서버를 기동/종료/재시작.

상태(프로세스 핸들, 실행 모드 등)를 관리하는 싱글톤 매니저.
model_registry.yaml 기반으로 base 모델과 LoRA adapter를 자동 구성한다.
"""

import os
import subprocess
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, IO

import httpx
import yaml
from loguru import logger

from vllm_serving.config import settings


class VLLMProcessManager:
    """vLLM subprocess 생명주기를 관리하는 싱글톤 매니저."""

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._log_file: Optional[IO] = None
        self._log_path: Optional[str] = None
        self._mode: Optional[str] = None
        self._started_at: Optional[float] = None

    @property
    def is_running(self) -> bool:
        if self._process is None:
            return False
        return self._process.poll() is None

    def start(self, mode: str = "single") -> Dict[str, Any]:
        """vLLM 프로세스 기동."""
        if self.is_running:
            return {
                "status": "already_running",
                "pid": self._process.pid,
                "url": settings.vllm_url,
            }

        registry = self._load_registry()
        cmd = self._build_cmd(registry, mode)

        logger.info(f"[ProcessManager] vLLM 기동: mode={mode}")
        logger.info(f"  cmd: {' '.join(cmd)}")

        # [advice from AI] stdout=PIPE는 버퍼(64KB)가 가득 차면 자식 프로세스가 블록되는
        # 데드락을 유발하므로, 파일로 리다이렉트하여 로그 보존과 안정성을 동시에 확보
        self._log_path = self._open_log_file()
        self._log_file = open(self._log_path, "w", buffering=1, encoding="utf-8")
        logger.info(f"[ProcessManager] vLLM 로그 파일: {self._log_path}")

        self._process = subprocess.Popen(
            cmd,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
        )
        self._mode = mode
        self._started_at = time.time()

        logger.info(f"[ProcessManager] vLLM 프로세스 시작: PID={self._process.pid}")

        healthy = self._wait_for_healthy()

        if healthy:
            logger.info(f"[ProcessManager] vLLM 준비 완료: {settings.vllm_url}")
            return {
                "status": "started",
                "pid": self._process.pid,
                "url": settings.vllm_url,
                "mode": mode,
            }
        else:
            logger.error("[ProcessManager] vLLM 시작 시간 초과")
            self.stop()
            return {"status": "error", "error": "startup_timeout"}

    def stop(self) -> Dict[str, Any]:
        """vLLM 프로세스 종료."""
        if not self.is_running:
            return {"status": "not_running"}

        pid = self._process.pid
        logger.info(f"[ProcessManager] vLLM 종료: PID={pid}")

        self._process.terminate()
        try:
            self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            logger.warning("[ProcessManager] SIGTERM 타임아웃, SIGKILL 전송")
            self._process.kill()
            self._process.wait(timeout=10)

        self._process = None
        self._mode = None
        self._started_at = None
        self._close_log_file()

        logger.info(f"[ProcessManager] vLLM 종료 완료: PID={pid}")
        return {"status": "stopped", "pid": pid}

    def restart(self, mode: Optional[str] = None) -> Dict[str, Any]:
        """종료 후 재기동."""
        prev_mode = self._mode or "single"
        restart_mode = mode or prev_mode

        stop_result = self.stop()
        time.sleep(2)
        start_result = self.start(mode=restart_mode)

        return {"stop": stop_result, "start": start_result}

    def status(self) -> Dict[str, Any]:
        """vLLM 프로세스 상태 + health check."""
        result: Dict[str, Any] = {
            "running": self.is_running,
            "mode": self._mode,
            "url": settings.vllm_url,
        }

        if self.is_running:
            result["pid"] = self._process.pid
            result["uptime_sec"] = round(time.time() - self._started_at, 1) if self._started_at else 0
            result["vllm_health"] = self._check_health()
            result["models"] = self._list_vllm_models()
            result["log_file"] = self._log_path
        else:
            result["pid"] = None
            result["vllm_health"] = False

        return result

    def _open_log_file(self) -> str:
        """vLLM 로그 파일 경로를 생성하고 디렉토리를 준비한다."""
        os.makedirs(settings.VLLM_LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(settings.VLLM_LOG_DIR, f"vllm_{timestamp}.log")

    def _close_log_file(self):
        """로그 파일 핸들을 안전하게 닫는다."""
        if self._log_file and not self._log_file.closed:
            self._log_file.close()
        self._log_file = None

    def _build_cmd(self, registry: Dict[str, Any], mode: str) -> List[str]:
        base = registry.get("base_model", {})
        base_path = base.get("local_path", "Qwen3.5-9B-FP8")
        if not os.path.isabs(base_path):
            base_path = os.path.join(settings.MODEL_CACHE_DIR, base_path)

        # [advice from AI] --served-model-name으로 환경 무관한 고정 모델명 사용
        served_name = base.get("name", os.path.basename(base_path))

        vllm_cmd = [
            "vllm", "serve", base_path,
            "--served-model-name", served_name,
            "--host", settings.VLLM_HOST,
            "--port", str(settings.VLLM_PORT),
            "--quantization", "fp8",
            "--kv-cache-dtype", "fp8",
            "--dtype", settings.VLLM_DTYPE,
            "--max-model-len", str(settings.VLLM_MAX_MODEL_LEN),
            "--gpu-memory-utilization", str(settings.VLLM_GPU_MEMORY_UTILIZATION),
            "-cc", '{"cudagraph_mode":"none"}',
            "--trust-remote-code",
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--max-num-batched-tokens", "8192",
            "--max-num-seqs", str(settings.VLLM_MAX_NUM_SEQS),
            "--disable-log-stats",
            "--language-model-only",
        ]

        adapters = registry.get("adapters", {})
        if adapters:
            vllm_cmd.append("--enable-lora")
            vllm_cmd.extend(["--max-lora-rank", str(settings.VLLM_MAX_LORA_RANK)])

            lora_modules = []
            for name, info in adapters.items():
                adapter_path = info.get("local_path", "")
                if not os.path.isabs(adapter_path):
                    adapter_path = os.path.join(settings.MODEL_CACHE_DIR, adapter_path)
                lora_modules.append(f"{name}={adapter_path}")

            vllm_cmd.extend(["--lora-modules", *lora_modules])

        # [advice from AI] conda 환경이 설정되어 있으면 conda run으로 감싸서 실행
        if settings.VLLM_CONDA_ENV:
            cmd = [
                "conda", "run", "-n", settings.VLLM_CONDA_ENV,
                "--no-capture-output", *vllm_cmd,
            ]
        else:
            cmd = vllm_cmd

        return cmd

    # [advice from AI] vLLM 프로세스 종료/타임아웃 시 로그 파일 내용을 loguru로 출력
    def _dump_vllm_log(self, tail_lines: int = 200):
        """vLLM 로그 파일의 마지막 N줄을 loguru로 출력."""
        if not self._log_path or not os.path.exists(self._log_path):
            logger.warning("[ProcessManager] vLLM 로그 파일을 찾을 수 없음")
            return

        try:
            if self._log_file and not self._log_file.closed:
                self._log_file.flush()

            with open(self._log_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            if not lines:
                logger.warning("[ProcessManager] vLLM 로그 파일이 비어 있음")
                return

            tail = lines[-tail_lines:]
            logger.error(f"[ProcessManager] vLLM 로그 (마지막 {len(tail)}줄):\n{''.join(tail)}")
        except Exception as e:
            logger.error(f"[ProcessManager] vLLM 로그 읽기 실패: {e}")

    def _wait_for_healthy(self) -> bool:
        deadline = time.time() + settings.VLLM_STARTUP_TIMEOUT
        check_interval = 5

        while time.time() < deadline:
            if not self.is_running:
                exit_code = self._process.returncode if self._process else "unknown"
                logger.error(f"[ProcessManager] vLLM 프로세스가 예기치 않게 종료됨 (exit_code={exit_code})")
                self._dump_vllm_log()
                return False
            if self._check_health():
                return True
            elapsed = time.time() - self._started_at
            logger.debug(f"[ProcessManager] vLLM 준비 대기 중... ({elapsed:.0f}초)")
            time.sleep(check_interval)

        logger.error("[ProcessManager] vLLM 시작 타임아웃 — 마지막 로그 확인:")
        self._dump_vllm_log()
        return False

    @staticmethod
    def _check_health() -> bool:
        try:
            resp = httpx.get(f"{settings.vllm_url}/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    @staticmethod
    def _list_vllm_models() -> List[str]:
        try:
            resp = httpx.get(f"{settings.vllm_url}/v1/models", timeout=5.0)
            if resp.status_code == 200:
                return [m.get("id", "") for m in resp.json().get("data", [])]
        except Exception:
            pass
        return []

    @staticmethod
    def _load_registry() -> Dict[str, Any]:
        with open(settings.REGISTRY_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}


process_manager = VLLMProcessManager()
