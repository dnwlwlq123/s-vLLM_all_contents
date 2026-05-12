"""
외부 API 호출 공통 베이스.

타임아웃, 지연 초기화, 리소스 정리를 공통으로 처리한다.
"""

from typing import Optional

import httpx
from loguru import logger

from core.config import settings


class BaseClient:
    """외부 API 호출 공통 베이스 클라이언트."""

    def __init__(self, base_url: str, timeout: Optional[float] = None):
        self.base_url = base_url
        self.timeout = timeout or settings.HTTP_TIMEOUT
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """지연 초기화된 HTTP 클라이언트 반환."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                http2=True,
                limits=httpx.Limits(
                    max_keepalive_connections=10,
                    keepalive_expiry=120,
                ),
            )
        return self._client

    async def close(self):
        """클라이언트 리소스 정리."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _check_response(resp: httpx.Response, label: str):
        """HTTP 응답 상태 확인 — 비정상 시 로깅 후 예외 발생."""
        if resp.status_code != 200:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text[:500]
            logger.error(f"[{label}] {resp.status_code} 에러: {detail}")
            resp.raise_for_status()
