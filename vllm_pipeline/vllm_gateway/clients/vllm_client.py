"""
vLLM OpenAI-compatible HTTP 클라이언트.

외부 vLLM 서버의 /v1/chat/completions 엔드포인트를 호출한다.
비스트리밍/스트리밍 모두 지원.
"""

import json
from typing import List, Dict, Any, Optional, AsyncGenerator

from loguru import logger

from clients.base_client import BaseClient


class VLLMClient(BaseClient):
    """vLLM 서버와의 HTTP 통신을 담당한다."""

    def __init__(self, base_url: str, timeout: Optional[float] = None):
        super().__init__(base_url=base_url, timeout=timeout)

    async def infer(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.9,
        json_schema: Optional[dict] = None,
    ) -> str:
        """
        vLLM 비스트리밍 추론 호출.

        Args:
            model: vLLM에 등록된 모델/어댑터 이름
            messages: ChatML messages
            max_tokens: 최대 생성 토큰 수
            temperature: 생성 온도
            top_p: top-p 샘플링
            json_schema: guided decoding용 JSON 스키마

        Returns:
            생성된 텍스트
        """
        body = self._build_body(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            json_schema=json_schema,
            stream=False,
        )

        url = f"{self.base_url}/v1/chat/completions"
        client = await self._get_client()
        resp = await client.post(url, json=body)
        self._check_response(resp, model)

        data = resp.json()
        return data["choices"][0]["message"]["content"]

    async def infer_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.3,
        top_p: float = 0.9,
        json_schema: Optional[dict] = None,
    ) -> AsyncGenerator[str, None]:
        """
        vLLM SSE 스트리밍 추론 호출.

        Yields:
            토큰 단위 텍스트 chunk
        """
        body = self._build_body(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            json_schema=json_schema,
            stream=True,
        )

        url = f"{self.base_url}/v1/chat/completions"
        client = await self._get_client()

        async with client.stream("POST", url, json=body) as resp:
            # [advice from AI] 스트리밍 에러 시 vLLM 응답 body를 로깅
            if resp.status_code != 200:
                error_body = await resp.aread()
                logger.error(f"[vllm-stream] {resp.status_code} 에러: {error_body.decode()}")
                resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    delta = chunk["choices"][0].get("delta", {}).get("content", "")
                    if delta:
                        yield delta
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    async def infer_raw(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.9,
        json_schema: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """
        vLLM 비스트리밍 추론 호출 — 전체 응답 dict 반환.

        id, choices, usage 등 메타데이터를 보존하여
        OpenAI 호환 응답 변환에 사용한다.
        """
        body = self._build_body(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            json_schema=json_schema,
            stream=False,
        )

        url = f"{self.base_url}/v1/chat/completions"
        client = await self._get_client()
        resp = await client.post(url, json=body)
        self._check_response(resp, model)

        return resp.json()

    async def infer_stream_raw(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.3,
        top_p: float = 0.9,
        json_schema: Optional[dict] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        vLLM SSE 스트리밍 추론 호출 — 청크 dict를 그대로 yield.

        id, delta, finish_reason, usage(마지막 청크) 등
        메타데이터를 보존하여 OpenAI 호환 SSE 변환에 사용한다.

        마지막에 None을 yield하여 스트림 종료([DONE])를 알린다.
        """
        body = self._build_body(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            json_schema=json_schema,
            stream=True,
        )

        url = f"{self.base_url}/v1/chat/completions"
        client = await self._get_client()

        async with client.stream("POST", url, json=body) as resp:
            # [advice from AI] 스트리밍 에러 시 vLLM 응답 body를 로깅
            if resp.status_code != 200:
                error_body = await resp.aread()
                logger.error(f"[vllm-stream-raw] {resp.status_code} 에러: {error_body.decode()}")
                resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    yield None
                    break
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    continue

    async def health_check(self) -> bool:
        """vLLM 서버 상태 확인."""
        try:
            client = await self._get_client()
            resp = await client.get(f"{self.base_url}/health")
            return resp.status_code == 200
        except Exception:
            return False

    @staticmethod
    def _build_body(
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        json_schema: Optional[dict],
        stream: bool,
    ) -> dict:
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        # [advice from AI] 스트리밍 시 마지막 chunk에 usage(prompt_tokens 등) 포함
        if stream:
            body["stream_options"] = {"include_usage": True}

        if json_schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": json_schema},
            }

        return body
