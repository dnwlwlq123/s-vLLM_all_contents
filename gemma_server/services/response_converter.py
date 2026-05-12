"""
응답 변환 서비스 — vLLM 응답을 CE-Service가 기대하는 OpenAI 호환 형식으로 변환.

1. Non-Streaming: vLLM 전체 응답 dict → ChatCompletionResponse
2. Streaming: vLLM 청크 dict → ChatCompletionChunk (SSE data 문자열)
3. 에러: HTTP 상태코드 + 에러 상세 → ErrorResponse
"""

import json
import orjson
import time
import uuid
from typing import Any, Dict, Optional

from api.schemas.chat_schemas import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChunkChoice,
    ChunkDelta,
    ErrorDetail,
    ErrorResponse,
    ResponseFormat,
    UsageInfo,
)


def _generate_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


def _now_timestamp() -> int:
    return int(time.time())


class ResponseConverter:
    """vLLM 응답을 OpenAI 호환 형식으로 변환."""

    SUPPORTED_WRAP_KEYS = {"answer", "intent", "gemma-4-31B-it", "gemma-4-26B-A4B-it"}

    @staticmethod
    def _wrap_content(
        content: str,
        response_format: Optional[ResponseFormat],
    ) -> str:
        """
        response_format의 properties 키를 확인하여
        plain text content를 JSON string으로 래핑.

        sLLM은 항상 plain text를 반환하므로,
        CE-Service가 요청한 response_format에 맞춰
        Gateway에서 구조화된 JSON string으로 변환한다.
        """
        if response_format is None:
            return content

        schema = response_format.json_schema.schema_
        properties = schema.get("properties", {})

        # [advice from AI] properties에서 지원하는 키를 찾아 단일 필드 래핑
        for key in ResponseConverter.SUPPORTED_WRAP_KEYS:
            if key in properties:
                return json.dumps({key: content}, ensure_ascii=False)

        return content

    @staticmethod
    def convert_completion(
        vllm_response: Dict[str, Any],
        requested_model: str,
        response_format: Optional[ResponseFormat] = None,
    ) -> ChatCompletionResponse:
        """
        Non-Streaming vLLM 응답 → ChatCompletionResponse.

        vLLM 응답의 id/usage를 최대한 보존하되,
        model 필드는 CE-Service가 요청한 값(intent/answer)으로 교체한다.
        response_format이 있으면 content를 구조화된 JSON string으로 래핑한다.
        """
        response_id = vllm_response.get("id", _generate_id())
        created = vllm_response.get("created", _now_timestamp())

        raw_choices = vllm_response.get("choices", [])
        choices = []
        for raw_choice in raw_choices:
            msg = raw_choice.get("message", {})
            content = ResponseConverter._wrap_content(
                msg.get("content", ""), response_format,
            )
            choices.append(ChatCompletionChoice(
                index=raw_choice.get("index", 0),
                message=AssistantMessage(
                    role=msg.get("role", "assistant"),
                    content=content,
                ),
                finish_reason=raw_choice.get("finish_reason", "stop"),
            ))

        raw_usage = vllm_response.get("usage", {})
        usage = UsageInfo(
            prompt_tokens=raw_usage.get("prompt_tokens", 0),
            completion_tokens=raw_usage.get("completion_tokens", 0),
            total_tokens=raw_usage.get("total_tokens", 0),
        )

        return ChatCompletionResponse(
            id=response_id,
            object="chat.completion",
            created=created,
            model=requested_model,
            choices=choices,
            usage=usage,
        )

    @staticmethod
    def convert_chunk(
        vllm_chunk: Optional[Dict[str, Any]],
        requested_model: str,
        stream_id: str,
        created: int,
    ) -> bytes:
        """[PERF-G2] Fast SSE pass-through: dict 재빌드 없이 id/model/created 만 교체 후 orjson 직접 직렬화."""
        if vllm_chunk is None:
            return b"data: [DONE]\n\n"
        vllm_chunk["id"] = stream_id
        vllm_chunk["model"] = requested_model
        vllm_chunk["created"] = created
        return b"data: " + orjson.dumps(vllm_chunk) + b"\n\n"

    @staticmethod
    def convert_error(
        status_code: int,
        message: str,
        error_type: str = "internal_error",
        code: Optional[str] = None,
    ) -> ErrorResponse:
        """에러 → OpenAI 호환 ErrorResponse."""
        return ErrorResponse(
            error=ErrorDetail(
                message=message,
                type=error_type,
                code=code,
            )
        )


response_converter = ResponseConverter()
