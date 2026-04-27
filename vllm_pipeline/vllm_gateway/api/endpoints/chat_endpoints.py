"""
Gateway API 라우터.

POST /completions — 2-Stage 통합 추론 (intent -> answer)
POST /intent      — intent 단독 추론
POST /answer      — answer 단독 추론
GET  /health      — Gateway + vLLM 상태 확인
"""

import json
import time
from typing import Dict, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from core.config import settings
from api.schemas.chat_schemas import (
    ChatRequest, ChatResponse, IntentResult,
    IntentRequest, IntentResponse,
    AnswerRequest, AnswerResponse,
)
from services.inference_service import InferenceService
from clients.vllm_client import VLLMClient

router = APIRouter()

_intent_client = VLLMClient(base_url=settings.INTENT_VLLM_URL)
_answer_client = VLLMClient(base_url=settings.ANSWER_VLLM_URL)
_service = InferenceService(intent_client=_intent_client, answer_client=_answer_client)


def get_service() -> InferenceService:
    return _service


@router.post("/completions", response_model=ChatResponse)
async def completions(request: ChatRequest):
    """2-Stage 통합 추론: intent(LoRA) -> answer(바닐라)."""
    messages = [m.model_dump() for m in request.messages]
    intent_messages = (
        [m.model_dump() for m in request.intent_messages]
        if request.intent_messages else None
    )

    if request.stream:
        return StreamingResponse(
            _stream_response(messages, intent_messages),
            media_type="text/event-stream",
        )

    result = await _service.process(messages, intent_messages)

    return ChatResponse(
        intent=IntentResult(**result["intent"]),
        response=result["response"],
        latency_ms=result["latency_ms"],
    )


async def _stream_response(messages, intent_messages):
    """SSE 스트리밍 응답 생성."""
    async for chunk in _service.process_stream(messages, intent_messages):
        yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/intent", response_model=IntentResponse)
async def intent_only(request: IntentRequest):
    """intent 단독 추론."""
    start = time.time()
    messages = [m.model_dump() for m in request.messages]

    intent_result = await _service.run_intent(messages)
    latency_ms = (time.time() - start) * 1000

    return IntentResponse(
        intent=IntentResult(**intent_result),
        latency_ms=round(latency_ms, 1),
    )


@router.post("/answer", response_model=AnswerResponse)
async def answer_only(request: AnswerRequest):
    """answer 단독 추론."""
    start = time.time()
    messages = [m.model_dump() for m in request.messages]

    if request.stream:
        return StreamingResponse(
            _answer_stream(messages),
            media_type="text/event-stream",
        )

    answer_text = await _service.run_answer(messages)
    latency_ms = (time.time() - start) * 1000

    return AnswerResponse(
        response=answer_text,
        latency_ms=round(latency_ms, 1),
    )


async def _answer_stream(messages):
    async for chunk in _service.run_answer_stream(messages):
        yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Gateway + vLLM 서버 상태 확인."""
    intent_ok = await _intent_client.health_check()
    answer_ok = await _answer_client.health_check()
    return {
        "gateway": "ok",
        "intent_vllm": "ok" if intent_ok else "unavailable",
        "answer_vllm": "ok" if answer_ok else "unavailable",
        "intent_vllm_url": settings.INTENT_VLLM_URL,
        "answer_vllm_url": settings.ANSWER_VLLM_URL,
    }
