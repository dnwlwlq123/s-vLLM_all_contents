"""
OpenAI 호환 Chat Completions 엔드포인트.

POST /v1/chat/completions
- model 필드("intent"/"answer")로 vLLM 서버 라우팅
- Non-Streaming / Streaming(SSE) 모두 지원
"""

import json
import time
from typing import Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from api.schemas.chat_schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from clients.vllm_client import VLLMClient
from services.request_converter import request_converter, RoutingInfo
from services.response_converter import response_converter, _generate_id, _now_timestamp
from services.metrics_collector import metrics_collector

router = APIRouter()

_clients: Dict[str, VLLMClient] = {}


def _get_client(routing: RoutingInfo) -> VLLMClient:
    """라우팅 URL별 VLLMClient 싱글톤 관리."""
    if routing.vllm_url not in _clients:
        _clients[routing.vllm_url] = VLLMClient(base_url=routing.vllm_url)
    return _clients[routing.vllm_url]


async def close_clients():
    """모든 클라이언트 리소스 정리 (lifespan 종료 시 호출)."""
    for client in _clients.values():
        await client.close()
    _clients.clear()


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI 호환 Chat Completion 엔드포인트.

    model 필드를 라우팅 구분자로 사용하여 해당 vLLM 서버로 전달하고,
    OpenAI 호환 형식으로 응답을 반환한다.
    """
    try:
        routing = request_converter.resolve_routing(request.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    client = _get_client(routing)
    vllm_params = request_converter.build_vllm_params(request, routing)

    # [advice from AI] 디버깅용: 원본 요청 및 vLLM 전달 파라미터 로그
    # logger.info(f"[completions] 라우팅: model={request.model} → vllm_url={routing.vllm_url}, vllm_model={routing.vllm_model_name}")
    # logger.info(f"[completions] request body: {json.dumps(request.model_dump(), ensure_ascii=False)}")
    logger.info(f"[completions] vllm_params: {json.dumps(vllm_params, ensure_ascii=False)}")

    if request.stream:
        return StreamingResponse(
            _stream_completions(client, vllm_params, request.model),
            media_type="text/event-stream",
        )

    t0 = time.perf_counter()
    try:
        raw_response = await client.infer_raw(**vllm_params)
    except Exception as e:
        logger.error(f"[completions] vLLM 호출 실패 ({request.model}): {repr(e)}")
        raise HTTPException(status_code=502, detail=f"vLLM 서버 호출 실패: {repr(e)}")
    latency_ms = (time.perf_counter() - t0) * 1000

    # [advice from AI] response_format을 전달하여 Gateway에서 구조화된 응답 래핑 처리
    res = response_converter.convert_completion(
        raw_response, request.model, request.response_format,
    )

    prompt_tokens = raw_response.get("usage", {}).get("prompt_tokens", 0)
    logger.info(f"[completions] {request.model} 완료: {latency_ms:.0f}ms (prompt={prompt_tokens}tok)")
    metrics_collector.record(
        model=request.model, prompt_tokens=prompt_tokens,
        ttft_ms=latency_ms, total_ms=latency_ms, stream=False,
    )
    return res


async def _stream_completions(
    client: VLLMClient,
    vllm_params: dict,
    requested_model: str,
):
    """SSE 스트리밍 응답 생성 — vLLM 청크를 OpenAI 호환 형식으로 변환하여 전달."""
    stream_id = _generate_id()
    created = _now_timestamp()
    t0 = time.perf_counter()
    ttft_ms = 0.0
    first_flush_ms = 0.0
    first_token_logged = False
    first_flush_logged = False
    prompt_tokens = 0
    chunk_count = 0
    last_chunk_ms = 0.0
    first_punct_ms = 0.0
    first_punct_logged = False
    full_response = []

    logger.info(f"[stream-timing] {requested_model} upstream 요청 시작: {time.perf_counter() - t0:.1f}ms")

    try:
        async for raw_chunk in client.infer_stream_raw(**vllm_params):
            if not first_token_logged and raw_chunk is not None:
                ttft_ms = (time.perf_counter() - t0) * 1000
                logger.info(f"[stream-timing] {requested_model} upstream 첫 바이트 수신: {ttft_ms:.0f}ms")
                first_token_logged = True

            chunk_data = response_converter.convert_chunk(
                raw_chunk, requested_model, stream_id, created,
            )
            yield chunk_data

            if raw_chunk is not None:
                chunk_count += 1
                last_chunk_ms = (time.perf_counter() - t0) * 1000
                if not first_punct_logged:
                    delta = raw_chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta and any(p in delta for p in ".,?!。，"):
                        first_punct_ms = last_chunk_ms
                        first_punct_logged = True

            if not first_flush_logged and raw_chunk is not None:
                first_flush_ms = (time.perf_counter() - t0) * 1000
                logger.info(f"[stream-timing] {requested_model} client 첫 바이트 flush: {first_flush_ms:.0f}ms")
                first_flush_logged = True

            if raw_chunk and "usage" in raw_chunk:
                prompt_tokens = raw_chunk["usage"].get("prompt_tokens", 0)
    except Exception as e:
        logger.error(f"[completions-stream] vLLM 스트리밍 실패 ({requested_model}): {repr(e)}")
        error_resp = response_converter.convert_error(
            502, f"vLLM 스트리밍 실패: {e}", "server_error",
        )
        yield f"data: {json.dumps(error_resp.model_dump(), ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    total_ms = (time.perf_counter() - t0) * 1000
    decode_ms = total_ms - ttft_ms
    itl_ms = decode_ms / chunk_count if chunk_count > 0 else 0
    logger.info(
        f"[stream-timing] {requested_model} 전체 완료: {total_ms:.0f}ms | "
        f"첫토큰(TTFT): {ttft_ms:.0f}ms | "
        f"첫토큰→클라이언트: {first_flush_ms:.0f}ms | "
        f"생성구간: {decode_ms:.0f}ms | "
        f"총 chunk: {chunk_count}개 | "
        f"토큰당 평균: {itl_ms:.1f}ms | "
        f"첫 구두점: {first_punct_ms:.0f}ms | "
        f"프롬프트: {prompt_tokens}tok"
    )
    metrics_collector.record(
        model=requested_model, prompt_tokens=prompt_tokens,
        ttft_ms=ttft_ms, total_ms=total_ms, stream=True,
    )
