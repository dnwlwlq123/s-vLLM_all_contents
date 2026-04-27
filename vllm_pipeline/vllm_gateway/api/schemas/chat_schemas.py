"""
Gateway API 요청/응답 스키마.

Timbel sLLM API 명세 (OpenAI Chat Completion API 호환) 기반.
참조: dev/reference/timbel-sllm-api-spec.md
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


# ══════════════════════════════════════════════════════════════
#  OpenAI 호환 스키마 — CE-Service ↔ Gateway 인터페이스
# ══════════════════════════════════════════════════════════════

# --- 공통 ---

class ChatMessage(BaseModel):
    """개별 대화 메시지."""
    role: str = Field(..., description="system, user, assistant 중 하나")
    content: str = Field(..., description="메시지 내용")


# --- 요청 (Request) ---

class JsonSchemaSpec(BaseModel):
    """response_format.json_schema 내부 스키마 정의."""
    name: str = Field(..., description="스키마 이름")
    strict: bool = Field(default=True)
    schema_: Dict[str, Any] = Field(..., alias="schema", description="JSON Schema 정의")

    model_config = {"populate_by_name": True}


class ResponseFormat(BaseModel):
    """구조화된 출력 형식 — response_format 파라미터."""
    type: str = Field(..., description="json_schema")
    json_schema: JsonSchemaSpec = Field(..., description="JSON Schema 스펙")


class ChatCompletionRequest(BaseModel):
    """
    OpenAI 호환 Chat Completion 요청.

    model 필드를 "intent" / "answer" 라우팅 구분자로 사용한다.
    """
    model: str = Field(..., description="라우팅 구분자: intent 또는 answer")
    messages: List[ChatMessage] = Field(..., min_length=1, description="ChatML 메시지 배열")
    temperature: Optional[float] = Field(default=None, ge=0, le=2, description="샘플링 온도")
    max_tokens: Optional[int] = Field(default=None, gt=0, description="최대 생성 토큰 수")
    top_p: Optional[float] = Field(default=None, ge=0, le=1, description="Top-p 샘플링")
    stream: bool = Field(default=False, description="True면 SSE 스트리밍 응답")
    response_format: Optional[ResponseFormat] = Field(default=None, description="구조화된 출력 형식")


# --- 응답 (Response) — Non-Streaming ---

class UsageInfo(BaseModel):
    """토큰 사용량 정보."""
    prompt_tokens: int = Field(..., description="입력 토큰 수")
    completion_tokens: int = Field(..., description="출력 토큰 수")
    total_tokens: int = Field(..., description="총 토큰 수")


class AssistantMessage(BaseModel):
    """choices[].message — 어시스턴트 응답 메시지."""
    role: str = Field(default="assistant")
    content: str = Field(..., description="생성된 텍스트")


class ChatCompletionChoice(BaseModel):
    """choices[] 배열 요소."""
    index: int = Field(default=0)
    message: AssistantMessage
    finish_reason: Optional[str] = Field(default="stop", description="종료 사유: stop, length 등")


class ChatCompletionResponse(BaseModel):
    """OpenAI 호환 Non-Streaming 응답."""
    id: str = Field(..., description="응답 고유 ID")
    object: str = Field(default="chat.completion")
    created: int = Field(..., description="생성 Unix timestamp")
    model: str = Field(..., description="요청 시 사용한 모델명 (intent/answer)")
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


# --- 응답 (Response) — Streaming (SSE) ---

class ChunkDelta(BaseModel):
    """choices[].delta — 스트리밍 청크 내용."""
    role: Optional[str] = Field(default=None, description="첫 번째 청크에만 포함")
    content: Optional[str] = Field(default=None, description="생성된 텍스트 조각")


class ChunkChoice(BaseModel):
    """스트리밍 choices[] 배열 요소."""
    index: int = Field(default=0)
    delta: ChunkDelta
    finish_reason: Optional[str] = Field(default=None, description="마지막 청크에서 stop 등")


class ChatCompletionChunk(BaseModel):
    """OpenAI 호환 Streaming 청크 응답."""
    id: str = Field(..., description="응답 고유 ID")
    object: str = Field(default="chat.completion.chunk")
    created: int = Field(..., description="생성 Unix timestamp")
    model: str = Field(..., description="요청 시 사용한 모델명")
    choices: List[ChunkChoice]
    usage: Optional[UsageInfo] = Field(default=None, description="마지막 청크에 포함")


# --- 에러 응답 ---

class ErrorDetail(BaseModel):
    """에러 상세 정보."""
    message: str = Field(..., description="에러 상세 메시지")
    type: str = Field(..., description="에러 타입 (invalid_request_error 등)")
    code: Optional[str] = Field(default=None, description="에러 코드 (invalid_api_key 등)")


class ErrorResponse(BaseModel):
    """OpenAI 호환 에러 응답."""
    error: ErrorDetail


# ══════════════════════════════════════════════════════════════
#  레거시 스키마 — 2-Stage 통합 추론용 (레거시 엔드포인트 유지용)
# ══════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    """추론 요청. 프롬프트는 messages에 포함되어 전달된다."""
    messages: List[ChatMessage] = Field(
        ..., min_length=1,
        description="answer용 ChatML 메시지 (프롬프트 포함)",
    )
    stream: bool = Field(default=False, description="True면 SSE 스트리밍 응답")
    intent_messages: Optional[List[ChatMessage]] = Field(
        default=None,
        description="intent용 별도 메시지 (None이면 messages 사용)",
    )


class IntentResult(BaseModel):
    """intent 분류 결과."""
    name: str = Field(..., description="intent 이름 (faq, clarify, agent, end 등)")
    confidence: float = Field(default=0.0, description="신뢰도")


class ChatResponse(BaseModel):
    """비스트리밍 통합 응답."""
    intent: IntentResult = Field(..., description="intent 분류 결과")
    response: str = Field(..., description="answer 응답 텍스트")
    latency_ms: float = Field(default=0.0, description="처리 시간 (ms)")


class IntentRequest(BaseModel):
    """intent 단독 요청."""
    messages: List[ChatMessage] = Field(..., min_length=1)


class IntentResponse(BaseModel):
    """intent 단독 응답."""
    intent: IntentResult
    latency_ms: float = 0.0


class AnswerRequest(BaseModel):
    """answer 단독 요청."""
    messages: List[ChatMessage] = Field(..., min_length=1)
    stream: bool = False


class AnswerResponse(BaseModel):
    """answer 단독 비스트리밍 응답."""
    response: str
    latency_ms: float = 0.0
