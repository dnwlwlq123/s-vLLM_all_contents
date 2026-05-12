"""
요청 변환 서비스 — CE-Service 요청을 vLLM 내부 형식으로 변환.

CE-Service가 보내는 OpenAI 호환 요청에서:
1. model 필드 기반 라우팅 (intent/answer → vLLM URL + 실제 모델명)
2. 멀티턴 messages → sLLM 학습 포맷 2턴 구조로 변환
3. 추론 파라미터 결정 (요청 값 우선, 없으면 config 기본값)
4. response_format 변환 (명세 형식 → vLLM 내부 형식)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.config import settings
from api.schemas.chat_schemas import ChatCompletionRequest
from services.message_converter import message_converter


@dataclass
class RoutingInfo:
    """model 필드 기반 라우팅 결정 결과."""
    vllm_url: str
    vllm_model_name: str
    default_max_tokens: int
    default_temperature: float
    default_top_p: float


ROUTING_TABLE: Dict[str, callable] = {
    "intent": lambda: RoutingInfo(
        vllm_url=settings.INTENT_VLLM_URL,
        vllm_model_name=settings.INTENT_MODEL_NAME,
        default_max_tokens=settings.INTENT_MAX_TOKENS,
        default_temperature=settings.INTENT_TEMPERATURE,
        default_top_p=settings.DEFAULT_TOP_P,
    ),
    "answer": lambda: RoutingInfo(
        vllm_url=settings.ANSWER_VLLM_URL,
        vllm_model_name=settings.ANSWER_MODEL_NAME,
        default_max_tokens=settings.ANSWER_MAX_TOKENS,
        default_temperature=settings.ANSWER_TEMPERATURE,
        default_top_p=settings.DEFAULT_TOP_P,
    ),
    "qwen3.5-9b_intent": lambda: RoutingInfo(
        vllm_url=settings.INTENT_VLLM_URL,
        vllm_model_name=settings.INTENT_MODEL_NAME,
        default_max_tokens=settings.INTENT_MAX_TOKENS,
        default_temperature=settings.INTENT_TEMPERATURE,
        default_top_p=settings.DEFAULT_TOP_P,
    ),
    "qwen3.5-9b_answer": lambda: RoutingInfo(
        vllm_url=settings.ANSWER_VLLM_URL,
        vllm_model_name=settings.ANSWER_MODEL_NAME,
        default_max_tokens=settings.ANSWER_MAX_TOKENS,
        default_temperature=settings.ANSWER_TEMPERATURE,
        default_top_p=settings.DEFAULT_TOP_P,
    ),
    "qwen3.5-9b": lambda: RoutingInfo(
        vllm_url=settings.ANSWER_VLLM_URL,
        vllm_model_name=settings.BASE_MODEL_NAME,
        default_max_tokens=settings.ANSWER_MAX_TOKENS,
        default_temperature=settings.ANSWER_TEMPERATURE,
        default_top_p=settings.DEFAULT_TOP_P,
    ),
    "Qwen3.5-9B": lambda: RoutingInfo(
        vllm_url=settings.ANSWER_VLLM_URL,
        vllm_model_name=settings.BASE_MODEL_NAME,
        default_max_tokens=settings.ANSWER_MAX_TOKENS,
        default_temperature=settings.ANSWER_TEMPERATURE,
        default_top_p=settings.DEFAULT_TOP_P,
    ),
    "qwen3.5-27b": lambda: RoutingInfo(
        vllm_url=settings.ANSWER_VLLM_URL,
        vllm_model_name=settings.BASE_MODEL_NAME,
        default_max_tokens=settings.ANSWER_MAX_TOKENS,
        default_temperature=settings.ANSWER_TEMPERATURE,
        default_top_p=settings.DEFAULT_TOP_P,
    ),
    "Qwen3.5-27B": lambda: RoutingInfo(
        vllm_url=settings.ANSWER_VLLM_URL,
        vllm_model_name=settings.BASE_MODEL_NAME,
        default_max_tokens=settings.ANSWER_MAX_TOKENS,
        default_temperature=settings.ANSWER_TEMPERATURE,
        default_top_p=settings.DEFAULT_TOP_P,
    ),
    "Qwen3.5-27B-FP8": lambda: RoutingInfo(
        vllm_url=settings.ANSWER_VLLM_URL,
        vllm_model_name=settings.BASE_MODEL_NAME,
        default_max_tokens=settings.ANSWER_MAX_TOKENS,
        default_temperature=settings.ANSWER_TEMPERATURE,
        default_top_p=settings.DEFAULT_TOP_P,
    ),
    "Qwen3.5-35B-A3B": lambda: RoutingInfo(
        vllm_url=settings.ANSWER_VLLM_URL,
        vllm_model_name=settings.BASE_MODEL_NAME,
        default_max_tokens=settings.ANSWER_MAX_TOKENS,
        default_temperature=settings.ANSWER_TEMPERATURE,
        default_top_p=settings.DEFAULT_TOP_P,
    ),
    "Qwen3.5-35B-A3B-FP8": lambda: RoutingInfo(
        vllm_url=settings.ANSWER_VLLM_URL,
        vllm_model_name=settings.BASE_MODEL_NAME,
        default_max_tokens=settings.ANSWER_MAX_TOKENS,
        default_temperature=settings.ANSWER_TEMPERATURE,
        default_top_p=settings.DEFAULT_TOP_P,
    ),
    "Qwen3.6-35B-A3B": lambda: RoutingInfo(
        vllm_url=settings.ANSWER_VLLM_URL,
        vllm_model_name=settings.BASE_MODEL_NAME,
        default_max_tokens=settings.ANSWER_MAX_TOKENS,
        default_temperature=settings.ANSWER_TEMPERATURE,
        default_top_p=settings.DEFAULT_TOP_P,
    ),
    "Qwen3.6-35B-A3B-FP8": lambda: RoutingInfo(
        vllm_url=settings.ANSWER_VLLM_URL,
        vllm_model_name=settings.BASE_MODEL_NAME,
        default_max_tokens=settings.ANSWER_MAX_TOKENS,
        default_temperature=settings.ANSWER_TEMPERATURE,
        default_top_p=settings.DEFAULT_TOP_P,
    ),
}

ALLOWED_MODELS = set(ROUTING_TABLE.keys())


class RequestConverter:
    """CE-Service 요청(OpenAI 형식)을 vLLM 호출 파라미터로 변환."""

    @staticmethod
    def resolve_routing(model: str) -> RoutingInfo:
        """
        model 필드 기반으로 라우팅 정보를 결정.

        Raises:
            ValueError: 허용되지 않은 model 값
        """
        factory = ROUTING_TABLE.get(model)
        if factory is None:
            raise ValueError(
                f"지원하지 않는 model: '{model}'. "
                f"허용 값: {', '.join(sorted(ALLOWED_MODELS))}"
            )
        return factory()

    CONVERTER_MAP = {
        "intent": lambda msgs: message_converter.convert_for_intent(msgs),
        "answer": lambda msgs: message_converter.convert_for_answer(msgs),
        "qwen3.5-9b_intent": lambda msgs: message_converter.convert_for_intent(msgs),
        "qwen3.5-9b_answer": lambda msgs: message_converter.convert_for_answer(msgs),
    }

    @staticmethod
    def build_vllm_params(
        request: ChatCompletionRequest,
        routing: RoutingInfo,
    ) -> Dict[str, Any]:
        """
        CE-Service 요청 + 라우팅 정보 → vLLM 호출 파라미터.

        1. 멀티턴 messages → model별 학습 포맷 2턴 구조로 변환
        2. 요청에 값이 있으면 그대로 사용, 없으면 config 기본값 적용
        """
        raw_messages = [m.model_dump() for m in request.messages]

        # 멀티턴 메시지를 변환 없이 그대로 전달 (Qwen3.5는 멀티턴 네이티브 지원)
        messages = raw_messages

        params: Dict[str, Any] = {
            "model": routing.vllm_model_name,
            "messages": messages,
            "max_tokens": request.max_tokens if request.max_tokens is not None else routing.default_max_tokens,
            "temperature": request.temperature if request.temperature is not None else routing.default_temperature,
            "top_p": request.top_p if request.top_p is not None else routing.default_top_p,
        }

        return params


request_converter = RequestConverter()
