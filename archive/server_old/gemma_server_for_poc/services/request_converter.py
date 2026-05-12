"""
요청 변환 서비스 — CE-Service 요청을 vLLM 내부 형식으로 변환.

CE-Service가 보내는 OpenAI 호환 요청에서:
1. model 필드 기반 라우팅 (intent/answer → vLLM URL + 실제 모델명)
2. 멀티턴 messages → sLLM 학습 포맷 2턴 구조로 변환
3. 추론 파라미터 결정 (요청 값 우선, 없으면 config 기본값)
4. response_format 변환 (명세 형식 → vLLM 내부 형식)
5. Prefix cache 최적화: system 프롬프트 끝의 RAG 섹션을 마지막 user 메시지로 이동
6. Intent 요청 감지 시 guided_choice 로 라벨 고정 (GUIDED_CHOICE_ENABLED 로 on/off)
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from core.config import settings
from api.schemas.chat_schemas import ChatCompletionRequest
from services.message_converter import message_converter


# ══════════════════════════════════════════════════════════════
#  Prefix cache 최적화 헬퍼
# ══════════════════════════════════════════════════════════════

# CE-Service 가 system 프롬프트에 포함해서 보내는 RAG 섹션 헤더
# (message_converter.py 의 dead-code 정규식과 다름 — 실제 포맷 기반)
_RAG_HEADER_LINE = "# input - RAG 검색 결과"


# ══════════════════════════════════════════════════════════════
#  Intent guided_choice 설정
#  서버 한 대에 한 번에 하나의 모델/도메인만 서빙되므로 union 으로 넣어두면
#  현재 올라간 도메인에 해당하는 라벨만 실제로 매칭됨. 라벨 추가/삭제 시
#  이 리스트 업데이트 + gateway 재시작.
# ══════════════════════════════════════════════════════════════

# Intent 분류 요청 식별자 — system 프롬프트 내 "의도분류" / "의도 분류" 둘 다 매칭
# (실제 프롬프트엔 "의도분류기" 와 "[의도 분류 지침]" 두 표현이 같이 쓰임)
_INTENT_MARKER_RE = re.compile(r"의도\s*분류")

# Intent 라벨 외부 JSON 파일 경로 — 프롬프팅팀이 코드 안 건드리고 수정 가능
# 파일 구조: { "shared": [...], "finance": [...], "homeshopping": [...], ... }
# 수정 후 gateway 재시작 하면 반영됨.
_INTENT_LABELS_PATH = Path(__file__).resolve().parent.parent / "config" / "intent_labels.json"


def _load_intent_labels() -> List[str]:
    """JSON 에서 라벨 로드 + 순서 유지 dedup. 파일 없거나 잘못되면 빈 리스트."""
    try:
        with open(_INTENT_LABELS_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(
            f"[intent_labels] {_INTENT_LABELS_PATH} 로드 실패: {e}. "
            f"guided_choice 비활성 상태로 동작."
        )
        return []

    labels = []
    for key, group in data.items():
        if key.startswith("_") or not isinstance(group, list):
            continue  # '_comment' 같은 메타필드 스킵
        labels.extend(group)
    deduped = list(dict.fromkeys(labels))  # 순서 유지 dedup
    logger.info(
        f"[intent_labels] {len(deduped)}개 라벨 로드 from {_INTENT_LABELS_PATH.name}"
    )
    return deduped


_INTENT_LABELS = _load_intent_labels()

# Intent 응답 최대 토큰 — 가장 긴 라벨 (예: 'search_previous', 'confirm_modify') 이 약 4~5 토큰
_INTENT_MAX_TOKENS = 10


def _is_guided_choice_enabled() -> bool:
    """
    GUIDED_CHOICE_ENABLED 환경변수로 intent guided_choice 주입 on/off 결정.

    기본값: true (켜짐). 프롬프팅팀이 새 라벨 테스트 시:
        GUIDED_CHOICE_ENABLED=false
    로 끄면 자유 생성 → 라벨 확정 후 _INTENT_LABELS 업데이트 + 재시작.
    """
    return os.environ.get("GUIDED_CHOICE_ENABLED", "true").strip().lower() not in (
        "false", "0", "no", "off", ""
    )


def _is_intent_request(messages: List[Dict[str, str]]) -> bool:
    """system 프롬프트에 '의도분류' 또는 '의도 분류' 가 있으면 intent 요청으로 판정."""
    for m in messages:
        if m.get("role") == "system" and _INTENT_MARKER_RE.search(m.get("content") or ""):
            return True
    return False


def _move_rag_to_last_user(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    system 프롬프트 끝의 '# input - RAG 검색 결과' 섹션을 추출해서
    마지막 user 메시지 앞에 [RAG 검색 결과] 블록으로 붙임.

    목적: system 을 순수 정적 프롬프트로 만들어 vLLM prefix cache 100% hit 유도.
    턴마다 바뀌는 건 RAG 뿐이므로, RAG 를 뒤로 빼면 긴 system prompt (수천 토큰)
    가 전부 재사용됨 → TTFT 대폭 감소.

    RAG 섹션이 없으면 (intent 요청 등) messages 그대로 반환.
    """
    result = [dict(m) for m in messages]
    for i, m in enumerate(result):
        if m.get("role") != "system":
            continue
        content = m.get("content", "") or ""
        idx = content.find(_RAG_HEADER_LINE)
        if idx == -1:
            continue

        # 헤더 이후 전체를 RAG 본문으로 잘라냄 (CE-Service 포맷상 RAG 는 system 맨 끝)
        rag_text = content[idx + len(_RAG_HEADER_LINE):].lstrip("\n").rstrip()
        clean_system = content[:idx].rstrip()
        result[i] = {**m, "content": clean_system}

        # 가장 마지막 user 메시지 앞에 RAG 블록 prepend
        for j in range(len(result) - 1, -1, -1):
            if result[j].get("role") == "user":
                orig = result[j].get("content", "") or ""
                result[j] = {
                    **result[j],
                    "content": (
                        f"[RAG 검색 결과]\n{rag_text}\n\n"
                        f"[고객 질문]\n{orig}"
                    ),
                }
                break
        break  # system 메시지는 하나라고 가정
    return result


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
    "gemma-4-31B-it": lambda: RoutingInfo(
        vllm_url=settings.ANSWER_VLLM_URL,
        vllm_model_name=settings.BASE_MODEL_NAME,
        default_max_tokens=settings.ANSWER_MAX_TOKENS,
        default_temperature=settings.ANSWER_TEMPERATURE,
        default_top_p=settings.DEFAULT_TOP_P,
    ),
    "gemma-4-26B-A4B-it": lambda: RoutingInfo(
        vllm_url=settings.ANSWER_VLLM_URL,
        vllm_model_name=settings.BASE_MODEL_NAME,
        default_max_tokens=settings.ANSWER_MAX_TOKENS,
        default_temperature=settings.ANSWER_TEMPERATURE,
        default_top_p=settings.DEFAULT_TOP_P,
    ),
    "Qwen3.5-35B-A3B": lambda: RoutingInfo(
        vllm_url=settings.ANSWER_VLLM_URL,
        vllm_model_name="Qwen3.5-35B-A3B-FP8",
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
        3. system 프롬프트 끝의 RAG 섹션을 마지막 user 메시지로 이동 (prefix cache 최적화)
        4. Intent 요청 감지 시 guided_choice 로 라벨 고정 (GUIDED_CHOICE_ENABLED=true 일 때)
        """
        raw_messages = [m.model_dump() for m in request.messages]

        # Prefix cache 최적화: system 순수 정적화를 위해 RAG 를 user 쪽으로 이동
        # (RAG 없는 요청 — intent 등 — 은 원본 그대로 반환됨)
        messages = _move_rag_to_last_user(raw_messages)

        params: Dict[str, Any] = {
            "model": routing.vllm_model_name,
            "messages": messages,
            "max_tokens": request.max_tokens if request.max_tokens is not None else routing.default_max_tokens,
            "temperature": request.temperature if request.temperature is not None else routing.default_temperature,
            "top_p": request.top_p if request.top_p is not None else routing.default_top_p,
        }

        if request.top_k is not None:
            params["top_k"] = request.top_k
        if request.presence_penalty is not None:
            params["presence_penalty"] = request.presence_penalty
        if request.response_format is not None:
            params["response_format"] = request.response_format.model_dump(by_alias=True)
            if params["max_tokens"] > 128:
                params["max_tokens"] = 128
            params["stop"] = params.get("stop", []) + ["\n\n"]

        # Intent 분류 요청 + guided_choice 기능 켜짐 + 라벨 존재 → 라벨 강제 + max_tokens 축소
        # 프롬프팅팀 새 라벨 테스트 시 GUIDED_CHOICE_ENABLED=false 로 우회 가능
        # intent_labels.json 없거나 빈 리스트면 주입 스킵 (vLLM 이 빈 guided_choice 거부함)
        if (
            _is_guided_choice_enabled()
            and _INTENT_LABELS
            and _is_intent_request(messages)
        ):
            params["guided_choice"] = _INTENT_LABELS
            params["max_tokens"] = min(params["max_tokens"], _INTENT_MAX_TOKENS)

        return params


request_converter = RequestConverter()
