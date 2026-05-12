"""
Answer 모델 런타임 스위칭 — 인메모리 상태 관리.

answer LoRA adapter ↔ base 모델을 런타임에 전환한다.
서버 재시작 시 LoRA adapter(answer)로 초기화된다.
"""

_LORA_NAME = "answer"
_BASE_NAME = "Qwen3.5-9B"

_current_answer_model: str = _LORA_NAME


def get_answer_model() -> str:
    """현재 answer 추론에 사용 중인 모델명 반환."""
    return _current_answer_model


def toggle_answer_model() -> dict:
    """answer ↔ base 모델을 토글하고 변경 결과를 반환."""
    global _current_answer_model
    previous = _current_answer_model

    _current_answer_model = _BASE_NAME if _current_answer_model == _LORA_NAME else _LORA_NAME

    return {
        "answer_model": _current_answer_model,
        "previous": previous,
        "is_base": _current_answer_model == _BASE_NAME,
    }
