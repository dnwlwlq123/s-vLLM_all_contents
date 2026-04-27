"""
2-Stage 추론 서비스 — intent(LoRA) -> answer(바닐라) 파이프라인.

외부 클라이언트가 보낸 messages(프롬프트 포함)를 그대로 vLLM에 전달한다.
Gateway는 프롬프트 변환 없이 라우팅만 담당한다.
"""

import json
import re
import time
from typing import Dict, Any, Optional, List, AsyncGenerator

from loguru import logger

from core.config import settings
from clients.vllm_client import VLLMClient


class InferenceService:
    """intent -> answer 2-Stage 추론 파이프라인."""

    def __init__(
        self,
        intent_client: Optional[VLLMClient] = None,
        answer_client: Optional[VLLMClient] = None,
    ):
        self._intent_client = intent_client or VLLMClient(base_url=settings.INTENT_VLLM_URL)
        self._answer_client = answer_client or VLLMClient(base_url=settings.ANSWER_VLLM_URL)

    async def close(self):
        await self._intent_client.close()
        await self._answer_client.close()

    # ──────────────────────────────────────────────
    #  통합 추론 (intent -> answer)
    # ──────────────────────────────────────────────
    async def process(
        self,
        messages: List[Dict[str, str]],
        intent_messages: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        비스트리밍 2-Stage 추론.

        Args:
            messages: answer용 messages (프롬프트 포함)
            intent_messages: intent용 별도 messages (None이면 messages 사용)

        Returns:
            {"intent": {...}, "response": "...", "latency_ms": ...}
        """
        start = time.time()

        intent_result = await self.run_intent(intent_messages or messages)

        answer_text = await self.run_answer(messages)

        latency_ms = (time.time() - start) * 1000

        return {
            "intent": intent_result,
            "response": answer_text,
            "latency_ms": round(latency_ms, 1),
        }

    async def process_stream(
        self,
        messages: List[Dict[str, str]],
        intent_messages: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        스트리밍 2-Stage 추론.

        1. intent 비스트리밍 호출
        2. intent 결과를 JSON으로 yield
        3. answer 스트리밍 호출, 토큰 단위로 yield

        Yields:
            "intent:{json}" — intent 결과
            이후 answer 토큰 chunk들
        """
        intent_result = await self.run_intent(intent_messages or messages)

        yield f"intent:{json.dumps(intent_result, ensure_ascii=False)}"

        async for chunk in self._answer_client.infer_stream(
            model=settings.ANSWER_MODEL_NAME,
            messages=messages,
            max_tokens=settings.ANSWER_MAX_TOKENS,
            temperature=settings.ANSWER_TEMPERATURE,
            top_p=settings.DEFAULT_TOP_P,
        ):
            # [advice from AI] 스트리밍 chunk에서 한자/일본어 할루시네이션 필터링
            chunk = self._filter_hallucination_chars(chunk)
            if chunk:
                yield chunk

    # ──────────────────────────────────────────────
    #  개별 단계
    # ──────────────────────────────────────────────
    async def run_intent(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """intent 추론 단독 실행."""
        raw = await self._intent_client.infer(
            model=settings.INTENT_MODEL_NAME,
            messages=messages,
            max_tokens=settings.INTENT_MAX_TOKENS,
            temperature=settings.INTENT_TEMPERATURE,
            top_p=settings.DEFAULT_TOP_P,
        )

        return self._parse_intent(raw)

    async def run_answer(self, messages: List[Dict[str, str]]) -> str:
        """answer 추론 단독 실행."""
        raw = await self._answer_client.infer(
            model=settings.ANSWER_MODEL_NAME,
            messages=messages,
            max_tokens=settings.ANSWER_MAX_TOKENS,
            temperature=settings.ANSWER_TEMPERATURE,
            top_p=settings.DEFAULT_TOP_P,
        )

        return self._parse_answer(raw)

    async def run_answer_stream(
        self, messages: List[Dict[str, str]],
    ) -> AsyncGenerator[str, None]:
        """answer 스트리밍 추론 단독 실행."""
        async for chunk in self._answer_client.infer_stream(
            model=settings.ANSWER_MODEL_NAME,
            messages=messages,
            max_tokens=settings.ANSWER_MAX_TOKENS,
            temperature=settings.ANSWER_TEMPERATURE,
            top_p=settings.DEFAULT_TOP_P,
        ):
            # [advice from AI] 스트리밍 chunk에서 한자/일본어 할루시네이션 필터링
            chunk = self._filter_hallucination_chars(chunk)
            if chunk:
                yield chunk

    # ──────────────────────────────────────────────
    #  출력 파싱
    # ──────────────────────────────────────────────
    # [advice from AI] 한자/일본어 할루시네이션 필터용 정규식 (컴파일 1회)
    _HALLUCINATION_PATTERN = re.compile(
        r'[\u4E00-\u9FFF\u3400-\u4DBF\u3040-\u309F\u30A0-\u30FF]'
    )

    @staticmethod
    def _filter_hallucination_chars(text: str) -> str:
        """
        한자(CJK) 및 일본어(히라가나/가타카나) 문자 제거.
        sLLM 응답에 해당 문자가 포함되면 할루시네이션으로 간주한다.
        """
        filtered = InferenceService._HALLUCINATION_PATTERN.sub('', text)
        if len(filtered) != len(text):
            logger.warning(
                "Hallucination filtered | before={!r} | after={!r}",
                text, filtered,
            )
        return filtered

    @staticmethod
    def _clean_thinking_tags(text: str) -> str:
        """Qwen3 thinking mode 태그 제거."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _parse_intent(self, raw_output: str) -> Dict[str, Any]:
        """intent 출력 파싱 -> {"name": "...", "confidence": ...}"""
        cleaned = self._clean_thinking_tags(raw_output)

        parsed = self._try_json_parse(cleaned)
        if parsed:
            intent = parsed.get("intent", parsed)
            name = intent.get("name", "unknown") if isinstance(intent, dict) else str(intent)
            confidence = intent.get("confidence", 0.0) if isinstance(intent, dict) else 0.0
            return {"name": name.lower(), "confidence": confidence}

        for candidate in ["faq", "clarify", "agent", "end"]:
            if candidate in cleaned.lower():
                return {"name": candidate, "confidence": 1.0}

        return {"name": cleaned.strip().lower() or "unknown", "confidence": 0.0}

    def _parse_answer(self, raw_output: str) -> str:
        """answer 출력 파싱 -> 응답 텍스트"""
        cleaned = self._clean_thinking_tags(raw_output)

        parsed = self._try_json_parse(cleaned)
        if parsed and "response" in parsed:
            return parsed["response"]

        return cleaned if cleaned else "죄송합니다. 응답을 생성할 수 없습니다."

    @staticmethod
    def _try_json_parse(text: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        return None
