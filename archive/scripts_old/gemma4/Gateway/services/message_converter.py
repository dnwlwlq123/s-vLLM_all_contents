"""
메시지 변환 서비스 — CE-Service 멀티턴 messages를 sLLM 학습 포맷으로 변환.

CE-Service가 보내는 OpenAI 멀티턴 형식:
    system + user + assistant + user + ...

sLLM 학습 포맷 (2턴 구조):
    system + user([conversation_history] + 현재 발화)

model별 변환 규칙:
- intent: system(RAG 없음) + user([conversation_history] + 현재 발화)
- answer: system(RAG 제거) + user([conversation_history] + [수집된 고객 정보] + [FAQ / RAG 검색결과] + 현재 발화)
"""

import re
from typing import Dict, List, Optional, Tuple

from loguru import logger


RAG_PATTERN = re.compile(r"<RAG>(.*?)</RAG>", re.DOTALL)
# [advice from AI] "## RAG 검색 결과" 헤더 + 이후 <RAG>...</RAG> + 잔여 줄바꿈까지 제거
RAG_SECTION_PATTERN = re.compile(r"\n*##\s*RAG 검색 결과\s*\n*", re.DOTALL)


class MessageConverter:
    """CE-Service 멀티턴 messages를 sLLM 학습 포맷 2턴 구조로 변환."""

    def convert_for_intent(
        self, messages: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        Intent 모델용 변환.

        - system content 그대로 사용 (intent 요청에는 RAG 없음)
        - 멀티턴 → [conversation_history] + 고객 현재 발화
        """
        system_content, turns = self._split_messages(messages)

        user_content = self._build_user_content(turns)

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def convert_for_answer(
        self, messages: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        Answer 모델용 변환.

        - system content에서 <RAG>...</RAG> 추출 후 제거
        - 멀티턴 → [conversation_history] + [수집된 고객 정보] + [FAQ / RAG 검색결과] + 현재 발화
        """
        system_content, turns = self._split_messages(messages)

        rag_content = self._extract_rag(system_content)
        clean_system = self._remove_rag(system_content)

        user_content = self._build_user_content(turns, rag_content=rag_content)

        return [
            {"role": "system", "content": clean_system},
            {"role": "user", "content": user_content},
        ]

    # ──────────────────────────────────────────────
    #  내부 유틸리티
    # ──────────────────────────────────────────────

    @staticmethod
    def _split_messages(
        messages: List[Dict[str, str]],
    ) -> Tuple[str, List[Dict[str, str]]]:
        """system 메시지와 user/assistant 턴 목록을 분리."""
        system_content = ""
        turns = []
        for m in messages:
            if m["role"] == "system":
                system_content = m["content"]
            else:
                turns.append(m)
        return system_content, turns

    @staticmethod
    def _extract_rag(text: str) -> Optional[str]:
        """system content에서 <RAG>...</RAG> 내용 추출."""
        match = RAG_PATTERN.search(text)
        return match.group(1).strip() if match else None

    @staticmethod
    def _remove_rag(text: str) -> str:
        """system content에서 '## RAG 검색 결과' 헤더 + <RAG>...</RAG> 태그 제거."""
        text = RAG_PATTERN.sub("", text)
        text = RAG_SECTION_PATTERN.sub("", text)
        return text.strip()

    @staticmethod
    def _build_user_content(
        turns: List[Dict[str, str]],
        rag_content: Optional[str] = None,
    ) -> str:
        """
        멀티턴 messages → 학습 포맷 user content.

        RAG를 앞에 배치하여 vLLM prefix caching 효율을 높인다.
        동일 주제 질문 시 system_prompt + RAG 부분까지 KV cache 재사용 가능.

        구조:
            [FAQ / RAG 검색결과]           ← answer + RAG 있을 때만 (prefix caching 대상)
            RAG 내용

            [수집된 고객 정보]              ← answer + RAG 있을 때만
            (수집된 정보 없음)

            [conversation_history]        ← 이전 턴이 있을 때만
            고객: ...\nAI: ...\n고객: ...

            고객: 현재 발화
        """
        if not turns:
            return ""

        last_user = turns[-1]
        history = turns[:-1]

        parts = []

        if rag_content is not None:
            parts.append("[FAQ / RAG 검색결과]\n" + rag_content)
            parts.append("[수집된 고객 정보]\n(수집된 정보 없음)")

        if history:
            lines = []
            for m in history:
                prefix = "고객" if m["role"] == "user" else "AI"
                lines.append(f"{prefix}: {m['content']}")
            parts.append("[conversation_history]\n" + "\n".join(lines))

        parts.append(f"고객: {last_user['content']}")

        return "\n\n".join(parts)


message_converter = MessageConverter()
