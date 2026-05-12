"""
비동기 메트릭 수집기 — 요청별 레이턴시/토큰 수를 JSONL + Summary JSON으로 기록.

요청 처리 경로에 영향을 주지 않도록 asyncio.Queue 기반 fire-and-forget 패턴을 사용한다.
record() 호출은 큐에 dict를 넣기만 하므로 마이크로초 수준.
백그라운드 태스크가 큐에서 꺼내 파일에 기록한다.
"""

import asyncio
import json
import os
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


class MetricsCollector:
    """비동기 큐 기반 메트릭 수집기."""

    def __init__(self, log_dir: str = "logs"):
        self._log_dir = Path(log_dir)
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._records: Dict[str, list] = defaultdict(list)
        self._current_date: Optional[date] = None

    def record(
        self,
        model: str,
        prompt_tokens: int,
        ttft_ms: float,
        total_ms: float,
        stream: bool = False,
    ):
        """
        메트릭을 큐에 추가 (fire-and-forget, 논블로킹).

        Args:
            model: 요청 모델명 (intent/answer)
            prompt_tokens: 입력 토큰 수
            ttft_ms: 첫 토큰 시간 (ms)
            total_ms: 전체 처리 시간 (ms)
            stream: 스트리밍 여부
        """
        entry = {
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "model": model,
            "prompt_tokens": prompt_tokens,
            "ttft_ms": round(ttft_ms, 1),
            "total_ms": round(total_ms, 1),
            "stream": stream,
        }
        try:
            self._queue.put_nowait(entry)
        except asyncio.QueueFull:
            logger.warning("[metrics] 큐 가득 참, 메트릭 드롭")

    async def start(self):
        """백그라운드 flush 워커 시작."""
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._task = asyncio.create_task(self._flush_worker())
        logger.info(f"[metrics] 수집 시작: {self._log_dir}")

    async def stop(self):
        """잔여 큐 flush 후 summary 저장, 워커 종료."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        while not self._queue.empty():
            entry = self._queue.get_nowait()
            self._write_and_accumulate(entry)

        self._save_summary()
        logger.info("[metrics] 수집 종료, summary 저장 완료")

    async def _flush_worker(self):
        """백그라운드 루프: 큐에서 꺼내 JSONL 파일에 기록."""
        try:
            while True:
                entry = await self._queue.get()
                self._write_and_accumulate(entry)
        except asyncio.CancelledError:
            pass

    def _write_and_accumulate(self, entry: Dict[str, Any]):
        """JSONL append + 메모리 누적 (summary용)."""
        today = date.today()
        if self._current_date != today:
            if self._current_date is not None:
                self._save_summary()
                self._records.clear()
            self._current_date = today

        jsonl_path = self._log_dir / f"metrics_{today:%Y%m%d}.jsonl"
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        self._records[entry["model"]].append(entry)

        # [advice from AI] 3건마다 중간 summary 갱신
        total_count = sum(len(v) for v in self._records.values())
        if total_count % 3 == 0:
            self._save_summary()

    def _save_summary(self):
        """현재 누적 데이터로 summary JSON 파일 저장."""
        if not self._records or self._current_date is None:
            return

        summary: Dict[str, Any] = {
            "period": str(self._current_date),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }

        for model, records in self._records.items():
            if not records:
                continue

            prompt_tokens = [r["prompt_tokens"] for r in records]
            ttft_values = [r["ttft_ms"] for r in records]
            total_values = [r["total_ms"] for r in records]

            summary[model] = {
                "count": len(records),
                "prompt_tokens": _calc_stats(prompt_tokens),
                "ttft_ms": _calc_stats(ttft_values),
                "total_ms": _calc_stats(total_values),
            }

        summary_path = self._log_dir / f"metrics_{self._current_date:%Y%m%d}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


def _calc_stats(values: list) -> Dict[str, float]:
    """평균/min/max/p95 계산."""
    if not values:
        return {"avg": 0, "min": 0, "max": 0, "p95": 0}

    sorted_v = sorted(values)
    n = len(sorted_v)
    p95_idx = min(int(n * 0.95), n - 1)

    return {
        "avg": round(sum(sorted_v) / n, 1),
        "min": round(sorted_v[0], 1),
        "max": round(sorted_v[-1], 1),
        "p95": round(sorted_v[p95_idx], 1),
    }


metrics_collector = MetricsCollector()
