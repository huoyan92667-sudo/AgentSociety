"""Deterministic fake adapter used before any real provider call."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from yelp_agent.agent.llm import LLMCallResult, LLMMessage


class FakeChatGenerator:
    def __init__(self, results: Iterable[LLMCallResult]) -> None:
        self._results = deque(results)
        self.messages: list[list[LLMMessage]] = []

    @property
    def call_count(self) -> int:
        return len(self.messages)

    def generate(self, messages: list[LLMMessage]) -> LLMCallResult:
        self.messages.append(messages)
        if not self._results:
            raise AssertionError("FakeChatGenerator has no queued result")
        return self._results.popleft()


class CacheOnlyChatGenerator:
    """Refuse provider access while allowing validated cache replay."""

    def generate(self, messages: list[LLMMessage]) -> LLMCallResult:
        del messages
        return LLMCallResult(
            status="disabled",
            content=None,
            model=None,
            latency_ms=0.0,
            attempt_count=0,
            failure_reason="cache_only_miss",
        )
