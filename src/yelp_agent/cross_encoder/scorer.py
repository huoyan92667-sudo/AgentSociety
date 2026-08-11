"""Small interface seam between ranking policy and local model inference."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class CrossEncoderProviderError(RuntimeError):
    def __init__(self, reason: str, *, retryable: bool) -> None:
        super().__init__(reason)
        self.reason = reason
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class ScoredPairBatch:
    scores: tuple[float, ...]
    input_tokens: int
    per_pair_input_tokens: tuple[int, ...]
    truncated_pair_count: int
    latency_ms: float


class PairScorer(Protocol):
    provider: str
    model: str
    batch_size: int

    def score(self, query_text: str, documents: Sequence[str]) -> ScoredPairBatch: ...
