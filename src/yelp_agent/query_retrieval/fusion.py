"""Rank-level fusion between frozen history and current-Query channels."""

from __future__ import annotations

from yelp_agent.retrieval import RetrievalResult

from .config import QueryRetrievalConfig
from .schema import (
    DualChannelCandidate,
    DualChannelRetrievalResult,
    QueryRetrievalResult,
)


class DualChannelFusion:
    """Combine two independently testable retrieval channels without raw scores."""

    def __init__(self, config: QueryRetrievalConfig) -> None:
        self._config = config

    def fuse(
        self,
        history: RetrievalResult,
        query: QueryRetrievalResult,
    ) -> DualChannelRetrievalResult:
        history_by_id = {item.business_id: item for item in history.candidates}
        query_by_id = {item.business_id: item for item in query.candidates}
        all_ids = set(history_by_id).union(query_by_id)
        scores: dict[str, float] = {}
        for business_id in all_ids:
            score = 0.0
            history_item = history_by_id.get(business_id)
            query_item = query_by_id.get(business_id)
            if history_item is not None and self._config.dual_history_weight > 0:
                score += self._config.dual_history_weight / (
                    self._config.rrf_constant + history_item.rank
                )
            if query_item is not None and self._config.dual_query_weight > 0:
                score += self._config.dual_query_weight / (
                    self._config.rrf_constant + query_item.rank
                )
            if score > 0:
                scores[business_id] = score
        ordered = sorted(
            scores,
            key=lambda business_id: (-scores[business_id], business_id),
        )[: self._config.candidate_limit]
        candidates = []
        for rank, business_id in enumerate(ordered, start=1):
            history_item = history_by_id.get(business_id)
            query_item = query_by_id.get(business_id)
            candidates.append(
                DualChannelCandidate(
                    business_id=business_id,
                    rank=rank,
                    fusion_score=scores[business_id],
                    channel_count=int(history_item is not None)
                    + int(query_item is not None),
                    history_rank=(None if history_item is None else history_item.rank),
                    history_score=(
                        None if history_item is None else history_item.fusion_score
                    ),
                    query_rank=None if query_item is None else query_item.rank,
                    query_score=(
                        None if query_item is None else query_item.fusion_score
                    ),
                )
            )
        return DualChannelRetrievalResult(
            candidates=candidates,
            history_candidate_count=len(history_by_id),
            query_candidate_count=len(query_by_id),
            overlap_count=len(set(history_by_id).intersection(query_by_id)),
            latency_ms=history.latency_ms + query.latency_ms,
        )
