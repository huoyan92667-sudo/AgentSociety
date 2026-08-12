"""Pure coarse scoring functions used by runtime and development selection."""

from __future__ import annotations

from collections.abc import Sequence

from .config import QueryAwareRankingPolicy
from .schema import CandidateEvidence, CoarseCandidateScore, PreparedQueryAwareCase


def rank_percentile(rank: int, size: int) -> float:
    if size < 1 or not 1 <= rank <= size:
        raise ValueError("rank must fall inside the ranked population")
    return 1.0 if size == 1 else (size - rank) / (size - 1)


def score_coarse_candidates(
    candidates: Sequence[CandidateEvidence],
    *,
    query_weight: float,
    query_retrieval_signal_weight: float,
    embedding_signal_weight: float,
) -> list[CoarseCandidateScore]:
    if not candidates or len({item.business_id for item in candidates}) != len(
        candidates
    ):
        raise ValueError("candidate evidence must be nonempty and unique")
    if not 0 <= query_weight <= 1:
        raise ValueError("query weight must be between zero and one")
    if abs(query_retrieval_signal_weight + embedding_signal_weight - 1.0) > 1e-9:
        raise ValueError("query signal weights must sum to one")
    intermediate: dict[str, tuple[float, float]] = {}
    for item in candidates:
        query_signal = (
            query_retrieval_signal_weight * item.query_retrieval_percentile
            + embedding_signal_weight * item.embedding_percentile
        )
        coarse = (
            (1.0 - query_weight) * item.lightgbm_percentile
            + query_weight * query_signal
        )
        intermediate[item.business_id] = (query_signal, coarse)
    ordered = sorted(
        candidates,
        key=lambda item: (-intermediate[item.business_id][1], item.business_id),
    )
    rank_by_id = {
        item.business_id: rank for rank, item in enumerate(ordered, start=1)
    }
    return [
        CoarseCandidateScore(
            business_id=item.business_id,
            source_channels=item.source_channels,
            history_rank=item.history_rank,
            query_rank=item.query_rank,
            lightgbm_rank=item.lightgbm_rank,
            embedding_rank=item.embedding_rank,
            query_signal=intermediate[item.business_id][0],
            coarse_score=intermediate[item.business_id][1],
            coarse_rank=rank_by_id[item.business_id],
        )
        for item in sorted(candidates, key=lambda value: rank_by_id[value.business_id])
    ]


def apply_coarse_policy(
    prepared: PreparedQueryAwareCase,
    policy: QueryAwareRankingPolicy,
) -> list[CoarseCandidateScore]:
    if prepared.eligible_candidate_count > policy.union_candidate_limit:
        raise ValueError("prepared case exceeds the frozen union limit")
    return score_coarse_candidates(
        prepared.candidates,
        query_weight=policy.selected_query_weight,
        query_retrieval_signal_weight=policy.query_retrieval_signal_weight,
        embedding_signal_weight=policy.embedding_signal_weight,
    )
