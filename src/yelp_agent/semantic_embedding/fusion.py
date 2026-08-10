"""Deterministic policy that consumes semantic evidence without hiding ranking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def fuse_hybrid_and_semantic(
    hybrid_ranking: Sequence[str],
    semantic_ranks: Mapping[str, int],
    *,
    alpha: float,
) -> list[str]:
    """Rerank only a Hybrid prefix; preserve the unscored tail exactly."""

    ranking = list(hybrid_ranking)
    if not ranking or len(ranking) != len(set(ranking)):
        raise ValueError("hybrid ranking must be a nonempty unique sequence")
    if not 0 <= alpha <= 1:
        raise ValueError("fusion alpha must be between zero and one")
    if not semantic_ranks:
        return ranking
    size = len(semantic_ranks)
    prefix = ranking[:size]
    if set(prefix) != set(semantic_ranks):
        raise ValueError("semantic evidence must cover one complete Hybrid prefix")
    if sorted(semantic_ranks.values()) != list(range(1, size + 1)):
        raise ValueError("semantic ranks must be contiguous")
    if size == 1 or alpha == 0:
        return ranking
    hybrid_rank = {business_id: rank for rank, business_id in enumerate(prefix, 1)}

    def percentile(rank: int) -> float:
        return (size - rank) / (size - 1)

    fused = sorted(
        prefix,
        key=lambda business_id: (
            -(
                (1.0 - alpha) * percentile(hybrid_rank[business_id])
                + alpha * percentile(semantic_ranks[business_id])
            ),
            business_id,
        ),
    )
    return [*fused, *ranking[size:]]
