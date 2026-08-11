"""Transparent sequential rank fusion for Step 26."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def fuse_ranking_and_cross_encoder(
    ranking: Sequence[str],
    cross_encoder_ranks: Mapping[str, int],
    *,
    beta: float,
) -> list[str]:
    """Rerank one complete prefix and preserve its unscored tail exactly."""

    values = list(ranking)
    if not values or len(values) != len(set(values)):
        raise ValueError("input ranking must be a nonempty unique sequence")
    if not 0 <= beta <= 1:
        raise ValueError("fusion beta must be between zero and one")
    if not cross_encoder_ranks:
        return values
    size = len(cross_encoder_ranks)
    prefix = values[:size]
    if set(prefix) != set(cross_encoder_ranks):
        raise ValueError("Cross-Encoder evidence must cover one complete ranking prefix")
    if sorted(cross_encoder_ranks.values()) != list(range(1, size + 1)):
        raise ValueError("Cross-Encoder ranks must be contiguous")
    if size == 1 or beta == 0:
        return values
    original_rank = {business_id: rank for rank, business_id in enumerate(prefix, 1)}

    def percentile(rank: int) -> float:
        return (size - rank) / (size - 1)

    fused = sorted(
        prefix,
        key=lambda business_id: (
            -(
                (1.0 - beta) * percentile(original_rank[business_id])
                + beta * percentile(cross_encoder_ranks[business_id])
            ),
            business_id,
        ),
    )
    return [*fused, *values[size:]]
