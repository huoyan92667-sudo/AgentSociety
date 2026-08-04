"""Single-positive recommendation ranking metrics."""

from __future__ import annotations

import math

from pydantic import Field

from yelp_agent.models import StrictModel


class SingleTaskMetrics(StrictModel):
    target_rank: int | None = Field(default=None, ge=1)
    hit_at_1: float = Field(ge=0, le=1)
    hit_at_3: float = Field(ge=0, le=1)
    hit_at_5: float = Field(ge=0, le=1)
    reciprocal_rank: float = Field(ge=0, le=1)
    ndcg_at_5: float = Field(ge=0, le=1)


def compute_single_positive_metrics(
    target_business_id: str,
    ranking: list[str],
) -> SingleTaskMetrics:
    """Score one ranking containing at most one relevant business."""

    try:
        rank = ranking.index(target_business_id) + 1
    except ValueError:
        rank = None
    return SingleTaskMetrics(
        target_rank=rank,
        hit_at_1=float(rank is not None and rank <= 1),
        hit_at_3=float(rank is not None and rank <= 3),
        hit_at_5=float(rank is not None and rank <= 5),
        reciprocal_rank=0.0 if rank is None else 1.0 / rank,
        ndcg_at_5=(
            0.0
            if rank is None or rank > 5
            else 1.0 / math.log2(rank + 1)
        ),
    )
