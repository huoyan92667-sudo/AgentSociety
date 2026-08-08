"""Small in-memory ranking interface for future Agent tool integration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import Field

from yelp_agent.learning_to_rank.artifacts import load_frozen_hybrid_v2
from yelp_agent.learning_to_rank.lambdamart import LambdaMARTModel
from yelp_agent.learning_to_rank.lambdamart_artifacts import (
    load_frozen_lambdamart,
)
from yelp_agent.learning_to_rank.model import PairwiseLogisticModel
from yelp_agent.models import StrictModel


class HybridV2ScoredCandidate(StrictModel):
    business_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    model_rank: int = Field(ge=1)
    hybrid_v1_rank: int = Field(ge=1)
    model_score: float
    hybrid_v1_score: float
    blend_score: float = Field(ge=0, le=1)


@dataclass(frozen=True, slots=True)
class FrozenHybridV2Ranker:
    """Hide model scaling and conservative rank-percentile fusion."""

    model: PairwiseLogisticModel | LambdaMARTModel
    blend_alpha: float

    def __post_init__(self) -> None:
        if not 0 <= self.blend_alpha <= 1:
            raise ValueError("blend_alpha must be between zero and one")

    @classmethod
    def from_artifacts(cls, artifact_root: str | Path) -> FrozenHybridV2Ranker:
        model, manifest = load_frozen_hybrid_v2(artifact_root)
        return cls(model=model, blend_alpha=manifest.selected_blend_alpha)

    def rank(
        self,
        candidates: Sequence[Mapping[str, object]],
    ) -> tuple[HybridV2ScoredCandidate, ...]:
        """Return every unique candidate exactly once in deterministic order."""

        if not candidates:
            raise ValueError("candidates cannot be empty")
        business_ids = [
            str(candidate.get("business_id") or "") for candidate in candidates
        ]
        if any(not business_id for business_id in business_ids):
            raise ValueError("every candidate requires a business_id")
        if len(set(business_ids)) != len(business_ids):
            raise ValueError("candidate business IDs must be unique")
        try:
            matrix = np.asarray(
                [
                    [float(candidate[name]) for name in self.model.feature_names]
                    for candidate in candidates
                ],
                dtype=np.float64,
            )
            v1_scores = np.asarray(
                [float(candidate["hybrid_v1_score"]) for candidate in candidates],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "candidate feature values are incomplete or invalid"
            ) from exc
        if not np.all(np.isfinite(v1_scores)):
            raise ValueError("Hybrid V1 scores must be finite")
        model_scores = self.model.score(matrix, feature_names=self.model.feature_names)

        def ranks(values: np.ndarray) -> np.ndarray:
            order = sorted(
                range(len(values)),
                key=lambda index: (-float(values[index]), business_ids[index]),
            )
            output = np.empty(len(values), dtype=np.int64)
            for rank, index in enumerate(order, start=1):
                output[index] = rank
            return output

        model_ranks = ranks(model_scores)
        v1_ranks = ranks(v1_scores)
        count = len(candidates)
        if count == 1:
            model_percentiles = np.ones(1, dtype=np.float64)
            v1_percentiles = np.ones(1, dtype=np.float64)
        else:
            model_percentiles = (count - model_ranks) / (count - 1)
            v1_percentiles = (count - v1_ranks) / (count - 1)
        blend_scores = (
            self.blend_alpha * model_percentiles
            + (1.0 - self.blend_alpha) * v1_percentiles
        )
        final_order = sorted(
            range(count),
            key=lambda index: (-float(blend_scores[index]), business_ids[index]),
        )
        return tuple(
            HybridV2ScoredCandidate(
                business_id=business_ids[index],
                rank=rank,
                model_rank=int(model_ranks[index]),
                hybrid_v1_rank=int(v1_ranks[index]),
                model_score=float(model_scores[index]),
                hybrid_v1_score=float(v1_scores[index]),
                blend_score=min(1.0, max(0.0, float(blend_scores[index]))),
            )
            for rank, index in enumerate(final_order, start=1)
        )


@dataclass(frozen=True, slots=True)
class FrozenLambdaMARTRanker:
    """Expose the same complete-ranking interface for the nonlinear model."""

    model: LambdaMARTModel
    blend_alpha: float

    def __post_init__(self) -> None:
        if not 0 <= self.blend_alpha <= 1:
            raise ValueError("blend_alpha must be between zero and one")

    @classmethod
    def from_artifacts(cls, artifact_root: str | Path) -> FrozenLambdaMARTRanker:
        model, manifest = load_frozen_lambdamart(artifact_root)
        return cls(model=model, blend_alpha=manifest.selected_blend_alpha)

    def rank(
        self,
        candidates: Sequence[Mapping[str, object]],
    ) -> tuple[HybridV2ScoredCandidate, ...]:
        """Return every unique candidate exactly once in deterministic order."""

        return FrozenHybridV2Ranker(
            model=self.model,
            blend_alpha=self.blend_alpha,
        ).rank(candidates)
