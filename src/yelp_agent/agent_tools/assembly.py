"""Assemble exact-cutoff knowledge and frozen Hybrid V2 ranking online."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Protocol

from yelp_agent.business_profiles.schema import BusinessProfileV1
from yelp_agent.config import BusinessProfileConfig
from yelp_agent.learning_to_rank.features import (
    HybridV1Weights,
    build_online_hybrid_v2_features,
)
from yelp_agent.learning_to_rank.runtime import HybridV2ScoredCandidate
from yelp_agent.profiles.schema import UserProfileV1


class ExactUserProfileReader(Protocol):
    def get(self, user_id: str, cutoff_time: datetime) -> UserProfileV1: ...


class ExactBusinessProfileReader(Protocol):
    def get(
        self,
        business_ids: list[str],
        cutoff_time: datetime,
    ) -> dict[str, BusinessProfileV1]: ...


class FrozenRanker(Protocol):
    def rank(
        self,
        candidates: Sequence[Mapping[str, object]],
    ) -> Sequence[HybridV2ScoredCandidate]: ...


class OnlineHybridV2RankingService:
    """Hide exact profile joins, feature assembly, and frozen ranking."""

    def __init__(
        self,
        *,
        user_profiles: ExactUserProfileReader,
        business_profiles: ExactBusinessProfileReader,
        ranker: FrozenRanker,
        weights: HybridV1Weights,
        broad_categories: set[str],
        business_profile_config: BusinessProfileConfig,
    ) -> None:
        self._user_profiles = user_profiles
        self._business_profiles = business_profiles
        self._ranker = ranker
        self._weights = weights
        self._broad_categories = set(broad_categories)
        self._business_profile_config = business_profile_config

    def rank(
        self,
        *,
        request_id: str,
        user_id: str,
        cutoff_time: datetime,
        candidates: Sequence[Mapping[str, object]],
    ) -> list[dict[str, object]]:
        rows = [dict(candidate) for candidate in candidates]
        business_ids = [str(row.get("business_id") or "") for row in rows]
        profile = self._user_profiles.get(user_id, cutoff_time)
        business_profiles = self._business_profiles.get(business_ids, cutoff_time)
        features = build_online_hybrid_v2_features(
            request_id=request_id,
            profile=profile,
            business_profiles=business_profiles,
            candidates=rows,
            weights=self._weights,
            broad_categories=self._broad_categories,
            business_profile_config=self._business_profile_config,
        )
        return [item.model_dump() for item in self._ranker.rank(features)]
