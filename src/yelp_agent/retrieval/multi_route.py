"""Retrieve candidates through four independent signals and rank-level fusion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Literal

from pydantic import Field

from yelp_agent.collaborative.item_knn import (
    ItemKNNHistoryEvent,
    ItemKNNRequest,
    TemporalItemKNNStore,
)
from yelp_agent.config import RetrievalConfig
from yelp_agent.data.temporal_view import TemporalDataView
from yelp_agent.features.category import TemporalCategoryStore
from yelp_agent.features.location import TemporalLocationStore
from yelp_agent.features.quality import TemporalQualityStore
from yelp_agent.features.text import TemporalTextStore
from yelp_agent.models import StrictModel

RouteName = Literal[
    "quality",
    "category",
    "text",
    "location",
    "item_knn",
]
ROUTE_NAMES: tuple[RouteName, ...] = (
    "quality",
    "category",
    "text",
    "location",
    "item_knn",
)


class RetrievalError(RuntimeError):
    """Raised when a retrieval task cannot be evaluated safely."""


class RetrievalTaskContext(StrictModel):
    """A label-free recommendation moment used by candidate retrievers."""

    task_id: str = Field(min_length=1)
    split: Literal["train", "validation", "test", "development"]
    user_id: str = Field(min_length=1)
    cutoff_time: datetime
    history_count: int = Field(ge=1)


@dataclass(frozen=True, slots=True)
class _ScoringRequest:
    task_id: str
    user_id: str
    cutoff_time: datetime
    candidate_business_ids: list[str]


@dataclass(frozen=True, slots=True)
class RouteCandidate:
    """One business and its position within a single retrieval route."""

    business_id: str
    route: RouteName
    rank: int
    score: float


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    """One fused candidate with route evidence retained for later models."""

    business_id: str
    rank: int
    fusion_score: float
    route_count: int
    quality_rank: int | None
    quality_score: float | None
    category_rank: int | None
    category_score: float | None
    text_rank: int | None
    text_score: float | None
    location_rank: int | None
    location_score: float | None
    distance_km: float | None
    item_knn_rank: int | None
    item_knn_positive_score: float
    item_knn_negative_evidence: float
    item_knn_positive_support_count: int
    item_knn_negative_support_count: int
    item_knn_positive_neighbor_count: int
    item_knn_negative_neighbor_count: int
    item_knn_missing: bool


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """A complete label-free result returned by the retrieval interface."""

    task: RetrievalTaskContext
    catalog_size: int
    eligible_candidate_count: int
    excluded_history_businesses: int
    candidates: tuple[RetrievalCandidate, ...]
    route_candidates: tuple[RouteCandidate, ...]
    route_result_counts: dict[RouteName, int]
    latency_ms: float


def _rank_scores(
    scores: dict[str, float],
    *,
    limit: int,
    include_zero: bool,
) -> list[tuple[str, float]]:
    return sorted(
        (
            (business_id, score)
            for business_id, score in scores.items()
            if include_zero or score > 0.0
        ),
        key=lambda item: (-item[1], item[0]),
    )[:limit]


class MultiRouteRetriever:
    """Hide full-catalog filtering, four routes and RRF behind one call."""

    def __init__(
        self,
        data_view: TemporalDataView,
        *,
        category_store: TemporalCategoryStore,
        text_store: TemporalTextStore,
        quality_store: TemporalQualityStore,
        location_store: TemporalLocationStore,
        config: RetrievalConfig,
        item_knn_store: TemporalItemKNNStore | None = None,
    ) -> None:
        self._data_view = data_view
        self._category_store = category_store
        self._text_store = text_store
        self._quality_store = quality_store
        self._location_store = location_store
        self._item_knn_store = item_knn_store
        self._config = config
        self._all_business_ids = sorted(
            business.business_id for business in data_view.businesses()
        )

    def retrieve(
        self,
        task: RetrievalTaskContext,
        *,
        include_route_provenance: bool = False,
    ) -> RetrievalResult:
        """Return target-blind Top-K candidates for one frozen moment."""

        started = perf_counter()
        history = self._data_view.user_history(task.user_id, task.cutoff_time)
        if len(history) != task.history_count:
            raise RetrievalError(
                f"Task {task.task_id!r} expected {task.history_count} history "
                f"rows but the temporal view returned {len(history)}"
            )

        quality = self._quality_store.score_catalog(task.cutoff_time)
        if list(quality.business_ids) != self._all_business_ids:
            raise RetrievalError("Quality catalog order does not match businesses")
        quality_by_business = {
            business_id: float(score)
            for business_id, score in zip(
                quality.business_ids,
                quality.quality_scores,
                strict=True,
            )
        }
        review_count_by_business = dict(
            zip(
                quality.business_ids,
                (int(value) for value in quality.review_counts),
                strict=True,
            )
        )
        catalog_ids = [
            business_id
            for business_id in self._all_business_ids
            if review_count_by_business[business_id] > 0
        ]
        history_ids = {interaction.business_id for interaction in history}
        if self._config.exclude_history_businesses:
            eligible_ids = [
                business_id
                for business_id in catalog_ids
                if business_id not in history_ids
            ]
        else:
            eligible_ids = catalog_ids
        excluded_history = len(catalog_ids) - len(eligible_ids)
        if not eligible_ids:
            return RetrievalResult(
                task=task,
                catalog_size=len(catalog_ids),
                eligible_candidate_count=0,
                excluded_history_businesses=excluded_history,
                candidates=(),
                route_candidates=(),
                route_result_counts={route: 0 for route in ROUTE_NAMES},
                latency_ms=(perf_counter() - started) * 1000.0,
            )

        request = _ScoringRequest(
            task_id=task.task_id,
            user_id=task.user_id,
            cutoff_time=task.cutoff_time,
            candidate_business_ids=eligible_ids,
        )
        category_scores = self._category_store.score_candidates(request)
        text = self._text_store.score_candidates(request)
        text_scores = dict(
            zip(text.business_ids, map(float, text.text_scores), strict=True)
        )
        location = self._location_store.score_candidates(request)
        location_details = {
            business_id: (
                float(score),
                None if math.isnan(float(distance)) else float(distance),
            )
            for business_id, score, distance in zip(
                location.business_ids,
                location.location_scores,
                location.distances_km,
                strict=True,
            )
        }
        item_knn_result = (
            None
            if self._item_knn_store is None
            else self._item_knn_store.score_candidates(
                ItemKNNRequest(
                    user_id=task.user_id,
                    cutoff_time=task.cutoff_time,
                    candidate_business_ids=tuple(eligible_ids),
                    history=tuple(
                        ItemKNNHistoryEvent(
                            business_id=interaction.business_id,
                            stars=interaction.stars,
                            date=interaction.date,
                        )
                        for interaction in history
                    ),
                )
            )
        )
        item_knn_by_business = (
            {}
            if item_knn_result is None
            else {score.business_id: score for score in item_knn_result.scores}
        )

        route_lists: dict[RouteName, list[tuple[str, float]]] = {}
        route_lists["quality"] = _rank_scores(
            {
                business_id: quality_by_business[business_id]
                for business_id in eligible_ids
            },
            limit=self._config.per_route_limit,
            include_zero=True,
        )
        route_lists["category"] = _rank_scores(
            category_scores,
            limit=self._config.per_route_limit,
            include_zero=False,
        )
        route_lists["text"] = (
            _rank_scores(
                text_scores,
                limit=self._config.per_route_limit,
                include_zero=True,
            )
            if text.positive_review_count or text.negative_review_count
            else []
        )
        location_scores = {
            business_id: score
            for business_id, (score, distance) in location_details.items()
            if distance is not None
        }
        route_lists["location"] = _rank_scores(
            location_scores,
            limit=self._config.per_route_limit,
            include_zero=True,
        )
        route_lists["item_knn"] = (
            _rank_scores(
                {
                    business_id: score.positive_score
                    for business_id, score in item_knn_by_business.items()
                },
                limit=self._config.per_route_limit,
                include_zero=False,
            )
            if item_knn_result is not None
            else []
        )

        by_business: dict[
            str,
            dict[RouteName, tuple[int, float]],
        ] = {}
        fusion_scores: dict[str, float] = {}
        route_candidates: list[RouteCandidate] = []
        for route in ROUTE_NAMES:
            for rank, (business_id, score) in enumerate(
                route_lists[route],
                start=1,
            ):
                if include_route_provenance:
                    route_candidates.append(
                        RouteCandidate(
                            business_id=business_id,
                            route=route,
                            rank=rank,
                            score=float(score),
                        )
                    )
                by_business.setdefault(business_id, {})[route] = (
                    rank,
                    float(score),
                )
                fusion_scores[business_id] = fusion_scores.get(
                    business_id, 0.0
                ) + 1.0 / (self._config.rrf_constant + rank)

        fused_ids = sorted(
            fusion_scores,
            key=lambda business_id: (-fusion_scores[business_id], business_id),
        )[: self._config.candidate_limit]
        fused: list[RetrievalCandidate] = []
        for rank, business_id in enumerate(fused_ids, start=1):
            evidence = by_business[business_id]
            quality_item = evidence.get("quality")
            category_item = evidence.get("category")
            text_item = evidence.get("text")
            location_item = evidence.get("location")
            item_knn_item = evidence.get("item_knn")
            location_score, distance_km = location_details[business_id]
            item_knn_score = item_knn_by_business.get(business_id)
            fused.append(
                RetrievalCandidate(
                    business_id=business_id,
                    rank=rank,
                    fusion_score=fusion_scores[business_id],
                    route_count=len(evidence),
                    quality_rank=(None if quality_item is None else quality_item[0]),
                    quality_score=quality_by_business[business_id],
                    category_rank=(None if category_item is None else category_item[0]),
                    category_score=category_scores[business_id],
                    text_rank=None if text_item is None else text_item[0],
                    text_score=text_scores[business_id],
                    location_rank=(None if location_item is None else location_item[0]),
                    location_score=location_score,
                    distance_km=distance_km,
                    item_knn_rank=(None if item_knn_item is None else item_knn_item[0]),
                    item_knn_positive_score=(
                        0.0 if item_knn_score is None else item_knn_score.positive_score
                    ),
                    item_knn_negative_evidence=(
                        0.0
                        if item_knn_score is None
                        else item_knn_score.negative_evidence
                    ),
                    item_knn_positive_support_count=(
                        0
                        if item_knn_score is None
                        else item_knn_score.positive_support_count
                    ),
                    item_knn_negative_support_count=(
                        0
                        if item_knn_score is None
                        else item_knn_score.negative_support_count
                    ),
                    item_knn_positive_neighbor_count=(
                        0
                        if item_knn_score is None
                        else item_knn_score.positive_neighbor_count
                    ),
                    item_knn_negative_neighbor_count=(
                        0
                        if item_knn_score is None
                        else item_knn_score.negative_neighbor_count
                    ),
                    item_knn_missing=(
                        True if item_knn_result is None else item_knn_result.missing
                    ),
                )
            )

        if len(fused) != min(
            self._config.candidate_limit,
            len(eligible_ids),
        ):
            raise RetrievalError(
                f"Task {task.task_id!r} produced an incomplete fused candidate set"
            )
        if any(
            not math.isfinite(candidate.fusion_score) or candidate.fusion_score <= 0
            for candidate in fused
        ):
            raise RetrievalError(
                f"Task {task.task_id!r} produced an invalid fusion score"
            )
        return RetrievalResult(
            task=task,
            catalog_size=len(catalog_ids),
            eligible_candidate_count=len(eligible_ids),
            excluded_history_businesses=excluded_history,
            candidates=tuple(fused),
            route_candidates=tuple(route_candidates),
            route_result_counts={
                route: len(route_lists[route]) for route in ROUTE_NAMES
            },
            latency_ms=(perf_counter() - started) * 1000.0,
        )
