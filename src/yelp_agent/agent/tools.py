"""Task-bound, read-only tools exposed to the recommendation Agent."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from pydantic import Field

from yelp_agent.data.temporal_view import TemporalDataError, TemporalDataView
from yelp_agent.features.quality import BusinessQuality
from yelp_agent.models import (
    Prediction,
    RecommendationTask,
    ScoreBreakdown,
    StrictModel,
    UserProfile,
)
from yelp_agent.rankers.hybrid_ranker import HybridTaskScore


MAX_HISTORY_REVIEWS = 30
REPRESENTATIVE_REVIEWS_PER_SENTIMENT = 4


class AgentToolError(RuntimeError):
    """Raised when an Agent tool request is invalid or crosses task scope."""


class HistoryReview(StrictModel):
    review_id: str = Field(min_length=1)
    business_id: str = Field(min_length=1)
    business_name: str = Field(min_length=1)
    categories: list[str]
    stars: int = Field(ge=1, le=5)
    text: str
    date: datetime


class UserHistoryResult(StrictModel):
    user_id: str = Field(min_length=1)
    cutoff_time: datetime
    reviews: list[HistoryReview]
    recent_positive: list[HistoryReview]
    recent_negative: list[HistoryReview]


class BusinessDetails(StrictModel):
    business_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    address: str
    city: str = Field(min_length=1)
    state: str = Field(min_length=1)
    postal_code: str
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    categories: list[str]
    attributes: dict[str, Any]
    quality: BusinessQuality
    score_breakdown: ScoreBreakdown


class BusinessDetailsResult(StrictModel):
    task_id: str = Field(min_length=1)
    cutoff_time: datetime
    businesses: list[BusinessDetails]


class HybridRankingResult(StrictModel):
    task_id: str = Field(min_length=1)
    ranking: list[str]
    score_breakdowns: dict[str, ScoreBreakdown]


class _HybridRanker(Protocol):
    def score(self, task: RecommendationTask) -> HybridTaskScore: ...

    def rank(self, task: RecommendationTask) -> Prediction: ...


class _QualityStore(Protocol):
    def score_businesses(
        self,
        business_ids: list[str],
        cutoff_time: datetime,
    ) -> dict[str, BusinessQuality]: ...


class AgentToolbox:
    """Load shared read-only data once, then bind tools to one frozen task."""

    def __init__(
        self,
        data_view: TemporalDataView,
        *,
        hybrid_ranker: _HybridRanker,
        quality_store: _QualityStore,
    ) -> None:
        self._data_view = data_view
        self._hybrid_ranker = hybrid_ranker
        self._quality_store = quality_store

    def for_task(self, task: RecommendationTask) -> "TaskAgentTools":
        """Create an isolated tool session authorized for exactly one task."""

        return TaskAgentTools(
            task,
            data_view=self._data_view,
            hybrid_ranker=self._hybrid_ranker,
            quality_store=self._quality_store,
        )


class TaskAgentTools:
    """Four read-only Agent tools constrained to a single frozen task."""

    def __init__(
        self,
        task: RecommendationTask,
        *,
        data_view: TemporalDataView,
        hybrid_ranker: _HybridRanker,
        quality_store: _QualityStore,
    ) -> None:
        self._task = task
        self._data_view = data_view
        self._hybrid_ranker = hybrid_ranker
        self._quality_store = quality_store
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    def _validate_identity(self, user_id: str, cutoff_time: datetime) -> None:
        if user_id != self._task.user_id:
            raise AgentToolError("Agent tool user_id is outside the bound task")
        if cutoff_time != self._task.cutoff_time:
            raise AgentToolError("Agent tool cutoff_time is outside the bound task")

    def _validate_cutoff(self, cutoff_time: datetime) -> None:
        if cutoff_time != self._task.cutoff_time:
            raise AgentToolError("Agent tool cutoff_time is outside the bound task")

    def get_user_history(
        self,
        user_id: str,
        cutoff_time: datetime,
        limit: int = MAX_HISTORY_REVIEWS,
    ) -> UserHistoryResult:
        """Return recent frozen history and up to four positive/negative examples."""

        self._call_count += 1
        self._validate_identity(user_id, cutoff_time)
        if not 1 <= limit <= MAX_HISTORY_REVIEWS:
            raise AgentToolError(
                f"history limit must be between 1 and {MAX_HISTORY_REVIEWS}"
            )
        history = self._data_view.user_history(user_id, cutoff_time)
        if not history:
            raise AgentToolError(
                f"No frozen history exists for task {self._task.task_id!r}"
            )
        reviews: list[HistoryReview] = []
        for item in reversed(history[-limit:]):
            try:
                business = self._data_view.business(item.business_id)
            except TemporalDataError as exc:
                raise AgentToolError(str(exc)) from exc
            reviews.append(
                HistoryReview(
                    review_id=item.review_id,
                    business_id=item.business_id,
                    business_name=business.name,
                    categories=list(business.categories),
                    stars=int(item.stars),
                    text=item.text,
                    date=item.date,
                )
            )
        return UserHistoryResult(
            user_id=user_id,
            cutoff_time=cutoff_time,
            reviews=reviews,
            recent_positive=[item for item in reviews if item.stars >= 4][
                :REPRESENTATIVE_REVIEWS_PER_SENTIMENT
            ],
            recent_negative=[item for item in reviews if item.stars <= 2][
                :REPRESENTATIVE_REVIEWS_PER_SENTIMENT
            ],
        )

    def get_business_details(
        self,
        business_ids: list[str],
        cutoff_time: datetime,
    ) -> BusinessDetailsResult:
        """Return static and point-in-time details for authorized candidates."""

        self._call_count += 1
        self._validate_cutoff(cutoff_time)
        if not business_ids:
            raise AgentToolError("business_ids cannot be empty")
        if len(set(business_ids)) != len(business_ids):
            raise AgentToolError("business_ids must be unique")
        outside = sorted(
            set(business_ids).difference(self._task.candidate_business_ids)
        )
        if outside:
            raise AgentToolError(
                "Business request is outside the bound candidates: "
                f"{outside[:3]}"
            )

        scored = self._hybrid_ranker.score(self._task)
        quality = self._quality_store.score_businesses(
            business_ids,
            cutoff_time,
        )
        details: list[BusinessDetails] = []
        for business_id in business_ids:
            try:
                business = self._data_view.business(business_id)
            except TemporalDataError as exc:
                raise AgentToolError(str(exc)) from exc
            breakdown = scored.score_breakdowns.get(business_id)
            point_in_time_quality = quality.get(business_id)
            if (
                breakdown is None
                or point_in_time_quality is None
            ):
                raise AgentToolError(
                    f"Candidate details are incomplete for {business_id!r}"
                )
            details.append(
                BusinessDetails(
                    business_id=business_id,
                    name=business.name,
                    address=business.address,
                    city=business.city,
                    state=business.state,
                    postal_code=business.postal_code,
                    latitude=business.latitude,
                    longitude=business.longitude,
                    categories=list(business.categories),
                    attributes=business.attributes_dict(),
                    quality=point_in_time_quality,
                    score_breakdown=breakdown,
                )
            )
        return BusinessDetailsResult(
            task_id=self._task.task_id,
            cutoff_time=cutoff_time,
            businesses=details,
        )

    def get_user_profile(
        self,
        user_id: str,
        cutoff_time: datetime,
    ) -> UserProfile:
        """Return the merged point-in-time profile for the bound task."""

        self._call_count += 1
        self._validate_identity(user_id, cutoff_time)
        profile = self._hybrid_ranker.score(self._task).profile
        if profile.user_id != self._task.user_id:
            raise AgentToolError("Hybrid profile user does not match bound task")
        return profile

    def get_hybrid_ranking(
        self,
        user_id: str,
        candidate_ids: list[str],
        cutoff_time: datetime,
    ) -> HybridRankingResult:
        """Return the complete frozen Hybrid order and explainable scores."""

        self._call_count += 1
        self._validate_identity(user_id, cutoff_time)
        if (
            len(candidate_ids) != len(self._task.candidate_business_ids)
            or len(set(candidate_ids)) != len(candidate_ids)
            or set(candidate_ids) != set(self._task.candidate_business_ids)
        ):
            raise AgentToolError(
                "Hybrid ranking requires the complete bound candidate set"
            )
        prediction = self._hybrid_ranker.rank(self._task)
        scored = self._hybrid_ranker.score(self._task)
        expected = set(self._task.candidate_business_ids)
        if (
            prediction.task_id != self._task.task_id
            or set(prediction.ranking) != expected
            or set(scored.score_breakdowns) != expected
        ):
            raise AgentToolError(
                "Hybrid ranker output does not match the bound task"
            )
        return HybridRankingResult(
            task_id=self._task.task_id,
            ranking=prediction.ranking,
            score_breakdowns=scored.score_breakdowns,
        )
