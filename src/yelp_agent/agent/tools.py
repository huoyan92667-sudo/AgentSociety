"""Task-bound, read-only tools exposed to the recommendation Agent."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Protocol

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field

from yelp_agent.features.quality import BusinessQuality
from yelp_agent.models import (
    Prediction,
    RecommendationTask,
    ScoreBreakdown,
    StrictModel,
    UserProfile,
)
from yelp_agent.rankers.hybrid_ranker import HybridTaskScore


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


@dataclass(frozen=True)
class _StaticHistoryBusiness:
    name: str
    address: str
    city: str
    state: str
    postal_code: str
    latitude: float | None
    longitude: float | None
    categories: tuple[str, ...]
    attributes: dict[str, Any]


@dataclass(frozen=True)
class _FrozenHistoryReview:
    position: int
    review_id: str
    user_id: str
    business_id: str
    stars: int
    text: str
    date: datetime


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
        businesses_path: str | Path,
        interactions_path: str | Path,
        histories_path: str | Path,
        *,
        hybrid_ranker: _HybridRanker,
        quality_store: _QualityStore,
    ) -> None:
        self._businesses = self._load_businesses(Path(businesses_path))
        self._histories = self._load_histories(
            Path(interactions_path),
            Path(histories_path),
        )
        self._hybrid_ranker = hybrid_ranker
        self._quality_store = quality_store

    @staticmethod
    def _load_businesses(
        path: Path,
    ) -> dict[str, _StaticHistoryBusiness]:
        if not path.is_file():
            raise FileNotFoundError(f"Business Parquet does not exist: {path}")
        try:
            rows = pq.read_table(
                path,
                columns=[
                    "business_id",
                    "name",
                    "address",
                    "city",
                    "state",
                    "postal_code",
                    "latitude",
                    "longitude",
                    "categories",
                    "attributes_json",
                ],
            ).to_pylist()
        except (OSError, pa.ArrowException) as exc:
            raise AgentToolError(
                f"Could not read Agent business data from {path}: {exc}"
            ) from exc
        businesses: dict[str, _StaticHistoryBusiness] = {}
        for row in rows:
            business_id = row.get("business_id")
            name = row.get("name")
            categories = row.get("categories")
            attributes_json = row.get("attributes_json")
            if (
                not isinstance(business_id, str)
                or not business_id
                or business_id in businesses
                or not isinstance(name, str)
                or not name
                or not isinstance(categories, list)
                or not isinstance(attributes_json, str)
            ):
                raise AgentToolError("Agent business data contains an invalid row")
            try:
                attributes = json.loads(attributes_json)
            except json.JSONDecodeError as exc:
                raise AgentToolError(
                    f"Business {business_id!r} has invalid attributes JSON"
                ) from exc
            if not isinstance(attributes, dict):
                raise AgentToolError(
                    f"Business {business_id!r} attributes must be an object"
                )
            businesses[business_id] = _StaticHistoryBusiness(
                name=name,
                address=str(row.get("address") or ""),
                city=str(row.get("city") or ""),
                state=str(row.get("state") or ""),
                postal_code=str(row.get("postal_code") or ""),
                latitude=(
                    float(row["latitude"])
                    if row.get("latitude") is not None
                    else None
                ),
                longitude=(
                    float(row["longitude"])
                    if row.get("longitude") is not None
                    else None
                ),
                categories=tuple(
                    str(category).strip()
                    for category in categories
                    if str(category).strip()
                ),
                attributes=attributes,
            )
        if not businesses:
            raise AgentToolError("Agent business data is empty")
        return businesses

    def _load_histories(
        self,
        interactions_path: Path,
        histories_path: Path,
    ) -> dict[str, tuple[_FrozenHistoryReview, ...]]:
        for label, path in (
            ("Interaction", interactions_path),
            ("Temporal history", histories_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} Parquet does not exist: {path}")
        try:
            with duckdb.connect() as connection:
                rows = connection.execute(
                    """
                    SELECT
                        history.task_id,
                        history.position,
                        interaction.review_id,
                        interaction.user_id,
                        interaction.business_id,
                        interaction.stars,
                        interaction.text,
                        interaction.date
                    FROM read_parquet(?) AS history
                    JOIN read_parquet(?) AS interaction USING (review_id)
                    ORDER BY history.task_id, history.position
                    """,
                    [str(histories_path), str(interactions_path)],
                ).fetchall()
                history_count = connection.execute(
                    "SELECT count(*) FROM read_parquet(?)",
                    [str(histories_path)],
                ).fetchone()[0]
        except duckdb.Error as exc:
            raise AgentToolError(
                f"Could not join frozen Agent histories: {exc}"
            ) from exc
        if len(rows) != history_count:
            raise AgentToolError(
                "A frozen Agent history review is missing from interactions"
            )

        grouped: defaultdict[str, list[_FrozenHistoryReview]] = defaultdict(list)
        for (
            task_id,
            position,
            review_id,
            user_id,
            business_id,
            stars,
            text,
            date,
        ) in rows:
            rating = float(stars)
            if (
                not task_id
                or not review_id
                or not user_id
                or not business_id
                or str(business_id) not in self._businesses
                or not rating.is_integer()
                or not 1 <= rating <= 5
                or not isinstance(text, str)
                or not isinstance(date, datetime)
            ):
                raise AgentToolError("Frozen Agent history contains an invalid row")
            grouped[str(task_id)].append(
                _FrozenHistoryReview(
                    position=int(position),
                    review_id=str(review_id),
                    user_id=str(user_id),
                    business_id=str(business_id),
                    stars=int(rating),
                    text=text,
                    date=date,
                )
            )
        return {
            task_id: tuple(history)
            for task_id, history in grouped.items()
        }

    def for_task(self, task: RecommendationTask) -> "TaskAgentTools":
        """Create an isolated tool session authorized for exactly one task."""

        return TaskAgentTools(
            task,
            businesses=self._businesses,
            histories=self._histories,
            hybrid_ranker=self._hybrid_ranker,
            quality_store=self._quality_store,
        )


class TaskAgentTools:
    """Four read-only Agent tools constrained to a single frozen task."""

    def __init__(
        self,
        task: RecommendationTask,
        *,
        businesses: dict[str, _StaticHistoryBusiness],
        histories: dict[str, tuple[_FrozenHistoryReview, ...]],
        hybrid_ranker: _HybridRanker,
        quality_store: _QualityStore,
    ) -> None:
        self._task = task
        self._businesses = businesses
        self._histories = histories
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
        limit: int = 30,
    ) -> UserHistoryResult:
        """Return recent frozen history and up to four positive/negative examples."""

        self._call_count += 1
        self._validate_identity(user_id, cutoff_time)
        if not 1 <= limit <= 30:
            raise AgentToolError("history limit must be between 1 and 30")
        history = self._histories.get(self._task.task_id)
        if not history:
            raise AgentToolError(
                f"No frozen history exists for task {self._task.task_id!r}"
            )
        reviews: list[HistoryReview] = []
        for item in reversed(history[-limit:]):
            if item.user_id != self._task.user_id:
                raise AgentToolError("Frozen history user does not match bound task")
            if item.date >= self._task.cutoff_time:
                raise AgentToolError("Frozen history is not strictly before cutoff")
            business = self._businesses[item.business_id]
            reviews.append(
                HistoryReview(
                    review_id=item.review_id,
                    business_id=item.business_id,
                    business_name=business.name,
                    categories=list(business.categories),
                    stars=item.stars,
                    text=item.text,
                    date=item.date,
                )
            )
        return UserHistoryResult(
            user_id=user_id,
            cutoff_time=cutoff_time,
            reviews=reviews,
            recent_positive=[item for item in reviews if item.stars >= 4][:4],
            recent_negative=[item for item in reviews if item.stars <= 2][:4],
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
            business = self._businesses.get(business_id)
            breakdown = scored.score_breakdowns.get(business_id)
            point_in_time_quality = quality.get(business_id)
            if (
                business is None
                or breakdown is None
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
                    attributes=business.attributes,
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
