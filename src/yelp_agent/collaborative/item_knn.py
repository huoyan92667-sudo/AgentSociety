"""Point-in-time item-item collaborative evidence from Yelp ratings."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field, field_validator

from yelp_agent.config import ItemKNNConfig
from yelp_agent.models import StrictModel

Feedback = Literal["positive", "negative", "neutral"]
GraphFeedback = Literal["positive", "negative"]


class ItemKNNError(RuntimeError):
    """Raised when collaborative inputs cannot be used safely."""


class ItemKNNHistoryEvent(StrictModel):
    """One interaction already visible to the current recommendation task."""

    business_id: str = Field(min_length=1)
    stars: float = Field(ge=1.0, le=5.0)
    date: datetime


class ItemKNNRequest(StrictModel):
    """The small public request understood by the collaborative module."""

    user_id: str = Field(min_length=1)
    cutoff_time: datetime
    candidate_business_ids: tuple[str, ...]
    history: tuple[ItemKNNHistoryEvent, ...]

    @field_validator("candidate_business_ids")
    @classmethod
    def validate_candidates(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("candidate_business_ids cannot be empty")
        if any(not business_id for business_id in value):
            raise ValueError("candidate business IDs cannot be empty")
        if len(value) != len(set(value)):
            raise ValueError("candidate business IDs must be unique")
        return value


@dataclass(frozen=True, slots=True)
class ItemKNNCandidateScore:
    business_id: str
    positive_score: float
    negative_evidence: float
    positive_support_count: int
    negative_support_count: int
    positive_neighbor_count: int
    negative_neighbor_count: int


@dataclass(frozen=True, slots=True)
class ItemKNNScoreResult:
    scores: tuple[ItemKNNCandidateScore, ...]
    positive_history_count: int
    negative_history_count: int
    missing: bool


@dataclass(frozen=True, slots=True)
class _Interaction:
    user_id: str
    review_id: str
    business_id: str
    stars: float
    date: datetime


def _feedback(stars: float) -> Feedback:
    if stars >= 4.0:
        return "positive"
    if stars <= 2.0:
        return "negative"
    return "neutral"


class TemporalItemKNNStore:
    """Hide temporal deduplication, graph construction and scoring."""

    def __init__(
        self,
        interactions_path: str | Path,
        config: ItemKNNConfig,
        *,
        excluded_user_ids: set[str] | frozenset[str] = frozenset(),
    ) -> None:
        self._config = config
        self._interactions = self._load_interactions(
            (Path(interactions_path),),
            excluded_user_ids=frozenset(excluded_user_ids),
            apply_reservation=True,
        )
        self._prepare_graph()

    @classmethod
    def from_event_artifacts(
        cls,
        positive_events_path: str | Path,
        negative_events_path: str | Path,
        neutral_events_path: str | Path,
        config: ItemKNNConfig,
    ) -> TemporalItemKNNStore:
        """Load graph-safe events that were already split and reserved."""

        instance = cls.__new__(cls)
        instance._config = config
        instance._interactions = instance._load_interactions(
            (
                Path(positive_events_path),
                Path(negative_events_path),
                Path(neutral_events_path),
            ),
            excluded_user_ids=frozenset(),
            apply_reservation=False,
        )
        instance._prepare_graph()
        return instance

    def _load_interactions(
        self,
        paths: tuple[Path, ...],
        *,
        excluded_user_ids: frozenset[str],
        apply_reservation: bool,
    ) -> tuple[_Interaction, ...]:
        rows: list[dict[str, object]] = []
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Interaction Parquet does not exist: {path}")
            try:
                rows.extend(
                    pq.read_table(
                        path,
                        columns=[
                            "user_id",
                            "review_id",
                            "business_id",
                            "stars",
                            "date",
                        ],
                    ).to_pylist()
                )
            except (OSError, pa.ArrowException) as exc:
                raise ItemKNNError(
                    f"Could not read Item-KNN interactions: {path}"
                ) from exc

        by_user: dict[str, list[_Interaction]] = defaultdict(list)
        seen_review_ids: set[str] = set()
        for row in rows:
            user_id = str(row.get("user_id") or "")
            review_id = str(row.get("review_id") or "")
            business_id = str(row.get("business_id") or "")
            stars = row.get("stars")
            date = row.get("date")
            if (
                not user_id
                or not review_id
                or review_id in seen_review_ids
                or not business_id
                or not isinstance(stars, (int, float))
                or not 1.0 <= float(stars) <= 5.0
                or not isinstance(date, datetime)
            ):
                raise ItemKNNError("Item-KNN interactions contain an invalid row")
            seen_review_ids.add(review_id)
            if user_id in excluded_user_ids:
                continue
            by_user[user_id].append(
                _Interaction(
                    user_id=user_id,
                    review_id=review_id,
                    business_id=business_id,
                    stars=float(stars),
                    date=date,
                )
            )

        retained: list[_Interaction] = []
        reserve = self._config.reserved_tail_interactions if apply_reservation else 0
        for user_id in sorted(by_user):
            history = sorted(
                by_user[user_id],
                key=lambda row: (row.date, row.review_id),
            )
            if reserve:
                history = history[:-reserve]
            retained.extend(history)
        retained.sort(key=lambda row: (row.date, row.review_id))
        return tuple(retained)

    def _prepare_graph(self) -> None:
        self._reference_time = (
            None if not self._interactions else self._interactions[-1].date
        )
        self._business_ids = tuple(
            sorted({item.business_id for item in self._interactions})
        )
        self._business_index = {
            business_id: index for index, business_id in enumerate(self._business_ids)
        }
        self._reset_graph()

    def _reset_graph(self) -> None:
        self._next_interaction = 0
        self._current_cutoff: datetime | None = None
        self._active: dict[tuple[str, str], _Interaction] = {}
        self._user_items: dict[tuple[str, GraphFeedback], dict[str, _Interaction]] = (
            defaultdict(dict)
        )
        business_count = len(self._business_ids)
        self._item_popularity: dict[GraphFeedback, np.ndarray] = {
            "positive": np.zeros(business_count, dtype=np.float64),
            "negative": np.zeros(business_count, dtype=np.float64),
        }
        self._cooccurrence: dict[GraphFeedback, np.ndarray] = {
            feedback: np.zeros(
                (business_count, business_count),
                dtype=np.float64,
            )
            for feedback in ("positive", "negative")
        }
        self._support: dict[GraphFeedback, np.ndarray] = {
            feedback: np.zeros(
                (business_count, business_count),
                dtype=np.int32,
            )
            for feedback in ("positive", "negative")
        }

    def _weight(self, interaction_time: datetime) -> float:
        if self._config.half_life_days is None or self._reference_time is None:
            return 1.0
        offset_days = (
            interaction_time - self._reference_time
        ).total_seconds() / 86_400.0
        return 2.0 ** (offset_days / self._config.half_life_days)

    def _change_pairs(
        self,
        feedback: GraphFeedback,
        interaction: _Interaction,
        others: tuple[_Interaction, ...],
        *,
        direction: Literal[-1, 1],
    ) -> None:
        if not others:
            return
        item_index = self._business_index[interaction.business_id]
        other_indices = np.fromiter(
            (self._business_index[other.business_id] for other in others),
            dtype=np.intp,
            count=len(others),
        )
        contributions = np.fromiter(
            (
                math.sqrt(self._weight(interaction.date) * self._weight(other.date))
                for other in others
            ),
            dtype=np.float64,
            count=len(others),
        )
        cooccurrence = self._cooccurrence[feedback]
        support = self._support[feedback]
        cooccurrence[item_index, other_indices] += direction * contributions
        cooccurrence[other_indices, item_index] += direction * contributions
        support[item_index, other_indices] += direction
        support[other_indices, item_index] += direction
        updated_support = support[item_index, other_indices]
        updated_cooccurrence = cooccurrence[item_index, other_indices]
        if np.any(updated_support < 0):
            raise ItemKNNError("Item-KNN graph update became inconsistent")
        empty = updated_support == 0
        if np.any(empty):
            empty_indices = other_indices[empty]
            cooccurrence[item_index, empty_indices] = 0.0
            cooccurrence[empty_indices, item_index] = 0.0
        nonempty = ~empty
        if np.any(updated_cooccurrence[nonempty] <= 0.0):
            raise ItemKNNError("Item-KNN graph update became inconsistent")

    def _remove_active(self, interaction: _Interaction) -> None:
        feedback = _feedback(interaction.stars)
        if feedback == "neutral":
            return
        graph_feedback: GraphFeedback = feedback
        user_items = self._user_items[(interaction.user_id, graph_feedback)]
        others = tuple(
            other
            for other in user_items.values()
            if other.business_id != interaction.business_id
        )
        self._change_pairs(
            graph_feedback,
            interaction,
            others,
            direction=-1,
        )
        user_items.pop(interaction.business_id, None)
        item_index = self._business_index[interaction.business_id]
        popularity = self._item_popularity[graph_feedback][item_index] - self._weight(
            interaction.date
        )
        if popularity < -1e-9:
            raise ItemKNNError("Item-KNN popularity became inconsistent")
        self._item_popularity[graph_feedback][item_index] = max(
            0.0,
            popularity,
        )

    def _add_active(self, interaction: _Interaction) -> None:
        feedback = _feedback(interaction.stars)
        if feedback == "neutral":
            return
        graph_feedback: GraphFeedback = feedback
        user_items = self._user_items[(interaction.user_id, graph_feedback)]
        self._change_pairs(
            graph_feedback,
            interaction,
            tuple(user_items.values()),
            direction=1,
        )
        user_items[interaction.business_id] = interaction
        item_index = self._business_index[interaction.business_id]
        self._item_popularity[graph_feedback][item_index] += self._weight(
            interaction.date
        )

    def _apply_interaction(self, interaction: _Interaction) -> None:
        key = (interaction.user_id, interaction.business_id)
        previous = self._active.get(key)
        if previous is not None:
            self._remove_active(previous)
        self._active[key] = interaction
        self._add_active(interaction)

    def _advance_to(self, cutoff_time: datetime) -> None:
        if self._current_cutoff is not None and cutoff_time < self._current_cutoff:
            self._reset_graph()
        while self._next_interaction < len(self._interactions):
            interaction = self._interactions[self._next_interaction]
            if interaction.date >= cutoff_time:
                break
            self._apply_interaction(interaction)
            self._next_interaction += 1
        self._current_cutoff = cutoff_time

    def score_candidates(self, request: ItemKNNRequest) -> ItemKNNScoreResult:
        """Return independent positive and negative evidence at one cutoff."""

        self._advance_to(request.cutoff_time)
        latest_history: dict[str, ItemKNNHistoryEvent] = {}
        for interaction in sorted(
            request.history,
            key=lambda item: (item.date, item.business_id),
        ):
            if interaction.date >= request.cutoff_time:
                raise ItemKNNError(
                    "Item-KNN request history must be strictly before cutoff"
                )
            latest_history[interaction.business_id] = interaction
        seeds: dict[Feedback, list[ItemKNNHistoryEvent]] = {
            "positive": [],
            "negative": [],
            "neutral": [],
        }
        for interaction in latest_history.values():
            feedback = _feedback(interaction.stars)
            if feedback != "neutral":
                seeds[feedback].append(interaction)

        def evidence(
            feedback: GraphFeedback,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            output_size = len(request.candidate_business_ids)
            values = np.zeros(output_size, dtype=np.float64)
            support_counts = np.zeros(output_size, dtype=np.int64)
            neighbor_counts = np.zeros(output_size, dtype=np.int64)
            seed_weight_sum = sum(self._weight(seed.date) for seed in seeds[feedback])
            known_seeds = [
                seed
                for seed in seeds[feedback]
                if seed.business_id in self._business_index
            ]
            known_candidates = [
                (position, self._business_index[business_id])
                for position, business_id in enumerate(request.candidate_business_ids)
                if business_id in self._business_index
            ]
            if not known_seeds or not known_candidates or not seed_weight_sum:
                return values, support_counts, neighbor_counts
            seed_indices = np.fromiter(
                (self._business_index[seed.business_id] for seed in known_seeds),
                dtype=np.intp,
                count=len(known_seeds),
            )
            candidate_positions = np.fromiter(
                (position for position, _ in known_candidates),
                dtype=np.intp,
                count=len(known_candidates),
            )
            candidate_indices = np.fromiter(
                (index for _, index in known_candidates),
                dtype=np.intp,
                count=len(known_candidates),
            )
            seed_weights = np.fromiter(
                (self._weight(seed.date) for seed in known_seeds),
                dtype=np.float64,
                count=len(known_seeds),
            )
            pair_support = self._support[feedback][
                np.ix_(seed_indices, candidate_indices)
            ]
            cooccurrence = self._cooccurrence[feedback][
                np.ix_(seed_indices, candidate_indices)
            ]
            denominator = np.sqrt(
                self._item_popularity[feedback][seed_indices, np.newaxis]
                * self._item_popularity[feedback][
                    np.newaxis,
                    candidate_indices,
                ]
            )
            valid = (pair_support > 0) & (denominator > 0.0)
            similarity = np.zeros_like(cooccurrence)
            similarity[valid] = (
                cooccurrence[valid]
                / denominator[valid]
                * pair_support[valid]
                / (pair_support[valid] + self._config.shrinkage_beta)
            )
            values[candidate_positions] = seed_weights @ similarity / seed_weight_sum
            support_counts[candidate_positions] = pair_support.sum(
                axis=0,
                dtype=np.int64,
            )
            neighbor_counts[candidate_positions] = (pair_support > 0).sum(
                axis=0,
                dtype=np.int64,
            )
            return values, support_counts, neighbor_counts

        positive_values, positive_supports, positive_neighbors = evidence("positive")
        negative_values, negative_supports, negative_neighbors = evidence("negative")
        has_collaborative_evidence = bool(
            np.any(positive_values > 0.0) or np.any(negative_values > 0.0)
        )
        scores = tuple(
            ItemKNNCandidateScore(
                business_id=business_id,
                positive_score=float(positive_values[index]),
                negative_evidence=float(negative_values[index]),
                positive_support_count=int(positive_supports[index]),
                negative_support_count=int(negative_supports[index]),
                positive_neighbor_count=int(positive_neighbors[index]),
                negative_neighbor_count=int(negative_neighbors[index]),
            )
            for index, business_id in enumerate(request.candidate_business_ids)
        )
        return ItemKNNScoreResult(
            scores=scores,
            positive_history_count=len(seeds["positive"]),
            negative_history_count=len(seeds["negative"]),
            missing=not has_collaborative_evidence,
        )
