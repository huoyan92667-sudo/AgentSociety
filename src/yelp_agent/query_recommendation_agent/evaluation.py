"""Hidden-label evaluation performed only after Agent predictions are frozen."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
from pydantic import Field

from yelp_agent.evaluation import EndToEndCaseResult, UnifiedEndToEndReport
from yelp_agent.evaluation.unified_end_to_end import evaluate_end_to_end
from yelp_agent.models import StrictModel
from yelp_agent.query.benchmark import ExpectedRequestCondition
from yelp_agent.query_recommendation_benchmark import (
    QueryRecommendationFrame,
    QueryRecommendationGroundTruth,
    VisibleQueryRecommendationCase,
)

from .schema import QueryRecommendationAgentPrediction


class QueryRecommendationAgentCaseAudit(StrictModel):
    """Human-readable joined record; unlike predictions, this contains truth."""

    case_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: str
    query_text: str
    target_business_id: str
    target_business_name: str | None = None
    target_retrieval_rank: int | None = Field(default=None, ge=1)
    target_final_rank: int | None = Field(default=None, ge=1)
    status: str
    response_kind: str
    displayed_businesses: list[dict[str, object]]
    tool_sequence: list[str]
    parsed_conditions: list[dict[str, object]]
    evidence_card_count: int = Field(ge=0)
    fallback_reason: str | None = None
    failure_codes: list[str] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class QueryRecommendationAgentEvaluation:
    end_to_end_cases: tuple[EndToEndCaseResult, ...]
    case_audits: tuple[QueryRecommendationAgentCaseAudit, ...]
    report: UnifiedEndToEndReport


class BusinessConditionIndex:
    """Cutoff-safe facts used for exact and transparent compliance checks."""

    def __init__(
        self,
        businesses: Mapping[str, Mapping[str, object]],
        aspects: Mapping[tuple[str, str], Sequence[Mapping[str, object]]],
    ) -> None:
        self.businesses = dict(businesses)
        self.aspects = {key: tuple(values) for key, values in aspects.items()}

    @classmethod
    def from_parquet(
        cls,
        *,
        businesses_path: str | Path,
        aspect_events_path: str | Path,
    ) -> "BusinessConditionIndex":
        business_rows = pq.read_table(
            businesses_path,
            columns=[
                "business_id",
                "name",
                "latitude",
                "longitude",
                "categories",
                "attributes_json",
            ],
        ).to_pylist()
        aspect_rows = pq.read_table(
            aspect_events_path,
            columns=[
                "business_id",
                "review_time",
                "aspect",
                "sentiment",
                "confidence",
            ],
        ).to_pylist()
        aspects: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
        for row in aspect_rows:
            aspects[(str(row["business_id"]), str(row["aspect"]))].append(row)
        return cls(
            {str(row["business_id"]): row for row in business_rows},
            aspects,
        )

    def name(self, business_id: str) -> str | None:
        row = self.businesses.get(business_id)
        return None if row is None else str(row.get("name") or business_id)

    def satisfies(
        self,
        case: VisibleQueryRecommendationCase,
        business_id: str,
        condition: ExpectedRequestCondition,
    ) -> bool:
        business = self.businesses.get(business_id)
        if business is None:
            return False
        if condition.field == "category":
            categories = {
                str(item).casefold() for item in (business.get("categories") or [])
            }
            present = str(condition.value).casefold() in categories
            return not present if condition.operator == "excludes" else present
        if condition.field == "distance_km":
            if case.user_latitude is None or case.user_longitude is None:
                return False
            latitude = business.get("latitude")
            longitude = business.get("longitude")
            if latitude is None or longitude is None:
                return False
            distance = _haversine_km(
                case.user_latitude,
                case.user_longitude,
                float(latitude),
                float(longitude),
            )
            return distance <= float(condition.value) + 1e-9
        if condition.field == "price_level":
            observed = _price_level(business.get("attributes_json"))
            if observed is None:
                return False
            if condition.operator == "greater_than_or_equal":
                return observed >= int(condition.value)
            return observed <= int(condition.value)
        rows = [
            row
            for row in self.aspects.get((business_id, condition.field), ())
            if _as_utc_naive(row["review_time"]) < _as_utc_naive(case.cutoff_time)
        ]
        positive = sum(
            float(row["confidence"])
            for row in rows
            if row.get("sentiment") == "positive"
        )
        negative = sum(
            float(row["confidence"])
            for row in rows
            if row.get("sentiment") == "negative"
        )
        if condition.operator == "avoid":
            return negative > positive
        return positive > 0 and positive > negative


def evaluate_frozen_predictions(
    *,
    visible_cases: Sequence[VisibleQueryRecommendationCase],
    ground_truth: Sequence[QueryRecommendationGroundTruth],
    frames: Sequence[QueryRecommendationFrame],
    predictions: Sequence[QueryRecommendationAgentPrediction],
    facts: BusinessConditionIndex,
) -> QueryRecommendationAgentEvaluation:
    visible_by_id = _unique_index(visible_cases, "visible cases")
    truth_by_id = _unique_index(ground_truth, "ground truth")
    frame_by_id = _unique_index(frames, "structured frames")
    prediction_by_id = _unique_index(predictions, "Agent predictions")
    case_ids = set(visible_by_id)
    if any(set(values) != case_ids for values in (truth_by_id, frame_by_id, prediction_by_id)):
        raise ValueError("visible cases, frozen predictions, truth, and frames must align")

    end_to_end: list[EndToEndCaseResult] = []
    audits: list[QueryRecommendationAgentCaseAudit] = []
    for case_id in sorted(case_ids):
        case = visible_by_id[case_id]
        truth = truth_by_id[case_id]
        frame = frame_by_id[case_id]
        prediction = prediction_by_id[case_id]
        hard_conditions = [
            condition for condition in frame.conditions if condition.enforcement == "filter"
        ]
        hard_allowed = (
            None
            if not hard_conditions
            else [
                business_id
                for business_id in prediction.final_ranking
                if all(facts.satisfies(case, business_id, item) for item in hard_conditions)
            ]
        )
        compliant = [
            business_id
            for business_id in prediction.final_ranking
            if all(facts.satisfies(case, business_id, item) for item in frame.conditions)
        ]
        evidence_has_time = any(
            item.source_time is not None
            for card in prediction.evidence_cards
            for item in card.supporting_evidence + card.contradicting_evidence
        )
        reported_recency = any(turn.reported_evidence_recency for turn in prediction.turns)
        end_to_end.append(
            EndToEndCaseResult(
                case_id=case_id,
                target_business_id=truth.target_business_id,
                retrieval_ranking=prediction.retrieval_ranking,
                final_ranking=prediction.final_ranking,
                hard_constraint_satisfied_business_ids=hard_allowed,
                query_compliant_business_ids=compliant,
                evidence_cards=prediction.evidence_cards,
                agent_status=prediction.status,
                response_kind=prediction.response_kind,
                fallback=prediction.fallback,
                reported_evidence_recency=(
                    reported_recency if evidence_has_time else None
                ),
                action_count=prediction.action_count,
                invalid_action_count=prediction.invalid_action_count,
                tool_call_count=prediction.tool_call_count,
                latency_ms=prediction.latency_ms,
                input_tokens=prediction.input_tokens,
                output_tokens=prediction.output_tokens,
                cost_usd=prediction.cost_usd,
            )
        )
        audits.append(
            QueryRecommendationAgentCaseAudit(
                case_id=case_id,
                split=case.split,
                query_text=case.query_text,
                target_business_id=truth.target_business_id,
                target_business_name=facts.name(truth.target_business_id),
                target_retrieval_rank=_rank(
                    prediction.retrieval_ranking,
                    truth.target_business_id,
                ),
                target_final_rank=_rank(
                    prediction.final_ranking,
                    truth.target_business_id,
                ),
                status=prediction.status,
                response_kind=prediction.response_kind,
                displayed_businesses=[
                    {
                        "rank": index,
                        "business_id": business_id,
                        "business_name": facts.name(business_id),
                    }
                    for index, business_id in enumerate(
                        prediction.displayed_business_ids,
                        start=1,
                    )
                ],
                tool_sequence=[
                    call.tool_name
                    for turn in prediction.turns
                    for call in turn.tool_calls
                ],
                parsed_conditions=[
                    item.model_dump(mode="json") for item in prediction.request.conditions
                ],
                evidence_card_count=len(prediction.evidence_cards),
                fallback_reason=prediction.fallback_reason,
                failure_codes=prediction.failure_codes,
            )
        )
    rows = tuple(end_to_end)
    return QueryRecommendationAgentEvaluation(
        end_to_end_cases=rows,
        case_audits=tuple(audits),
        report=evaluate_end_to_end(rows),
    )


def _unique_index(values: Sequence[object], label: str) -> dict[str, object]:
    output = {str(getattr(item, "case_id")): item for item in values}
    if not output or len(output) != len(values):
        raise ValueError(f"{label} must be nonempty and unique")
    return output


def _rank(ranking: Sequence[str], target: str) -> int | None:
    try:
        return list(ranking).index(target) + 1
    except ValueError:
        return None


def _price_level(raw: object) -> int | None:
    try:
        payload = json.loads(str(raw or "{}"))
        return int(payload["RestaurantsPriceRange2"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _as_utc_naive(value: object) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    value = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(delta_lambda / 2) ** 2
    )
    return 6371.0 * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))
