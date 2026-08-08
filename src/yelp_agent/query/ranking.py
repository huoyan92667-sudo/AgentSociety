"""Deterministic hard filtering and static Query-aware ranking."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from yelp_agent.models import StrictModel
from yelp_agent.query.schema import RecommendationRequest, RequestCondition
from yelp_agent.reviews.schema import AspectName

type QueryRankingMode = Literal["query_only", "hybrid_query"]
type ConstraintStatus = Literal["eligible", "eligible_with_unknowns"]


class CandidateAspectEvidence(StrictModel):
    """Compact point-in-time aspect evidence consumed by static ranking."""

    status: Literal["known", "unknown"]
    positive_ratio: float | None = Field(default=None, ge=0, le=1)
    negative_ratio: float | None = Field(default=None, ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    evidence_count: int = Field(ge=0)
    conflict: bool = False
    latest_evidence_time: datetime | None = None

    @model_validator(mode="after")
    def validate_evidence(self) -> CandidateAspectEvidence:
        if self.status == "unknown":
            if self.positive_ratio is not None or self.negative_ratio is not None:
                raise ValueError("unknown aspect evidence cannot expose ratios")
        else:
            if self.positive_ratio is None or self.negative_ratio is None:
                raise ValueError("known aspect evidence requires both ratios")
            if not math.isclose(
                self.positive_ratio + self.negative_ratio,
                1.0,
                abs_tol=1e-9,
            ):
                raise ValueError("aspect evidence ratios must sum to one")
        return self


class QueryAwareCandidate(StrictModel):
    """Target-blind candidate view shared by Query-only and fused ranking."""

    business_id: str = Field(min_length=1)
    hybrid_rank: int = Field(ge=1)
    categories: tuple[str, ...]
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    price_level: int | None = Field(default=None, ge=1, le=4)
    aspect_evidence: dict[AspectName, CandidateAspectEvidence] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_candidate(self) -> QueryAwareCandidate:
        if not self.categories or len(set(self.categories)) != len(self.categories):
            raise ValueError("candidate categories must be nonempty and unique")
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("candidate coordinates must be both present or absent")
        return self


class QueryAwareScoredCandidate(StrictModel):
    business_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    hybrid_rank: int = Field(ge=1)
    query_rank: int = Field(ge=1)
    query_score: float = Field(ge=0, le=1)
    fusion_score: float = Field(ge=0)
    constraint_status: ConstraintStatus
    matched_fields: list[str]
    unmatched_fields: list[str]
    unknown_fields: list[str]


class ExcludedQueryCandidate(StrictModel):
    business_id: str = Field(min_length=1)
    reason_codes: list[str] = Field(min_length=1)


class QueryAwareRankingResult(StrictModel):
    request_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    mode: QueryRankingMode
    ranking: list[QueryAwareScoredCandidate]
    excluded: list[ExcludedQueryCandidate]
    warnings: list[str]


def _haversine_km(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    radius = 6371.0088
    lat_a = math.radians(latitude_a)
    lat_b = math.radians(latitude_b)
    delta_lat = lat_b - lat_a
    delta_lon = math.radians(longitude_b - longitude_a)
    value = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * radius * math.asin(math.sqrt(value))


def _has_category(candidate: QueryAwareCandidate, value: object) -> bool:
    expected = str(value).casefold()
    return any(category.casefold() == expected for category in candidate.categories)


def _filter_failures(
    request: RecommendationRequest,
    candidate: QueryAwareCandidate,
) -> list[str]:
    failures: list[str] = []
    required_categories = [
        condition
        for condition in request.hard_constraints
        if condition.field == "category" and condition.operator == "includes"
    ]
    if required_categories and not any(
        _has_category(candidate, condition.value) for condition in required_categories
    ):
        allowed = "|".join(sorted(str(item.value) for item in required_categories))
        failures.append(f"HARD_CATEGORY_MISSING:{allowed}")
    for condition in request.hard_constraints:
        if condition.field == "category":
            present = _has_category(candidate, condition.value)
            if condition.operator == "excludes" and present:
                failures.append(f"HARD_CATEGORY_EXCLUDED:{condition.value}")
        elif condition.field == "distance_km":
            if request.location_center is None:
                failures.append("HARD_DISTANCE_REQUEST_LOCATION_UNKNOWN")
            elif candidate.latitude is None:
                failures.append("HARD_DISTANCE_BUSINESS_LOCATION_UNKNOWN")
            else:
                distance = _haversine_km(
                    request.location_center.latitude,
                    request.location_center.longitude,
                    candidate.latitude,
                    candidate.longitude,
                )
                if distance > float(condition.value):
                    failures.append("HARD_DISTANCE_EXCEEDED")
    return failures


_IMPORTANCE_WEIGHT = {"mandatory": 1.0, "strong": 0.75, "preferred": 0.5}


def _condition_match(
    condition: RequestCondition,
    candidate: QueryAwareCandidate,
) -> tuple[float, Literal["matched", "unmatched", "unknown"]]:
    if condition.field == "category":
        matched = _has_category(candidate, condition.value)
        if condition.operator == "excludes":
            matched = not matched
        return (1.0 if matched else 0.0), ("matched" if matched else "unmatched")
    evidence = candidate.aspect_evidence.get(condition.field)  # type: ignore[arg-type]
    if evidence is None or evidence.status == "unknown":
        return 0.5, "unknown"
    ratio = (
        float(evidence.negative_ratio)
        if condition.operator == "avoid"
        else float(evidence.positive_ratio)
    )
    score = 0.5 + evidence.confidence * (ratio - 0.5)
    return min(1.0, max(0.0, score)), ("matched" if score >= 0.5 else "unmatched")


def _query_score(
    request: RecommendationRequest,
    candidate: QueryAwareCandidate,
) -> tuple[float, list[str], list[str], list[str]]:
    conditions = [
        condition
        for condition in request.conditions
        if condition.enforcement in {"rank", "evidence"}
        or (condition.enforcement == "filter" and condition.field == "category")
    ]
    if not conditions:
        return 0.5, [], [], []
    weighted_sum = 0.0
    total_weight = 0.0
    matched: list[str] = []
    unmatched: list[str] = []
    unknown: list[str] = []
    for condition in conditions:
        weight = _IMPORTANCE_WEIGHT[condition.importance]
        score, status = _condition_match(condition, candidate)
        weighted_sum += weight * score
        total_weight += weight
        target = {"matched": matched, "unmatched": unmatched, "unknown": unknown}[
            status
        ]
        if condition.field not in target:
            target.append(condition.field)
    return weighted_sum / total_weight, matched, unmatched, unknown


class QueryAwareStaticRanker:
    """Apply hard policy, then rank by Query alone or conservative rank fusion."""

    def __init__(
        self,
        *,
        query_rrf_weight: float = 1.0,
        rrf_constant: float = 60.0,
    ) -> None:
        if query_rrf_weight < 0:
            raise ValueError("query_rrf_weight cannot be negative")
        if rrf_constant <= 0:
            raise ValueError("rrf_constant must be positive")
        self._query_rrf_weight = query_rrf_weight
        self._rrf_constant = rrf_constant

    def rank(
        self,
        request: RecommendationRequest,
        candidates: Sequence[QueryAwareCandidate],
        *,
        mode: QueryRankingMode,
    ) -> QueryAwareRankingResult:
        if not candidates:
            raise ValueError("candidates cannot be empty")
        ids = [candidate.business_id for candidate in candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("candidate business IDs must be unique")
        hybrid_ranks = [candidate.hybrid_rank for candidate in candidates]
        if len(set(hybrid_ranks)) != len(hybrid_ranks):
            raise ValueError("hybrid ranks must be unique")

        eligible: list[
            tuple[QueryAwareCandidate, float, list[str], list[str], list[str]]
        ] = []
        excluded: list[ExcludedQueryCandidate] = []
        for candidate in candidates:
            failures = _filter_failures(request, candidate)
            if failures:
                excluded.append(
                    ExcludedQueryCandidate(
                        business_id=candidate.business_id,
                        reason_codes=failures,
                    )
                )
                continue
            score, matched, unmatched, unknown = _query_score(request, candidate)
            eligible.append((candidate, score, matched, unmatched, unknown))

        query_order = sorted(
            eligible,
            key=lambda item: (-item[1], item[0].business_id),
        )
        query_rank = {
            item[0].business_id: rank for rank, item in enumerate(query_order, start=1)
        }
        has_query_evidence = any(
            condition.enforcement in {"rank", "evidence"}
            for condition in request.conditions
        )

        def fusion(
            item: tuple[QueryAwareCandidate, float, list[str], list[str], list[str]],
        ) -> float:
            candidate = item[0]
            if mode == "query_only":
                return item[1]
            hybrid_part = 1.0 / (self._rrf_constant + candidate.hybrid_rank)
            if not has_query_evidence:
                return hybrid_part
            query_part = self._query_rrf_weight / (
                self._rrf_constant + query_rank[candidate.business_id]
            )
            return hybrid_part + query_part

        final_order = sorted(
            eligible,
            key=lambda item: (-fusion(item), item[0].business_id),
        )
        ranking = [
            QueryAwareScoredCandidate(
                business_id=item[0].business_id,
                rank=rank,
                hybrid_rank=item[0].hybrid_rank,
                query_rank=query_rank[item[0].business_id],
                query_score=item[1],
                fusion_score=fusion(item),
                constraint_status=("eligible_with_unknowns" if item[4] else "eligible"),
                matched_fields=item[2],
                unmatched_fields=item[3],
                unknown_fields=item[4],
            )
            for rank, item in enumerate(final_order, start=1)
        ]
        warnings = sorted(
            {
                f"UNKNOWN_EVIDENCE:{field}"
                for item in ranking
                for field in item.unknown_fields
            }
        )
        return QueryAwareRankingResult(
            request_id=request.request_id,
            mode=mode,
            ranking=ranking,
            excluded=sorted(excluded, key=lambda item: item.business_id),
            warnings=warnings,
        )
