"""Cutoff-safe structured and local-model semantic scoring."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from yelp_agent.business_profiles.schema import BusinessProfileV1
from yelp_agent.query.schema import RecommendationRequest, RequestCondition
from yelp_agent.reviews.schema import ASPECT_NAMES

from .config import SemanticRankingPolicy
from .schema import CandidateSemanticScore, ConditionMatch


_IMPORTANCE = {"mandatory": 1.0, "strong": 0.75, "preferred": 0.5}


def _haversine_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    radius = 6371.0088
    a_lat_r = math.radians(a_lat)
    b_lat_r = math.radians(b_lat)
    delta_lat = b_lat_r - a_lat_r
    delta_lon = math.radians(b_lon - a_lon)
    value = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(a_lat_r) * math.cos(b_lat_r) * math.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * radius * math.asin(math.sqrt(value))


def _price_level(profile: BusinessProfileV1) -> int | None:
    value = profile.structured_attributes.get("RestaurantsPriceRange2")
    try:
        parsed = int(str(value).strip(" '\""))
    except (TypeError, ValueError):
        return None
    return parsed if 1 <= parsed <= 4 else None


def _condition_match(
    request: RecommendationRequest,
    condition: RequestCondition,
    profile: BusinessProfileV1,
) -> ConditionMatch:
    if condition.field == "category":
        expected = str(condition.value).casefold()
        present = any(value.casefold() == expected for value in profile.categories)
        matched = not present if condition.operator == "excludes" else present
        return ConditionMatch(
            field=condition.field,
            status="matched" if matched else "unmatched",
            score=1.0 if matched else 0.0,
            evidence_confidence=1.0,
        )
    if condition.field == "distance_km":
        if request.location_center is None or profile.latitude is None:
            return ConditionMatch(
                field=condition.field,
                status="unknown",
                score=0.5,
                evidence_confidence=0.0,
            )
        distance = _haversine_km(
            request.location_center.latitude,
            request.location_center.longitude,
            profile.latitude,
            profile.longitude,
        )
        limit = float(condition.value)
        matched = distance <= limit
        score = math.exp(-distance / max(limit, 1.0))
        return ConditionMatch(
            field=condition.field,
            status="matched" if matched else "unmatched",
            score=min(1.0, max(0.0, score)),
            evidence_confidence=1.0,
        )
    if condition.field == "price_level":
        actual = _price_level(profile)
        if actual is None:
            return ConditionMatch(
                field=condition.field,
                status="unknown",
                score=0.5,
                evidence_confidence=0.0,
            )
        expected = float(condition.value)
        if condition.operator == "less_than_or_equal":
            matched = actual <= expected
        elif condition.operator == "greater_than_or_equal":
            matched = actual >= expected
        else:
            matched = actual == expected
        return ConditionMatch(
            field=condition.field,
            status="matched" if matched else "unmatched",
            score=1.0 if matched else 0.0,
            evidence_confidence=1.0,
        )
    if condition.field == "budget_per_person":
        return ConditionMatch(
            field=condition.field,
            status="unknown",
            score=0.5,
            evidence_confidence=0.0,
        )
    if condition.field in ASPECT_NAMES:
        summary = profile.aspect_summaries[condition.field]  # type: ignore[index]
        if summary.status == "unknown":
            return ConditionMatch(
                field=condition.field,
                status="unknown",
                score=0.5,
                evidence_confidence=0.0,
            )
        directional = (
            float(summary.weighted_negative_ratio)
            if condition.operator == "avoid"
            else float(summary.weighted_positive_ratio)
        )
        score = 0.5 + summary.confidence * (directional - 0.5)
        return ConditionMatch(
            field=condition.field,
            status="matched" if score >= 0.5 else "unmatched",
            score=min(1.0, max(0.0, score)),
            evidence_confidence=summary.confidence,
        )
    return ConditionMatch(
        field=condition.field,
        status="unknown",
        score=0.5,
        evidence_confidence=0.0,
    )


def score_candidates(
    *,
    request: RecommendationRequest,
    base_ranking: Sequence[str],
    profiles: Mapping[str, BusinessProfileV1],
    embedding_scores: Mapping[str, float],
    cross_encoder_scores: Mapping[str, float],
    policy: SemanticRankingPolicy,
) -> list[CandidateSemanticScore]:
    rankable = [
        condition
        for condition in request.conditions
        if condition.enforcement in {"rank", "evidence"}
    ]
    rows: list[CandidateSemanticScore] = []
    for base_rank, business_id in enumerate(base_ranking, start=1):
        matches = [
            _condition_match(request, condition, profiles[business_id])
            for condition in rankable
        ]
        weights = [_IMPORTANCE[item.importance] for item in rankable]
        total = sum(weights)
        structured = (
            sum(weight * match.score for weight, match in zip(weights, matches, strict=True))
            / total
            if total
            else 0.5
        )
        coverage = (
            sum(
                weight * match.evidence_confidence
                for weight, match in zip(weights, matches, strict=True)
            )
            / total
            if total
            else 0.0
        )
        embedding = min(1.0, max(0.0, embedding_scores[business_id]))
        cross = min(1.0, max(0.0, cross_encoder_scores[business_id]))
        semantic = (
            policy.embedding_weight * embedding
            + policy.cross_encoder_weight * cross
            + policy.structured_weight * structured
        )
        rows.append(
            CandidateSemanticScore(
                business_id=business_id,
                base_rank=base_rank,
                embedding_score=embedding,
                cross_encoder_score=cross,
                structured_score=structured,
                evidence_coverage=coverage,
                semantic_score=min(1.0, max(0.0, semantic)),
                semantic_rank=1,
                fused_score=0.0,
                final_rank=base_rank,
                rank_movement=0,
                matches=matches,
            )
        )
    ordered = sorted(rows, key=lambda item: (-item.semantic_score, item.business_id))
    semantic_rank = {
        item.business_id: rank for rank, item in enumerate(ordered, start=1)
    }
    return [
        item.model_copy(update={"semantic_rank": semantic_rank[item.business_id]})
        for item in rows
    ]
