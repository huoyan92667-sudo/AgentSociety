"""Full-catalog Query retrieval behind one small public interface."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime
from time import perf_counter
from typing import Protocol

from yelp_agent.data.temporal_view import BusinessRecord, ReviewCatalogSnapshot
from yelp_agent.query.schema import RecommendationRequest, RequestCondition
from yelp_agent.business_profiles.schema import BusinessAspectSummary
from yelp_agent.reviews.schema import ASPECT_NAMES, AspectName
from yelp_agent.semantic_ranking.intent_compiler import RankingIntentCompiler

from .config import QueryRetrievalConfig
from .schema import (
    ExcludedQueryBusiness,
    QueryRetrievalCandidate,
    QueryRetrievalResult,
    QueryRetrievalTask,
    QueryRetrievalUsage,
)


class TemporalBusinessCatalog(Protocol):
    def businesses(self) -> Sequence[BusinessRecord]: ...

    def review_catalog_before(self, cutoff_time: datetime) -> ReviewCatalogSnapshot: ...


class BusinessAspectReader(Protocol):
    @property
    def source_scope(self) -> str: ...

    def get_aspects(
        self,
        business_ids: list[str],
        aspects: list[AspectName],
        cutoff_time: datetime,
    ) -> dict[str, dict[AspectName, BusinessAspectSummary]]: ...


class EmbeddingMatcher(Protocol):
    def match(
        self,
        *,
        query_text: str,
        business_ids: Sequence[str],
        cutoff_time: datetime,
        usage_scope: str | None = None,
    ) -> object: ...


class QueryCandidateRetriever:
    """Retrieve point-in-time businesses from accepted current-request semantics."""

    def __init__(
        self,
        *,
        catalog: TemporalBusinessCatalog,
        profiles: BusinessAspectReader,
        embedding_matcher: EmbeddingMatcher | None,
        config: QueryRetrievalConfig,
    ) -> None:
        self._catalog = catalog
        self._profiles = profiles
        self._embedding = embedding_matcher
        self._config = config
        self._intent_compiler = RankingIntentCompiler()
        businesses = list(catalog.businesses())
        self._businesses = {item.business_id: item for item in businesses}
        if len(self._businesses) != len(businesses):
            raise ValueError("business catalog IDs must be unique")

    def retrieve(self, task: QueryRetrievalTask) -> QueryRetrievalResult:
        started = perf_counter()
        request = task.request
        snapshot = self._catalog.review_catalog_before(request.cutoff_time)
        counts = dict(
            zip(snapshot.business_ids, snapshot.review_counts, strict=True)
        )
        pre_cutoff = [
            business_id
            for business_id in sorted(self._businesses)
            if counts.get(business_id, 0) > 0
        ]
        eligible: list[str] = []
        excluded: list[ExcludedQueryBusiness] = []
        for business_id in pre_cutoff:
            reasons = _static_hard_failures(
                request,
                self._businesses[business_id],
            )
            if reasons:
                excluded.append(
                    ExcludedQueryBusiness(
                        business_id=business_id,
                        reason_codes=reasons,
                    )
                )
            else:
                eligible.append(business_id)

        aspect_conditions = [
            condition
            for condition in request.conditions
            if condition.field in ASPECT_NAMES
            and condition.enforcement in {"filter", "rank", "evidence"}
        ]
        requested_aspects = sorted(
            {condition.field for condition in aspect_conditions}
        )
        aspect_cache: dict[str, dict[AspectName, BusinessAspectSummary]] = {}
        if requested_aspects and eligible:
            _validate_profile_reader_scope(
                self._profiles,
                self._config.aspect_source_scope,
            )
            aspect_cache = self._profiles.get_aspects(
                eligible,
                requested_aspects,
                request.cutoff_time,
            )
        hard_aspect_conditions = [
            condition
            for condition in request.hard_constraints
            if condition.field in ASPECT_NAMES
        ]
        if hard_aspect_conditions and eligible:
            aspect_eligible: list[str] = []
            for business_id in eligible:
                reasons = _aspect_hard_failures(
                    hard_aspect_conditions,
                    aspect_cache[business_id],
                )
                if reasons:
                    excluded.append(
                        ExcludedQueryBusiness(
                            business_id=business_id,
                            reason_codes=reasons,
                        )
                    )
                else:
                    aspect_eligible.append(business_id)
            eligible = aspect_eligible

        required = {value.casefold() for value in request.desired_categories}
        category_scores: dict[str, float] = {}
        if required:
            for business_id in eligible:
                categories = {
                    value.casefold()
                    for value in self._businesses[business_id].categories
                }
                overlap = len(categories.intersection(required))
                if overlap:
                    category_scores[business_id] = overlap / len(required)
        category_order = sorted(
            category_scores,
            key=lambda business_id: (-category_scores[business_id], business_id),
        )[: self._config.per_route_limit]
        category_rank = {
            business_id: rank
            for rank, business_id in enumerate(category_order, start=1)
        }
        distances = {
            business_id: _haversine_km(
                request.location_center.latitude,
                request.location_center.longitude,
                float(self._businesses[business_id].latitude),
                float(self._businesses[business_id].longitude),
            )
            for business_id in eligible
            if request.location_center is not None
            and self._businesses[business_id].latitude is not None
        }
        location_scores = {
            business_id: math.exp(-distance / self._config.location_scale_km)
            for business_id, distance in distances.items()
        }
        location_order = sorted(
            location_scores,
            key=lambda business_id: (-location_scores[business_id], business_id),
        )[: self._config.per_route_limit]
        location_rank = {
            business_id: rank
            for rank, business_id in enumerate(location_order, start=1)
        }
        embedding_scores: dict[str, float] = {}
        embedding_usage = QueryRetrievalUsage()
        if self._embedding is not None and eligible:
            intent = self._intent_compiler.compile(request)
            match_result = self._embedding.match(
                query_text=intent.document,
                business_ids=eligible,
                cutoff_time=request.cutoff_time,
                usage_scope=f"{task.usage_scope}:query-retrieval",
            )
            embedding_scores = {
                item.business_id: float(item.normalized_score)
                for item in match_result.matches
            }
            if (
                len(match_result.matches) != len(eligible)
                or set(embedding_scores) != set(eligible)
            ):
                raise ValueError("Embedding route did not score the complete eligible scope")
            embedding_usage = QueryRetrievalUsage(
                embedding_input_tokens=match_result.usage.input_tokens,
                embedding_logical_tokens=match_result.usage.logical_input_tokens,
                cache_hits=match_result.usage.cache_hits,
                cache_misses=match_result.usage.cache_misses,
                provider_calls=match_result.usage.api_calls,
            )
        embedding_order = sorted(
            embedding_scores,
            key=lambda business_id: (-embedding_scores[business_id], business_id),
        )[: self._config.per_route_limit]
        embedding_rank = {
            business_id: rank
            for rank, business_id in enumerate(embedding_order, start=1)
        }
        aspect_scores: dict[str, float] = {}
        aspect_matched: dict[str, list[str]] = {}
        aspect_unknown: dict[str, list[str]] = {}
        warnings: list[str] = []
        if aspect_conditions and eligible:
            unknown_counts = {condition.field: 0 for condition in aspect_conditions}
            for business_id in eligible:
                summaries = aspect_cache[business_id]
                weighted_sum = 0.0
                total_weight = 0.0
                known_count = 0
                matched_fields: list[str] = []
                unknown_fields: list[str] = []
                for condition in aspect_conditions:
                    weight = _importance_weight(condition)
                    total_weight += weight
                    summary = summaries[condition.field]
                    if summary.status == "unknown":
                        weighted_sum += 0.5 * weight
                        unknown_counts[condition.field] += 1
                        if condition.field not in unknown_fields:
                            unknown_fields.append(condition.field)
                        continue
                    known_count += 1
                    directional = (
                        float(summary.weighted_negative_ratio)
                        if _wants_negative(condition)
                        else float(summary.weighted_positive_ratio)
                    )
                    score = min(
                        1.0,
                        max(0.0, 0.5 + summary.confidence * (directional - 0.5)),
                    )
                    weighted_sum += score * weight
                    if score > 0.5 and condition.field not in matched_fields:
                        matched_fields.append(condition.field)
                aspect_matched[business_id] = matched_fields
                aspect_unknown[business_id] = unknown_fields
                if known_count:
                    combined_score = weighted_sum / total_weight
                    if combined_score > 0.5:
                        aspect_scores[business_id] = combined_score
            warnings.append("ASPECT_COVERAGE_LIMITED_TO_SELECTED_5000_USERS")
            warnings.extend(
                f"ASPECT_EVIDENCE_UNKNOWN:{field}:{count}"
                for field, count in sorted(unknown_counts.items())
                if count
            )
        aspect_order = sorted(
            aspect_scores,
            key=lambda business_id: (-aspect_scores[business_id], business_id),
        )[: self._config.per_route_limit]
        aspect_rank = {
            business_id: rank
            for rank, business_id in enumerate(aspect_order, start=1)
        }
        fusion: dict[str, float] = {}
        _add_rrf(
            fusion,
            category_rank,
            self._config.category_weight,
            self._config.rrf_constant,
        )
        _add_rrf(
            fusion,
            location_rank,
            self._config.location_weight,
            self._config.rrf_constant,
        )
        _add_rrf(
            fusion,
            aspect_rank,
            self._config.aspect_weight,
            self._config.rrf_constant,
        )
        _add_rrf(
            fusion,
            embedding_rank,
            self._config.embedding_weight,
            self._config.rrf_constant,
        )
        final_ids = sorted(
            fusion,
            key=lambda business_id: (-fusion[business_id], business_id),
        )[: self._config.candidate_limit]
        candidates: list[QueryRetrievalCandidate] = []
        for rank, business_id in enumerate(final_ids, start=1):
            matched = []
            if business_id in category_rank:
                matched.append("category")
            if business_id in location_rank:
                matched.append("distance_km")
            for field in aspect_matched.get(business_id, []):
                if field not in matched:
                    matched.append(field)
            candidates.append(QueryRetrievalCandidate(
                business_id=business_id,
                rank=rank,
                fusion_score=fusion[business_id],
                route_count=int(business_id in category_rank)
                + int(business_id in location_rank)
                + int(business_id in aspect_rank)
                + int(business_id in embedding_rank),
                category_rank=category_rank.get(business_id),
                category_score=category_scores.get(business_id),
                location_rank=location_rank.get(business_id),
                location_score=location_scores.get(business_id),
                distance_km=distances.get(business_id),
                aspect_rank=aspect_rank.get(business_id),
                aspect_score=aspect_scores.get(business_id),
                embedding_rank=embedding_rank.get(business_id),
                embedding_score=embedding_scores.get(business_id),
                matched_fields=matched,
                unknown_fields=aspect_unknown.get(business_id, []),
                source_scope=self._config.aspect_source_scope,
            ))
        return QueryRetrievalResult(
            request_id=request.request_id,
            cutoff_time=request.cutoff_time,
            catalog_size=len(self._businesses),
            pre_cutoff_business_count=len(pre_cutoff),
            eligible_business_count=len(eligible),
            candidates=candidates,
            excluded=excluded,
            route_result_counts={
                "query_category": len(category_order),
                "query_embedding": len(embedding_order),
                "query_aspect": len(aspect_order),
                "query_location": len(location_order),
            },
            warnings=warnings,
            usage=embedding_usage,
            latency_ms=(perf_counter() - started) * 1000.0,
        )


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


def _add_rrf(
    destination: dict[str, float],
    ranks: dict[str, int],
    weight: float,
    constant: float,
) -> None:
    if weight <= 0:
        return
    for business_id, rank in ranks.items():
        destination[business_id] = destination.get(business_id, 0.0) + (
            weight / (constant + rank)
        )


def _price_level(business: BusinessRecord) -> int | None:
    raw = business.attributes_dict().get("RestaurantsPriceRange2")
    try:
        value = int(str(raw).strip().strip("'\""))
    except (TypeError, ValueError):
        return None
    return value if 1 <= value <= 4 else None


def _unknown_is_failure(condition: RequestCondition) -> bool:
    return condition.unknown_policy in {"exclude", "ask"}


def _importance_weight(condition: RequestCondition) -> float:
    return {"mandatory": 1.0, "strong": 0.75, "preferred": 0.5}[
        condition.importance
    ]


def _wants_negative(condition: RequestCondition) -> bool:
    return condition.operator == "avoid" or condition.value is False


def _validate_profile_reader_scope(reader: object, expected_scope: str) -> None:
    if getattr(reader, "source_scope", None) != expected_scope:
        raise ValueError("Aspect profile source scope does not match config")


def _aspect_hard_failures(
    conditions: Sequence[RequestCondition],
    summaries: dict[AspectName, BusinessAspectSummary],
) -> list[str]:
    failures: list[str] = []
    for condition in conditions:
        summary = summaries[condition.field]
        if summary.status == "unknown":
            if _unknown_is_failure(condition):
                failures.append(f"HARD_ASPECT_UNKNOWN:{condition.field}")
            continue
        directional = (
            float(summary.weighted_negative_ratio)
            if _wants_negative(condition)
            else float(summary.weighted_positive_ratio)
        )
        score = 0.5 + summary.confidence * (directional - 0.5)
        if score <= 0.5:
            failures.append(f"HARD_ASPECT_MISMATCH:{condition.field}")
    return failures


def _static_hard_failures(
    request: RecommendationRequest,
    business: BusinessRecord,
) -> list[str]:
    categories = {value.casefold() for value in business.categories}
    failures: list[str] = []
    for condition in request.hard_constraints:
        if condition.field == "category":
            value = str(condition.value)
            present = value.casefold() in categories
            if condition.operator == "includes" and not present:
                failures.append(f"HARD_CATEGORY_MISSING:{value}")
            elif condition.operator == "excludes" and present:
                failures.append("HARD_CATEGORY_EXCLUDED")
        elif condition.field == "distance_km":
            if request.location_center is None:
                failures.append("HARD_DISTANCE_REQUEST_LOCATION_UNKNOWN")
            elif business.latitude is None:
                if _unknown_is_failure(condition):
                    failures.append("HARD_DISTANCE_BUSINESS_LOCATION_UNKNOWN")
            else:
                distance = _haversine_km(
                    request.location_center.latitude,
                    request.location_center.longitude,
                    business.latitude,
                    float(business.longitude),
                )
                if distance > float(condition.value):
                    failures.append("HARD_DISTANCE_EXCEEDED")
        elif condition.field == "price_level":
            actual = _price_level(business)
            if actual is None:
                if _unknown_is_failure(condition):
                    failures.append("HARD_PRICE_LEVEL_UNKNOWN")
            else:
                expected = float(condition.value)
                matched = (
                    actual <= expected
                    if condition.operator == "less_than_or_equal"
                    else actual >= expected
                    if condition.operator == "greater_than_or_equal"
                    else actual == expected
                )
                if not matched:
                    failures.append("HARD_PRICE_LEVEL_EXCEEDED")
        elif condition.field == "budget_per_person" and _unknown_is_failure(condition):
            failures.append("HARD_BUDGET_PER_PERSON_UNVERIFIABLE")
    return failures
