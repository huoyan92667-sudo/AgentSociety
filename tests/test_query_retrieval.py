"""Behavior tests for full-catalog Query candidate retrieval."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from yelp_agent.agent_tools import (
    AgentToolRegistry,
    ExpandCandidatesTool,
    ToolExecutionContext,
)
from yelp_agent.data.temporal_view import BusinessRecord, ReviewCatalogSnapshot
from yelp_agent.models import LocationCenter
from yelp_agent.query.schema import RecommendationRequest, RequestCondition
from yelp_agent.query_retrieval import (
    DualChannelFusion,
    QueryCandidateRetriever,
    QueryRetrievalConfig,
    QueryRetrievalTask,
    load_query_retrieval_config,
)
from yelp_agent.retrieval import (
    RetrievalCandidate,
    RetrievalResult,
    RetrievalTaskContext,
)


def _business(
    business_id: str,
    *categories: str,
    latitude: float | None = 39.95,
    longitude: float | None = -75.16,
    attributes_json: str = "{}",
) -> BusinessRecord:
    return BusinessRecord(
        business_id=business_id,
        name=business_id,
        address="",
        city="Philadelphia",
        state="PA",
        postal_code="",
        latitude=latitude,
        longitude=longitude,
        categories=categories,
        attributes_json=attributes_json,
    )


class _Catalog:
    def __init__(self) -> None:
        self.rows = (
            _business("old-steak", "Restaurants", "Steakhouses"),
            _business("future-steak", "Restaurants", "Steakhouses"),
            _business("old-sushi", "Restaurants", "Sushi Bars"),
        )

    def businesses(self) -> tuple[BusinessRecord, ...]:
        return self.rows

    def review_catalog_before(self, cutoff_time: datetime) -> ReviewCatalogSnapshot:
        assert cutoff_time == datetime(2022, 1, 1)
        return ReviewCatalogSnapshot(
            business_ids=("future-steak", "old-steak", "old-sushi"),
            review_counts=(0, 3, 2),
            star_sums=(0.0, 14.0, 8.0),
            global_count=5,
            global_star_sum=22.0,
        )


class _Profiles:
    def get(self, business_ids: list[str], cutoff_time: datetime):
        raise AssertionError("category-only retrieval must not load Aspect profiles")


class _AspectProfiles:
    source_scope = "selected_user_interactions"

    def get(self, business_ids: list[str], cutoff_time: datetime):
        raise AssertionError("Query retrieval must not build complete business profiles")

    def get_aspects(
        self,
        business_ids: list[str],
        aspects: list[str],
        cutoff_time: datetime,
    ):
        values = {
            "quiet-positive": {
                    "quiet_environment": SimpleNamespace(
                        status="known",
                        weighted_positive_ratio=0.9,
                        weighted_negative_ratio=0.1,
                        confidence=0.8,
                    )
                },
            "quiet-negative": {
                    "quiet_environment": SimpleNamespace(
                        status="known",
                        weighted_positive_ratio=0.1,
                        weighted_negative_ratio=0.9,
                        confidence=0.8,
                    )
                },
            "quiet-unknown": {
                    "quiet_environment": SimpleNamespace(
                        status="unknown",
                        weighted_positive_ratio=None,
                        weighted_negative_ratio=None,
                        confidence=0.0,
                    )
                },
        }
        assert aspects == ["quiet_environment"]
        return {business_id: values[business_id] for business_id in business_ids}


class _MutableTemporalProfiles:
    source_scope = "selected_user_interactions"

    def __init__(self) -> None:
        self.events = {
            "a": [(datetime(2021, 1, 1), 0.9)],
            "b": [(datetime(2021, 1, 1), 0.1)],
        }

    def get(self, business_ids: list[str], cutoff_time: datetime):
        raise AssertionError("Query retrieval must not build complete business profiles")

    def get_aspects(
        self,
        business_ids: list[str],
        aspects: list[str],
        cutoff_time: datetime,
    ):
        profiles = {}
        for business_id in business_ids:
            values = [
                value
                for event_time, value in self.events[business_id]
                if event_time < cutoff_time
            ]
            ratio = sum(values) / len(values) if values else None
            profiles[business_id] = {
                    "quiet_environment": SimpleNamespace(
                        status="known" if ratio is not None else "unknown",
                        weighted_positive_ratio=ratio,
                        weighted_negative_ratio=(None if ratio is None else 1 - ratio),
                        confidence=0.8 if ratio is not None else 0.0,
                    )
                }
        return profiles


class _EmbeddingMatcher:
    def __init__(self) -> None:
        self.query_text: str | None = None
        self.business_ids: list[str] | None = None

    def match(
        self,
        *,
        query_text: str,
        business_ids: list[str],
        cutoff_time: datetime,
        usage_scope: str,
    ):
        self.query_text = query_text
        self.business_ids = list(business_ids)
        scores = {"semantic-first": 0.9, "semantic-second": 0.6}
        ordered = sorted(business_ids, key=lambda value: (-scores[value], value))
        return SimpleNamespace(
            matches=[
                SimpleNamespace(
                    business_id=business_id,
                    normalized_score=scores[business_id],
                    semantic_rank=rank,
                )
                for rank, business_id in enumerate(ordered, start=1)
            ],
            usage=SimpleNamespace(
                input_tokens=11,
                logical_input_tokens=35,
                cache_hits=2,
                cache_misses=1,
                api_calls=0,
            ),
        )


class _CustomCatalog:
    def __init__(
        self,
        rows: tuple[BusinessRecord, ...],
        counts: tuple[int, ...],
    ) -> None:
        self.rows = rows
        self.counts = counts

    def businesses(self) -> tuple[BusinessRecord, ...]:
        return self.rows

    def review_catalog_before(self, cutoff_time: datetime) -> ReviewCatalogSnapshot:
        return ReviewCatalogSnapshot(
            business_ids=tuple(row.business_id for row in self.rows),
            review_counts=self.counts,
            star_sums=tuple(float(count * 4) for count in self.counts),
            global_count=sum(self.counts),
            global_star_sum=float(sum(self.counts) * 4),
        )


def _condition(
    *,
    field: str,
    operator: str,
    value: str | int | float | bool,
    enforcement: str,
) -> RequestCondition:
    span = str(value)
    return RequestCondition(
        field=field,
        operator=operator,
        value=value,
        importance="mandatory" if enforcement == "filter" else "preferred",
        enforcement=enforcement,
        explicit=True,
        confidence=1.0,
        evidence_span=span,
        evidence_start=0,
        evidence_end=len(span),
        source="rule",
        unknown_policy="exclude" if enforcement == "filter" else "not_applicable",
    )


def _request(
    *conditions: RequestCondition,
    location_center: LocationCenter | None = None,
) -> RecommendationRequest:
    return RecommendationRequest(
        request_id="a" * 64,
        user_id="user-1",
        session_id="session-1",
        cutoff_time=datetime(2022, 1, 1),
        query_text="I only want a steakhouse",
        intent="recommendation_request",
        conditions=list(conditions),
        location_center=location_center,
        parser_version="test-v1",
    )


def _config(**updates: object) -> QueryRetrievalConfig:
    values: dict[str, object] = {
        "schema_version": 1,
        "agent_version": "step31-dual-channel-query-recall-v1",
        "enabled": True,
        "candidate_limit": 10,
        "per_route_limit": 10,
        "rrf_constant": 60.0,
        "category_weight": 1.0,
        "embedding_weight": 1.0,
        "aspect_weight": 1.0,
        "location_weight": 1.0,
        "location_scale_km": 10.0,
        "dual_history_weight": 1.0,
        "dual_query_weight": 1.0,
        "aspect_source_scope": "selected_user_interactions",
    }
    values.update(updates)
    return QueryRetrievalConfig.model_validate(values)


def test_project_query_retrieval_config_freezes_selected_user_aspect_scope() -> None:
    config = load_query_retrieval_config("configs/query_retrieval.yaml")

    assert config.enabled is True
    assert config.agent_version == "step31-dual-channel-query-recall-v1"
    assert config.candidate_limit == 500
    assert config.per_route_limit == 500
    assert config.aspect_source_scope == "selected_user_interactions"
    assert config.dual_history_weight == 1.0
    assert config.dual_query_weight == 1.0


def test_category_query_retrieves_only_matching_pre_cutoff_businesses() -> None:
    retriever = QueryCandidateRetriever(
        catalog=_Catalog(),
        profiles=_Profiles(),
        embedding_matcher=None,
        config=_config(),
    )
    task = QueryRetrievalTask(
        request=_request(
            _condition(
                field="category",
                operator="includes",
                value="Steakhouses",
                enforcement="filter",
            )
        ),
        usage_scope="category-test",
    )

    result = retriever.retrieve(task)

    assert result.catalog_size == 3
    assert result.pre_cutoff_business_count == 2
    assert result.eligible_business_count == 1
    assert [item.business_id for item in result.candidates] == ["old-steak"]
    assert result.route_result_counts == {
        "query_category": 1,
        "query_embedding": 0,
        "query_aspect": 0,
        "query_location": 0,
    }
    assert result.candidates[0].category_rank == 1
    assert result.candidates[0].source_scope == "selected_user_interactions"


def test_soft_category_route_does_not_fill_its_tail_with_unrelated_businesses() -> None:
    retriever = QueryCandidateRetriever(
        catalog=_Catalog(),
        profiles=_Profiles(),
        embedding_matcher=None,
        config=_config(),
    )

    result = retriever.retrieve(
        QueryRetrievalTask(
            request=_request(
                _condition(
                    field="category",
                    operator="includes",
                    value="Steakhouses",
                    enforcement="rank",
                )
            ),
            usage_scope="soft-category-test",
        )
    )

    assert [item.business_id for item in result.candidates] == ["old-steak"]
    assert result.eligible_business_count == 2
    assert result.route_result_counts["query_category"] == 1


def test_every_mandatory_included_category_must_be_present() -> None:
    rows = (
        _business("steak-only", "Steakhouses"),
        _business("restaurant-steak", "Restaurants", "Steakhouses"),
    )
    retriever = QueryCandidateRetriever(
        catalog=_CustomCatalog(rows, (2, 2)),
        profiles=_Profiles(),
        embedding_matcher=None,
        config=_config(),
    )

    result = retriever.retrieve(
        QueryRetrievalTask(
            request=_request(
                _condition(
                    field="category",
                    operator="includes",
                    value="Restaurants",
                    enforcement="filter",
                ),
                _condition(
                    field="category",
                    operator="includes",
                    value="Steakhouses",
                    enforcement="filter",
                ),
            ),
            usage_scope="multiple-hard-categories-test",
        )
    )

    assert [item.business_id for item in result.candidates] == [
        "restaurant-steak"
    ]
    assert [item.model_dump() for item in result.excluded] == [
        {
            "business_id": "steak-only",
            "reason_codes": ["HARD_CATEGORY_MISSING:Restaurants"],
        }
    ]


def test_distance_and_price_filters_run_before_current_location_recall() -> None:
    rows = (
        _business(
            "eligible",
            "Restaurants",
            latitude=39.9526,
            longitude=-75.1652,
            attributes_json='{"RestaurantsPriceRange2":"2"}',
        ),
        _business(
            "too-far",
            "Restaurants",
            latitude=40.05,
            longitude=-75.1652,
            attributes_json='{"RestaurantsPriceRange2":"2"}',
        ),
        _business(
            "too-expensive",
            "Restaurants",
            latitude=39.9527,
            longitude=-75.1652,
            attributes_json='{"RestaurantsPriceRange2":"4"}',
        ),
        _business(
            "unknown-location",
            "Restaurants",
            latitude=None,
            longitude=None,
            attributes_json='{"RestaurantsPriceRange2":"2"}',
        ),
    )
    retriever = QueryCandidateRetriever(
        catalog=_CustomCatalog(rows, (2, 2, 2, 2)),
        profiles=_Profiles(),
        embedding_matcher=None,
        config=_config(),
    )
    task = QueryRetrievalTask(
        request=_request(
            _condition(
                field="distance_km",
                operator="less_than_or_equal",
                value=5.0,
                enforcement="filter",
            ),
            _condition(
                field="price_level",
                operator="less_than_or_equal",
                value=2,
                enforcement="filter",
            ),
            location_center=LocationCenter(latitude=39.9526, longitude=-75.1652),
        ),
        usage_scope="distance-test",
    )

    result = retriever.retrieve(task)

    assert [item.business_id for item in result.candidates] == ["eligible"]
    assert result.candidates[0].location_rank == 1
    reasons = {item.business_id: item.reason_codes for item in result.excluded}
    assert reasons == {
        "too-expensive": ["HARD_PRICE_LEVEL_EXCEEDED"],
        "too-far": ["HARD_DISTANCE_EXCEEDED"],
        "unknown-location": ["HARD_DISTANCE_BUSINESS_LOCATION_UNKNOWN"],
    }


def test_aspect_recall_uses_only_selected_user_evidence_and_keeps_unknown_neutral() -> None:
    rows = tuple(
        _business(business_id, "Restaurants")
        for business_id in (
            "quiet-positive",
            "quiet-negative",
            "quiet-unknown",
        )
    )
    retriever = QueryCandidateRetriever(
        catalog=_CustomCatalog(rows, (3, 3, 3)),
        profiles=_AspectProfiles(),
        embedding_matcher=None,
        config=_config(),
    )
    task = QueryRetrievalTask(
        request=_request(
            _condition(
                field="quiet_environment",
                operator="prefer",
                value=True,
                enforcement="rank",
            ),
            location_center=LocationCenter(latitude=39.95, longitude=-75.16),
        ),
        usage_scope="aspect-test",
    )

    result = retriever.retrieve(task)

    assert [item.business_id for item in result.candidates] == [
        "quiet-positive",
        "quiet-negative",
        "quiet-unknown",
    ]
    assert result.eligible_business_count == 3
    assert not result.excluded
    assert result.candidates[0].aspect_score > 0.5
    assert result.route_result_counts["query_aspect"] == 1
    unknown = next(
        item for item in result.candidates if item.business_id == "quiet-unknown"
    )
    assert unknown.aspect_rank is None
    assert unknown.unknown_fields == ["quiet_environment"]
    assert result.warnings == [
        "ASPECT_COVERAGE_LIMITED_TO_SELECTED_5000_USERS",
        "ASPECT_EVIDENCE_UNKNOWN:quiet_environment:1",
    ]


def test_mandatory_aspect_excludes_known_mismatch_and_respects_unknown_policy() -> None:
    rows = tuple(
        _business(business_id, "Restaurants")
        for business_id in (
            "quiet-positive",
            "quiet-negative",
            "quiet-unknown",
        )
    )
    retriever = QueryCandidateRetriever(
        catalog=_CustomCatalog(rows, (3, 3, 3)),
        profiles=_AspectProfiles(),
        embedding_matcher=None,
        config=_config(),
    )

    result = retriever.retrieve(
        QueryRetrievalTask(
            request=_request(
                _condition(
                    field="quiet_environment",
                    operator="equals",
                    value=True,
                    enforcement="filter",
                )
            ),
            usage_scope="hard-aspect-test",
        )
    )

    assert [item.business_id for item in result.candidates] == ["quiet-positive"]
    reasons = {item.business_id: item.reason_codes for item in result.excluded}
    assert reasons == {
        "quiet-negative": ["HARD_ASPECT_MISMATCH:quiet_environment"],
        "quiet-unknown": ["HARD_ASPECT_UNKNOWN:quiet_environment"],
    }


def test_future_aspect_evidence_cannot_change_an_existing_cutoff_result() -> None:
    profiles = _MutableTemporalProfiles()
    retriever = QueryCandidateRetriever(
        catalog=_CustomCatalog(
            (_business("a", "Restaurants"), _business("b", "Restaurants")),
            (2, 2),
        ),
        profiles=profiles,
        embedding_matcher=None,
        config=_config(),
    )
    task = QueryRetrievalTask(
        request=_request(
            _condition(
                field="quiet_environment",
                operator="prefer",
                value=True,
                enforcement="rank",
            )
        ),
        usage_scope="leakage-test",
    )

    before = retriever.retrieve(task)
    profiles.events["b"].append((datetime(2023, 1, 1), 1.0))
    after = retriever.retrieve(task)

    assert before.candidates == after.candidates
    assert before.route_result_counts == after.route_result_counts
    assert before.excluded == after.excluded


def test_embedding_route_scores_the_complete_pre_cutoff_scope_with_structured_intent() -> None:
    rows = (
        _business("semantic-first", "Restaurants"),
        _business("semantic-second", "Restaurants"),
    )
    matcher = _EmbeddingMatcher()
    retriever = QueryCandidateRetriever(
        catalog=_CustomCatalog(rows, (2, 2)),
        profiles=_Profiles(),
        embedding_matcher=matcher,
        config=_config(),
    )
    task = QueryRetrievalTask(
        request=_request(
            _condition(
                field="budget_per_person",
                operator="prefer",
                value=50,
                enforcement="rank",
            )
        ),
        usage_scope="embedding-test",
    )

    result = retriever.retrieve(task)

    assert [item.business_id for item in result.candidates] == [
        "semantic-first",
        "semantic-second",
    ]
    assert matcher.business_ids == ["semantic-first", "semantic-second"]
    assert "Original request: I only want a steakhouse" in str(matcher.query_text)
    assert "budget per person prefer 50" in str(matcher.query_text)
    assert result.candidates[0].embedding_rank == 1
    assert result.usage.embedding_input_tokens == 11
    assert result.usage.embedding_logical_tokens == 35
    assert result.usage.provider_calls == 0


def _history_candidate(business_id: str, rank: int) -> RetrievalCandidate:
    return RetrievalCandidate(
        business_id=business_id,
        rank=rank,
        fusion_score=1.0 / (60 + rank),
        route_count=1,
        quality_rank=rank,
        quality_score=0.8,
        category_rank=None,
        category_score=None,
        text_rank=None,
        text_score=None,
        location_rank=None,
        location_score=0.5,
        distance_km=None,
        item_knn_rank=None,
        item_knn_positive_score=0.0,
        item_knn_negative_evidence=0.0,
        item_knn_positive_support_count=0,
        item_knn_negative_support_count=0,
        item_knn_positive_neighbor_count=0,
        item_knn_negative_neighbor_count=0,
        item_knn_missing=True,
    )


def _history_result(*business_ids: str) -> RetrievalResult:
    return RetrievalResult(
        task=RetrievalTaskContext(
            task_id="a" * 64,
            split="development",
            user_id="user-1",
            cutoff_time=datetime(2022, 1, 1),
            history_count=3,
        ),
        catalog_size=10,
        eligible_candidate_count=8,
        excluded_history_businesses=2,
        candidates=tuple(
            _history_candidate(business_id, rank)
            for rank, business_id in enumerate(business_ids, start=1)
        ),
        route_candidates=(),
        route_result_counts={
            "quality": len(business_ids),
            "category": 0,
            "text": 0,
            "location": 0,
            "item_knn": 0,
        },
        latency_ms=1.0,
    )


def test_dual_channel_fusion_keeps_unique_candidates_and_rewards_consensus() -> None:
    query = QueryCandidateRetriever(
        catalog=_CustomCatalog(
            (
                _business("shared", "Steakhouses"),
                _business("query-only", "Steakhouses"),
            ),
            (2, 2),
        ),
        profiles=_Profiles(),
        embedding_matcher=None,
        config=_config(),
    ).retrieve(
        QueryRetrievalTask(
            request=_request(
                _condition(
                    field="category",
                    operator="includes",
                    value="Steakhouses",
                    enforcement="filter",
                )
            ),
            usage_scope="dual-test",
        )
    )

    result = DualChannelFusion(_config()).fuse(
        _history_result("history-only", "shared"),
        query,
    )

    assert [item.business_id for item in result.candidates] == [
        "shared",
        "history-only",
        "query-only",
    ]
    assert result.candidates[0].channel_count == 2
    assert result.candidates[0].history_rank == 2
    assert result.candidates[0].query_rank == 2
    assert result.candidates[1].history_rank == 1
    assert result.candidates[1].query_rank is None
    assert result.candidates[2].history_rank is None
    assert result.candidates[2].query_rank == 1


def test_dual_channel_fusion_preserves_history_order_when_query_is_empty() -> None:
    query = QueryCandidateRetriever(
        catalog=_CustomCatalog((_business("sushi", "Sushi Bars"),), (2,)),
        profiles=_Profiles(),
        embedding_matcher=None,
        config=_config(),
    ).retrieve(
        QueryRetrievalTask(
            request=_request(),
            usage_scope="empty-query-test",
        )
    )

    result = DualChannelFusion(_config()).fuse(
        _history_result("history-a", "history-b"),
        query,
    )

    assert [item.business_id for item in result.candidates] == [
        "history-a",
        "history-b",
    ]


class _HistoryRetriever:
    def retrieve(self, task, *, include_route_provenance=False):
        assert include_route_provenance is True
        return _history_result("history-only", "shared")


class _HistoryReader:
    def user_history(self, user_id: str, cutoff_time: datetime):
        return (object(), object(), object())


class _FailingQueryRetriever:
    def retrieve(self, task):
        raise RuntimeError("simulated Query route failure")


def test_agent_candidate_tool_returns_dual_channel_scope_and_query_provenance() -> None:
    request = _request(
        _condition(
            field="category",
            operator="includes",
            value="Steakhouses",
            enforcement="filter",
        )
    )
    query_retriever = QueryCandidateRetriever(
        catalog=_CustomCatalog(
            (
                _business("shared", "Steakhouses"),
                _business("query-only", "Steakhouses"),
            ),
            (2, 2),
        ),
        profiles=_Profiles(),
        embedding_matcher=None,
        config=_config(),
    )
    registry = AgentToolRegistry(
        [
            ExpandCandidatesTool(
                _HistoryRetriever(),
                _HistoryReader(),
                query_retriever=query_retriever,
                dual_fusion=DualChannelFusion(_config()),
            )
        ]
    )
    context = ToolExecutionContext(
        request_id=request.request_id,
        user_id=request.user_id,
        cutoff_time=request.cutoff_time,
        action="retrieve_candidates",
        state_snapshot={
            "split": "development",
            "request": request.model_dump(mode="json", exclude_computed_fields=True),
        },
    )

    result = registry.execute("EXPAND_CANDIDATES", {}, context)

    assert result.status == "success"
    assert result.data["retrieval_mode"] == "dual_channel"
    assert result.data["candidate_business_ids"] == [
        "shared",
        "history-only",
        "query-only",
    ]
    assert result.data["history_candidate_count"] == 2
    assert result.data["query_candidate_count"] == 2
    assert result.data["overlap_count"] == 1
    shared = result.data["candidates"][0]
    assert shared["history_rank"] == 2
    assert shared["query_rank"] == 2
    assert shared["query_category_rank"] == 2


def test_agent_candidate_tool_falls_back_to_exact_history_order_on_query_failure() -> None:
    registry = AgentToolRegistry(
        [
            ExpandCandidatesTool(
                _HistoryRetriever(),
                _HistoryReader(),
                query_retriever=_FailingQueryRetriever(),  # type: ignore[arg-type]
                dual_fusion=DualChannelFusion(_config()),
            )
        ]
    )
    request = _request()
    context = ToolExecutionContext(
        request_id=request.request_id,
        user_id=request.user_id,
        cutoff_time=request.cutoff_time,
        action="retrieve_candidates",
        state_snapshot={
            "split": "development",
            "request": request.model_dump(mode="json", exclude_computed_fields=True),
        },
    )

    result = registry.execute("EXPAND_CANDIDATES", {}, context)

    assert result.status == "success"
    assert result.data["retrieval_mode"] == "history_fallback"
    assert result.data["candidate_business_ids"] == ["history-only", "shared"]
    assert result.warnings == ["QUERY_RETRIEVAL_FALLBACK:RuntimeError"]
