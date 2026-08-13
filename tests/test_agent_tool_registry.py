"""Public-interface tests for the Step 23 Agent tool registry."""

from __future__ import annotations

from datetime import datetime
from dataclasses import asdict
from pathlib import Path

from pydantic import Field

from yelp_agent.agent_tools import (
    AgentToolRegistry,
    GetUserProfileTool,
    GetSessionMemoryTool,
    GetBusinessDetailsTool,
    GetBusinessProfileTool,
    ExpandCandidatesTool,
    ApplyConstraintsTool,
    CompareBusinessesTool,
    GetHybridRankingTool,
    OnlineHybridV2RankingService,
    build_step23_tool_registry,
    HybridV2FallbackHandler,
    load_agent_tool_runtime_config,
    RegistryActionExecutor,
    ToolDefinition,
    ToolExecutionContext,
    ToolObservation,
    UnavailableTool,
)
from yelp_agent.agent_benchmark import VisibleAgentScenario
from yelp_agent.business_profiles.schema import BusinessRatingEvent
from yelp_agent.business_profiles.store import BusinessKnowledgeStore
from yelp_agent.config import load_business_profile_config
from yelp_agent.data.temporal_view import BusinessRecord
from yelp_agent.agent_harness import (
    ActionOutcome,
    AgentDecision,
    AgentHarness,
    FakeClock,
    RuleBasedRequestInterpreter,
)
from yelp_agent.models import StrictModel
from yelp_agent.profiles.schema import ProfileEvidenceSummary, UserProfileV1
from yelp_agent.query import QueryParseInput, build_rule_based_request_parser
from yelp_agent.query.ranking import QueryAwareCandidate
from yelp_agent.learning_to_rank.features import ALL_FEATURE_NAMES, HybridV1Weights
from yelp_agent.learning_to_rank.runtime import HybridV2ScoredCandidate
from yelp_agent.retrieval import (
    RetrievalCandidate,
    RetrievalResult,
    RetrievalTaskContext,
)

PROJECT_CONFIG_DIR = Path(__file__).parents[1] / "configs"


class _EchoInput(StrictModel):
    text: str = Field(min_length=1)


class _EchoOutput(StrictModel):
    normalized_text: str = Field(min_length=1)


class _EchoTool:
    definition = ToolDefinition(
        name="ECHO_TEXT",
        version="1",
        kind="deterministic",
        allowed_actions=("get_business_details",),
        input_model=_EchoInput,
        output_model=_EchoOutput,
        public_summary="Normalize one visible text value.",
    )

    def run(
        self,
        arguments: _EchoInput,
        context: ToolExecutionContext,
    ) -> ToolObservation:
        del context
        return ToolObservation.success(
            tool_name=self.definition.name,
            data={"normalized_text": arguments.text.strip().lower()},
            confidence=1.0,
        )


def _context() -> ToolExecutionContext:
    return ToolExecutionContext(
        request_id="a" * 64,
        user_id="user-1",
        cutoff_time="2022-01-01T00:00:00",
        action="get_business_details",
        business_scope=("business-1",),
        business_scope_known=True,
        state_snapshot={"turn": 1},
    )


def test_registry_executes_a_registered_tool_through_one_public_interface() -> None:
    registry = AgentToolRegistry([_EchoTool()])

    result = registry.execute(
        "ECHO_TEXT",
        {"text": "  HELLO  "},
        _context(),
    )

    assert result.tool_name == "ECHO_TEXT"
    assert result.status == "success"
    assert result.data == {"normalized_text": "hello"}
    assert result.confidence == 1.0
    assert result.latency_ms >= 0
    assert registry.describe("ECHO_TEXT").public_summary == (
        "Normalize one visible text value."
    )


def test_registry_rejects_a_tool_that_does_not_match_the_current_action() -> None:
    registry = AgentToolRegistry([_EchoTool()])
    context = _context().model_copy(update={"action": "rank_candidates"})

    result = registry.execute("ECHO_TEXT", {"text": "hello"}, context)

    assert result.status == "permanent_error"
    assert result.error_code == "ACTION_NOT_ALLOWED"


def test_future_tool_is_described_but_cannot_be_executed_early() -> None:
    tool = UnavailableTool(
        ToolDefinition(
            name="SEARCH_BUSINESS_REVIEWS",
            version="planned-step-27",
            kind="review_rag",
            allowed_actions=("retrieve_business_reviews",),
            input_model=_EchoInput,
            output_model=_EchoOutput,
            public_summary="Search point-in-time reviews for one business.",
            availability={"available": False, "reason": "planned_for_step_27"},
        )
    )
    registry = AgentToolRegistry([tool])
    context = _context().model_copy(update={"action": "retrieve_business_reviews"})

    result = registry.execute(
        "SEARCH_BUSINESS_REVIEWS",
        {"text": "quiet"},
        context,
    )

    assert result.status == "unavailable"
    assert result.error_code == "TOOL_NOT_IMPLEMENTED"
    assert registry.describe("SEARCH_BUSINESS_REVIEWS").availability.reason == (
        "planned_for_step_27"
    )


class _ToolThenReturnPolicy:
    def allowed_actions(self, state):
        return ("get_business_details", "return_recommendation")


class _ToolThenReturnRouter:
    def choose_action(self, state):
        if state.step_count == 0:
            return AgentDecision(
                action="get_business_details",
                arguments={"text": " BUSINESS-1 "},
                reason_code="DETAILS_REQUIRED",
                tool_name="ECHO_TEXT",
                tool_kind="deterministic",
            )
        return AgentDecision(
            action="return_recommendation",
            reason_code="READY_TO_FINALIZE",
        )


class _TerminalExecutor:
    def execute(self, state, decision):
        del state, decision
        return ActionOutcome(
            status="completed",
            response_kind="recommendation",
            candidate_ranking=["business-1"],
            recommended_business_ids=["business-1"],
        )


def test_registry_executor_connects_a_tool_to_the_controlled_harness() -> None:
    registry = AgentToolRegistry([_EchoTool()])
    harness = AgentHarness(
        agent_version="step23-test",
        interpreter=RuleBasedRequestInterpreter(),
        action_policy=_ToolThenReturnPolicy(),
        router=_ToolThenReturnRouter(),
        executor=RegistryActionExecutor(registry, fallback=_TerminalExecutor()),
        clock=FakeClock(),
    )
    scenario = VisibleAgentScenario(
        scenario_id="b" * 64,
        split="development",
        language="en-US",
        user_id="user-1",
        session_id="session-1",
        cutoff_time=datetime(2022, 1, 1),
        query_text="Recommend a restaurant",
    )

    result = harness.start(scenario)

    assert result.run is not None
    assert result.run.fallback is False
    assert result.session.observations[0].payload["status"] == "success"
    assert result.session.observations[0].payload["data"] == {
        "normalized_text": "business-1"
    }
    assert result.run.turns[0].tool_calls[0].tool_name == "ECHO_TEXT"


def test_invalid_tool_arguments_become_a_structured_error() -> None:
    registry = AgentToolRegistry([_EchoTool()])

    result = registry.execute("ECHO_TEXT", {"unknown": "value"}, _context())

    assert result.status == "permanent_error"
    assert result.error_code == "INVALID_ARGUMENTS"
    assert result.data == {}


def test_unknown_tool_name_becomes_a_structured_error() -> None:
    registry = AgentToolRegistry([_EchoTool()])

    result = registry.execute("MISSING_TOOL", {}, _context())

    assert result.status == "permanent_error"
    assert result.error_code == "UNKNOWN_TOOL"


class _WrongIdentityTool(_EchoTool):
    def run(self, arguments, context):
        return ToolObservation.success(
            tool_name="WRONG_TOOL",
            data={"normalized_text": arguments.text},
        )


def test_malformed_tool_identity_becomes_a_structured_error() -> None:
    registry = AgentToolRegistry([_WrongIdentityTool()])

    result = registry.execute("ECHO_TEXT", {"text": "hello"}, _context())

    assert result.status == "permanent_error"
    assert result.error_code == "MALFORMED_TOOL_OUTPUT"


class _ProfileStore:
    def __init__(self, profile: UserProfileV1) -> None:
        self.profile = profile
        self.calls = []

    def get(self, user_id, cutoff_time):
        self.calls.append((user_id, cutoff_time))
        return self.profile


def _profile() -> UserProfileV1:
    return UserProfileV1(
        profile_id="c" * 64,
        user_id="user-1",
        cutoff_time=datetime(2022, 1, 1),
        history_length=1,
        average_rating=5,
        rating_distribution={"1": 0, "2": 0, "3": 0, "4": 0, "5": 1},
        category_preferences=[],
        category_dislikes=[],
        aspect_preferences=[],
        aspect_dislikes=[],
        frequent_areas=[],
        reliability=0.4,
        evidence_summary=ProfileEvidenceSummary(
            category_evidence_count=0,
            aspect_evidence_count=0,
            price_evidence_count=0,
            area_evidence_count=0,
            first_interaction=datetime(2021, 1, 1),
            last_interaction=datetime(2021, 1, 1),
        ),
        profile_version="1.0.0",
    )


def test_user_profile_tool_uses_session_identity_and_exact_cutoff() -> None:
    store = _ProfileStore(_profile())
    registry = AgentToolRegistry([GetUserProfileTool(store)])
    context = _context().model_copy(update={"action": "rank_candidates"})

    result = registry.execute("GET_USER_PROFILE", {}, context)

    assert result.status == "success"
    assert result.data["profile"]["profile_id"] == "c" * 64
    assert store.calls == [("user-1", datetime(2022, 1, 1))]


def test_session_memory_tool_returns_only_visible_prior_observations() -> None:
    registry = AgentToolRegistry([GetSessionMemoryTool()])
    context = _context().model_copy(
        update={
            "action": "apply_feedback",
            "state_snapshot": {
                "session_id": "session-1",
                "turn_index": 2,
                "observations": [{"payload": {"candidate_business_ids": ["b1"]}}],
            },
        }
    )

    result = registry.execute("GET_SESSION_MEMORY", {}, context)

    assert result.status == "success"
    assert result.data == {
        "session_id": "session-1",
        "turn_index": 2,
        "observations": [{"payload": {"candidate_business_ids": ["b1"]}}],
        "memory_context": None,
        "effective_request": None,
    }


def test_session_memory_tool_prefers_compact_authoritative_context() -> None:
    registry = AgentToolRegistry([GetSessionMemoryTool()])
    context = _context().model_copy(
        update={
            "action": "apply_feedback",
            "state_snapshot": {
                "session_id": "session-1",
                "turn_index": 2,
                "observations": [{"payload": {"large": "legacy"}}],
                "memory_context": {
                    "revision": 2,
                    "task_type": "feedback_refinement",
                    "hard_constraints": [],
                    "soft_preferences": [],
                    "information_gaps": [],
                    "rejected_business_ids": ["b1"],
                    "last_presented_business_ids": ["b1", "b2"],
                    "current_business_scope": ["b1", "b2"],
                    "business_scope_known": True,
                    "clarification_answers": {},
                    "semantic_summary": "Find another restaurant.",
                    "recent_turn_summaries": ["turn=2; task=feedback_refinement"],
                },
            },
        }
    )

    result = registry.execute("GET_SESSION_MEMORY", {}, context)

    assert result.status == "success"
    assert result.data["observations"] == []
    assert result.data["memory_context"]["rejected_business_ids"] == ["b1"]


class _BusinessStoreThatMustNotRun:
    def get(self, business_ids, cutoff_time):
        raise AssertionError("out-of-scope IDs must be rejected before store access")


def test_business_tool_rejects_ids_outside_the_current_candidate_scope() -> None:
    registry = AgentToolRegistry(
        [GetBusinessDetailsTool(_BusinessStoreThatMustNotRun())]
    )

    result = registry.execute(
        "GET_BUSINESS_DETAILS",
        {"business_ids": ["outside-business"]},
        _context(),
    )

    assert result.status == "permanent_error"
    assert result.error_code == "BUSINESS_OUT_OF_SCOPE"


def test_known_empty_scope_rejects_every_business_id() -> None:
    registry = AgentToolRegistry(
        [GetBusinessDetailsTool(_BusinessStoreThatMustNotRun())]
    )
    context = _context().model_copy(update={"business_scope": ()})

    result = registry.execute(
        "GET_BUSINESS_DETAILS",
        {"business_ids": ["business-1"]},
        context,
    )

    assert result.status == "permanent_error"
    assert result.error_code == "BUSINESS_OUT_OF_SCOPE"


def _business_store() -> BusinessKnowledgeStore:
    return BusinessKnowledgeStore.from_records(
        businesses=(
            BusinessRecord(
                business_id="business-1",
                name="Test Steakhouse",
                address="1 Test Street",
                city="Philadelphia",
                state="PA",
                postal_code="19107",
                latitude=39.95,
                longitude=-75.16,
                categories=("Restaurants", "Steakhouses"),
                attributes_json='{"RestaurantsPriceRange2":"3"}',
            ),
        ),
        rating_events=(
            BusinessRatingEvent(
                review_id="review-1",
                business_id="business-1",
                user_id="reviewer-1",
                stars=5.0,
                review_time=datetime(2021, 1, 1),
            ),
        ),
        aspect_events=(),
        config=load_business_profile_config(PROJECT_CONFIG_DIR),
    )


def test_business_details_tool_returns_cutoff_safe_static_and_quality_data() -> None:
    registry = AgentToolRegistry([GetBusinessDetailsTool(_business_store())])

    result = registry.execute(
        "GET_BUSINESS_DETAILS",
        {"business_ids": ["business-1"]},
        _context(),
    )

    assert result.status == "success"
    detail = result.data["businesses"][0]
    assert detail["name"] == "Test Steakhouse"
    assert detail["review_count"] == 1
    assert detail["structured_attributes"] == {"RestaurantsPriceRange2": "3"}


def test_business_profile_tool_returns_aggregates_without_raw_review_text() -> None:
    registry = AgentToolRegistry([GetBusinessProfileTool(_business_store())])

    result = registry.execute(
        "GET_BUSINESS_PROFILE",
        {"business_ids": ["business-1"]},
        _context(),
    )

    profile = result.data["profiles"][0]
    assert profile["business_id"] == "business-1"
    assert "aspect_summaries" in profile
    assert "review_text" not in str(profile)


class _HistoryReader:
    def user_history(self, user_id, cutoff_time):
        assert (user_id, cutoff_time) == ("user-1", datetime(2022, 1, 1))
        return (object(), object(), object())


class _Retriever:
    def __init__(self) -> None:
        self.task = None

    def retrieve(self, task, *, include_route_provenance=False):
        self.task = task
        assert include_route_provenance is True
        candidate = RetrievalCandidate(
            business_id="business-1",
            rank=1,
            fusion_score=0.05,
            route_count=2,
            quality_rank=1,
            quality_score=0.8,
            category_rank=2,
            category_score=0.7,
            text_rank=None,
            text_score=0.5,
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
        return RetrievalResult(
            task=task,
            catalog_size=10,
            eligible_candidate_count=7,
            excluded_history_businesses=3,
            candidates=(candidate,),
            route_candidates=(),
            route_result_counts={
                "quality": 1,
                "category": 1,
                "text": 0,
                "location": 0,
                "item_knn": 0,
            },
            latency_ms=4.0,
        )


def test_candidate_tool_builds_a_label_free_runtime_retrieval_context() -> None:
    retriever = _Retriever()
    registry = AgentToolRegistry(
        [ExpandCandidatesTool(retriever, _HistoryReader())]
    )
    context = _context().model_copy(
        update={"action": "retrieve_candidates", "business_scope": ()}
    )

    result = registry.execute("EXPAND_CANDIDATES", {}, context)

    assert result.status == "success"
    assert result.data["candidate_business_ids"] == ["business-1"]
    assert result.data["candidates"][0]["fusion_score"] == 0.05
    assert retriever.task == RetrievalTaskContext(
        task_id="a" * 64,
        split="development",
        user_id="user-1",
        cutoff_time=datetime(2022, 1, 1),
        history_count=3,
    )


class _ConstraintCandidateReader:
    def get_candidates(self, business_ids, cutoff_time):
        assert cutoff_time == datetime(2022, 1, 1)
        by_id = {
            "eligible": QueryAwareCandidate(
                business_id="eligible",
                hybrid_rank=1,
                categories=("Steakhouses",),
            ),
            "wrong": QueryAwareCandidate(
                business_id="wrong",
                hybrid_rank=2,
                categories=("Japanese",),
            ),
        }
        return [by_id[business_id] for business_id in business_ids]


def test_constraint_tool_preserves_order_and_removes_proven_failures() -> None:
    request = build_rule_based_request_parser().parse(
        QueryParseInput(
            user_id="user-1",
            session_id="session-1",
            cutoff_time=datetime(2022, 1, 1),
            query_text="I only want a steakhouse",
        )
    )
    context = _context().model_copy(
        update={
            "action": "apply_hard_constraints",
            "business_scope": ("eligible", "wrong"),
            "state_snapshot": {"request": request.model_dump(mode="json")},
        }
    )
    registry = AgentToolRegistry(
        [ApplyConstraintsTool(_ConstraintCandidateReader())]
    )

    result = registry.execute(
        "APPLY_CONSTRAINTS",
        {"business_ids": ["eligible", "wrong"]},
        context,
    )

    assert result.status == "success"
    assert result.data["candidate_business_ids"] == ["eligible"]
    assert result.data["excluded"] == [
        {"business_id": "wrong", "reason_codes": ["HARD_CATEGORY_MISSING:Steakhouses"]}
    ]


def test_compare_tool_uses_the_same_query_rules_as_static_ranking() -> None:
    request = build_rule_based_request_parser().parse(
        QueryParseInput(
            user_id="user-1",
            session_id="session-1",
            cutoff_time=datetime(2022, 1, 1),
            query_text="I only want a steakhouse",
        )
    )
    registry = AgentToolRegistry(
        [CompareBusinessesTool(_ConstraintCandidateReader())]
    )
    context = _context().model_copy(
        update={
            "action": "compare_candidates",
            "business_scope": ("wrong", "eligible"),
            "state_snapshot": {"request": request.model_dump(mode="json")},
        }
    )

    result = registry.execute(
        "COMPARE_BUSINESSES",
        {"business_ids": ["wrong", "eligible"]},
        context,
    )

    assert result.status == "success"
    assert result.data["ranking"] == ["eligible"]
    assert result.data["excluded"][0]["business_id"] == "wrong"


class _HybridRankingService:
    def __init__(self) -> None:
        self.candidates = None

    def rank(self, *, request_id, user_id, cutoff_time, candidates):
        self.candidates = candidates
        self.request_id = request_id
        assert user_id == "user-1"
        assert cutoff_time == datetime(2022, 1, 1)
        return [
            {
                "business_id": "business-1",
                "rank": 1,
                "model_rank": 1,
                "hybrid_v1_rank": 1,
                "model_score": 0.7,
                "hybrid_v1_score": 0.6,
                "blend_score": 1.0,
            }
        ]


def test_hybrid_tool_ranks_only_scoped_rows_from_retrieval_observation() -> None:
    service = _HybridRankingService()
    registry = AgentToolRegistry([GetHybridRankingTool(service)])
    retrieval_row = _Retriever().retrieve(
        RetrievalTaskContext(
            task_id="a" * 64,
            split="development",
            user_id="user-1",
            cutoff_time=datetime(2022, 1, 1),
            history_count=3,
        ),
        include_route_provenance=True,
    ).candidates[0]
    context = _context().model_copy(
        update={
            "action": "rank_candidates",
            "state_snapshot": {
                "observations": [
                    {
                        "payload": {
                            "tool_name": "EXPAND_CANDIDATES",
                            "status": "success",
                            "data": {"candidates": [asdict(retrieval_row)]},
                        }
                    }
                ]
            },
        }
    )

    result = registry.execute(
        "GET_HYBRID_RANKING",
        {"business_ids": ["business-1"]},
        context,
    )

    assert result.status == "success"
    assert result.data["ranking"] == ["business-1"]
    assert [row["business_id"] for row in service.candidates] == ["business-1"]


class _FrozenRanker:
    def __init__(self) -> None:
        self.rows = None

    def rank(self, candidates):
        self.rows = candidates
        return (
            HybridV2ScoredCandidate(
                business_id="business-1",
                rank=1,
                model_rank=1,
                hybrid_v1_rank=1,
                model_score=0.2,
                hybrid_v1_score=float(candidates[0]["hybrid_v1_score"]),
                blend_score=1.0,
            ),
        )


def test_online_hybrid_service_builds_the_frozen_feature_contract() -> None:
    profile_store = _ProfileStore(_profile())
    ranker = _FrozenRanker()
    service = OnlineHybridV2RankingService(
        user_profiles=profile_store,
        business_profiles=_business_store(),
        ranker=ranker,
        weights=HybridV1Weights(
            category=0.4,
            text=0.3,
            quality=0.2,
            location=0.1,
        ),
        broad_categories={"Restaurants", "Food", "Nightlife", "Shopping"},
        business_profile_config=load_business_profile_config(PROJECT_CONFIG_DIR),
    )
    candidate = asdict(
        _Retriever().retrieve(
            RetrievalTaskContext(
                task_id="a" * 64,
                split="development",
                user_id="user-1",
                cutoff_time=datetime(2022, 1, 1),
                history_count=3,
            ),
            include_route_provenance=True,
        ).candidates[0]
    )

    result = service.rank(
        request_id="a" * 64,
        user_id="user-1",
        cutoff_time=datetime(2022, 1, 1),
        candidates=[candidate],
    )

    assert result[0]["business_id"] == "business-1"
    assert set(ALL_FEATURE_NAMES).issubset(ranker.rows[0])
    assert ranker.rows[0]["business_rating_count_log"] > 0
    assert ranker.rows[0]["hybrid_v1_score"] == 0.4 * 0.7 + 0.3 * 0.5 + 0.2 * 0.8 + 0.1 * 0.5


def test_step23_catalog_separates_ready_tools_from_future_capabilities() -> None:
    registry = build_step23_tool_registry(
        user_profiles=_ProfileStore(_profile()),
        business_profiles=_business_store(),
        retriever=_Retriever(),
        history_reader=_HistoryReader(),
        hybrid_ranking=_HybridRankingService(),
    )

    descriptors = {item.name: item for item in registry.list_tools()}

    assert descriptors["EXPAND_CANDIDATES"].availability.available is True
    assert descriptors["COMPUTE_EMBEDDING_MATCH"].availability.reason == (
        "planned_for_step_25"
    )
    assert descriptors["SEARCH_BUSINESS_REVIEWS"].availability.reason == (
        "planned_for_step_27"
    )
    profile_descriptor = descriptors["GET_USER_PROFILE"]
    assert profile_descriptor.input_schema["additionalProperties"] is False
    assert profile_descriptor.output_schema["properties"]["profile"]
    assert profile_descriptor.cache_scope == "request"
    assert profile_descriptor.max_attempts == 1
    assert profile_descriptor.timeout_ms == 30_000
    assert descriptors["APPLY_CONSTRAINTS"].input_schema["properties"][
        "business_ids"
    ]["maxItems"] == 500
    assert descriptors["GET_BUSINESS_DETAILS"].input_schema["properties"][
        "business_ids"
    ]["maxItems"] == 100
    assert "ASK_CLARIFICATION" not in descriptors
    assert "FINALIZE" not in descriptors


def test_agent_tool_runtime_config_is_explicit_and_bounded() -> None:
    config = load_agent_tool_runtime_config(PROJECT_CONFIG_DIR / "agent_tools.yaml")

    assert config.registry_version == "1.0.0"
    assert config.cache_max_entries == 512
    assert config.max_retry_attempts == 2
    assert config.max_timeout_ms == 90_000


def test_request_cache_reuses_a_safe_read_and_reports_cache_hit() -> None:
    store = _ProfileStore(_profile())
    registry = AgentToolRegistry([GetUserProfileTool(store)])
    context = _context().model_copy(update={"action": "rank_candidates"})

    first = registry.execute("GET_USER_PROFILE", {}, context)
    second = registry.execute("GET_USER_PROFILE", {}, context)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert store.calls == [("user-1", datetime(2022, 1, 1))]


class _RetryOnceTool(_EchoTool):
    definition = ToolDefinition(
        name="RETRY_ONCE",
        version="1",
        kind="deterministic",
        allowed_actions=("get_business_details",),
        input_model=_EchoInput,
        output_model=_EchoOutput,
        public_summary="Retry one transient failure inside one registry call.",
        max_attempts=2,
    )

    def __init__(self) -> None:
        self.calls = 0

    def run(self, arguments, context):
        self.calls += 1
        if self.calls == 1:
            return ToolObservation.error(
                tool_name=self.definition.name,
                status="retryable_error",
                error_code="TEMPORARY_FAILURE",
                warning="temporary failure",
            )
        return ToolObservation.success(
            tool_name=self.definition.name,
            data={"normalized_text": arguments.text},
        )


def test_retryable_failure_retries_inside_one_logical_tool_call() -> None:
    tool = _RetryOnceTool()
    registry = AgentToolRegistry([tool])

    result = registry.execute("RETRY_ONCE", {"text": "hello"}, _context())

    assert result.status == "success"
    assert result.attempt_count == 2
    assert tool.calls == 2


class _AlwaysRetrieveRouter:
    def choose_action(self, state):
        return AgentDecision(
            action="retrieve_candidates",
            arguments={},
            reason_code="CANDIDATES_REQUIRED",
            tool_name="EXPAND_CANDIDATES",
            tool_kind="deterministic",
        )


class _RetrievePolicy:
    def allowed_actions(self, state):
        return ("retrieve_candidates",)


def test_hybrid_fallback_reuses_retrieval_without_reintroducing_filtered_ids() -> None:
    service = _HybridRankingService()
    registry = AgentToolRegistry(
        [ExpandCandidatesTool(_Retriever(), _HistoryReader())]
    )
    harness = AgentHarness(
        agent_version="step23-fallback-test",
        interpreter=RuleBasedRequestInterpreter(),
        action_policy=_RetrievePolicy(),
        router=_AlwaysRetrieveRouter(),
        executor=RegistryActionExecutor(registry),
        fallback_handler=HybridV2FallbackHandler(service),
        clock=FakeClock(),
    )

    result = harness.start(
        VisibleAgentScenario(
            scenario_id="d" * 64,
            split="development",
            language="en-US",
            user_id="user-1",
            session_id="session-1",
            cutoff_time=datetime(2022, 1, 1),
            query_text="Recommend a restaurant",
        )
    )

    assert result.run is not None
    assert result.run.fallback is True
    assert result.run.fallback_reason == "duplicate_tool_call"
    assert result.run.turns[0].candidate_ranking == ["business-1"]


class _UnavailableReviewRouter:
    def choose_action(self, state):
        return AgentDecision(
            action="retrieve_business_reviews",
            arguments={"business_id": "business-1"},
            reason_code="REVIEW_EVIDENCE_REQUIRED",
            tool_name="SEARCH_BUSINESS_REVIEWS",
            tool_kind="review_rag",
        )


class _ReviewPolicy:
    def allowed_actions(self, state):
        return ("retrieve_business_reviews",)


class _ReviewInput(StrictModel):
    business_id: str


def test_failed_tool_keeps_its_structured_error_in_session_state() -> None:
    unavailable = UnavailableTool(
        ToolDefinition(
            name="SEARCH_BUSINESS_REVIEWS",
            version="planned-step-27",
            kind="review_rag",
            allowed_actions=("retrieve_business_reviews",),
            input_model=_ReviewInput,
            output_model=_EchoOutput,
            public_summary="Future review search.",
            availability={"available": False, "reason": "planned_for_step_27"},
        )
    )
    harness = AgentHarness(
        agent_version="step23-error-test",
        interpreter=RuleBasedRequestInterpreter(),
        action_policy=_ReviewPolicy(),
        router=_UnavailableReviewRouter(),
        executor=RegistryActionExecutor(AgentToolRegistry([unavailable])),
        clock=FakeClock(),
    )

    result = harness.start(
        VisibleAgentScenario(
            scenario_id="e" * 64,
            split="development",
            language="en-US",
            user_id="user-1",
            session_id="session-1",
            cutoff_time=datetime(2022, 1, 1),
            query_text="Is business one quiet?",
            referenced_business_ids=["business-1"],
        )
    )

    assert result.run is not None and result.run.fallback is True
    assert result.session.observations[-1].payload["status"] == "unavailable"
    assert result.session.observations[-1].payload["error_code"] == (
        "TOOL_NOT_IMPLEMENTED"
    )


class _RealChainPolicy:
    def allowed_actions(self, state):
        return (
            "retrieve_candidates",
            "apply_hard_constraints",
            "rank_candidates",
            "get_business_details",
            "return_recommendation",
        )


class _RealChainRouter:
    _decisions = (
        AgentDecision(
            action="retrieve_candidates",
            reason_code="CANDIDATES_REQUIRED",
            tool_name="EXPAND_CANDIDATES",
            tool_kind="deterministic",
        ),
        AgentDecision(
            action="apply_hard_constraints",
            arguments={"business_ids": ["business-1"]},
            reason_code="HARD_CONSTRAINT_PRESENT",
            tool_name="APPLY_CONSTRAINTS",
            tool_kind="deterministic",
        ),
        AgentDecision(
            action="rank_candidates",
            arguments={"business_ids": ["business-1"]},
            reason_code="RANKING_REQUIRED",
            tool_name="GET_HYBRID_RANKING",
            tool_kind="deterministic",
        ),
        AgentDecision(
            action="get_business_details",
            arguments={"business_ids": ["business-1"]},
            reason_code="DETAILS_REQUIRED",
            tool_name="GET_BUSINESS_DETAILS",
            tool_kind="deterministic",
        ),
        AgentDecision(
            action="return_recommendation",
            reason_code="READY_TO_FINALIZE",
        ),
    )

    def choose_action(self, state):
        return self._decisions[state.step_count]


def test_real_registry_chain_runs_inside_one_controlled_agent_session() -> None:
    registry = build_step23_tool_registry(
        user_profiles=_ProfileStore(_profile()),
        business_profiles=_business_store(),
        retriever=_Retriever(),
        history_reader=_HistoryReader(),
        hybrid_ranking=_HybridRankingService(),
    )
    harness = AgentHarness(
        agent_version="step23-integration",
        interpreter=RuleBasedRequestInterpreter(),
        action_policy=_RealChainPolicy(),
        router=_RealChainRouter(),
        executor=RegistryActionExecutor(registry, fallback=_TerminalExecutor()),
        clock=FakeClock(),
    )

    result = harness.start(
        VisibleAgentScenario(
            scenario_id="f" * 64,
            split="development",
            language="en-US",
            user_id="user-1",
            session_id="session-1",
            cutoff_time=datetime(2022, 1, 1),
            query_text="I only want a steakhouse",
        )
    )

    assert result.run is not None and result.run.fallback is False
    assert [call.tool_name for call in result.run.turns[0].tool_calls] == [
        "EXPAND_CANDIDATES",
        "APPLY_CONSTRAINTS",
        "GET_HYBRID_RANKING",
        "GET_BUSINESS_DETAILS",
    ]
    assert result.run.turns[0].candidate_ranking == ["business-1"]
