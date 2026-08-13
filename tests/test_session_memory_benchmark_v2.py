from __future__ import annotations

from datetime import datetime
import json

from yelp_agent.session_memory_benchmark import (
    FrozenPresentation,
    MemoryBenchmarkInitialSession,
    PlanningContext,
    PresentedBusinessSnapshot,
    derive_relative_behavior,
)
from yelp_agent.session_memory_benchmark.config import (
    LanguagePlan,
    SessionMemoryBenchmarkV2Config,
    SplitPlan,
)
from yelp_agent.session_memory_benchmark.planner import BenchmarkV2Planner
from yelp_agent.session_memory_benchmark.generation import generate_and_review_turns
from yelp_agent.session_memory_benchmark.evaluation import (
    MemoryStateView,
    score_memory_transition,
)
from yelp_agent.agent.llm import LLMCallResult


def _business(
    business_id: str,
    rank: int,
    *,
    price: int | None,
    distance: float | None,
) -> PresentedBusinessSnapshot:
    return PresentedBusinessSnapshot(
        business_id=business_id,
        rank=rank,
        name=business_id,
        categories=["Restaurants"],
        price_level=price,
        distance_km=distance,
    )


def test_relative_behavior_is_derived_from_the_presented_business() -> None:
    presented = _business("shown", 1, price=3, distance=5.7)
    candidates = [
        presented,
        _business("cheaper-and-closer", 2, price=2, distance=2.1),
        _business("same-price", 3, price=3, distance=4.0),
        _business("unknown", 4, price=None, distance=None),
    ]

    cheaper = derive_relative_behavior("cheaper", presented, candidates)
    closer = derive_relative_behavior("closer", presented, candidates)

    assert cheaper.baseline_value == 3
    assert cheaper.acceptable_business_ids == ["cheaper-and-closer"]
    assert closer.baseline_value == 5.7
    assert closer.acceptable_business_ids == [
        "cheaper-and-closer",
        "same-price",
    ]


def test_planner_builds_three_context_grounded_turns_per_recommendation_session() -> None:
    digest = "a" * 64
    businesses = [
        _business("shown", 1, price=3, distance=5.7),
        _business("better", 2, price=2, distance=2.1),
        _business("other", 3, price=4, distance=8.0),
    ]
    initial = MemoryBenchmarkInitialSession(
        session_case_id=digest,
        source_scenario_id="b" * 64,
        split="development",
        language="zh-CN",
        user_id="user",
        session_id="session",
        cutoff_time=datetime(2020, 1, 1),
        query_text="帮我推荐一家餐厅",
        user_latitude=39.9,
        user_longitude=116.4,
    )
    context = PlanningContext(
        initial_session=initial,
        source_category="hard_constraint",
        source_frame_family="test",
        initial_task_type="recommendation_request",
        presentation=FrozenPresentation(
            session_case_id=digest,
            turn_index=1,
            pipeline_version="test",
            source_run_sha256="c" * 64,
            candidate_business_ids=[item.business_id for item in businesses],
            presented_businesses=businesses,
        ),
        candidate_businesses=businesses,
    )
    empty = LanguagePlan(
        recommendation_sessions=0,
        recommendation_families={},
        support_families={},
    )
    config = SessionMemoryBenchmarkV2Config(
        formal=False,
        benchmark_version="test-v2",
        pipeline_version="test",
        generator_prompt_version="generator",
        reviewer_prompt_version="reviewer",
        split_plans={
            "development": SplitPlan(
                languages={
                    "zh-CN": LanguagePlan(
                        recommendation_sessions=1,
                        recommendation_families={
                            "relative_preference": 1,
                            "reference_rejection": 1,
                            "no_state_change": 1,
                        },
                        support_families={},
                    ),
                    "en-US": empty,
                }
            ),
            "validation": SplitPlan(
                languages={"zh-CN": empty, "en-US": empty}
            ),
        },
    )

    plan = BenchmarkV2Planner(config).plan([context])

    assert len(plan.initial_sessions) == 1
    assert [item.turn_index for item in plan.turn_specs] == [2, 3, 4]
    assert {item.family for item in plan.turn_specs} == {
        "relative_preference",
        "reference_rejection",
        "no_state_change",
    }
    relative = next(
        item for item in plan.turn_specs if item.family == "relative_preference"
    )
    assert relative.behaviors[0].baseline_business_id == "shown"
    assert relative.behaviors[0].baseline_value == 3
    assert relative.behaviors[0].acceptable_business_ids == ["better"]


class _FakeLLM:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = list(payloads)
        self.call_count = 0

    def generate(self, _messages: object) -> LLMCallResult:
        self.call_count += 1
        return LLMCallResult(
            status="success",
            content=json.dumps(self.payloads.pop(0), ensure_ascii=False),
            model="fake-model",
            latency_ms=1,
            attempt_count=1,
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
        )


def test_generation_requires_independent_semantic_review() -> None:
    spec = _one_turn_spec()
    public_id = __import__("hashlib").sha256(b"benchmark-public-job:1").hexdigest()
    generator = _FakeLLM(
        [
            {"turns": [{"turn_case_id": public_id, "query_text": "再便宜一点吧"}]},
            {"turns": [{"turn_case_id": public_id, "query_text": "换个更实惠的选择"}]},
        ]
    )
    reviewer = _FakeLLM(
        [
            {"decisions": [{
                "turn_case_id": public_id,
                "intent_consistent": False,
                "extra_constraint_detected": False,
                "hidden_value_leaked": False,
                "reference_answerable": True,
                "naturalness_score": 4,
                "issue_codes": ["WRONG_INTENT"],
            }]},
            {"decisions": [{
                "turn_case_id": public_id,
                "intent_consistent": True,
                "extra_constraint_detected": False,
                "hidden_value_leaked": False,
                "reference_answerable": True,
                "naturalness_score": 5,
                "issue_codes": [],
            }]},
        ]
    )
    config = _small_config()

    result = generate_and_review_turns(
        [spec], generator=generator, reviewer=reviewer, config=config
    )

    assert "换个更实惠的选择" in result.turns[0].query_text
    assert result.rejected_generation_count == 1
    assert generator.call_count == 2
    assert reviewer.call_count == 2
    assert result.provider_call_count == 4


def _small_config() -> SessionMemoryBenchmarkV2Config:
    empty = LanguagePlan(
        recommendation_sessions=0,
        recommendation_families={},
        support_families={},
    )
    return SessionMemoryBenchmarkV2Config(
        formal=False,
        benchmark_version="test-v2",
        pipeline_version="test",
        generator_prompt_version="generator",
        reviewer_prompt_version="reviewer",
        batch_size=1,
        maximum_generation_rounds=2,
        split_plans={
            "development": SplitPlan(languages={"zh-CN": empty, "en-US": empty}),
            "validation": SplitPlan(languages={"zh-CN": empty, "en-US": empty}),
        },
    )


def _one_turn_spec():
    from yelp_agent.session_memory_benchmark import (
        ExpectedMemoryDeltaV2,
        ExpectedRelativePreference,
        TurnGenerationSpec,
    )

    return TurnGenerationSpec(
        turn_case_id="d" * 64,
        session_case_id="e" * 64,
        split="development",
        turn_index=2,
        language="zh-CN",
        family="relative_preference",
        intent_code="relative_cheaper",
        visible_context={"presented_businesses": [{"rank": 1, "name": "A"}]},
        required_meaning={"relative_preference": {"field": "price", "direction": "lower"}},
        forbidden_meaning=["exact numeric threshold"],
        expected_delta=ExpectedMemoryDeltaV2(
            task_type="feedback_refinement",
            relative_preferences=[
                ExpectedRelativePreference(field="price", direction="lower")
            ],
        ),
        trigger_action="return_recommendation",
    )


def test_transition_evaluator_scores_real_state_changes_not_keyword_labels() -> None:
    from yelp_agent.session_memory_benchmark import (
        ExpectedConditionDelta,
        ExpectedMemoryDeltaV2,
        ExpectedRelativePreference,
        FrozenTurnGroundTruthV2,
    )

    before = MemoryStateView(
        task_type="recommendation_request",
        conditions=frozenset({("price_level", "less_than_or_equal", "3", "strong")}),
        relative_preferences=frozenset(),
        clarification_answers=(),
        rejected_business_ids=frozenset(),
        resolved_business_ids=frozenset(),
        information_gaps=frozenset(),
        party_size=None,
    )
    after = MemoryStateView(
        task_type="feedback_refinement",
        conditions=frozenset({("price_level", "less_than_or_equal", "2", "strong")}),
        relative_preferences=frozenset({("distance", "closer")}),
        clarification_answers=(),
        rejected_business_ids=frozenset(),
        resolved_business_ids=frozenset({"shown"}),
        information_gaps=frozenset(),
        party_size=None,
    )
    truth = FrozenTurnGroundTruthV2(
        turn_case_id="f" * 64,
        session_case_id="0" * 64,
        split="development",
        turn_index=2,
        family="combined_update",
        expected_delta=ExpectedMemoryDeltaV2(
            task_type="feedback_refinement",
            condition_deltas=[
                ExpectedConditionDelta(
                    operation="replace",
                    field="price_level",
                    operator="less_than_or_equal",
                    value=2,
                    importance="strong",
                )
            ],
            relative_preferences=[
                ExpectedRelativePreference(field="distance", direction="closer")
            ],
            resolved_business_ids=["shown"],
        ),
    )

    score = score_memory_transition(before, after, truth, "更近一点，价格等级最多2")

    assert score["task_type_correct"] is True
    assert score["condition_expected"] == 1
    assert score["condition_hits"] == 1
    assert score["relative_expected"] == 1
    assert score["relative_hits"] == 1
    assert score["resolved_reference_hits"] == 1
    assert score["numeric_hallucinations"] == 0
