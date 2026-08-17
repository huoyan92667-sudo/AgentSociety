from __future__ import annotations

from datetime import datetime

from yelp_agent.agent_harness import (
    ActionOutcome,
    AgentDecision,
    AgentHarness,
    HarnessBudget,
    RuleBasedRequestInterpreter,
)
from yelp_agent.query.benchmark import ExpectedRequestCondition
from yelp_agent.query_recommendation_agent import (
    BusinessConditionIndex,
    QueryRecommendationAgentRunner,
    evaluate_frozen_predictions,
    load_predictions,
    verify_visible_run,
    write_visible_run,
)
from yelp_agent.query_recommendation_benchmark import (
    QueryRecommendationFrame,
    QueryRecommendationGroundTruth,
    VisibleQueryRecommendationCase,
)


class _Clock:
    def now_ms(self) -> float:
        return 100.0


class _Policy:
    def allowed_actions(self, state):
        return (
            ("retrieve_candidates",)
            if not state.observations
            else ("return_recommendation",)
        )


class _Router:
    def choose_action(self, state):
        if not state.observations:
            return AgentDecision(
                action="retrieve_candidates",
                reason_code="CANDIDATES_REQUIRED",
                tool_name="EXPAND_CANDIDATES",
                tool_kind="deterministic",
            )
        return AgentDecision(
            action="return_recommendation",
            arguments={"business_ids": ["b2", "b1"]},
            reason_code="READY_TO_FINALIZE",
        )


class _Executor:
    def execute(self, state, decision):
        if decision.action == "retrieve_candidates":
            return ActionOutcome(
                status="completed",
                observation={
                    "tool_name": "EXPAND_CANDIDATES",
                    "status": "success",
                    "data": {"candidate_business_ids": ["b1", "b2"]},
                },
                business_scope=["b1", "b2"],
            )
        return ActionOutcome(
            status="completed",
            response_kind="recommendation",
            candidate_ranking=["b2", "b1"],
            recommended_business_ids=["b2", "b1"],
        )


def _case() -> VisibleQueryRecommendationCase:
    return VisibleQueryRecommendationCase(
        case_id="a" * 64,
        split="development",
        language="en-US",
        user_id="user-1",
        session_id="session-1",
        cutoff_time=datetime(2022, 1, 1),
        query_text="Recommend a steakhouse.",
        generator_kind="deterministic",
    )


def _harness() -> AgentHarness:
    return AgentHarness(
        agent_version="query-agent-test-v1",
        interpreter=RuleBasedRequestInterpreter(),
        action_policy=_Policy(),
        router=_Router(),
        executor=_Executor(),
        budget=HarnessBudget(),
        clock=_Clock(),
    )


def test_visible_query_runs_through_harness_and_freezes_full_trace() -> None:
    prediction = QueryRecommendationAgentRunner(_harness()).run((_case(),))[0]

    assert prediction.hidden_labels_loaded is False
    assert prediction.retrieval_ranking == ["b1", "b2"]
    assert prediction.final_ranking == ["b2", "b1"]
    assert prediction.displayed_business_ids == ["b2", "b1"]
    assert [call.tool_name for call in prediction.turns[0].tool_calls] == [
        "EXPAND_CANDIDATES"
    ]
    assert prediction.response_kind == "recommendation"


def test_visible_run_manifest_is_target_blind_and_hash_verified(tmp_path) -> None:
    case = _case()
    visible_path = tmp_path / "cases.jsonl"
    visible_path.write_text(case.model_dump_json() + "\n", encoding="utf-8")
    prediction = QueryRecommendationAgentRunner(_harness()).run((case,))[0]

    manifest = write_visible_run(
        (prediction,),
        output_root=tmp_path / "run",
        visible_cases_path=visible_path,
    )

    assert manifest.hidden_labels_loaded is False
    assert verify_visible_run(tmp_path / "run") == manifest
    payload = (tmp_path / "run" / "predictions.jsonl").read_text(encoding="utf-8")
    assert "target_business_id" not in payload
    assert load_predictions(tmp_path / "run" / "predictions.jsonl") == (prediction,)


def test_hidden_evaluation_is_joined_only_after_predictions_are_frozen() -> None:
    case = _case()
    prediction = QueryRecommendationAgentRunner(_harness()).run((case,))[0]
    truth = QueryRecommendationGroundTruth(
        case_id=case.case_id,
        source_task_id="task-1",
        target_business_id="b2",
        target_review_id="review-1",
        target_stars=5,
        target_time=datetime(2022, 2, 1),
    )
    frame = QueryRecommendationFrame(
        case_id=case.case_id,
        frame_family="category_only",
        conditions=[
            ExpectedRequestCondition(
                field="category",
                operator="includes",
                value="Steakhouses",
                importance="preferred",
                enforcement="rank",
            )
        ],
        location_source="none",
        target_support_sources=["static_category"],
    )
    facts = BusinessConditionIndex(
        {
            "b1": {"business_id": "b1", "name": "One", "categories": ["Pizza"]},
            "b2": {
                "business_id": "b2",
                "name": "Two",
                "categories": ["Steakhouses"],
            },
        },
        {},
    )

    evaluation = evaluate_frozen_predictions(
        visible_cases=(case,),
        ground_truth=(truth,),
        frames=(frame,),
        predictions=(prediction,),
        facts=facts,
    )

    assert evaluation.report.metrics["Recall@50"].value == 1.0
    assert evaluation.report.metrics["HR@1"].value == 1.0
    assert evaluation.report.metrics["QueryCompliance@1"].value == 1.0
    assert evaluation.report.metrics["RecommendationCompletionRate"].value == 1.0
    assert evaluation.case_audits[0].target_final_rank == 1
    assert evaluation.report.metrics["CitationCorrectness"].status == "unavailable"
