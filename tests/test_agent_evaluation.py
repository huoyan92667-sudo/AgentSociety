from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from yelp_agent.agent_benchmark import (
    EvidenceLabel,
    ScenarioGroundTruth,
    VisibleAgentScenario,
)
from yelp_agent.agent_evaluation import (
    AgentActionTrace,
    AgentScenarioRun,
    AgentTurnTrace,
    ClarificationQuestionTrace,
    EvidenceReference,
    ResponseClaimTrace,
    RetrievedEvidenceTrace,
    ToolCallTrace,
    evaluate_agent_scenario_runs,
    freeze_agent_evaluation_contract,
    load_agent_scenario_runs,
    metric_definitions,
    write_agent_evaluation_report,
    write_agent_scenario_runs,
)
from yelp_agent.config import load_agent_evaluation_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks" / "agent_scenarios_v1"


def _visible(*, scenario_id: str = "a" * 64) -> VisibleAgentScenario:
    return VisibleAgentScenario(
        scenario_id=scenario_id,
        split="development",
        language="zh-CN",
        user_id="user-1",
        session_id="session-1",
        cutoff_time=datetime(2021, 1, 1),
        query_text="推荐一家满足条件的餐厅",
    )


def _truth(*, scenario_id: str = "a" * 64) -> ScenarioGroundTruth:
    return ScenarioGroundTruth(
        scenario_id=scenario_id,
        scenario_category="hard_constraint",
        frame_family="development:hard-constraint",
        source_task_id="validation:user-1:1",
        source_profile_id="b" * 64,
        task_type="recommendation_request",
        expected_conditions=[],
        expected_information_gaps=[],
        allowed_actions=[
            "retrieve_candidates",
            "apply_hard_constraints",
            "rank_candidates",
            "return_recommendation",
        ],
        required_actions=[
            "apply_hard_constraints",
            "return_recommendation",
        ],
        forbidden_actions=["ask_clarification"],
        business_scope=["business-1", "business-2"],
        acceptable_business_ids=["business-1"],
        scripted_user_turns=[],
        uncertainty_policy="proceed",
        current_request_overrides_profile=False,
    )


def _perfect_run(*, scenario_id: str = "a" * 64) -> AgentScenarioRun:
    return AgentScenarioRun(
        scenario_id=scenario_id,
        agent_version="fake-perfect-v1",
        turns=[
            AgentTurnTrace(
                turn_index=1,
                predicted_task_type="recommendation_request",
                detected_information_gaps=[],
                actions=[
                    AgentActionTrace(
                        step_index=1,
                        action="retrieve_candidates",
                        status="completed",
                        reason_code="NEED_CANDIDATES",
                    ),
                    AgentActionTrace(
                        step_index=2,
                        action="apply_hard_constraints",
                        status="completed",
                        reason_code="HARD_CONSTRAINT_PRESENT",
                    ),
                    AgentActionTrace(
                        step_index=3,
                        action="rank_candidates",
                        status="completed",
                        reason_code="READY_TO_RANK",
                    ),
                    AgentActionTrace(
                        step_index=4,
                        action="return_recommendation",
                        status="completed",
                        reason_code="READY_TO_FINALIZE",
                    ),
                ],
                candidate_ranking=["business-1", "business-2"],
                recommended_business_ids=["business-1"],
                response_kind="recommendation",
            )
        ],
        latency_ms=20,
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.001,
    )


def test_perfect_recommendation_trace_scores_through_public_interface() -> None:
    report = evaluate_agent_scenario_runs(
        [_perfect_run()],
        visible_scenarios=[_visible()],
        ground_truth=[_truth()],
        evidence_labels=[],
    )

    assert report.scenario_count == 1
    assert report.metrics["task_type_accuracy"].value == 1.0
    assert report.metrics["action_accuracy"].value == 1.0
    assert report.metrics["hr_at_1"].value == 1.0
    assert report.metrics["hr_at_3"].value == 1.0
    assert report.metrics["hr_at_5"].value == 1.0
    assert report.metrics["mrr"].value == 1.0
    assert report.metrics["ndcg_at_5"].value == 1.0
    assert report.metrics["recall_at_50"].status == "unavailable"
    assert report.metrics["recall_at_100"].status == "unavailable"
    assert report.metrics["recall_at_500"].status == "unavailable"
    assert report.metrics["hard_constraint_satisfaction"].value == 1.0
    assert report.metrics["valid_candidate_rate"].value == 1.0
    assert report.metrics["empty_result_rate"].value == 0.0
    assert report.by_split["development"].scenario_count == 1
    assert report.by_category["hard_constraint"].metrics[
        "hard_constraint_satisfaction"
    ].value == 1.0


def test_clarification_metrics_score_detected_answerable_gap_and_follow_up() -> None:
    visible = _visible(scenario_id="c" * 64)
    truth_payload = _truth(scenario_id="c" * 64).model_dump()
    truth_payload.update(
        {
            "scenario_category": "information_gap",
            "frame_family": "development:missing-location",
            "expected_information_gaps": ["missing_location"],
            "allowed_actions": ["ask_clarification", "safe_fallback"],
            "required_actions": ["ask_clarification"],
            "forbidden_actions": ["return_recommendation"],
            "business_scope": [],
            "acceptable_business_ids": [],
            "scripted_user_turns": [
                {
                    "turn_index": 2,
                    "trigger_action": "ask_clarification",
                    "query_text": "我在费城市政厅附近",
                    "expected_task_type": "recommendation_request",
                    "expected_information_gaps": [],
                    "state_updates": {
                        "user_location": "Philadelphia City Hall"
                    },
                    "expected_allowed_actions": [
                        "retrieve_candidates",
                        "apply_hard_constraints",
                        "rank_candidates",
                        "return_recommendation",
                    ],
                }
            ],
            "uncertainty_policy": "clarify_before_action",
        }
    )
    truth = ScenarioGroundTruth.model_validate(truth_payload)
    run = AgentScenarioRun(
        scenario_id="c" * 64,
        agent_version="fake-perfect-v1",
        turns=[
            AgentTurnTrace(
                turn_index=1,
                predicted_task_type="recommendation_request",
                detected_information_gaps=["missing_location"],
                actions=[
                    AgentActionTrace(
                        step_index=1,
                        action="ask_clarification",
                        status="completed",
                        reason_code="MISSING_LOCATION",
                    )
                ],
                clarification_questions=[
                    ClarificationQuestionTrace(
                        question_text="你现在在哪里？",
                        requested_information_gaps=["missing_location"],
                    )
                ],
                response_kind="clarification",
            ),
            AgentTurnTrace(
                turn_index=2,
                predicted_task_type="recommendation_request",
                detected_information_gaps=[],
                actions=[
                    AgentActionTrace(
                        step_index=1,
                        action="return_recommendation",
                        status="completed",
                        reason_code="READY_TO_FINALIZE",
                    )
                ],
                response_kind="recommendation",
            ),
        ],
        latency_ms=10,
    )

    report = evaluate_agent_scenario_runs(
        [run],
        visible_scenarios=[visible],
        ground_truth=[truth],
        evidence_labels=[],
    )

    assert report.metrics["missing_field_detection_precision"].value == 1.0
    assert report.metrics["missing_field_detection_recall"].value == 1.0
    assert report.metrics["unnecessary_question_rate"].value == 0.0
    assert report.metrics["question_answerability_rate"].value == 1.0
    assert report.metrics["average_questions_before_finalize"].value == 1.0


def test_routing_metrics_detect_invalid_repeated_and_unnecessary_tool_calls() -> None:
    run = AgentScenarioRun(
        scenario_id="a" * 64,
        agent_version="fake-bad-router-v1",
        turns=[
            AgentTurnTrace(
                turn_index=1,
                predicted_task_type="recommendation_request",
                actions=[
                    AgentActionTrace(
                        step_index=1,
                        action="retrieve_business_reviews",
                        status="completed",
                        reason_code="UNNECESSARY_RAG",
                    ),
                    AgentActionTrace(
                        step_index=2,
                        action="return_recommendation",
                        status="completed",
                        reason_code="PREMATURE_FINALIZE",
                    ),
                ],
                tool_calls=[
                    ToolCallTrace(
                        call_id="call-1",
                        action_step_index=1,
                        action="retrieve_business_reviews",
                        tool_name="SEARCH_BUSINESS_REVIEWS",
                        tool_kind="review_rag",
                        arguments_sha256="1" * 64,
                        status="completed",
                        latency_ms=5,
                    ),
                    ToolCallTrace(
                        call_id="call-2",
                        action_step_index=1,
                        action="retrieve_business_reviews",
                        tool_name="SEARCH_BUSINESS_REVIEWS",
                        tool_kind="review_rag",
                        arguments_sha256="1" * 64,
                        status="completed",
                        latency_ms=5,
                    ),
                ],
                candidate_ranking=["business-1"],
                recommended_business_ids=["business-1"],
                response_kind="recommendation",
            )
        ],
        latency_ms=10,
    )

    report = evaluate_agent_scenario_runs(
        [run],
        visible_scenarios=[_visible()],
        ground_truth=[_truth()],
        evidence_labels=[],
    )

    assert report.metrics["invalid_action_rate"].value == 0.5
    assert report.metrics["tool_selection_accuracy"].value == 0.0
    assert report.metrics["repeated_tool_call_rate"].value == 0.5
    assert report.metrics["unnecessary_rag_call_rate"].value == 1.0
    assert report.metrics["direct_return_precision"].value == 0.0
    assert report.metrics["fallback_rate"].value == 0.0


def test_evidence_metrics_check_scope_retrieval_grounding_and_conflict() -> None:
    visible = _visible(scenario_id="d" * 64)
    truth_payload = _truth(scenario_id="d" * 64).model_dump()
    truth_payload.update(
        {
            "scenario_category": "evidence_uncertainty",
            "frame_family": "development:conflicting-reviews",
            "task_type": "review_experience_question",
            "allowed_actions": [
                "retrieve_business_reviews",
                "return_uncertain_answer",
            ],
            "required_actions": [
                "retrieve_business_reviews",
                "return_uncertain_answer",
            ],
            "forbidden_actions": ["retrieve_candidates"],
            "business_scope": ["business-1"],
            "acceptable_business_ids": [],
            "uncertainty_policy": "report_conflict",
        }
    )
    truth = ScenarioGroundTruth.model_validate(truth_payload)
    labels = [
        EvidenceLabel(
            scenario_id="d" * 64,
            business_id="business-1",
            source_type="review",
            review_id="review-1",
            aspect="quiet_environment",
            relevance="relevant",
            stance="supports",
            event_time=datetime(2020, 1, 1),
            confidence=0.9,
            source_text_sha256="1" * 64,
        ),
        EvidenceLabel(
            scenario_id="d" * 64,
            business_id="business-1",
            source_type="review",
            review_id="review-2",
            aspect="quiet_environment",
            relevance="relevant",
            stance="contradicts",
            event_time=datetime(2020, 2, 1),
            confidence=0.9,
            source_text_sha256="2" * 64,
        ),
        EvidenceLabel(
            scenario_id="d" * 64,
            business_id="business-2",
            source_type="review",
            review_id="review-decoy",
            aspect="quiet_environment",
            relevance="out_of_scope",
            stance="supports",
            event_time=datetime(2020, 1, 1),
            confidence=0.9,
            source_text_sha256="3" * 64,
        ),
    ]
    relevant_reference = EvidenceReference(
        business_id="business-1",
        source_type="review",
        review_id="review-1",
    )
    run = AgentScenarioRun(
        scenario_id="d" * 64,
        agent_version="fake-evidence-v1",
        turns=[
            AgentTurnTrace(
                turn_index=1,
                predicted_task_type="review_experience_question",
                actions=[
                    AgentActionTrace(
                        step_index=1,
                        action="retrieve_business_reviews",
                        status="completed",
                        reason_code="REVIEW_EVIDENCE_REQUIRED",
                    ),
                    AgentActionTrace(
                        step_index=2,
                        action="return_uncertain_answer",
                        status="completed",
                        reason_code="CONFLICTING_EVIDENCE",
                    ),
                ],
                tool_calls=[
                    ToolCallTrace(
                        call_id="call-evidence",
                        action_step_index=1,
                        action="retrieve_business_reviews",
                        tool_name="SEARCH_BUSINESS_REVIEWS",
                        tool_kind="review_rag",
                        arguments_sha256="4" * 64,
                        status="completed",
                        latency_ms=5,
                        retrieved_evidence=[
                            RetrievedEvidenceTrace(
                                rank=1,
                                evidence=relevant_reference,
                                relevance_score=0.95,
                            ),
                            RetrievedEvidenceTrace(
                                rank=2,
                                evidence=EvidenceReference(
                                    business_id="business-2",
                                    source_type="review",
                                    review_id="review-decoy",
                                ),
                                relevance_score=0.8,
                            ),
                        ],
                    )
                ],
                claims=[
                    ResponseClaimTrace(
                        claim_id="claim-1",
                        text="评论对安静程度存在不同看法。",
                        business_id="business-1",
                        evidence_refs=[relevant_reference],
                    )
                ],
                response_kind="uncertain_answer",
                reported_conflict=True,
                reported_evidence_recency=True,
            )
        ],
        latency_ms=10,
    )

    report = evaluate_agent_scenario_runs(
        [run],
        visible_scenarios=[visible],
        ground_truth=[truth],
        evidence_labels=labels,
    )

    assert report.metrics["business_scope_isolation_rate"].value == 0.5
    assert report.metrics["review_retrieval_recall_at_1"].value == 0.5
    assert report.metrics["evidence_precision_at_1"].value == 1.0
    assert report.metrics["evidence_precision_at_3"].value == 0.5
    assert report.metrics["grounded_answer_rate"].value == 1.0
    assert report.metrics["unsupported_claim_rate"].value == 0.0
    assert report.metrics["citation_correctness"].value == 1.0
    assert report.metrics["evidence_recency_reporting_rate"].value == 1.0
    assert report.metrics["conflict_detection_accuracy"].value == 1.0


def test_cost_metrics_use_frozen_linear_percentiles_and_success_definition() -> None:
    second_id = "e" * 64
    second_run = _perfect_run(scenario_id=second_id).model_copy(
        update={
            "latency_ms": 80.0,
            "input_tokens": 40,
            "output_tokens": 20,
            "cost_usd": 0.003,
        }
    )

    report = evaluate_agent_scenario_runs(
        [_perfect_run(), second_run],
        visible_scenarios=[_visible(), _visible(scenario_id=second_id)],
        ground_truth=[_truth(), _truth(scenario_id=second_id)],
        evidence_labels=[],
    )

    assert report.metrics["mean_latency_ms"].value == 50.0
    assert report.metrics["p50_latency_ms"].value == 50.0
    assert report.metrics["p95_latency_ms"].value == 77.0
    assert report.metrics["mean_tokens"].value == 37.5
    assert report.metrics["p95_tokens"].value == 57.75
    assert report.metrics["cost_per_successful_recommendation"].value == 0.002
    assert report.metrics["cache_hit_rate"].status == "not_applicable"


def test_detection_precision_uses_micro_counts_not_average_of_scenario_rates() -> None:
    scenario_ids = ["f" * 64, "9" * 64]
    truths = []
    runs = []
    for index, scenario_id in enumerate(scenario_ids):
        payload = _truth(scenario_id=scenario_id).model_dump()
        payload.update(
            {
                "scenario_category": "information_gap",
                "frame_family": f"development:micro-{index}",
                "expected_information_gaps": ["missing_location"],
                "allowed_actions": ["ask_clarification", "safe_fallback"],
                "required_actions": ["ask_clarification"],
                "forbidden_actions": ["return_recommendation"],
                "business_scope": [],
                "acceptable_business_ids": [],
                "uncertainty_policy": "clarify_before_action",
            }
        )
        truths.append(ScenarioGroundTruth.model_validate(payload))
        gaps = ["missing_location"]
        if index == 1:
            gaps.extend(["missing_budget", "missing_party_size"])
        runs.append(
            AgentScenarioRun(
                scenario_id=scenario_id,
                agent_version="fake-gap-v1",
                turns=[
                    AgentTurnTrace(
                        turn_index=1,
                        predicted_task_type="recommendation_request",
                        detected_information_gaps=gaps,
                        actions=[
                            AgentActionTrace(
                                step_index=1,
                                action="ask_clarification",
                                status="completed",
                                reason_code="MISSING_INFORMATION",
                            )
                        ],
                        clarification_questions=[
                            ClarificationQuestionTrace(
                                question_text="你在哪里？",
                                requested_information_gaps=["missing_location"],
                            )
                        ],
                        response_kind="clarification",
                    )
                ],
                latency_ms=1,
            )
        )

    report = evaluate_agent_scenario_runs(
        runs,
        visible_scenarios=[_visible(scenario_id=value) for value in scenario_ids],
        ground_truth=truths,
        evidence_labels=[],
    )

    assert report.metrics["missing_field_detection_precision"].value == 0.5
    assert report.metrics["missing_field_detection_recall"].value == 1.0


def test_post_clarification_utility_gain_compares_rankings_before_and_after() -> None:
    scenario_id = "8" * 64
    payload = _truth(scenario_id=scenario_id).model_dump()
    payload.update(
        {
            "scenario_category": "information_gap",
            "frame_family": "development:clarification-gain",
            "expected_information_gaps": ["missing_location"],
            "allowed_actions": ["ask_clarification", "safe_fallback"],
            "required_actions": ["ask_clarification"],
            "forbidden_actions": [],
            "scripted_user_turns": [
                {
                    "turn_index": 2,
                    "trigger_action": "ask_clarification",
                    "query_text": "我在市政厅附近",
                    "expected_task_type": "recommendation_request",
                    "expected_information_gaps": [],
                    "expected_allowed_actions": [
                        "rank_candidates",
                        "return_recommendation",
                    ],
                }
            ],
            "uncertainty_policy": "clarify_before_action",
        }
    )
    truth = ScenarioGroundTruth.model_validate(payload)
    run = AgentScenarioRun(
        scenario_id=scenario_id,
        agent_version="fake-clarification-gain-v1",
        turns=[
            AgentTurnTrace(
                turn_index=1,
                predicted_task_type="recommendation_request",
                detected_information_gaps=["missing_location"],
                actions=[
                    AgentActionTrace(
                        step_index=1,
                        action="ask_clarification",
                        status="completed",
                        reason_code="MISSING_LOCATION",
                    )
                ],
                clarification_questions=[
                    ClarificationQuestionTrace(
                        question_text="你在哪里？",
                        requested_information_gaps=["missing_location"],
                    )
                ],
                candidate_ranking=["business-2", "business-1"],
                response_kind="clarification",
            ),
            AgentTurnTrace(
                turn_index=2,
                predicted_task_type="recommendation_request",
                actions=[
                    AgentActionTrace(
                        step_index=1,
                        action="rank_candidates",
                        status="completed",
                        reason_code="UPDATED_CONTEXT",
                    ),
                    AgentActionTrace(
                        step_index=2,
                        action="return_recommendation",
                        status="completed",
                        reason_code="READY_TO_FINALIZE",
                    ),
                ],
                candidate_ranking=["business-1", "business-2"],
                recommended_business_ids=["business-1"],
                response_kind="recommendation",
            ),
        ],
        latency_ms=2,
    )

    report = evaluate_agent_scenario_runs(
        [run],
        visible_scenarios=[_visible(scenario_id=scenario_id)],
        ground_truth=[truth],
        evidence_labels=[],
    )

    assert report.metrics["post_clarification_utility_gain"].value == pytest.approx(
        1.0 - 1.0 / 1.584962500721156
    )


def test_step21_config_freezes_cutoffs_and_unavailable_metric_policy() -> None:
    config = load_agent_evaluation_config("configs")

    assert config.contract_version == "1.0.0"
    assert config.ranking_cutoffs == [1, 3, 5]
    assert config.full_retrieval_cutoffs == [50, 100, 500]
    assert config.evidence_cutoffs == [1, 3, 5]
    assert config.percentile_method == "linear"
    assert config.missing_observation_policy == "explicit_status"
    assert config.agent_ranking_relevance == "acceptable_business_ids"
    assert config.full_retrieval_source == "step11_full_retrieval_benchmark"


def test_metric_catalog_freezes_source_aggregation_and_direction() -> None:
    definitions = {item.name: item for item in metric_definitions()}

    required = {
        "hr_at_1",
        "mrr",
        "ndcg_at_5",
        "recall_at_500",
        "hard_constraint_satisfaction",
        "missing_field_detection_recall",
        "post_clarification_utility_gain",
        "action_accuracy",
        "tool_selection_accuracy",
        "unnecessary_rag_call_rate",
        "business_scope_isolation_rate",
        "review_retrieval_recall_at_5",
        "unsupported_claim_rate",
        "official_policy_caution_accuracy",
        "p95_latency_ms",
        "cost_per_successful_recommendation",
    }
    assert required.issubset(definitions)
    assert definitions["recall_at_500"].source == "full_retrieval"
    assert definitions["hr_at_1"].source == "agent_scenario"
    assert definitions["unsupported_claim_rate"].higher_is_better is False
    assert definitions["missing_field_detection_recall"].aggregation == "micro"

    report = evaluate_agent_scenario_runs(
        [_perfect_run()],
        visible_scenarios=[_visible()],
        ground_truth=[_truth()],
        evidence_labels=[],
    )
    assert set(report.metrics) == set(definitions)


def test_contract_manifest_hashes_benchmark_config_and_metric_catalog(tmp_path) -> None:
    output = tmp_path / "evaluation_contract.json"
    config = load_agent_evaluation_config(PROJECT_ROOT / "configs")

    first = freeze_agent_evaluation_contract(
        BENCHMARK_ROOT,
        config,
        output_path=output,
    )
    first_bytes = output.read_bytes()
    second = freeze_agent_evaluation_contract(
        BENCHMARK_ROOT,
        config,
        output_path=output,
    )

    assert first.status == "written"
    assert second.status == "reused"
    assert first.manifest.contract_version == "1.0.0"
    assert first.manifest.metric_count == len(metric_definitions())
    assert first.manifest.hidden_labels_visible_to_agent is False
    assert first.manifest.full_retrieval_metrics == [
        "recall_at_50",
        "recall_at_100",
        "recall_at_500",
    ]
    assert output.read_bytes() == first_bytes


def test_run_and_report_artifacts_are_atomic_typed_and_byte_stable(tmp_path) -> None:
    runs_path = tmp_path / "runs.jsonl"
    report_path = tmp_path / "report.json"
    runs = [_perfect_run()]

    write_agent_scenario_runs(runs, runs_path)
    first_runs = runs_path.read_bytes()
    assert load_agent_scenario_runs(runs_path) == tuple(runs)
    write_agent_scenario_runs(runs, runs_path)
    assert runs_path.read_bytes() == first_runs

    report = evaluate_agent_scenario_runs(
        runs,
        visible_scenarios=[_visible()],
        ground_truth=[_truth()],
        evidence_labels=[],
    )
    write_agent_evaluation_report(report, report_path)
    first_report = report_path.read_bytes()
    write_agent_evaluation_report(report, report_path)
    assert report_path.read_bytes() == first_report
    assert not list(tmp_path.glob("*.partial"))


def test_run_schema_rejects_hidden_answers_and_unranked_recommendations() -> None:
    hidden_payload = _perfect_run().model_dump()
    hidden_payload["expected_task_type"] = "recommendation_request"
    with pytest.raises(ValidationError, match="expected_task_type"):
        AgentScenarioRun.model_validate(hidden_payload)

    invalid_turn = _perfect_run().turns[0].model_copy(
        update={"recommended_business_ids": ["business-outside-ranking"]}
    )
    with pytest.raises(ValidationError, match="subset of candidate_ranking"):
        AgentTurnTrace.model_validate(invalid_turn.model_dump())


def test_evaluator_rejects_missing_or_extra_scenario_runs() -> None:
    with pytest.raises(ValueError, match="must align"):
        evaluate_agent_scenario_runs(
            [_perfect_run()],
            visible_scenarios=[_visible(), _visible(scenario_id="7" * 64)],
            ground_truth=[_truth(), _truth(scenario_id="7" * 64)],
            evidence_labels=[],
        )


def test_scripted_user_turn_cannot_be_released_without_its_trigger_action() -> None:
    scenario_id = "6" * 64
    payload = _truth(scenario_id=scenario_id).model_dump()
    payload.update(
        {
            "scenario_category": "information_gap",
            "frame_family": "development:trigger-guard",
            "expected_information_gaps": ["missing_location"],
            "allowed_actions": ["ask_clarification", "safe_fallback"],
            "required_actions": ["ask_clarification"],
            "forbidden_actions": ["return_recommendation"],
            "business_scope": [],
            "acceptable_business_ids": [],
            "scripted_user_turns": [
                {
                    "turn_index": 2,
                    "trigger_action": "ask_clarification",
                    "query_text": "我在市政厅附近",
                    "expected_task_type": "recommendation_request",
                    "expected_information_gaps": [],
                    "expected_allowed_actions": ["return_recommendation"],
                }
            ],
            "uncertainty_policy": "clarify_before_action",
        }
    )
    truth = ScenarioGroundTruth.model_validate(payload)
    run = AgentScenarioRun(
        scenario_id=scenario_id,
        agent_version="fake-invalid-dialogue-v1",
        turns=[
            AgentTurnTrace(
                turn_index=1,
                predicted_task_type="recommendation_request",
                detected_information_gaps=["missing_location"],
                actions=[
                    AgentActionTrace(
                        step_index=1,
                        action="safe_fallback",
                        status="completed",
                        reason_code="STOPPED_EARLY",
                    )
                ],
                response_kind="fallback",
            ),
            AgentTurnTrace(
                turn_index=2,
                predicted_task_type="recommendation_request",
                actions=[
                    AgentActionTrace(
                        step_index=1,
                        action="return_recommendation",
                        status="completed",
                        reason_code="INVALID_RELEASE",
                    )
                ],
                response_kind="recommendation",
            ),
        ],
        latency_ms=1,
    )

    report = evaluate_agent_scenario_runs(
        [run],
        visible_scenarios=[_visible(scenario_id=scenario_id)],
        ground_truth=[truth],
        evidence_labels=[],
    )

    assert "scripted_turn_released_without_trigger:2" in (
        report.scenario_results[0].violations
    )
