from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from yelp_agent.agent_benchmark import (
    AgentBenchmarkBundle,
    FakeScenarioRewriter,
    OpenAICompatibleScenarioRewriter,
    ScenarioGroundTruth,
    VisibleAgentScenario,
    audit_agent_benchmark_bundle,
    load_evidence_labels,
    load_scenario_ground_truth,
    load_visible_scenarios,
)
from yelp_agent.agent.llm import LLMCallResult, LLMMessage
from yelp_agent.agent_benchmark.baseline import (
    evaluate_decision_readiness_benchmark,
)
from yelp_agent.config import load_agent_benchmark_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks" / "agent_scenarios_v1"


def test_visible_scenario_cannot_accept_hidden_answers() -> None:
    payload = {
        "scenario_id": "a" * 64,
        "split": "development",
        "language": "zh-CN",
        "user_id": "user-1",
        "session_id": "session-1",
        "cutoff_time": datetime(2021, 1, 1),
        "query_text": "推荐一家安静的餐厅",
        "referenced_business_ids": [],
        "hidden_task_type": "recommendation_request",
    }

    with pytest.raises(ValidationError, match="hidden_task_type"):
        VisibleAgentScenario.model_validate(payload)


def test_ground_truth_requires_consistent_action_policy() -> None:
    payload = {
        "scenario_id": "a" * 64,
        "scenario_category": "information_gap",
        "frame_family": "missing-location",
        "source_task_id": "train:user-1:1",
        "source_profile_id": "b" * 64,
        "task_type": "recommendation_request",
        "expected_conditions": [],
        "expected_information_gaps": ["missing_location"],
        "allowed_actions": ["ask_clarification"],
        "required_actions": ["ask_clarification"],
        "forbidden_actions": ["ask_clarification"],
        "business_scope": [],
        "acceptable_business_ids": [],
        "scripted_user_turns": [],
        "uncertainty_policy": "clarify_before_action",
        "current_request_overrides_profile": False,
    }

    with pytest.raises(ValidationError, match="forbidden"):
        ScenarioGroundTruth.model_validate(payload)


def test_frozen_benchmark_passes_public_integrity_audit() -> None:
    bundle = AgentBenchmarkBundle(
        visible_scenarios=load_visible_scenarios(
            BENCHMARK_ROOT / "visible" / "scenarios.jsonl"
        ),
        ground_truth=load_scenario_ground_truth(
            BENCHMARK_ROOT / "hidden" / "ground_truth.jsonl"
        ),
        evidence_labels=load_evidence_labels(
            BENCHMARK_ROOT / "hidden" / "evidence_labels.parquet"
        ),
    )

    report = audit_agent_benchmark_bundle(
        bundle,
        load_agent_benchmark_config(PROJECT_ROOT / "configs"),
    )

    assert report.scenario_count == 500
    assert report.split_counts == {"development": 400, "validation": 100}
    assert report.language_counts == {"en-US": 200, "zh-CN": 300}
    assert report.evidence_label_count == 545
    assert report.scripted_user_turn_count == 250
    assert report.evidence_before_cutoff is True
    assert report.evidence_scope_isolated is True
    assert report.hidden_fields_absent_from_visible is True


def test_fake_rewriter_changes_only_visible_wording_without_provider_metadata() -> None:
    result = FakeScenarioRewriter().rewrite(
        scenario_id="a" * 64,
        language="en-US",
        visible_query="Find a quiet restaurant.",
    )

    assert result.text == "Please note: Find a quiet restaurant."
    assert result.generator_kind == "fake"
    assert result.generator_model is None
    assert result.prompt_sha256 is None


class _RewriteLLM:
    def __init__(self) -> None:
        self.messages: tuple[LLMMessage, ...] | None = None

    def generate(self, messages: tuple[LLMMessage, ...]) -> LLMCallResult:
        self.messages = messages
        return LLMCallResult(
            status="success",
            content='{"text":"Could you find a quiet restaurant?"}',
            model="deepseek-v4-flash",
            latency_ms=1,
            attempt_count=1,
        )


def test_provider_rewriter_receives_only_visible_language_payload() -> None:
    llm = _RewriteLLM()
    result = OpenAICompatibleScenarioRewriter(
        llm,
        model="deepseek-v4-flash",
    ).rewrite(
        scenario_id="a" * 64,
        language="en-US",
        visible_query="Find a quiet restaurant.",
    )

    assert result.text == "Could you find a quiet restaurant?"
    assert result.generator_kind == "openai_compatible"
    assert result.generator_model == "deepseek-v4-flash"
    assert llm.messages is not None
    serialized = "\n".join(message.content for message in llm.messages)
    assert "Find a quiet restaurant." in serialized
    assert "aaaaaaaa" not in serialized
    assert "expected_information_gaps" not in serialized
    assert "allowed_actions" not in serialized


def test_step19_baseline_runs_against_the_isolated_answer_key() -> None:
    report = evaluate_decision_readiness_benchmark(
        BENCHMARK_ROOT / "visible" / "scenarios.jsonl",
        BENCHMARK_ROOT / "hidden" / "ground_truth.jsonl",
    )

    assert report.scenario_count == 500
    assert report.task_type_accuracy == pytest.approx(0.734)
    assert report.information_gap_f1 == pytest.approx(0.6323529411764706)
    assert report.query_aware_confidence_refusal_rate == pytest.approx(
        0.7413793103448276
    )
