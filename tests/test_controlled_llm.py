from __future__ import annotations

import json
from datetime import UTC, datetime

from yelp_agent.agent.llm import LLMCallResult
from yelp_agent.agent_evaluation.schema import EvidenceReference, ResponseClaimTrace
from yelp_agent.agent_harness import RuleBasedRequestInterpreter
from yelp_agent.controlled_llm import (
    AnswerComposerConfig,
    AnswerCompositionInput,
    AnswerEvidenceItem,
    ControlledJSONCaller,
    ControlledLLMUsageLedger,
    ControlledRequestInterpreter,
    ControlledSemanticEnhancer,
    FakeChatGenerator,
    GroundedAnswerComposer,
    SemanticEnhancementInput,
    SemanticEscalationConfig,
    SqliteControlledLLMCache,
    build_controlled_llm_runtime,
    load_controlled_llm_config,
)
from yelp_agent.query import QueryParseInput


def _success(content: str, *, total: int = 120) -> LLMCallResult:
    return LLMCallResult(
        status="success",
        content=content,
        model="fake-model",
        latency_ms=12.5,
        attempt_count=1,
        input_tokens=80,
        output_tokens=total - 80,
        total_tokens=total,
    )


def _semantic_config(*, mode: str = "always") -> SemanticEscalationConfig:
    return SemanticEscalationConfig(
        enabled=True,
        mode=mode,
        prompt_version="test-semantic-v1",
        max_output_tokens=900,
        minimum_signal_confidence=0.7,
        minimum_task_type_confidence=0.8,
        call_for_task_types=["candidate_comparison", "review_experience_question"],
        complex_markers=["however"],
    )


def _answer_config() -> AnswerComposerConfig:
    return AnswerComposerConfig(
        enabled=True,
        prompt_version="test-answer-v1",
        max_output_tokens=1200,
        compose_grounded_answers=True,
        compose_uncertain_answers=True,
        maximum_evidence_items=12,
    )


def _base_interpretation(text: str):  # type: ignore[no-untyped-def]
    return RuleBasedRequestInterpreter().interpret(
        QueryParseInput(
            user_id="user-1",
            session_id="session-1",
            cutoff_time=datetime(2024, 1, 1, tzinfo=UTC),
            query_text=text,
        )
    )


def _caller(tmp_path, result: LLMCallResult):  # type: ignore[no-untyped-def]
    generator = FakeChatGenerator([result])
    ledger = ControlledLLMUsageLedger()
    cache = SqliteControlledLLMCache(tmp_path / "cache.sqlite3")
    return (
        ControlledJSONCaller(
            generator=generator,
            model_name="fake-model",
            cache=cache,
            ledger=ledger,
        ),
        generator,
        ledger,
        cache,
    )


def test_semantic_enhancer_adds_only_verbatim_validated_signal(tmp_path) -> None:
    base = _base_interpretation(
        "Find me somewhere cozy enough to talk for a first date."
    )
    output = {
        "task_type": "recommendation_request",
        "task_type_confidence": 0.96,
        "conditions": [
            {
                "field": "quiet_environment",
                "operator": "prefer",
                "value": "quiet",
                "importance": "preferred",
                "evidence_span": "cozy enough to talk",
                "confidence": 0.91,
            }
        ],
        "party_size": None,
        "missing_fields": ["desired_category"],
        "ambiguities": [],
        "overall_confidence": 0.92,
    }
    caller, generator, ledger, cache = _caller(
        tmp_path, _success(json.dumps(output))
    )
    try:
        result = ControlledSemanticEnhancer(
            config=_semantic_config(), caller=caller
        ).enhance(
            SemanticEnhancementInput(
                base_request=base.request,
                base_readiness=base.readiness,
                language="en-US",
            )
        )
        assert result.status == "success"
        assert result.accepted_signal_count == 1
        semantic = [item for item in result.request.conditions if item.source == "semantic_model"]
        assert len(semantic) == 1
        assert semantic[0].field == "quiet_environment"
        assert semantic[0].evidence_span == "cozy enough to talk"
        assert result.readiness.request_id == result.request.request_id
        assert generator.call_count == 1
        assert ledger.summary()["total_tokens"] == 120
    finally:
        cache.close()


def test_semantic_rule_field_wins_and_mandatory_is_downgraded(tmp_path) -> None:
    base = _base_interpretation("I must have a steakhouse that feels romantic.")
    output = {
        "task_type": "recommendation_request",
        "task_type_confidence": 0.99,
        "conditions": [
            {
                "field": "category",
                "operator": "includes",
                "value": "Japanese",
                "importance": "mandatory",
                "evidence_span": "steakhouse",
                "confidence": 0.99,
            },
            {
                "field": "date_suitable",
                "operator": "prefer",
                "value": True,
                "importance": "mandatory",
                "evidence_span": "romantic",
                "confidence": 0.9,
            },
        ],
        "party_size": None,
        "missing_fields": [],
        "ambiguities": [],
        "overall_confidence": 0.95,
    }
    caller, _, _, cache = _caller(tmp_path, _success(json.dumps(output)))
    try:
        result = ControlledSemanticEnhancer(
            config=_semantic_config(), caller=caller
        ).enhance(
            SemanticEnhancementInput(
                base_request=base.request,
                base_readiness=base.readiness,
                language="en-US",
            )
        )
        categories = result.request.desired_categories
        assert categories == ["Steakhouses"]
        date = [item for item in result.request.conditions if item.field == "date_suitable"]
        assert date
        assert date[-1].importance != "mandatory"
        assert any("explicit_rule_field_wins" in item for item in result.rejected_signals)
    finally:
        cache.close()


def test_invalid_semantic_json_preserves_rule_baseline(tmp_path) -> None:
    base = _base_interpretation("Find me a steakhouse.")
    caller, _, ledger, cache = _caller(tmp_path, _success("not-json"))
    try:
        result = ControlledSemanticEnhancer(
            config=_semantic_config(), caller=caller
        ).enhance(
            SemanticEnhancementInput(
                base_request=base.request,
                base_readiness=base.readiness,
                language="en-US",
            )
        )
        assert result.status == "invalid_output"
        assert result.request == base.request
        assert result.readiness == base.readiness
        assert ledger.summary()["failure_count"] == 1
    finally:
        cache.close()


def test_model_missing_field_without_evidence_cannot_force_clarification(
    tmp_path,
) -> None:
    base = _base_interpretation("Find me a steakhouse.")
    output = {
        "task_type": "recommendation_request",
        "task_type_confidence": 0.99,
        "conditions": [],
        "party_size": None,
        "missing_fields": ["budget_precision"],
        "ambiguities": ["The user did not state a budget."],
        "overall_confidence": 0.9,
    }
    caller, _, _, cache = _caller(tmp_path, _success(json.dumps(output)))
    try:
        result = ControlledSemanticEnhancer(
            config=_semantic_config(), caller=caller
        ).enhance(
            SemanticEnhancementInput(
                base_request=base.request,
                base_readiness=base.readiness,
                language="en-US",
            )
        )
        assert "budget_precision" not in result.request.missing_fields
        assert "ambiguous_requirement" not in result.request.missing_fields
        assert "missing_budget" not in result.readiness.information_gaps
        assert any("ungrounded_model_gap" in item for item in result.rejected_signals)
    finally:
        cache.close()


def _answer_input(*, uncertain: bool = False) -> AnswerCompositionInput:
    claim = ResponseClaimTrace(
        claim_id="aggregate:b1:quiet_environment",
        text=(
            "b1: quiet evidence supports; support=3, contradict=1; "
            "latest_evidence_time=2023-04-01"
        ),
        business_id="b1",
        evidence_refs=[
            EvidenceReference(
                business_id="b1", source_type="review", review_id="r1"
            )
        ],
    )
    return AnswerCompositionInput(
        context_id="scenario-1",
        turn_index=1,
        query_text="Is it quiet?",
        language="en-US",
        task_type="review_experience_question",
        response_kind="uncertain_answer" if uncertain else "grounded_answer",
        allowed_business_ids=["b1"],
        evidence=[AnswerEvidenceItem(evidence_code="E1", claim=claim)],
        reported_conflict=uncertain,
        reported_evidence_recency=True,
        recommended_official_verification=False,
    )


def test_answer_composer_binds_sentence_to_existing_evidence(tmp_path) -> None:
    output = {
        "sentences": [
            {
                "text": "Available review evidence suggests b1 is usually quiet.",
                "business_id": "b1",
                "evidence_codes": ["E1"],
            }
        ],
        "contains_uncertainty": False,
        "recommends_official_verification": False,
    }
    caller, _, _, cache = _caller(tmp_path, _success(json.dumps(output)))
    try:
        result = GroundedAnswerComposer(
            config=_answer_config(), caller=caller
        ).compose(_answer_input())
        assert result.status == "success"
        assert not result.used_deterministic_fallback
        assert result.claims[0].business_id == "b1"
        assert result.claims[0].evidence_refs[0].review_id == "r1"
    finally:
        cache.close()


def test_answer_overclaim_or_unknown_evidence_falls_back(tmp_path) -> None:
    output = {
        "sentences": [
            {
                "text": "This restaurant is definitely always quiet.",
                "business_id": "b1",
                "evidence_codes": ["E99"],
            }
        ],
        "contains_uncertainty": False,
        "recommends_official_verification": False,
    }
    caller, _, ledger, cache = _caller(tmp_path, _success(json.dumps(output)))
    try:
        value = _answer_input(uncertain=True)
        result = GroundedAnswerComposer(
            config=_answer_config(), caller=caller
        ).compose(value)
        assert result.status == "invalid_output"
        assert result.used_deterministic_fallback
        assert result.claims == [value.evidence[0].claim]
        assert ledger.traces[-1].status == "invalid_output"
    finally:
        cache.close()


def test_validated_output_is_replayed_from_cache_without_provider(tmp_path) -> None:
    output = {
        "sentences": [
            {
                "text": "Available review evidence suggests b1 is quiet.",
                "business_id": "b1",
                "evidence_codes": ["E1"],
            }
        ],
        "contains_uncertainty": False,
        "recommends_official_verification": False,
    }
    generator = FakeChatGenerator([_success(json.dumps(output))])
    ledger = ControlledLLMUsageLedger()
    cache = SqliteControlledLLMCache(tmp_path / "cache.sqlite3")
    caller = ControlledJSONCaller(
        generator=generator,
        model_name="fake-model",
        cache=cache,
        ledger=ledger,
    )
    try:
        composer = GroundedAnswerComposer(config=_answer_config(), caller=caller)
        assert composer.compose(_answer_input()).status == "success"
        replay = composer.compose(_answer_input())
        assert replay.status == "success"
        assert replay.trace.cache_hit
        assert not replay.trace.provider_called
        assert generator.call_count == 1
        assert ledger.summary()["cache_hit_count"] == 1
    finally:
        cache.close()


def test_controlled_interpreter_reports_provider_tokens(tmp_path) -> None:
    output = {
        "task_type": "recommendation_request",
        "task_type_confidence": 0.95,
        "conditions": [
            {
                "field": "quiet_environment",
                "operator": "prefer",
                "value": True,
                "importance": "preferred",
                "evidence_span": "cozy enough to talk",
                "confidence": 0.9,
            }
        ],
        "party_size": None,
        "missing_fields": ["desired_category"],
        "ambiguities": [],
        "overall_confidence": 0.9,
    }
    caller, _, _, cache = _caller(tmp_path, _success(json.dumps(output), total=140))
    try:
        interpreter = ControlledRequestInterpreter(
            ControlledSemanticEnhancer(config=_semantic_config(), caller=caller)
        )
        result = interpreter.interpret(
            QueryParseInput(
                user_id="user-1",
                session_id="session-1",
                cutoff_time=datetime(2024, 1, 1, tzinfo=UTC),
                query_text="Find me somewhere cozy enough to talk.",
            )
        )
        assert result.semantic_calls == 1
        assert result.input_tokens == 80
        assert result.output_tokens == 60
    finally:
        cache.close()


def test_missing_api_configuration_preserves_rules_and_reports_no_tokens(tmp_path) -> None:
    config = load_controlled_llm_config("configs/controlled_llm.yaml")
    with build_controlled_llm_runtime(
        project_root=tmp_path,
        config=config,
        environment={},
    ) as runtime:
        interpreter = ControlledRequestInterpreter(runtime.semantic_enhancer)
        value = QueryParseInput(
            user_id="user-1",
            session_id="session-1",
            cutoff_time=datetime(2024, 1, 1, tzinfo=UTC),
            query_text="Compare the first one and the second one.",
            referenced_business_ids=["b1", "b2"],
        )
        baseline = RuleBasedRequestInterpreter().interpret(value)
        result = interpreter.interpret(value)
        assert result.request == baseline.request
        assert result.readiness == baseline.readiness
        assert result.semantic_calls == 0
        assert result.input_tokens is None
        assert runtime.ledger.traces[-1].status == "disabled"


def test_soft_semantic_distance_is_ranked_instead_of_hard_filtered(tmp_path) -> None:
    base = _base_interpretation(
        "Find a place; roughly a five-kilometre radius would be nice."
    )
    output = {
        "task_type": "recommendation_request",
        "task_type_confidence": 0.95,
        "conditions": [
            {
                "field": "distance_km",
                "operator": "less_than_or_equal",
                "value": "5",
                "importance": "preferred",
                "evidence_span": "roughly a five-kilometre radius would be nice",
                "confidence": 0.91,
            }
        ],
        "party_size": None,
        "missing_fields": [],
        "ambiguities": [],
        "overall_confidence": 0.91,
    }
    caller, _, _, cache = _caller(tmp_path, _success(json.dumps(output)))
    try:
        result = ControlledSemanticEnhancer(
            config=_semantic_config(), caller=caller
        ).enhance(
            SemanticEnhancementInput(
                base_request=base.request,
                base_readiness=base.readiness,
                language="en-US",
            )
        )
        distances = [
            item for item in result.request.conditions if item.field == "distance_km"
        ]
        assert distances
        assert distances[-1].enforcement == "rank"
    finally:
        cache.close()


def test_candidate_comparison_may_cite_two_in_scope_businesses(tmp_path) -> None:
    left = ResponseClaimTrace(
        claim_id="left",
        text="b1 has limited positive service evidence",
        business_id="b1",
        evidence_refs=[
            EvidenceReference(business_id="b1", source_type="review", review_id="r1")
        ],
    )
    right = ResponseClaimTrace(
        claim_id="right",
        text="b2 has limited negative service evidence",
        business_id="b2",
        evidence_refs=[
            EvidenceReference(business_id="b2", source_type="review", review_id="r2")
        ],
    )
    value = AnswerCompositionInput(
        context_id="comparison-1",
        turn_index=1,
        query_text="Which one has better service?",
        language="en-US",
        task_type="candidate_comparison",
        response_kind="uncertain_answer",
        allowed_business_ids=["b1", "b2"],
        evidence=[
            AnswerEvidenceItem(evidence_code="E1", claim=left),
            AnswerEvidenceItem(evidence_code="E2", claim=right),
        ],
        reported_conflict=True,
        reported_evidence_recency=True,
        recommended_official_verification=False,
    )
    output = {
        "sentences": [
            {
                "text": "Available evidence is limited, so the comparison remains uncertain.",
                "business_id": None,
                "evidence_codes": ["E1", "E2"],
            }
        ],
        "contains_uncertainty": True,
        "recommends_official_verification": False,
    }
    caller, _, _, cache = _caller(tmp_path, _success(json.dumps(output)))
    try:
        result = GroundedAnswerComposer(
            config=_answer_config(), caller=caller
        ).compose(value)
        assert result.status == "success"
        assert result.claims[0].business_id is None
        assert {ref.business_id for ref in result.claims[0].evidence_refs} == {"b1", "b2"}
    finally:
        cache.close()
