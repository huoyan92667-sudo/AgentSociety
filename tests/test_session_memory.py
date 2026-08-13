from __future__ import annotations

from datetime import datetime
from pathlib import Path

from yelp_agent.agent.llm import LLMCallResult
from yelp_agent.agent_harness import RuleBasedRequestInterpreter
from yelp_agent.controlled_llm.cache import SqliteControlledLLMCache
from yelp_agent.controlled_llm.gateway import ControlledJSONCaller
from yelp_agent.controlled_llm.ledger import ControlledLLMUsageLedger
from yelp_agent.query import QueryParseInput
from yelp_agent.session_memory.config import SessionMemoryConfig
from yelp_agent.session_memory.context import compact_memory
from yelp_agent.session_memory.extractor import (
    DeepSeekMemoryExtractor,
    MemoryExtractionAttempt,
)
from yelp_agent.session_memory.manager import SessionMemoryManager
from yelp_agent.session_memory.reducer import record_memory_observation
from yelp_agent.session_memory.schema import (
    MemoryConditionPatch,
    MemoryExtractionTrace,
    MemoryProposal,
    MemoryReferenceMention,
    RelativePreference,
    MemoryTurnInput,
)


def _config(**updates: object) -> SessionMemoryConfig:
    value = SessionMemoryConfig(
        memory_version="step34-test",
        enabled=True,
        prompt_version="step34-test-prompt",
        cache_relative_path="memory.sqlite3",
    )
    return value.model_copy(update=updates)


def _base(text: str, *, references: list[str] | None = None):
    value = QueryParseInput(
        user_id="user-1",
        session_id="session-1",
        cutoff_time=datetime(2022, 1, 1),
        query_text=text,
        referenced_business_ids=references or [],
    )
    interpreted = RuleBasedRequestInterpreter().interpret(value)
    return value, interpreted


class _ScriptedExtractor:
    def __init__(self, *proposals: MemoryProposal) -> None:
        self._proposals = list(proposals)

    def extract(self, value: MemoryTurnInput) -> MemoryExtractionAttempt:
        del value
        proposal = self._proposals.pop(0)
        return MemoryExtractionAttempt(
            proposal=proposal,
            trace=MemoryExtractionTrace(
                status="success",
                extractor="deepseek",
                prompt_version="fake-memory-v1",
                provider_called=True,
                model="fake-deepseek",
                latency_ms=12,
                attempt_count=1,
                input_tokens=100,
                output_tokens=30,
                total_tokens=130,
            ),
        )


def _proposal(**updates: object) -> MemoryProposal:
    value = MemoryProposal(
        request_mode="patch",
        task_type="recommendation_request",
        task_type_confidence=0.98,
        confidence=0.95,
    )
    return value.model_copy(update=updates)


def _turn(
    text: str,
    *,
    previous=None,
    references: list[str] | None = None,
    turn_index: int = 1,
) -> MemoryTurnInput:
    _, interpreted = _base(text, references=references)
    return MemoryTurnInput(
        query_text=text,
        language="en-US",
        current_turn=turn_index,
        base_request=interpreted.request,
        base_readiness=interpreted.readiness,
        previous_memory=previous,
        explicit_referenced_business_ids=references or [],
    )


def test_llm_proposal_updates_memory_but_code_resolves_the_business_id() -> None:
    manager = SessionMemoryManager(
        config=_config(),
        primary_extractor=_ScriptedExtractor(
            _proposal(request_mode="replace", semantic_summary="Find a steakhouse."),
            _proposal(
                task_type="feedback_refinement",
                references=[
                    MemoryReferenceMention(
                        reference_id="R1",
                        expression="the first one",
                        ordinal=1,
                    )
                ],
                rejected_reference_ids=["R1"],
                condition_patches=[
                    MemoryConditionPatch(
                        operation="replace",
                        field="budget_per_person",
                        operator="less_than_or_equal",
                        value=40,
                        importance="mandatory",
                        evidence_span="$40",
                        confidence=0.98,
                    )
                ],
                semantic_summary="Find a steakhouse under $40 and exclude the first result.",
            ),
        ),
    )
    first = manager.update(_turn("Recommend a steakhouse", turn_index=1))
    presented = record_memory_observation(
        first.memory,
        turn_index=1,
        business_scope=["b1", "b2", "b3", "b4", "b5"],
        presented_business_ids=["b1", "b2", "b3", "b4", "b5"],
    )
    assert presented is not None

    second = manager.update(
        _turn(
            "The first one is too expensive; keep it under $40.",
            previous=presented,
            turn_index=2,
        )
    )

    assert second.readiness.task_type == "feedback_refinement"
    assert second.memory.rejected_business_ids == ["b1"]
    assert second.request.referenced_business_ids == ["b1"]
    assert "Steakhouses" in second.request.desired_categories
    assert any(
        item.field == "budget_per_person" and item.value == 40
        for item in second.request.conditions
    )
    assert second.memory.recent_turns[-1].resolved_references[0].business_id == "b1"


def test_model_cannot_remove_a_hard_constraint_without_explicit_user_evidence() -> None:
    manager = SessionMemoryManager(
        config=_config(),
        primary_extractor=_ScriptedExtractor(
            _proposal(request_mode="replace"),
            _proposal(
                condition_patches=[
                    MemoryConditionPatch(
                        operation="remove",
                        field="distance_km",
                        evidence_span="something else",
                        confidence=0.99,
                    )
                ]
            ),
        ),
    )
    first = manager.update(
        _turn("Find a steakhouse within 5 km", turn_index=1)
    )
    second = manager.update(
        _turn("Please show me something else", previous=first.memory, turn_index=2)
    )

    assert any(item.field == "distance_km" for item in second.request.conditions)
    assert any(
        "hard_constraint_removal_not_explicit" in item
        for item in second.rejected_changes
    )


def test_model_business_id_outside_visible_scope_is_rejected() -> None:
    manager = SessionMemoryManager(
        config=_config(),
        primary_extractor=_ScriptedExtractor(
            _proposal(request_mode="replace"),
            _proposal(
                task_type="business_detail_question",
                references=[
                    MemoryReferenceMention(
                        reference_id="R1",
                        expression="invented",
                        explicit_business_id="outside-business",
                    )
                ],
            ),
        ),
    )
    first = manager.update(_turn("Recommend a steakhouse"))
    presented = record_memory_observation(
        first.memory,
        turn_index=1,
        business_scope=["b1", "b2"],
        presented_business_ids=["b1", "b2"],
    )
    assert presented is not None

    second = manager.update(
        _turn("Is that place quiet?", previous=presented, turn_index=2)
    )

    assert second.request.referenced_business_ids == []
    assert "ambiguous_reference" in second.readiness.information_gaps
    assert any("business_out_of_visible_scope" in item for item in second.rejected_changes)


def test_relative_preference_is_stored_without_accepting_an_invented_threshold() -> None:
    manager = SessionMemoryManager(
        config=_config(),
        primary_extractor=_ScriptedExtractor(
            _proposal(request_mode="replace"),
            _proposal(
                task_type="feedback_refinement",
                references=[
                    MemoryReferenceMention(
                        reference_id="R1",
                        expression="the first one",
                        ordinal=1,
                    )
                ],
                condition_patches=[
                    MemoryConditionPatch(
                        operation="add",
                        field="distance_km",
                        operator="less_than_or_equal",
                        value=5,
                        importance="strong",
                        evidence_span="closer",
                        confidence=0.9,
                    )
                ],
                relative_preferences=[
                    RelativePreference(
                        field="distance",
                        direction="closer",
                        evidence_span="closer",
                        confidence=0.9,
                    )
                ],
            ),
        ),
    )
    first = manager.update(_turn("Recommend a steakhouse"))
    presented = record_memory_observation(
        first.memory,
        turn_index=1,
        business_scope=["b1", "b2"],
        presented_business_ids=["b1", "b2"],
    )
    assert presented is not None

    second = manager.update(
        _turn("The first one is too far; find something closer.", previous=presented, turn_index=2)
    )

    assert second.memory.rejected_business_ids == ["b1"]
    assert second.memory.relative_preferences[0].direction == "closer"
    assert not any(
        item.field == "distance_km" and item.value == 5
        for item in second.request.conditions
    )
    assert any("numeric_value_not_explicit" in item for item in second.rejected_changes)


def test_provider_failure_preserves_memory_and_uses_rule_reference_fallback() -> None:
    class _FailingExtractor:
        def extract(self, value: MemoryTurnInput) -> MemoryExtractionAttempt:
            del value
            return MemoryExtractionAttempt(
                proposal=None,
                trace=MemoryExtractionTrace(
                    status="provider_failure",
                    extractor="deepseek",
                    prompt_version="failed-v1",
                    provider_called=True,
                    model="deepseek",
                    latency_ms=90_000,
                    attempt_count=3,
                    failure_reason="timeout",
                ),
            )

    initial = SessionMemoryManager(config=_config()).update(
        _turn("Recommend a steakhouse")
    )
    presented = record_memory_observation(
        initial.memory,
        turn_index=1,
        business_scope=["b1", "b2"],
        presented_business_ids=["b1", "b2"],
    )
    assert presented is not None
    manager = SessionMemoryManager(
        config=_config(), primary_extractor=_FailingExtractor()
    )

    result = manager.update(
        _turn(
            "The first one is too expensive, show another.",
            previous=presented,
            turn_index=2,
        )
    )

    assert result.extraction.status == "rule_fallback"
    assert result.extraction.primary_failure_reason == "timeout"
    assert result.memory.rejected_business_ids == ["b1"]
    assert "Steakhouses" in result.request.desired_categories


class _JSONGenerator:
    def __init__(self) -> None:
        self.calls = 0
        self.contents: list[str] = []

    def generate(self, messages):
        self.calls += 1
        self.contents.append(messages[-1].content)
        assert "ground_truth" not in messages[-1].content
        assert "private-business-id" not in messages[-1].content
        return LLMCallResult(
            status="success",
            content=_proposal(
                request_mode="replace",
                semantic_summary=f"summary-{self.calls}",
            ).model_dump_json(),
            model="fake-deepseek",
            latency_ms=5,
            attempt_count=1,
            input_tokens=20,
            output_tokens=10,
            total_tokens=30,
        )


def test_deepseek_extractor_is_structured_and_cache_is_input_specific(
    tmp_path: Path,
) -> None:
    generator = _JSONGenerator()
    with SqliteControlledLLMCache(tmp_path / "cache.sqlite3") as cache:
        extractor = DeepSeekMemoryExtractor(
            caller=ControlledJSONCaller(
                generator=generator,
                model_name="fake-deepseek",
                cache=cache,
                ledger=ControlledLLMUsageLedger(),
            ),
            config=_config(),
        )
        first = extractor.extract(_turn("Recommend a steakhouse"))
        second = extractor.extract(_turn("Recommend an Italian restaurant"))
        repeated = extractor.extract(_turn("Recommend a steakhouse"))

    assert first.proposal is not None
    assert second.proposal is not None
    assert repeated.proposal is not None
    assert generator.calls == 2
    assert repeated.trace.cache_hit is True
    assert first.proposal.semantic_summary == repeated.proposal.semantic_summary


def test_deepseek_prompt_keeps_business_ids_local(tmp_path: Path) -> None:
    generator = _JSONGenerator()
    initial = SessionMemoryManager(config=_config()).update(
        _turn("Recommend a steakhouse")
    )
    presented = record_memory_observation(
        initial.memory,
        turn_index=1,
        business_scope=["private-business-id", "private-other-id"],
        presented_business_ids=["private-business-id"],
    )
    assert presented is not None
    with SqliteControlledLLMCache(tmp_path / "safe-cache.sqlite3") as cache:
        extractor = DeepSeekMemoryExtractor(
            caller=ControlledJSONCaller(
                generator=generator,
                model_name="fake-deepseek",
                cache=cache,
                ledger=ControlledLLMUsageLedger(),
            ),
            config=_config(),
        )
        result = extractor.extract(
            _turn(
                "Tell me whether private-business-id is quiet.",
                previous=presented,
                references=["private-business-id"],
                turn_index=2,
            )
        )

    assert result.proposal is not None
    assert "EXPLICIT_1" in generator.contents[-1]


def test_deepseek_extractor_normalizes_reference_labels_only(tmp_path: Path) -> None:
    class _Generator:
        def generate(self, messages):
            del messages
            payload = _proposal(
                references=[
                    MemoryReferenceMention(
                        reference_id="R1",
                        expression="first",
                        ordinal=1,
                    )
                ],
                rejected_reference_ids=["R1"],
            ).model_dump(mode="json")
            payload["references"][0]["reference_id"] = "ref_1"
            payload["rejected_reference_ids"] = ["ref_1"]
            import json

            return LLMCallResult(
                status="success",
                content=json.dumps(payload),
                model="fake-deepseek",
                latency_ms=2,
                attempt_count=1,
                input_tokens=10,
                output_tokens=10,
                total_tokens=20,
            )

    with SqliteControlledLLMCache(tmp_path / "cache.sqlite3") as cache:
        result = DeepSeekMemoryExtractor(
            caller=ControlledJSONCaller(
                generator=_Generator(),
                model_name="fake-deepseek",
                cache=cache,
                ledger=ControlledLLMUsageLedger(),
            ),
            config=_config(),
        ).extract(_turn("The first one is too expensive"))

    assert result.trace.status == "success"
    assert result.proposal is not None
    assert result.proposal.references[0].reference_id == "R1"
    assert result.proposal.rejected_reference_ids == ["R1"]


def test_deepseek_extractor_moves_relative_alias_out_of_numeric_patch(
    tmp_path: Path,
) -> None:
    class _Generator:
        def generate(self, messages):
            del messages
            payload = _proposal().model_dump(mode="json")
            payload["condition_patches"] = [
                {
                    "operation": "add",
                    "field": "distance",
                    "operator": "prefer",
                    "value": True,
                    "importance": "strong",
                    "evidence_span": "近一点",
                    "confidence": 0.9,
                    "lifetime": "session",
                }
            ]
            import json

            return LLMCallResult(
                status="success",
                content=json.dumps(payload, ensure_ascii=False),
                model="fake-deepseek",
                latency_ms=2,
                attempt_count=1,
                input_tokens=10,
                output_tokens=10,
                total_tokens=20,
            )

    with SqliteControlledLLMCache(tmp_path / "cache.sqlite3") as cache:
        result = DeepSeekMemoryExtractor(
            caller=ControlledJSONCaller(
                generator=_Generator(),
                model_name="fake-deepseek",
                cache=cache,
                ledger=ControlledLLMUsageLedger(),
            ),
            config=_config(),
        ).extract(_turn("换一家近一点的"))

    assert result.trace.status == "success"
    assert result.proposal is not None
    assert result.proposal.condition_patches == []
    assert result.proposal.relative_preferences[0].direction == "closer"


def test_deepseek_extractor_deduplicates_equivalent_relative_preferences(
    tmp_path: Path,
) -> None:
    class _Generator:
        def generate(self, messages):
            del messages
            payload = _proposal().model_dump(mode="json")
            item = {
                "field": "distance",
                "direction": "closer",
                "evidence_span": "closer",
                "confidence": 0.9,
                "lifetime": "session",
            }
            payload["relative_preferences"] = [item, item]
            import json

            return LLMCallResult(
                status="success",
                content=json.dumps(payload),
                model="fake-deepseek",
                latency_ms=2,
                attempt_count=1,
                input_tokens=10,
                output_tokens=10,
                total_tokens=20,
            )

    with SqliteControlledLLMCache(tmp_path / "cache.sqlite3") as cache:
        result = DeepSeekMemoryExtractor(
            caller=ControlledJSONCaller(
                generator=_Generator(),
                model_name="fake-deepseek",
                cache=cache,
                ledger=ControlledLLMUsageLedger(),
            ),
            config=_config(),
        ).extract(_turn("find something closer"))

    assert result.proposal is not None
    assert len(result.proposal.relative_preferences) == 1


def test_compact_context_contains_authoritative_fields_not_raw_observations() -> None:
    result = SessionMemoryManager(config=_config()).update(
        _turn("Recommend a steakhouse")
    )
    context = compact_memory(result.memory)

    assert context.revision == 1
    assert context.hard_constraints or context.soft_preferences
    assert "observations" not in context.model_dump()


def test_followups_only_skips_provider_for_initial_memory_bootstrap() -> None:
    extractor = _ScriptedExtractor(_proposal())
    manager = SessionMemoryManager(
        config=_config(call_mode="followups_only"),
        primary_extractor=extractor,
    )

    first = manager.update(_turn("Recommend a steakhouse"))
    second = manager.update(
        _turn("Make it quieter", previous=first.memory, turn_index=2)
    )

    assert first.extraction.status == "deterministic_bootstrap"
    assert first.extraction.provider_called is False
    assert second.extraction.status == "success"
    assert second.extraction.provider_called is True
