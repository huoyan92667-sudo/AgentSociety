"""Validate proposals and reduce them into authoritative SessionMemory."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from yelp_agent.decision_readiness import (
    DecisionReadinessAnalyzer,
    identify_information_gaps,
)
from yelp_agent.query.parser import ExtractedRequestSignal, condition_from_signal
from yelp_agent.query.schema import (
    QueryParseInput,
    RecommendationRequest,
    RequestCondition,
    stable_request_id,
)

from .config import SessionMemoryConfig
from .resolver import ReferenceResolutionResult
from .schema import (
    MemoryConditionPatch,
    MemoryExtractionTrace,
    MemoryProposal,
    MemoryTurnInput,
    MemoryTurnRecord,
    MemoryTurnResult,
    SessionMemory,
)


@dataclass(frozen=True, slots=True)
class _PatchApplication:
    conditions: tuple[RequestCondition, ...]
    accepted: tuple[str, ...]
    rejected: tuple[str, ...]


class SessionMemoryReducer:
    """The only module allowed to turn a model proposal into canonical state."""

    def __init__(
        self,
        config: SessionMemoryConfig,
        *,
        analyzer: DecisionReadinessAnalyzer | None = None,
    ) -> None:
        self._config = config
        self._analyzer = analyzer or DecisionReadinessAnalyzer()

    def reduce(
        self,
        value: MemoryTurnInput,
        *,
        proposal: MemoryProposal,
        extraction: MemoryExtractionTrace,
        references: ReferenceResolutionResult,
    ) -> MemoryTurnResult:
        previous = value.previous_memory
        reset_session_request = previous is None or proposal.request_mode == "replace"
        context_text = _context_text(previous, value.query_text, proposal)
        base_conditions = _base_conditions(value, proposal)
        applied = self._apply_condition_patches(
            base_conditions,
            proposal.condition_patches,
            current_text=value.query_text,
            context_text=context_text,
            has_location=(
                value.base_request.location_center is not None
                or (
                    previous is not None
                    and previous.current_request.location_center is not None
                )
            ),
        )
        resolved_by_id = {item.reference_id: item for item in references.resolved}
        rejected_businesses = set(
            [] if reset_session_request else previous.rejected_business_ids
        )
        accepted = list(applied.accepted)
        rejected = [*references.rejected, *applied.rejected]
        for reference_id in proposal.rejected_reference_ids:
            resolved = resolved_by_id.get(reference_id)
            if resolved is None:
                rejected.append(f"reject:{reference_id}:unresolved_reference")
                continue
            rejected_businesses.add(resolved.business_id)
            accepted.append(f"reject_business:{resolved.business_id}")
        if (
            not proposal.rejected_reference_ids
            and proposal.task_type == "feedback_refinement"
            and len(references.resolved) == 1
            and _explicit_rejection(value.query_text)
        ):
            business_id = references.resolved[0].business_id
            rejected_businesses.add(business_id)
            accepted.append(f"reject_business:{business_id}:validated_text_fallback")

        reference_ids = list(
            dict.fromkeys(
                [item.business_id for item in references.resolved]
                + value.explicit_referenced_business_ids
            )
        )
        party_size = (
            proposal.party_size
            if proposal.party_size is not None
            else value.base_request.party_size
            if value.base_request.party_size is not None
            else (
                None
                if reset_session_request
                else previous.current_request.party_size
            )
        )
        location = value.base_request.location_center
        if location is None and not reset_session_request:
            location = previous.current_request.location_center
        missing = _merged_missing_fields(
            value,
            proposal,
            references_resolved=bool(reference_ids),
            party_size=party_size,
            has_location=location is not None,
        )
        request_input = QueryParseInput(
            user_id=value.base_request.user_id,
            session_id=value.base_request.session_id,
            cutoff_time=value.base_request.cutoff_time,
            query_text=context_text,
            user_latitude=None if location is None else location.latitude,
            user_longitude=None if location is None else location.longitude,
            referenced_business_ids=reference_ids,
        )
        request = RecommendationRequest(
            request_id=stable_request_id(
                request_input,
                parser_version="session-memory-v1",
            ),
            user_id=value.base_request.user_id,
            session_id=value.base_request.session_id,
            cutoff_time=value.base_request.cutoff_time,
            query_text=context_text,
            intent=_request_intent(value, proposal),
            conditions=list(applied.conditions),
            party_size=party_size,
            location_center=location,
            missing_fields=missing,
            referenced_business_ids=reference_ids,
            parse_warnings=list(
                dict.fromkeys(
                    value.base_request.parse_warnings
                    + [f"MEMORY_REJECTED:{item}" for item in rejected]
                )
            ),
            parser_version="session-memory-v1",
        )
        readiness = self._analyzer.analyze(request, ranking_source="none")
        if proposal.task_type_confidence >= self._config.minimum_task_type_confidence:
            gaps, conflicts = identify_information_gaps(request, proposal.task_type)
            if references.rejected and "ambiguous_reference" not in gaps:
                gaps = tuple(sorted({*gaps, "ambiguous_reference"}))
            readiness = readiness.model_copy(
                update={
                    "task_type": proposal.task_type,
                    "task_type_reason_code": "step34_memory_proposal",
                    "information_gaps": list(gaps),
                    "conflict_fields": list(conflicts),
                }
            )
        answers = (
            {} if reset_session_request else dict(previous.clarification_answers)
        )
        for answer in proposal.clarification_answers:
            if answer.evidence_span not in value.query_text:
                rejected.append(
                    f"clarification:{answer.information_gap}:evidence_not_in_message"
                )
                continue
            answers[answer.information_gap] = answer.value
            accepted.append(f"clarification:{answer.information_gap}")
        relative = (
            [] if reset_session_request else list(previous.relative_preferences)
        )
        relative_references = (
            {}
            if reset_session_request
            else dict(previous.relative_preference_references)
        )
        for preference in proposal.relative_preferences:
            if preference.evidence_span not in value.query_text:
                rejected.append(
                    f"relative:{preference.field}:evidence_not_in_message"
                )
                continue
            relative = [item for item in relative if item.field != preference.field]
            relative.append(preference)
            resolved_business_ids = [
                item.business_id for item in references.resolved
            ]
            if resolved_business_ids:
                relative_references[preference.field] = resolved_business_ids[0]
            else:
                relative_references.pop(preference.field, None)
            accepted.append(
                f"relative:{preference.field}:{preference.direction}"
            )
        long_term = list(
            dict.fromkeys(
                ([] if previous is None else previous.long_term_candidates)
                + proposal.long_term_candidates
                + [
                    f"{item.field}:{item.operator}:{item.value}"
                    for item in proposal.condition_patches
                    if item.lifetime == "long_term_candidate"
                ]
            )
        )
        turn = MemoryTurnRecord(
            turn_index=value.current_turn,
            query_text=value.query_text,
            task_type=readiness.task_type,
            accepted_changes=accepted,
            rejected_changes=rejected,
            resolved_references=list(references.resolved),
            extraction=extraction,
        )
        memory = SessionMemory(
            session_id=request.session_id,
            user_id=request.user_id,
            cutoff_time=request.cutoff_time,
            revision=1 if previous is None else previous.revision + 1,
            current_request=request,
            current_task_type=readiness.task_type,
            information_gaps=list(readiness.information_gaps),
            rejected_business_ids=sorted(rejected_businesses),
            last_presented_business_ids=(
                [] if previous is None else previous.last_presented_business_ids
            ),
            presented_candidate_sets=(
                [] if previous is None else previous.presented_candidate_sets
            ),
            current_business_scope=(
                [] if previous is None else previous.current_business_scope
            ),
            business_scope_known=(
                False if previous is None else previous.business_scope_known
            ),
            clarification_answers=answers,
            relative_preferences=relative,
            relative_preference_references=relative_references,
            semantic_summary=(
                proposal.semantic_summary
                if proposal.semantic_summary is not None
                else context_text[:1200]
            ),
            long_term_candidates=long_term,
            recent_turns=(
                ([] if previous is None else previous.recent_turns) + [turn]
            )[-self._config.max_recent_turns :],
        )
        return MemoryTurnResult(
            request=request,
            readiness=readiness,
            memory=memory,
            proposal=proposal,
            extraction=extraction,
            accepted_changes=accepted,
            rejected_changes=rejected,
        )

    def _apply_condition_patches(
        self,
        base: list[RequestCondition],
        patches: list[MemoryConditionPatch],
        *,
        current_text: str,
        context_text: str,
        has_location: bool,
    ) -> _PatchApplication:
        conditions = list(base)
        accepted: list[str] = []
        rejected: list[str] = []
        for index, patch in enumerate(patches):
            prefix = f"patch_{index}:{patch.field}:{patch.operation}"
            if patch.confidence < self._config.minimum_patch_confidence:
                rejected.append(f"{prefix}:low_confidence")
                continue
            if patch.evidence_span not in current_text:
                rejected.append(f"{prefix}:evidence_not_in_message")
                continue
            if (
                patch.field in {"distance_km", "budget_per_person", "price_level"}
                and patch.operation in {"add", "replace"}
                and not _contains_number(patch.evidence_span)
            ):
                rejected.append(f"{prefix}:numeric_value_not_explicit")
                continue
            matching = [item for item in conditions if item.field == patch.field]
            if patch.operation == "remove":
                if not matching:
                    rejected.append(f"{prefix}:condition_not_found")
                    continue
                if any(_is_hard_condition(item) for item in matching) and not _explicit_removal(
                    current_text, patch.evidence_span
                ):
                    rejected.append(f"{prefix}:hard_constraint_removal_not_explicit")
                    continue
                conditions = [item for item in conditions if item.field != patch.field]
                accepted.append(prefix)
                continue
            assert patch.operator is not None
            assert patch.value is not None
            assert patch.importance is not None
            offset = context_text.rfind(patch.evidence_span)
            signal = ExtractedRequestSignal(
                field=patch.field,
                operator=patch.operator,
                value=patch.value,
                importance=patch.importance,
                evidence_span=patch.evidence_span,
                evidence_start=max(0, offset),
                evidence_end=max(0, offset) + len(patch.evidence_span),
                confidence=patch.confidence,
                source="semantic_model",
            )
            try:
                condition, _ = condition_from_signal(
                    signal,
                    has_user_location=has_location,
                )
            except (TypeError, ValueError) as exc:
                rejected.append(f"{prefix}:invalid_condition:{type(exc).__name__}")
                continue
            if patch.operation == "replace":
                if any(_is_hard_condition(item) for item in matching):
                    if not _is_hard_condition(condition) and not _explicit_removal(
                        current_text, patch.evidence_span
                    ):
                        rejected.append(f"{prefix}:hard_constraint_downgrade")
                        continue
                conditions = [item for item in conditions if item.field != patch.field]
            key = _condition_key(condition)
            conditions = [item for item in conditions if _condition_key(item) != key]
            conditions.append(condition)
            accepted.append(prefix)
        return _PatchApplication(
            tuple(_deduplicate_conditions(conditions)),
            tuple(accepted),
            tuple(rejected),
        )


def record_memory_observation(
    memory: SessionMemory | None,
    *,
    turn_index: int,
    business_scope: list[str] | None = None,
    presented_business_ids: list[str] | None = None,
    max_presented_sets: int = 5,
) -> SessionMemory | None:
    """Update only code-owned scope/presentation facts after a validated outcome."""

    if memory is None:
        return None
    updates: dict[str, object] = {}
    if business_scope is not None:
        updates["current_business_scope"] = list(business_scope)
        updates["business_scope_known"] = True
    if presented_business_ids:
        from .schema import PresentedCandidateSet

        sets = memory.presented_candidate_sets + [
            PresentedCandidateSet(
                turn_index=turn_index,
                business_ids=list(presented_business_ids),
            )
        ]
        updates["last_presented_business_ids"] = list(presented_business_ids)
        updates["presented_candidate_sets"] = sets[-max_presented_sets:]
    return memory if not updates else memory.model_copy(update=updates)


def _base_conditions(
    value: MemoryTurnInput,
    proposal: MemoryProposal,
) -> list[RequestCondition]:
    previous = value.previous_memory
    if previous is None or proposal.request_mode == "replace":
        return list(value.base_request.conditions)
    if proposal.request_mode == "no_change":
        return list(previous.current_request.conditions)
    conditions = list(previous.current_request.conditions)
    for current in value.base_request.conditions:
        if current.field in {
            "distance_km",
            "budget_per_person",
            "price_level",
        } or current.field not in {"category"}:
            conditions = [item for item in conditions if item.field != current.field]
        conditions.append(current)
    return _deduplicate_conditions(conditions)


def _context_text(
    previous: SessionMemory | None,
    current_text: str,
    proposal: MemoryProposal,
) -> str:
    if previous is None or proposal.request_mode == "replace":
        return current_text
    if proposal.request_mode == "no_change":
        return previous.current_request.query_text
    prior = previous.semantic_summary or previous.current_request.query_text
    combined = f"{prior}\n{current_text}".strip()
    return combined[-2000:]


def _merged_missing_fields(
    value: MemoryTurnInput,
    proposal: MemoryProposal,
    *,
    references_resolved: bool,
    party_size: int | None,
    has_location: bool,
) -> list[str]:
    if value.previous_memory is None or proposal.request_mode == "replace":
        missing = set(value.base_request.missing_fields)
    else:
        missing = set(value.previous_memory.current_request.missing_fields)
    fields = {item.field for item in value.base_request.conditions}
    fields.update(
        item.field for item in proposal.condition_patches if item.operation != "remove"
    )
    if "budget_per_person" in fields or "price_level" in fields:
        missing.discard("budget_precision")
    if has_location:
        missing.discard("user_location")
    if party_size is not None:
        missing.discard("party_size")
    if references_resolved:
        missing.discard("ambiguous_requirement")
    for answer in proposal.clarification_answers:
        mapping = {
            "missing_location": "user_location",
            "missing_budget": "budget_precision",
            "missing_party_size": "party_size",
            "ambiguous_reference": "ambiguous_requirement",
        }
        field = mapping.get(answer.information_gap)
        if field is not None:
            missing.discard(field)
    return sorted(missing)


def _request_intent(value: MemoryTurnInput, proposal: MemoryProposal) -> str:
    if proposal.task_type == "feedback_refinement":
        return "feedback_refinement"
    if proposal.task_type == "candidate_comparison":
        return "candidate_comparison"
    if proposal.task_type == "business_detail_question":
        return "business_detail_question"
    if proposal.task_type == "recommendation_request":
        return "recommendation_request"
    return value.base_request.intent


def _condition_key(item: RequestCondition) -> tuple[str, str, str, str]:
    return (item.field, item.operator, str(item.value), item.enforcement)


def _deduplicate_conditions(
    conditions: list[RequestCondition],
) -> list[RequestCondition]:
    unique: dict[tuple[str, str, str, str], RequestCondition] = {}
    for item in conditions:
        unique[_condition_key(item)] = item
    return list(unique.values())


def _explicit_removal(text: str, span: str) -> bool:
    lowered = f"{text} {span}".casefold()
    return any(
        marker in lowered
        for marker in (
            "取消",
            "不用",
            "不再",
            "去掉",
            "放宽",
            "remove",
            "drop",
            "no longer",
            "relax",
        )
    )


def _contains_number(text: str) -> bool:
    return re.search(r"\d+(?:\.\d+)?", text) is not None


def _explicit_rejection(text: str) -> bool:
    return re.search(
        r"(?:太贵|太遠|太远|太吵|不合适|不要(?:这|那|刚才)|换(?:一|个|家)|"
        r"too expensive|too far|too noisy|not suitable|something else|another)",
        text,
        re.IGNORECASE,
    ) is not None


def _is_hard_condition(condition: RequestCondition) -> bool:
    return condition.importance == "mandatory" or condition.enforcement in {
        "filter",
        "clarify",
    }


def memory_digest(memory: SessionMemory) -> str:
    """Stable digest useful for cache and deterministic regression tests."""

    return hashlib.sha256(memory.model_dump_json().encode("utf-8")).hexdigest()
