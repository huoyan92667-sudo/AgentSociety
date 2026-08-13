"""Compile canonical SessionMemory into one downstream executable request."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .schema import (
    EffectiveRelativePreference,
    EffectiveSessionRequest,
    RelativePreference,
    SessionMemory,
)


def compile_effective_request(memory: SessionMemory) -> EffectiveSessionRequest:
    """Return the only request representation downstream tools should execute."""

    relative = [_compile_relative(memory, item) for item in memory.relative_preferences]
    query_document = _query_document(memory, relative)
    effective_id = _effective_id(memory, relative)
    request = memory.current_request.model_copy(
        update={
            "request_id": effective_id,
            "query_text": query_document,
            "parse_warnings": list(
                dict.fromkeys(
                    [*memory.current_request.parse_warnings, "EFFECTIVE_SESSION_REQUEST_V1"]
                )
            ),
            "parser_version": "session-effective-v1",
        }
    )
    latest_query = (
        memory.recent_turns[-1].query_text
        if memory.recent_turns
        else memory.current_request.query_text
    )
    return EffectiveSessionRequest(
        effective_request_id=effective_id,
        source_request_id=memory.current_request.request_id,
        revision=memory.revision,
        task_type=memory.current_task_type,
        request=request,
        relative_preferences=relative,
        rejected_business_ids=list(memory.rejected_business_ids),
        clarification_answers=dict(memory.clarification_answers),
        latest_user_query=latest_query,
    )


def effective_request_from_snapshot(
    snapshot: dict[str, Any],
) -> EffectiveSessionRequest | None:
    """Parse the compiled request exposed in a validated Agent tool snapshot."""

    raw = snapshot.get("effective_request")
    if not isinstance(raw, dict):
        return None
    return EffectiveSessionRequest.model_validate(raw)


def _compile_relative(
    memory: SessionMemory,
    preference: RelativePreference,
) -> EffectiveRelativePreference:
    source_turn = None
    marker = f"relative:{preference.field}:{preference.direction}"
    for turn in reversed(memory.recent_turns):
        if marker in turn.accepted_changes:
            source_turn = turn.turn_index
            break
    return EffectiveRelativePreference(
        field=preference.field,
        direction=preference.direction,
        evidence_span=preference.evidence_span,
        confidence=preference.confidence,
        reference_business_id=memory.relative_preference_references.get(
            preference.field
        ),
        source_turn_index=source_turn,
    )


def _query_document(
    memory: SessionMemory,
    relative: list[EffectiveRelativePreference],
) -> str:
    request = memory.current_request
    lines = ["Active recommendation request reconstructed from accepted session state."]
    lines.append(f"Task: {memory.current_task_type}.")
    for condition in sorted(
        request.conditions,
        key=lambda item: (
            item.field,
            item.operator,
            str(item.value),
            item.enforcement,
        ),
    ):
        lines.append(
            "Requirement: "
            f"{condition.field} {condition.operator} {condition.value} "
            f"({condition.importance}; {condition.enforcement})."
        )
    if request.party_size is not None:
        lines.append(f"Party size: {request.party_size}.")
    if request.location_center is not None:
        lines.append(
            "User location: "
            f"{request.location_center.latitude:.6f}, "
            f"{request.location_center.longitude:.6f}."
        )
    for item in relative:
        lines.append(f"Relative preference: {_relative_phrase(item)}.")
    if memory.rejected_business_ids:
        lines.append(
            f"Exclude {len(memory.rejected_business_ids)} user-rejected businesses "
            "by their validated IDs."
        )
    return "\n".join(lines)


def _relative_phrase(item: EffectiveRelativePreference) -> str:
    phrases = {
        ("price", "lower"): "Prefer a lower price than the referenced result",
        ("price", "higher"): "Prefer a higher price level than the referenced result",
        ("distance", "closer"): "Prefer a closer business than the referenced result",
        ("distance", "farther"): "Prefer a farther business than the referenced result",
        ("noise", "quieter"): "Prefer a quieter business than the referenced result",
        ("crowding", "less_crowded"): "Prefer a less crowded business than the referenced result",
    }
    return phrases.get(
        (item.field, item.direction),
        f"Prefer {item.field} in the {item.direction} direction",
    )


def _effective_id(
    memory: SessionMemory,
    relative: list[EffectiveRelativePreference],
) -> str:
    request = memory.current_request
    payload = {
        "user_id": request.user_id,
        "session_id": request.session_id,
        "cutoff_time": request.cutoff_time.isoformat(),
        "intent": request.intent,
        "conditions": [
            item.model_dump(mode="json")
            for item in sorted(
                request.conditions,
                key=lambda value: (
                    value.field,
                    value.operator,
                    str(value.value),
                    value.enforcement,
                ),
            )
        ],
        "party_size": request.party_size,
        "location_center": (
            None
            if request.location_center is None
            else request.location_center.model_dump(mode="json")
        ),
        "missing_fields": sorted(request.missing_fields),
        "referenced_business_ids": sorted(request.referenced_business_ids),
        "relative_preferences": [item.model_dump(mode="json") for item in relative],
        "rejected_business_ids": sorted(memory.rejected_business_ids),
        "clarification_answers": memory.clarification_answers,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
