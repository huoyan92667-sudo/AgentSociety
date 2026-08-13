"""LLM-first and deterministic fallback adapters for memory proposals."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from yelp_agent.agent.llm import LLMMessage
from yelp_agent.controlled_llm.gateway import ControlledJSONCaller

from .config import SessionMemoryConfig
from .context import compact_memory
from .schema import (
    MemoryExtractionTrace,
    MemoryProposal,
    MemoryReferenceMention,
    MemoryTurnInput,
)


@dataclass(frozen=True, slots=True)
class MemoryExtractionAttempt:
    proposal: MemoryProposal | None
    trace: MemoryExtractionTrace


class MemoryProposalExtractor(Protocol):
    def extract(self, value: MemoryTurnInput) -> MemoryExtractionAttempt: ...


class DeepSeekMemoryExtractor:
    """Ask an OpenAI-compatible model for one strict, non-authoritative proposal."""

    def __init__(
        self,
        *,
        caller: ControlledJSONCaller,
        config: SessionMemoryConfig,
    ) -> None:
        self._caller = caller
        self._config = config

    def extract(self, value: MemoryTurnInput) -> MemoryExtractionAttempt:
        payload, explicit_aliases = _visible_prompt_payload(value)
        called = self._caller.call(
            capability="memory_update",
            prompt_version=self._config.prompt_version,
            input_payload=payload,
            messages=_messages(payload),
            output_model=MemoryProposal,
            normalize_payload=lambda raw: _restore_explicit_aliases(
                _normalize_memory_payload(raw), explicit_aliases
            ),
            context_id=value.base_request.session_id,
            turn_index=value.current_turn,
        )
        trace = called.trace
        status = trace.status
        mapped = (
            "success"
            if status == "success"
            else "disabled"
            if status == "disabled"
            else "invalid_output"
            if status == "invalid_output"
            else "provider_failure"
        )
        return MemoryExtractionAttempt(
            proposal=called.output,
            trace=MemoryExtractionTrace(
                status=mapped,
                extractor="deepseek",
                prompt_version=trace.prompt_version,
                provider_called=trace.provider_called,
                cache_hit=trace.cache_hit,
                model=trace.model,
                latency_ms=trace.latency_ms,
                attempt_count=trace.attempt_count,
                input_tokens=trace.input_tokens,
                output_tokens=trace.output_tokens,
                total_tokens=trace.total_tokens,
                failure_reason=trace.failure_reason,
            ),
        )


class RuleMemoryExtractor:
    """Small fallback for explicit references and obvious feedback language."""

    prompt_version = "step34-rule-fallback-v1"

    def extract(self, value: MemoryTurnInput) -> MemoryExtractionAttempt:
        text = value.query_text
        references: list[MemoryReferenceMention] = []
        for index, business_id in enumerate(
            value.explicit_referenced_business_ids, start=1
        ):
            references.append(
                MemoryReferenceMention(
                    reference_id=f"R{index}",
                    expression=business_id,
                    explicit_business_id=business_id,
                )
            )
        ordinal = _ordinal_reference(text)
        if ordinal is not None and not references:
            references.append(
                MemoryReferenceMention(
                    reference_id="R1",
                    expression=_ordinal_expression(text, ordinal),
                    ordinal=ordinal,
                )
            )
        rejects = _rejects_previous(text)
        if rejects and not references and value.previous_memory is not None:
            references.append(
                MemoryReferenceMention(
                    reference_id="R1",
                    expression="previous recommendation",
                    ordinal=1,
                )
            )
        request_mode = "replace" if value.previous_memory is None else "patch"
        lowered = text.casefold()
        if value.previous_memory is not None and any(
            marker in lowered
            for marker in ("重新开始", "全新需求", "start over", "new request")
        ):
            request_mode = "replace"
        task_type = value.base_readiness.task_type
        if (
            task_type == "unknown"
            and value.previous_memory is not None
            and value.previous_memory.information_gaps
        ):
            task_type = value.previous_memory.current_task_type
        proposal = MemoryProposal(
            request_mode=request_mode,
            task_type=task_type,
            task_type_confidence=1.0,
            references=references,
            rejected_reference_ids=(
                [item.reference_id for item in references] if rejects else []
            ),
            condition_patches=[],
            clarification_answers=[],
            relative_preferences=[],
            party_size=value.base_request.party_size,
            semantic_summary=None,
            long_term_candidates=[],
            confidence=0.72,
            uncertainty_reasons=[],
        )
        return MemoryExtractionAttempt(
            proposal=proposal,
            trace=MemoryExtractionTrace(
                status="rule_fallback",
                extractor="rule",
                prompt_version=self.prompt_version,
            ),
        )


def _visible_prompt_payload(
    value: MemoryTurnInput,
) -> tuple[dict[str, object], dict[str, str]]:
    aliases = {
        f"EXPLICIT_{index}": business_id
        for index, business_id in enumerate(
            value.explicit_referenced_business_ids, start=1
        )
    }
    redacted_message = value.query_text
    for alias, business_id in aliases.items():
        redacted_message = redacted_message.replace(business_id, alias)
    prior = None
    if value.previous_memory is not None:
        compact = compact_memory(value.previous_memory)
        prior = {
            "revision": compact.revision,
            "task_type": compact.task_type,
            "hard_constraints": compact.hard_constraints,
            "soft_preferences": compact.soft_preferences,
            "information_gaps": compact.information_gaps,
            "rejected_business_count": len(compact.rejected_business_ids),
            "presented_results": [
                {"ordinal": index}
                for index, _ in enumerate(
                    compact.last_presented_business_ids, start=1
                )
            ],
            "business_scope_known": compact.business_scope_known,
            "business_scope_count": len(compact.current_business_scope),
            "clarification_answers": compact.clarification_answers,
            "relative_preferences": [
                item.model_dump(mode="json")
                for item in compact.relative_preferences
            ],
            "recent_turn_count": len(compact.recent_turn_summaries),
        }
    return {
        "current_turn": value.current_turn,
        "language": value.language,
        "user_message": redacted_message,
        "explicit_referenced_business_ids": list(aliases),
        "rule_parser": {
            "task_type": value.base_readiness.task_type,
            "conditions": [
                item.model_dump(mode="json")
                for item in value.base_request.conditions
            ],
            "party_size": value.base_request.party_size,
            "information_gaps": value.base_readiness.information_gaps,
        },
        "prior_memory": prior,
    }, aliases


def _messages(payload: dict[str, object]) -> list[LLMMessage]:
    system = """You extract a proposed update to a restaurant recommendation session.
Return JSON only and follow the supplied schema exactly. The proposal is not trusted
and will be validated by code. Never invent a business ID. A phrase such as first,
second, this one, or that place must be represented as a reference with an ordinal;
only copy an explicit business ID when it appears in explicit_referenced_business_ids.
Use request_mode=patch for refinements, replace only for an explicit new request, and
no_change for a pure factual question. Keep temporary wishes session-scoped. Put a
durable preference only in long_term_candidates; do not claim it was persisted.
For condition patches, evidence_span must be copied exactly from user_message.
Never convert relative language such as closer, cheaper, or quieter into a made-up
number; put it in relative_preferences. If a user criticizes a referenced result and
asks for another, include that reference_id in rejected_reference_ids.
Do not reveal reasoning. semantic_summary should describe the current active goal in
one compact sentence without inventing facts."""
    user = "Create one MemoryProposal:\n" + json.dumps(
        {
            "visible_input": payload,
            "output_schema": MemoryProposal.model_json_schema(),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return [
        LLMMessage(role="system", content=system),
        LLMMessage(role="user", content=user),
    ]


def _compact_output_contract() -> dict[str, object]:
    """Keep prompts cheap while Pydantic remains the authoritative validator."""

    return {
        "required": [
            "request_mode",
            "task_type",
            "task_type_confidence",
            "references",
            "rejected_reference_ids",
            "condition_patches",
            "clarification_answers",
            "relative_preferences",
            "party_size",
            "semantic_summary",
            "long_term_candidates",
            "confidence",
            "uncertainty_reasons",
        ],
        "references": {
            "item": [
                "reference_id",
                "expression",
                "ordinal",
                "explicit_business_id",
            ],
            "rule": "exactly one of ordinal or explicit_business_id",
        },
        "condition_patches": {
            "item": [
                "operation",
                "field",
                "operator",
                "value",
                "importance",
                "evidence_span",
                "confidence",
                "lifetime",
            ],
            "fields": [
                "category",
                "distance_km",
                "budget_per_person",
                "price_level",
                "quiet_environment",
                "crowded",
                "queue_time",
                "parking",
                "pet_friendly",
                "family_friendly",
                "date_suitable",
                "group_suitable",
                "spiciness",
                "cleanliness",
                "food_quality",
                "service",
                "price_value",
            ],
            "operators": [
                "includes",
                "excludes",
                "equals",
                "less_than_or_equal",
                "greater_than_or_equal",
                "prefer",
                "avoid",
            ],
            "rule": "never emit an exact numeric value unless evidence_span has it",
        },
        "relative_preferences": {
            "item": ["field", "direction", "evidence_span", "confidence", "lifetime"],
            "fields": ["distance", "price", "noise", "crowding"],
            "directions": [
                "closer",
                "farther",
                "lower",
                "higher",
                "quieter",
                "less_crowded",
            ],
        },
        "nulls": "use null for unknown optional scalar fields",
        "arrays": "use [] when there are no items",
    }


def _normalize_memory_payload(raw: object) -> object:
    """Normalize harmless reference labels while leaving semantic fields strict."""

    if not isinstance(raw, dict):
        return raw
    payload = dict(raw)
    references = payload.get("references")
    if not isinstance(references, list):
        return payload
    normalized: list[object] = []
    reference_map: dict[str, str] = {}
    for item in references:
        if not isinstance(item, dict):
            continue
        value = dict(item)
        expression = str(value.get("expression") or "").strip()
        ordinal = value.get("ordinal")
        if not isinstance(ordinal, int):
            ordinal = _infer_reference_ordinal(expression)
        explicit = value.get("explicit_business_id")
        if not (
            isinstance(explicit, str)
            and re.fullmatch(r"EXPLICIT_[1-9][0-9]*", explicit)
        ):
            explicit = None
        if ordinal is not None:
            explicit = None
        if ordinal is None and explicit is None:
            continue
        canonical = f"R{len(normalized) + 1}"
        old = value.get("reference_id")
        if isinstance(old, str):
            reference_map[old] = canonical
        value["reference_id"] = canonical
        value["expression"] = expression or f"ordinal {ordinal}"
        value["ordinal"] = ordinal
        value["explicit_business_id"] = explicit
        normalized.append(value)
    payload["references"] = normalized
    rejected = payload.get("rejected_reference_ids")
    if isinstance(rejected, list):
        payload["rejected_reference_ids"] = [
            reference_map[str(item)] for item in rejected if str(item) in reference_map
        ]
    _normalize_condition_aliases(payload)
    _normalize_patch_operations(payload)
    return payload


def _normalize_patch_operations(payload: dict[str, object]) -> None:
    patches = payload.get("condition_patches")
    if not isinstance(patches, list):
        return
    normalized: list[object] = []
    for item in patches:
        if not isinstance(item, dict):
            normalized.append(item)
            continue
        value = dict(item)
        operator_aliases = {
            "<=": "less_than_or_equal",
            ">=": "greater_than_or_equal",
            "include": "includes",
            "exclude": "excludes",
            "preferred": "prefer",
        }
        raw_operator = value.get("operator")
        if isinstance(raw_operator, str):
            value["operator"] = operator_aliases.get(raw_operator, raw_operator)
        if value.get("operation") == "remove":
            value["operator"] = None
            value["value"] = None
            value["importance"] = None
        elif any(value.get(key) is None for key in ("operator", "value", "importance")):
            continue
        normalized.append(value)
    payload["condition_patches"] = normalized


def _infer_reference_ordinal(expression: str) -> int | None:
    lowered = expression.casefold()
    patterns = (
        (1, ("first", "1st", "\u7b2c\u4e00", "\u4e00\u5bb6", "\u4e00\u4e2a")),
        (2, ("second", "2nd", "\u7b2c\u4e8c", "\u4e8c\u5bb6", "\u4e24\u5bb6", "\u4e8c\u4e2a")),
        (3, ("third", "3rd", "\u7b2c\u4e09", "\u4e09\u5bb6", "\u4e09\u4e2a")),
        (4, ("fourth", "4th", "\u7b2c\u56db", "\u56db\u5bb6", "\u56db\u4e2a")),
        (5, ("fifth", "5th", "\u7b2c\u4e94", "\u4e94\u5bb6", "\u4e94\u4e2a")),
    )
    return next(
        (ordinal for ordinal, markers in patterns if any(marker in lowered for marker in markers)),
        None,
    )


def _restore_explicit_aliases(
    raw: object, aliases: dict[str, str]
) -> object:
    if not isinstance(raw, dict) or not aliases:
        return raw
    payload = dict(raw)
    references = payload.get("references")
    if not isinstance(references, list):
        return payload
    restored: list[object] = []
    for item in references:
        if not isinstance(item, dict):
            restored.append(item)
            continue
        value = dict(item)
        explicit = value.get("explicit_business_id")
        if isinstance(explicit, str) and explicit in aliases:
            value["explicit_business_id"] = aliases[explicit]
        restored.append(value)
    payload["references"] = restored
    return payload


def _normalize_condition_aliases(payload: dict[str, object]) -> None:
    patches = payload.get("condition_patches")
    if not isinstance(patches, list):
        return
    relative = payload.get("relative_preferences")
    relative_items = list(relative) if isinstance(relative, list) else []
    normalized: list[object] = []
    aliases = {
        "distance": "distance_km",
        "proximity": "distance_km",
        "distance_preference": "distance_km",
        "price": "budget_per_person",
        "budget": "budget_per_person",
        "quiet": "quiet_environment",
        "quietness": "quiet_environment",
        "noise": "quiet_environment",
    }
    for item in patches:
        if not isinstance(item, dict):
            normalized.append(item)
            continue
        value = dict(item)
        raw_field = str(value.get("field") or "")
        field = aliases.get(raw_field, raw_field)
        evidence = str(value.get("evidence_span") or "")
        if field in {"distance_km", "budget_per_person"} and not re.search(
            r"\d+(?:\.\d+)?", evidence
        ):
            direction = _relative_direction(field, evidence)
            if direction is not None:
                relative_items.append(
                    {
                        "field": "distance" if field == "distance_km" else "price",
                        "direction": direction,
                        "evidence_span": evidence,
                        "confidence": value.get("confidence", 0.7),
                        "lifetime": value.get("lifetime", "session"),
                    }
                )
                continue
        value["field"] = field
        normalized.append(value)
    payload["condition_patches"] = normalized
    unique_relative: list[object] = []
    seen: set[tuple[str, str, str]] = set()
    for item in relative_items:
        if not isinstance(item, dict):
            unique_relative.append(item)
            continue
        key = (
            str(item.get("field")),
            str(item.get("direction")),
            str(item.get("evidence_span")),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_relative.append(item)
    payload["relative_preferences"] = unique_relative


def _relative_direction(field: str, evidence: str) -> str | None:
    lowered = evidence.casefold()
    if field == "distance_km":
        if any(marker in lowered for marker in ("近", "closer", "nearer")):
            return "closer"
        if any(marker in lowered for marker in ("远一点", "farther")):
            return "farther"
    if field == "budget_per_person":
        if any(
            marker in lowered
            for marker in ("便宜", "低一点", "太贵", "cheaper", "lower", "expensive")
        ):
            return "lower"
        if any(marker in lowered for marker in ("贵一点", "higher")):
            return "higher"
    return None


def _ordinal_reference(text: str) -> int | None:
    lowered = text.casefold()
    patterns = (
        (1, r"(?:第一家|第一个|第一個|first one|1st one)"),
        (2, r"(?:第二家|第二个|第二個|second one|2nd one)"),
        (3, r"(?:第三家|第三个|第三個|third one|3rd one)"),
        (4, r"(?:第四家|第四个|第四個|fourth one|4th one)"),
        (5, r"(?:第五家|第五个|第五個|fifth one|5th one)"),
    )
    return next(
        (ordinal for ordinal, pattern in patterns if re.search(pattern, lowered)),
        None,
    )


def _ordinal_expression(text: str, ordinal: int) -> str:
    return f"ordinal:{ordinal}:{text[:80]}"


def _rejects_previous(text: str) -> bool:
    return re.search(
        r"(?:太贵|太遠|太远|太吵|不合适|不要(?:这|那|刚才)|换(?:一|个|家)|"
        r"too expensive|too far|too noisy|not suitable|something else|another)",
        text,
        re.IGNORECASE,
    ) is not None
