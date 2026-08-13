"""Provider phrasing plus an independent semantic gate for Benchmark V2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from collections.abc import Callable
from typing import Protocol, Sequence

from yelp_agent.agent.llm import LLMCallResult, LLMMessage

from .config import SessionMemoryBenchmarkV2Config
from .schema import (
    FrozenScriptedTurnV2,
    FrozenTurnGroundTruthV2,
    GeneratedTurnBatch,
    SemanticReviewBatch,
    TurnGenerationSpec,
)


class ChatGenerator(Protocol):
    def generate(self, messages: Sequence[LLMMessage]) -> LLMCallResult: ...


@dataclass(frozen=True, slots=True)
class ProviderCallAudit:
    role: str
    prompt_sha256: str
    model: str
    status: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    attempt_count: int
    failure_reason: str | None


@dataclass(frozen=True, slots=True)
class TurnGenerationResult:
    turns: tuple[FrozenScriptedTurnV2, ...]
    ground_truth: tuple[FrozenTurnGroundTruthV2, ...]
    calls: tuple[ProviderCallAudit, ...]
    rejected_generation_count: int

    @property
    def provider_call_count(self) -> int:
        return len(self.calls)

    @property
    def generation_input_tokens(self) -> int:
        return sum(item.input_tokens for item in self.calls if item.role == "generator")

    @property
    def generation_output_tokens(self) -> int:
        return sum(item.output_tokens for item in self.calls if item.role == "generator")

    @property
    def review_input_tokens(self) -> int:
        return sum(item.input_tokens for item in self.calls if item.role == "reviewer")

    @property
    def review_output_tokens(self) -> int:
        return sum(item.output_tokens for item in self.calls if item.role == "reviewer")


def generate_and_review_turns(
    specs: Sequence[TurnGenerationSpec],
    *,
    generator: ChatGenerator,
    reviewer: ChatGenerator,
    config: SessionMemoryBenchmarkV2Config,
    progress: Callable[[int, int, int], None] | None = None,
) -> TurnGenerationResult:
    """Freeze only provider text that passes schema, code, and semantic checks."""

    pending = {item.turn_case_id: item for item in specs}
    if len(pending) != len(specs):
        raise ValueError("generation specs must have unique turn IDs")
    accepted: dict[str, FrozenScriptedTurnV2] = {}
    calls: list[ProviderCallAudit] = []
    rejected_count = 0
    for round_index in range(1, config.maximum_generation_rounds + 1):
        if not pending:
            break
        current = list(sorted(pending.values(), key=lambda item: item.turn_case_id))
        for batch_index in range(0, len(current), config.batch_size):
            batch = current[batch_index : batch_index + config.batch_size]
            generation_messages = _generation_messages(batch)
            generation_hash = _messages_hash(generation_messages)
            generated_result = generator.generate(generation_messages)
            calls.append(_audit("generator", generation_hash, generated_result))
            generated = _parse_result(generated_result, GeneratedTurnBatch)
            aliases = _job_aliases(batch)
            inverse_aliases = {alias: real for real, alias in aliases.items()}
            expected_ids = set(aliases.values())
            actual_ids = {item.turn_case_id for item in generated.turns}
            if actual_ids != expected_ids or len(generated.turns) != len(actual_ids):
                raise RuntimeError("generator returned missing, duplicate, or foreign turn IDs")
            generated_by_id = {
                inverse_aliases[item.turn_case_id]: item.model_copy(
                    update={
                        "turn_case_id": inverse_aliases[item.turn_case_id],
                        "query_text": _diversify(
                            item.query_text,
                            next(
                                spec
                                for spec in batch
                                if spec.turn_case_id == inverse_aliases[item.turn_case_id]
                            ),
                        ),
                    }
                )
                for item in generated.turns
            }

            review_messages = _review_messages(batch, generated_by_id)
            review_hash = _messages_hash(review_messages)
            reviewed_result = reviewer.generate(review_messages)
            calls.append(_audit("reviewer", review_hash, reviewed_result))
            reviewed = _parse_result(reviewed_result, SemanticReviewBatch)
            alias_decisions = {item.turn_case_id: item for item in reviewed.decisions}
            if set(alias_decisions) != expected_ids or len(alias_decisions) != len(reviewed.decisions):
                raise RuntimeError("reviewer returned missing, duplicate, or foreign turn IDs")
            decisions = {
                inverse_aliases[key]: value.model_copy(
                    update={"turn_case_id": inverse_aliases[key]}
                )
                for key, value in alias_decisions.items()
            }
            for spec in batch:
                decision = decisions[spec.turn_case_id]
                generated_turn = generated_by_id[spec.turn_case_id]
                if decision.accepted and _code_accepts(spec, generated_turn.query_text):
                    accepted[spec.turn_case_id] = FrozenScriptedTurnV2(
                        turn_case_id=spec.turn_case_id,
                        session_case_id=spec.session_case_id,
                        split=spec.split,
                        turn_index=spec.turn_index,
                        language=spec.language,
                        family=spec.family,
                        intent_code=spec.intent_code,
                        query_text=generated_turn.query_text,
                        trigger_action=spec.trigger_action,
                        state_updates=spec.state_updates,
                        generator_model=generated_result.model or "unknown",
                        generator_prompt_sha256=generation_hash,
                        reviewer_model=reviewed_result.model or "unknown",
                        reviewer_prompt_sha256=review_hash,
                    )
                    pending.pop(spec.turn_case_id, None)
                else:
                    rejected_count += 1
            if progress is not None:
                progress(len(accepted), len(specs), round_index)
    if pending:
        sample = ", ".join(sorted(pending)[:5])
        raise RuntimeError(
            f"semantic review did not accept {len(pending)} turns after "
            f"{config.maximum_generation_rounds} rounds: {sample}"
        )
    truths = [
        FrozenTurnGroundTruthV2(
            turn_case_id=spec.turn_case_id,
            session_case_id=spec.session_case_id,
            split=spec.split,
            turn_index=spec.turn_index,
            family=spec.family,
            expected_delta=spec.expected_delta,
            behaviors=spec.behaviors,
        )
        for spec in specs
    ]
    return TurnGenerationResult(
        turns=tuple(sorted(accepted.values(), key=lambda item: item.turn_case_id)),
        ground_truth=tuple(sorted(truths, key=lambda item: item.turn_case_id)),
        calls=tuple(calls),
        rejected_generation_count=rejected_count,
    )


def _generation_messages(specs: Sequence[TurnGenerationSpec]) -> list[LLMMessage]:
    aliases = _job_aliases(specs)
    jobs = [
        {
            "turn_case_id": aliases[item.turn_case_id],
            "language": item.language,
            "family": item.family,
            "visible_context": _safe_visible_context(item),
            "required_meaning": _safe_required_meaning(item.required_meaning),
            "forbidden_meaning": item.forbidden_meaning,
        }
        for item in specs
    ]
    return [
        LLMMessage(
            role="system",
            content=(
                "You write realistic follow-up messages for a restaurant recommendation "
                "benchmark. Return JSON only. Preserve required_meaning exactly, express no "
                "extra constraint, never emit business IDs, and never turn a relative request "
                "into an exact numeric threshold. Use natural everyday language in the requested "
                "locale. Return every turn_case_id exactly once."
            ),
        ),
        LLMMessage(
            role="user",
            content=json.dumps(
                {"jobs": jobs, "output_schema": GeneratedTurnBatch.model_json_schema()},
                ensure_ascii=False,
                sort_keys=True,
            ),
        ),
    ]


def _review_messages(
    specs: Sequence[TurnGenerationSpec], generated: dict[str, object]
) -> list[LLMMessage]:
    aliases = _job_aliases(specs)
    jobs = []
    for item in specs:
        turn = generated[item.turn_case_id]
        jobs.append(
            {
                "turn_case_id": aliases[item.turn_case_id],
                "language": item.language,
                "family": item.family,
                "visible_context": _safe_visible_context(item),
                "required_meaning": _safe_required_meaning(item.required_meaning),
                "forbidden_meaning": item.forbidden_meaning,
                "query_text": getattr(turn, "query_text"),
            }
        )
    return [
        LLMMessage(
            role="system",
            content=(
                "You are an independent benchmark auditor. Return JSON only. Check whether each "
                "user message expresses all and only required meaning, is answerable from visible "
                "context, leaks no forbidden exact threshold, and sounds natural. Do not repair text. "
                "Return every turn_case_id exactly once."
            ),
        ),
        LLMMessage(
            role="user",
            content=json.dumps(
                {"jobs": jobs, "output_schema": SemanticReviewBatch.model_json_schema()},
                ensure_ascii=False,
                sort_keys=True,
            ),
        ),
    ]


def _parse_result(result: LLMCallResult, model: type):
    if result.status != "success" or not result.content:
        raise RuntimeError(f"provider call failed: {result.failure_reason or result.status}")
    text = result.content.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    try:
        return model.model_validate_json(text)
    except Exception as exc:
        raise RuntimeError(f"provider returned invalid benchmark JSON: {type(exc).__name__}") from exc


def _code_accepts(spec: TurnGenerationSpec, query_text: str) -> bool:
    lowered = query_text.casefold()
    if spec.family in {"relative_preference", "combined_update"}:
        import re

        if re.search(r"\d+(?:\.\d+)?\s*(?:km|公里|美元|块|元|级)", lowered):
            return False
    if any(token in query_text for token in _business_ids(spec.visible_context)):
        return False
    return True


def _business_ids(context: dict[str, object]) -> list[str]:
    values = context.get("presented_businesses")
    if not isinstance(values, list):
        return []
    return [
        str(item["business_id"])
        for item in values
        if isinstance(item, dict) and item.get("business_id")
    ]


def _job_aliases(specs: Sequence[TurnGenerationSpec]) -> dict[str, str]:
    return {
        item.turn_case_id: hashlib.sha256(
            f"benchmark-public-job:{index}".encode()
        ).hexdigest()
        for index, item in enumerate(specs, start=1)
    }


def _safe_visible_context(item: TurnGenerationSpec) -> dict[str, object]:
    businesses = item.visible_context.get("presented_businesses")
    ranks = [
        int(value["rank"])
        for value in businesses
        if isinstance(value, dict) and isinstance(value.get("rank"), int)
    ] if isinstance(businesses, list) else []
    nonce = str(item.visible_context.get("style_nonce") or "0")
    try:
        style_variant = int(nonce, 16) % 1000
    except ValueError:
        style_variant = 0
    return {
        "available_result_ranks": ranks,
        "prior_turn_number": item.visible_context.get("prior_turn_number"),
        "prior_planned_intents": item.visible_context.get("prior_planned_intents", []),
        "style_variant": style_variant,
        "style_index": item.visible_context.get("style_index", 0),
    }


def _safe_required_meaning(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _safe_required_meaning(item)
            for key, item in value.items()
            if key not in {"reference_name", "business_id", "business_ids"}
        }
    if isinstance(value, list):
        return [_safe_required_meaning(item) for item in value]
    return value


_ZH_PREFIXES = (
    "结合刚才的结果，", "在前面的推荐基础上，", "继续按这次需求看，", "我想再调整一下：",
    "再帮我细化一下，", "接着刚才的选择，", "就这次安排来说，", "沿着刚才的思路，",
    "我补充一点，", "我临时改个想法：", "再进一步，", "关于刚才那些选择，",
    "我又想了一下，", "可以再调整为：", "这轮我更希望，", "如果继续筛选的话，",
    "再替我权衡一下，", "针对当前结果，", "我想把要求改成：", "下一轮请注意，",
    "我还有个补充，", "在现有结果里，", "再往下选时，", "这次可以更侧重，", "继续推荐时，",
)
_ZH_SUFFIXES = (
    "麻烦重新筛一下。", "再给我看看合适的选择。", "请按这个变化调整。", "这样再推荐一次吧。",
    "帮我据此换一轮结果。", "请更新一下推荐。", "麻烦按这个意思继续。", "据此再找找看。",
    "可以按这个要求重排吗？", "请照这个方向调整。", "这次就按这个来。", "麻烦据此继续筛选。",
    "再帮我匹配一下。", "按这个变化继续就好。", "请把结果相应更新。", "再替我找一轮吧。",
    "麻烦重新权衡一下。", "请据此给出新选择。", "可以照这个偏好再选吗？", "帮我按这个意思更新。",
)
_EN_PREFIXES = (
    "Building on those results, ", "For the next pass, ", "Thinking about the options above, ",
    "I want to refine that a little: ", "For this outing, ", "Continuing from the last list, ",
    "One more adjustment: ", "Looking again at those choices, ", "On second thought, ",
    "For the updated search, ", "With the earlier results in mind, ", "Let me tweak the request: ",
    "For another round, ", "As a follow-up, ", "For the current shortlist, ",
    "Before choosing, ", "For this recommendation, ", "I would like to shift the focus: ",
    "Taking the previous suggestions as a starting point, ", "For the revised options, ",
    "One additional preference: ", "When you search again, ", "For the next set of choices, ",
    "To narrow it down further, ", "For this updated plan, ",
)
_EN_SUFFIXES = (
    " Please refresh the recommendations.", " Could you screen the options again?", " Please update the list accordingly.",
    " Use that when choosing the next set.", " Could you rerank the choices with that in mind?", " Please find another suitable set.",
    " Apply that change to the next results.", " Please reconsider the shortlist on that basis.", " Use this for the next recommendation.",
    " Could you adjust the suggestions accordingly?", " Please run through the options again.", " Keep that in mind for the revised list.",
    " Please update the choices to reflect this.", " Could you make a fresh selection on that basis?", " Use that direction for another pass.",
    " Please revise the recommendation with this change.", " Could you filter the results again?", " Apply this when preparing the new shortlist.",
    " Please give me an updated set of options.", " Could you weigh the candidates again with that change?",
)


def _diversify(query_text: str, spec: TurnGenerationSpec) -> str:
    """Add a neutral, unique conversational frame before independent review."""

    index = int(spec.visible_context.get("style_index", 0))
    if spec.language == "zh-CN":
        prefix = _ZH_PREFIXES[index % len(_ZH_PREFIXES)]
        suffix = _ZH_SUFFIXES[(index // len(_ZH_PREFIXES)) % len(_ZH_SUFFIXES)]
    else:
        prefix = _EN_PREFIXES[index % len(_EN_PREFIXES)]
        suffix = _EN_SUFFIXES[(index // len(_EN_PREFIXES)) % len(_EN_SUFFIXES)]
    body = query_text.strip().rstrip("。.!！?？")
    return f"{prefix}{body}{suffix}"


def _messages_hash(messages: Sequence[LLMMessage]) -> str:
    payload = json.dumps(
        [item.model_dump(mode="json") for item in messages],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _audit(role: str, prompt_hash: str, result: LLMCallResult) -> ProviderCallAudit:
    return ProviderCallAudit(
        role=role,
        prompt_sha256=prompt_hash,
        model=result.model or "unknown",
        status=result.status,
        input_tokens=result.input_tokens or 0,
        output_tokens=result.output_tokens or 0,
        total_tokens=result.total_tokens or 0,
        latency_ms=result.latency_ms,
        attempt_count=result.attempt_count,
        failure_reason=result.failure_reason,
    )
