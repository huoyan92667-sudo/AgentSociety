"""Three-layer Step 34.5 evaluation with observable, state-derived labels."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from yelp_agent.agent_harness import RuleBasedRequestInterpreter
from yelp_agent.query import QueryParseInput
from yelp_agent.session_memory.manager import SessionMemoryManager
from yelp_agent.session_memory.reducer import record_memory_observation
from yelp_agent.session_memory.schema import MemoryTurnInput, SessionMemory
from yelp_agent.session_memory.schema import MemoryProposal

from .schema import (
    FrozenPresentation,
    FrozenScriptedTurnV2,
    FrozenTurnGroundTruthV2,
    MemoryBenchmarkInitialSession,
    PresentedBusinessSnapshot,
)


@dataclass(frozen=True, slots=True)
class MemoryStateView:
    task_type: str
    conditions: frozenset[tuple[str, str, str, str]]
    relative_preferences: frozenset[tuple[str, str]]
    clarification_answers: tuple[tuple[str, str], ...]
    rejected_business_ids: frozenset[str]
    resolved_business_ids: frozenset[str]
    information_gaps: frozenset[str]
    party_size: int | None


@dataclass(frozen=True, slots=True)
class MemoryBenchmarkV2Result:
    output_root: Path
    cases_path: Path
    metrics_path: Path
    summary_path: Path
    metrics: dict[str, object]


def score_memory_transition(
    before: MemoryStateView,
    after: MemoryStateView,
    truth: FrozenTurnGroundTruthV2,
    query_text: str,
    proposal: MemoryProposal | None = None,
) -> dict[str, object]:
    expected = truth.expected_delta
    expected_conditions = expected.condition_deltas
    condition_hits = 0
    for item in expected_conditions:
        if item.operation == "remove":
            hit = not any(value[0] == item.field for value in after.conditions)
        else:
            key = (
                item.field,
                str(item.operator),
                _value(item.value),
                str(item.importance),
            )
            hit = key in after.conditions
        condition_hits += int(hit)
    added_conditions = after.conditions - before.conditions
    removed_conditions = before.conditions - after.conditions
    actual_condition_changes = len(added_conditions) + len(removed_conditions)
    expected_relative = {
        (item.field, item.direction) for item in expected.relative_preferences
    }
    actual_relative = after.relative_preferences - before.relative_preferences
    clarification = dict(after.clarification_answers)
    clarification_hits = 0
    for key, value in expected.clarification_answers.items():
        stored = _value(clarification.get(key)) == _value(value)
        resolved_gap = key not in after.information_gaps
        if key == "missing_party_size":
            resolved_gap = resolved_gap and after.party_size == int(value)
        clarification_hits += int(stored or resolved_gap)
    rejected_hits = len(
        set(expected.rejected_business_ids) & after.rejected_business_ids
    )
    resolved_hits = len(
        set(expected.resolved_business_ids) & after.resolved_business_ids
    )
    numeric_hallucinations = sum(
        _is_numeric_field(item[0])
        and _extract_number(item[2]) is not None
        and not _query_contains_number(query_text, _extract_number(item[2]))
        for item in added_conditions
    )
    preference_unchanged = (
        before.conditions == after.conditions
        and before.relative_preferences == after.relative_preferences
        and before.clarification_answers == after.clarification_answers
        and before.rejected_business_ids == after.rejected_business_ids
        and before.party_size == after.party_size
    )
    semantic = _semantic_score(proposal, truth) if proposal is not None else {}
    return {
        "task_type_correct": after.task_type == expected.task_type,
        "information_gaps_correct": after.information_gaps
        == frozenset(expected.expected_information_gaps),
        "condition_expected": len(expected_conditions),
        "condition_hits": condition_hits,
        "condition_actual_changes": actual_condition_changes,
        "relative_expected": len(expected_relative),
        "relative_hits": len(expected_relative & after.relative_preferences),
        "relative_actual_changes": len(actual_relative),
        "clarification_expected": len(expected.clarification_answers),
        "clarification_hits": clarification_hits,
        "rejected_business_expected": len(expected.rejected_business_ids),
        "rejected_business_hits": rejected_hits,
        "resolved_reference_expected": len(expected.resolved_business_ids),
        "resolved_reference_hits": resolved_hits,
        "no_state_change_expected": expected.no_state_change,
        "no_state_change_correct": (not expected.no_state_change) or preference_unchanged,
        "numeric_hallucinations": numeric_hallucinations,
        "actual_numeric_changes": sum(
            _is_numeric_field(item[0]) for item in added_conditions
        ),
        **semantic,
    }


def run_memory_benchmark_v2(
    manager: SessionMemoryManager,
    *,
    benchmark_root: str | Path,
    output_root: str | Path,
    progress: Callable[[int, int], None] | None = None,
    candidate_contexts: dict[str, Sequence[PresentedBusinessSnapshot]] | None = None,
) -> MemoryBenchmarkV2Result:
    root = Path(benchmark_root)
    output = Path(output_root)
    sessions = _load_models(
        root / "visible" / "initial_sessions.jsonl", MemoryBenchmarkInitialSession
    )
    turns = _load_models(
        root / "visible" / "scripted_turns.jsonl", FrozenScriptedTurnV2
    )
    truths = _load_models(
        root / "hidden" / "ground_truth.jsonl", FrozenTurnGroundTruthV2
    )
    presentations = _load_models(
        root / "visible" / "frozen_presentations.jsonl", FrozenPresentation
    )
    truth_by_id = {item.turn_case_id: item for item in truths}
    presentation_by_session = {item.session_case_id: item for item in presentations}
    turns_by_session: dict[str, list[FrozenScriptedTurnV2]] = defaultdict(list)
    for item in turns:
        turns_by_session[item.session_case_id].append(item)
    if set(truth_by_id) != {item.turn_case_id for item in turns}:
        raise ValueError("V2 visible and hidden turns do not align")
    rows: list[dict[str, object]] = []
    ordered_sessions = sorted(sessions, key=lambda item: item.session_case_id)
    for session_index, initial in enumerate(ordered_sessions, start=1):
        memory = _start(manager, initial)
        presentation = presentation_by_session.get(initial.session_case_id)
        if presentation is not None:
            memory = record_memory_observation(
                memory,
                turn_index=1,
                business_scope=presentation.candidate_business_ids,
                presented_business_ids=[
                    item.business_id for item in presentation.presented_businesses
                ],
            )
            assert memory is not None
        for turn in sorted(
            turns_by_session[initial.session_case_id], key=lambda item: item.turn_index
        ):
            before = memory_state_view(memory)
            result = _continue(manager, initial, memory, turn)
            memory = result.memory
            after = memory_state_view(memory, current_turn=turn.turn_index)
            truth = truth_by_id[turn.turn_case_id]
            score = score_memory_transition(
                before,
                after,
                truth,
                turn.query_text,
                proposal=result.proposal,
            )
            behavior = _behavior_score(
                after,
                truth,
                []
                if candidate_contexts is None
                else candidate_contexts.get(initial.source_scenario_id, []),
            )
            rows.append(
                {
                    "turn_case_id": turn.turn_case_id,
                    "session_case_id": turn.session_case_id,
                    "split": turn.split,
                    "language": turn.language,
                    "family": turn.family,
                    "query_text": turn.query_text,
                    **score,
                    **behavior,
                    "provider_called": result.extraction.provider_called,
                    "rule_fallback": result.extraction.status == "rule_fallback",
                    "extraction_status": result.extraction.status,
                    "input_tokens": result.extraction.input_tokens or 0,
                    "output_tokens": result.extraction.output_tokens or 0,
                    "latency_ms": result.extraction.latency_ms,
                    "accepted_changes": result.accepted_changes,
                    "rejected_changes": result.rejected_changes,
                }
            )
        if progress is not None:
            progress(session_index, len(ordered_sessions))
    metrics = _aggregate(rows)
    cases_path = _write_jsonl(output / "memory_cases.jsonl", rows)
    metrics_path = _write_text(
        output / "metrics.json",
        json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    summary_path = _write_text(output / "summary.md", _summary(metrics))
    return MemoryBenchmarkV2Result(output, cases_path, metrics_path, summary_path, metrics)


def memory_state_view(
    memory: SessionMemory, *, current_turn: int | None = None
) -> MemoryStateView:
    resolved: set[str] = set()
    if current_turn is not None and memory.recent_turns:
        record = memory.recent_turns[-1]
        if record.turn_index == current_turn:
            resolved = {item.business_id for item in record.resolved_references}
    return MemoryStateView(
        task_type=memory.current_task_type,
        conditions=frozenset(
            (
                item.field,
                item.operator,
                _value(item.value),
                item.importance,
            )
            for item in memory.current_request.conditions
        ),
        relative_preferences=frozenset(
            (item.field, item.direction) for item in memory.relative_preferences
        ),
        clarification_answers=tuple(
            sorted((key, _value(value)) for key, value in memory.clarification_answers.items())
        ),
        rejected_business_ids=frozenset(memory.rejected_business_ids),
        resolved_business_ids=frozenset(resolved),
        information_gaps=frozenset(memory.information_gaps),
        party_size=memory.current_request.party_size,
    )


def _start(manager: SessionMemoryManager, initial: MemoryBenchmarkInitialSession) -> SessionMemory:
    parsed = RuleBasedRequestInterpreter().interpret(_parse_input(initial, initial.query_text))
    return manager.update(
        MemoryTurnInput(
            query_text=initial.query_text,
            language=initial.language,
            current_turn=1,
            base_request=parsed.request,
            base_readiness=parsed.readiness,
            explicit_referenced_business_ids=initial.referenced_business_ids,
        )
    ).memory


def _continue(
    manager: SessionMemoryManager,
    initial: MemoryBenchmarkInitialSession,
    memory: SessionMemory,
    turn: FrozenScriptedTurnV2,
):
    parsed = RuleBasedRequestInterpreter().interpret(
        _parse_input(
            initial,
            turn.query_text,
            state_updates=turn.state_updates,
        )
    )
    return manager.update(
        MemoryTurnInput(
            query_text=turn.query_text,
            language=turn.language,
            current_turn=turn.turn_index,
            base_request=parsed.request,
            base_readiness=parsed.readiness,
            previous_memory=memory,
        )
    )


def _parse_input(
    initial: MemoryBenchmarkInitialSession,
    query_text: str,
    *,
    state_updates: dict[str, str | int | float | bool] | None = None,
) -> QueryParseInput:
    updates = state_updates or {}
    latitude = updates.get("user_latitude", initial.user_latitude)
    longitude = updates.get("user_longitude", initial.user_longitude)
    return QueryParseInput(
        user_id=initial.user_id,
        session_id=initial.session_id,
        cutoff_time=initial.cutoff_time,
        query_text=query_text,
        user_latitude=float(latitude) if latitude is not None else None,
        user_longitude=float(longitude) if longitude is not None else None,
        referenced_business_ids=(initial.referenced_business_ids if query_text == initial.query_text else []),
    )


def _aggregate(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("Benchmark V2 has no evaluated turns")
    overall = _slice(rows)
    by_split = {
        key: _slice([item for item in rows if item["split"] == key])
        for key in sorted({str(item["split"]) for item in rows})
    }
    by_family = {
        key: _slice([item for item in rows if item["family"] == key])
        for key in sorted({str(item["family"]) for item in rows})
    }
    return {
        "schema_version": 2,
        "turn_count": len(rows),
        "overall": overall,
        "by_split": by_split,
        "by_family": by_family,
        "hidden_labels_visible_to_manager": False,
    }


def _slice(rows: Sequence[dict[str, object]]) -> dict[str, float | int]:
    def recall(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 1.0

    def precision(numerator: float, denominator: float, expected: float) -> float:
        return numerator / denominator if denominator else (1.0 if not expected else 0.0)

    condition_expected = sum(int(item["semantic_condition_expected"]) for item in rows)
    condition_hits = sum(int(item["semantic_condition_hits"]) for item in rows)
    condition_core_hits = sum(
        int(item["semantic_condition_core_hits"]) for item in rows
    )
    condition_field_hits = sum(
        int(item["semantic_condition_field_operation_hits"]) for item in rows
    )
    condition_actual = sum(int(item["semantic_condition_predicted"]) for item in rows)
    relative_expected = sum(int(item["semantic_relative_expected"]) for item in rows)
    relative_hits = sum(int(item["semantic_relative_hits"]) for item in rows)
    relative_actual = sum(int(item["semantic_relative_predicted"]) for item in rows)
    condition_precision = precision(condition_hits, condition_actual, condition_expected)
    condition_recall = recall(condition_hits, condition_expected)
    relative_precision = precision(relative_hits, relative_actual, relative_expected)
    relative_recall = recall(relative_hits, relative_expected)
    provider = [item for item in rows if item["provider_called"]]
    return {
        "turn_count": len(rows),
        "task_type_accuracy": recall(sum(bool(item["semantic_task_type_correct"]) for item in rows), len(rows)),
        "task_goal_accuracy": recall(sum(bool(item["semantic_task_goal_correct"]) for item in rows), len(rows)),
        "information_gap_exact_match": recall(sum(bool(item["information_gaps_correct"]) for item in rows), len(rows)),
        "semantic_condition_precision": condition_precision,
        "semantic_condition_recall": condition_recall,
        "semantic_condition_f1": (2 * condition_precision * condition_recall / (condition_precision + condition_recall) if condition_precision + condition_recall else 0.0),
        "semantic_condition_core_recall": recall(condition_core_hits, condition_expected),
        "semantic_condition_field_operation_recall": recall(condition_field_hits, condition_expected),
        "memory_condition_application_recall": recall(sum(int(item["condition_hits"]) for item in rows), sum(int(item["condition_expected"]) for item in rows)),
        "semantic_relative_precision": relative_precision,
        "semantic_relative_recall": relative_recall,
        "memory_relative_application_recall": recall(sum(int(item["relative_hits"]) for item in rows), sum(int(item["relative_expected"]) for item in rows)),
        "semantic_clarification_recall": recall(sum(int(item["semantic_clarification_hits"]) for item in rows), sum(int(item["semantic_clarification_expected"]) for item in rows)),
        "memory_clarification_application_recall": recall(sum(int(item["clarification_hits"]) for item in rows), sum(int(item["clarification_expected"]) for item in rows)),
        "rejected_business_recall": recall(sum(int(item["rejected_business_hits"]) for item in rows), sum(int(item["rejected_business_expected"]) for item in rows)),
        "reference_resolution_recall": recall(sum(int(item["resolved_reference_hits"]) for item in rows), sum(int(item["resolved_reference_expected"]) for item in rows)),
        "no_state_change_accuracy": recall(sum(bool(item["no_state_change_correct"]) for item in rows if item["no_state_change_expected"]), sum(bool(item["no_state_change_expected"]) for item in rows)),
        "numeric_hallucination_rate": (sum(int(item["numeric_hallucinations"]) for item in rows) / sum(int(item["actual_numeric_changes"]) for item in rows) if sum(int(item["actual_numeric_changes"]) for item in rows) else 0.0),
        "behavior_compliance_at_1": recall(sum(int(item["behavior_at_1_hits"]) for item in rows), sum(int(item["behavior_expected"]) for item in rows)),
        "behavior_compliance_at_5": recall(sum(int(item["behavior_at_5_hits"]) for item in rows), sum(int(item["behavior_expected"]) for item in rows)),
        "rejected_business_exclusion_rate": recall(sum(int(item["rejection_exclusion_hits"]) for item in rows), sum(int(item["rejection_exclusion_expected"]) for item in rows)),
        "provider_call_count": len(provider),
        "rule_fallback_rate": recall(sum(bool(item["rule_fallback"]) for item in rows), len(rows)),
        "input_tokens": sum(int(item["input_tokens"]) for item in provider),
        "output_tokens": sum(int(item["output_tokens"]) for item in provider),
        "mean_provider_latency_ms": (sum(float(item["latency_ms"]) for item in provider) / len(provider) if provider else 0.0),
    }


def _summary(metrics: dict[str, object]) -> str:
    overall = metrics["overall"]
    assert isinstance(overall, dict)
    return "\n".join(
        [
            "# 第 34.5 步：Session Memory Benchmark V2",
            "",
            f"- 后续用户轮次：{metrics['turn_count']}",
            f"- 任务类型准确率：{float(overall['task_type_accuracy']):.2%}",
            f"- 语义条件变更 Precision / Recall / F1：{float(overall['semantic_condition_precision']):.2%} / {float(overall['semantic_condition_recall']):.2%} / {float(overall['semantic_condition_f1']):.2%}",
            f"- 语义相对偏好 Precision / Recall：{float(overall['semantic_relative_precision']):.2%} / {float(overall['semantic_relative_recall']):.2%}",
            f"- Memory 条件 / 相对偏好应用召回：{float(overall['memory_condition_application_recall']):.2%} / {float(overall['memory_relative_application_recall']):.2%}",
            f"- 推荐行为 Compliance@1 / @5：{float(overall['behavior_compliance_at_1']):.2%} / {float(overall['behavior_compliance_at_5']):.2%}",
            f"- 商家拒绝召回率：{float(overall['rejected_business_recall']):.2%}",
            f"- 指代解析召回率：{float(overall['reference_resolution_recall']):.2%}",
            f"- 数值幻觉率：{float(overall['numeric_hallucination_rate']):.2%}",
            f"- API 调用 / 输入 Token / 输出 Token：{overall['provider_call_count']} / {overall['input_tokens']} / {overall['output_tokens']}",
            "- Ground Truth 只由离线评测器读取，Memory Manager 不接收隐藏标签。",
            "",
        ]
    )


def _load_models(path: Path, model: type) -> tuple[Any, ...]:
    return tuple(
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _value(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _extract_number(value: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    return float(match.group()) if match else None


def _query_contains_number(text: str, value: float | None) -> bool:
    if value is None:
        return False
    return any(abs(float(item) - value) < 1e-9 for item in re.findall(r"\d+(?:\.\d+)?", text))


def _is_numeric_field(field: str) -> bool:
    return field in {"distance_km", "budget_per_person", "price_level", "queue_time"}


def _semantic_score(
    proposal: MemoryProposal | None, truth: FrozenTurnGroundTruthV2
) -> dict[str, object]:
    if proposal is None:
        raise ValueError("semantic score requires the actual proposal used by memory")
    expected = truth.expected_delta
    expected_conditions = {
        (
            item.operation,
            item.field,
            str(item.operator),
            _value(item.value),
            str(item.importance),
        )
        for item in expected.condition_deltas
    }
    predicted_conditions = {
        (
            item.operation,
            item.field,
            str(item.operator),
            _value(item.value),
            str(item.importance),
        )
        for item in proposal.condition_patches
    }
    expected_relative = {
        (item.field, item.direction) for item in expected.relative_preferences
    }
    predicted_relative = {
        (item.field, item.direction) for item in proposal.relative_preferences
    }
    expected_answers = {
        (key, _value(value)) for key, value in expected.clarification_answers.items()
    }
    predicted_answers = {
        (item.information_gap, _value(item.value))
        for item in proposal.clarification_answers
    }
    if (
        "missing_party_size" in expected.clarification_answers
        and proposal.party_size is not None
    ):
        predicted_answers.add(("missing_party_size", _value(proposal.party_size)))
    return {
        "semantic_task_type_correct": proposal.task_type == expected.task_type,
        "semantic_task_goal_correct": _task_goal(proposal.task_type)
        == _task_goal(expected.task_type),
        "semantic_condition_expected": len(expected_conditions),
        "semantic_condition_predicted": len(predicted_conditions),
        "semantic_condition_hits": len(expected_conditions & predicted_conditions),
        "semantic_condition_core_hits": len(
            {_condition_core(item) for item in expected_conditions}
            & {_condition_core(item) for item in predicted_conditions}
        ),
        "semantic_condition_field_operation_hits": len(
            {_condition_field_operation(item) for item in expected_conditions}
            & {_condition_field_operation(item) for item in predicted_conditions}
        ),
        "semantic_relative_expected": len(expected_relative),
        "semantic_relative_predicted": len(predicted_relative),
        "semantic_relative_hits": len(expected_relative & predicted_relative),
        "semantic_clarification_expected": len(expected_answers),
        "semantic_clarification_predicted": len(predicted_answers),
        "semantic_clarification_hits": len(expected_answers & predicted_answers),
    }


def _condition_core(value: tuple[str, str, str, str, str]) -> tuple[str, str, str, str]:
    operation, field, operator, condition_value, _importance = value
    return operation, field, operator, condition_value


def _condition_field_operation(
    value: tuple[str, str, str, str, str]
) -> tuple[str, str]:
    operation, field, _operator, _condition_value, _importance = value
    return operation, field


def _task_goal(task_type: str) -> str:
    return (
        "recommendation_goal"
        if task_type in {"recommendation_request", "feedback_refinement"}
        else task_type
    )


def _behavior_score(
    state: MemoryStateView,
    truth: FrozenTurnGroundTruthV2,
    candidates: Sequence[PresentedBusinessSnapshot],
) -> dict[str, int | list[str]]:
    if not candidates:
        return {
            "behavior_expected": 0,
            "behavior_at_1_hits": 0,
            "behavior_at_5_hits": 0,
            "rejection_exclusion_expected": 0,
            "rejection_exclusion_hits": 0,
            "simulated_recommendations": [],
        }
    ranked = _rerank_from_memory(state, candidates)
    top_ids = [item.business_id for item in ranked[:5]]
    behavior_expected = 0
    hit_at_1 = 0
    hit_at_5 = 0
    rejection_expected = 0
    rejection_hits = 0
    for behavior in truth.behaviors:
        if behavior.kind == "none":
            continue
        acceptable = set(behavior.acceptable_business_ids)
        excluded = set(behavior.excluded_business_ids)
        if acceptable:
            behavior_expected += 1
            hit_at_1 += int(bool(top_ids) and top_ids[0] in acceptable)
            hit_at_5 += int(bool(set(top_ids) & acceptable))
        if excluded:
            rejection_expected += 1
            rejection_hits += int(not (set(top_ids) & excluded))
    return {
        "behavior_expected": behavior_expected,
        "behavior_at_1_hits": hit_at_1,
        "behavior_at_5_hits": hit_at_5,
        "rejection_exclusion_expected": rejection_expected,
        "rejection_exclusion_hits": rejection_hits,
        "simulated_recommendations": top_ids,
    }


def _rerank_from_memory(
    state: MemoryStateView,
    candidates: Sequence[PresentedBusinessSnapshot],
) -> list[PresentedBusinessSnapshot]:
    values = [
        item for item in candidates if item.business_id not in state.rejected_business_ids
    ]
    conditions = list(state.conditions)
    for field, operator, raw_value, importance in conditions:
        value = json.loads(raw_value)
        if importance == "mandatory" and operator in {"less_than_or_equal", "includes", "excludes", "equals"}:
            values = [item for item in values if _condition_matches(item, field, operator, value)]

    relative = set(state.relative_preferences)

    def key(item: PresentedBusinessSnapshot) -> tuple[float, ...]:
        signals: list[float] = []
        if ("price", "lower") in relative:
            signals.append(float(item.price_level if item.price_level is not None else 99))
        if ("distance", "closer") in relative:
            signals.append(float(item.distance_km if item.distance_km is not None else 1e9))
        if ("noise", "quieter") in relative:
            signals.append(float({"quiet": 0, "average": 1, "loud": 2, "very_loud": 3}.get(item.noise_level, 99)))
        for field, operator, raw_value, importance in conditions:
            if importance != "mandatory":
                signals.append(0.0 if _condition_matches(item, field, operator, json.loads(raw_value)) else 1.0)
        return (*signals, float(item.rank))

    return sorted(values, key=key)


def _condition_matches(
    item: PresentedBusinessSnapshot, field: str, operator: str, value: object
) -> bool:
    if field == "price_level":
        actual = item.price_level
        return actual is not None and (actual <= float(value) if operator == "less_than_or_equal" else actual == value)
    if field == "distance_km":
        return item.distance_km is not None and item.distance_km <= float(value)
    if field == "category":
        present = str(value).casefold() in {part.casefold() for part in item.categories}
        return not present if operator == "excludes" else present
    if field == "quiet_environment":
        return item.noise_level == "quiet"
    return True


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> Path:
    return _write_text(path, "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in rows))


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(content, encoding="utf-8", newline="\n")
    partial.replace(path)
    return path
