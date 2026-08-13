"""Context-grounded replay evaluation and human-readable failure cases.

V2 froze behavior labels against an older presentation.  V3 keeps the hidden
intent frozen, but rebinds ordinal references (for example, "the first one")
to the businesses that the evaluated Agent actually showed.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Literal

import pyarrow.parquet as pq

from yelp_agent.agent_evaluation import (
    AgentScenarioRun,
    AgentTurnTrace,
    load_agent_scenario_runs,
)

from .dynamics import derive_relative_behavior
from .schema import (
    BehaviorExpectation,
    ExpectedConditionDelta,
    FrozenPresentation,
    FrozenScriptedTurnV2,
    FrozenTurnGroundTruthV2,
    MemoryBenchmarkInitialSession,
    PresentedBusinessSnapshot,
)
from .sources import _business_catalog, _snapshot

type ContextStatus = Literal["not_required", "aligned", "rebound", "unavailable"]


@dataclass(frozen=True, slots=True)
class ContextualBehaviorBinding:
    behaviors: tuple[BehaviorExpectation, ...]
    context_status: ContextStatus
    reference_mappings: tuple[dict[str, object], ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ContextGroundedEvaluationResult:
    output_root: Path
    cases_path: Path
    explorer_path: Path
    failures_path: Path
    explorer_index_path: Path
    metrics_path: Path
    summary_path: Path
    metrics: dict[str, object]


def rebind_behavior_expectations(
    truth: FrozenTurnGroundTruthV2,
    *,
    frozen_presentation: FrozenPresentation | None,
    previous_presented_business_ids: Sequence[str],
    candidate_snapshots: Sequence[PresentedBusinessSnapshot],
    reference_snapshots: Sequence[PresentedBusinessSnapshot] = (),
) -> ContextualBehaviorBinding:
    """Rebuild observable answer sets from the evaluated Agent's real context."""

    candidates = list(candidate_snapshots)
    snapshots = {item.business_id: item for item in (*candidates, *reference_snapshots)}
    condition_deltas = iter(truth.expected_delta.condition_deltas)
    rebound: list[BehaviorExpectation] = []
    mappings: list[dict[str, object]] = []
    notes: list[str] = []
    statuses: list[ContextStatus] = []

    for behavior in truth.behaviors:
        if behavior.kind in {"cheaper", "closer", "quieter"}:
            mapped = _actual_reference(
                behavior.baseline_business_id,
                frozen_presentation=frozen_presentation,
                previous_presented_business_ids=previous_presented_business_ids,
                snapshots=snapshots,
            )
            if mapped is None:
                rebound.append(_empty_behavior(behavior))
                statuses.append("unavailable")
                notes.append("ordinal_reference_could_not_be_bound_to_actual_presentation")
                continue
            reference, mapping = mapped
            mappings.append(mapping)
            try:
                rebound.append(derive_relative_behavior(behavior.kind, reference, candidates))
            except ValueError as exc:
                rebound.append(
                    behavior.model_copy(
                        update={
                            "baseline_business_id": reference.business_id,
                            "baseline_value": _relative_value(behavior.kind, reference),
                            "acceptable_business_ids": [],
                        }
                    )
                )
                notes.append(str(exc))
            statuses.append(
                "aligned"
                if reference.business_id == behavior.baseline_business_id
                else "rebound"
            )
            continue

        if behavior.excluded_business_ids:
            excluded: list[str] = []
            unavailable = False
            for frozen_id in behavior.excluded_business_ids:
                mapped = _actual_reference(
                    frozen_id,
                    frozen_presentation=frozen_presentation,
                    previous_presented_business_ids=previous_presented_business_ids,
                    snapshots=snapshots,
                )
                if mapped is None:
                    unavailable = True
                    notes.append("rejected_ordinal_could_not_be_bound_to_actual_presentation")
                    continue
                reference, mapping = mapped
                mappings.append(mapping)
                excluded.append(reference.business_id)
                statuses.append(
                    "aligned" if reference.business_id == frozen_id else "rebound"
                )
            rebound.append(behavior.model_copy(update={"excluded_business_ids": excluded}))
            if unavailable:
                statuses.append("unavailable")
            continue

        if behavior.kind == "constraint_satisfaction":
            delta = next(condition_deltas, None)
            if delta is None:
                rebound.append(_empty_behavior(behavior))
                statuses.append("unavailable")
                notes.append("constraint_behavior_has_no_matching_expected_delta")
            else:
                derived = _behavior_for_condition(delta, candidates)
                rebound.append(derived)
                statuses.append("not_required")
                if not _condition_is_observable(delta):
                    notes.append(f"condition_{delta.field}_is_not_observable_from_yelp_facts")
            continue

        rebound.append(behavior)
        statuses.append("not_required")

    return ContextualBehaviorBinding(
        behaviors=tuple(rebound),
        context_status=_combined_status(statuses),
        reference_mappings=tuple(mappings),
        notes=tuple(dict.fromkeys(notes)),
    )


def evaluate_context_grounded_replay_v3(
    *,
    runs_path: str | Path,
    benchmark_root: str | Path,
    businesses_path: str | Path,
    output_root: str | Path,
    profile_snapshots_path: str | Path | None = None,
    preference_signals_path: str | Path | None = None,
) -> ContextGroundedEvaluationResult:
    """Rescore saved Agent traces without rerunning retrieval, models, or APIs."""

    benchmark = Path(benchmark_root)
    output = Path(output_root)
    sessions = _load(benchmark / "visible" / "initial_sessions.jsonl", MemoryBenchmarkInitialSession)
    turns = _load(benchmark / "visible" / "scripted_turns.jsonl", FrozenScriptedTurnV2)
    truths = _load(benchmark / "hidden" / "ground_truth.jsonl", FrozenTurnGroundTruthV2)
    presentations = _load(
        benchmark / "visible" / "frozen_presentations.jsonl", FrozenPresentation
    )
    runs = load_agent_scenario_runs(runs_path)
    run_by_session = {item.scenario_id: item for item in runs}
    truth_by_turn = {item.turn_case_id: item for item in truths}
    presentation_by_session = {item.session_case_id: item for item in presentations}
    turns_by_session: dict[str, list[FrozenScriptedTurnV2]] = defaultdict(list)
    for turn in turns:
        turns_by_session[turn.session_case_id].append(turn)
    catalog = _business_catalog(Path(businesses_path))
    profiles = _load_profiles(
        sessions,
        profile_snapshots_path=profile_snapshots_path,
        preference_signals_path=preference_signals_path,
    )

    rows: list[dict[str, object]] = []
    explorers: list[dict[str, object]] = []
    for initial in sorted(sessions, key=lambda item: item.session_case_id):
        run = run_by_session.get(initial.session_case_id)
        if run is None:
            continue
        trace_by_turn = {item.turn_index: item for item in run.turns}
        initial_trace = trace_by_turn.get(1)
        last_presented = (
            [] if initial_trace is None else list(initial_trace.recommended_business_ids)
        )
        last_candidate_ranking = (
            [] if initial_trace is None else list(initial_trace.candidate_ranking)
        )
        query_history = [initial.query_text]
        frozen = presentation_by_session.get(initial.session_case_id)
        for turn in sorted(
            turns_by_session[initial.session_case_id], key=lambda item: item.turn_index
        ):
            trace = trace_by_turn.get(turn.turn_index)
            query_history.append(turn.query_text)
            if trace is None:
                rows.append(_unreleased_row(turn, last_presented))
                continue
            evaluation_scope, scope_source = evaluation_candidate_scope(
                trace.candidate_ranking,
                last_candidate_ranking,
            )
            current_snapshots = _snapshots(
                evaluation_scope,
                catalog,
                initial,
            )
            reference_snapshots = _snapshots(last_presented, catalog, initial)
            truth = truth_by_turn[turn.turn_case_id]
            binding = rebind_behavior_expectations(
                truth,
                frozen_presentation=frozen,
                previous_presented_business_ids=last_presented,
                candidate_snapshots=current_snapshots,
                reference_snapshots=reference_snapshots,
            )
            scored = _score_binding(
                binding,
                recommendations=trace.recommended_business_ids[:5],
                ranking=evaluation_scope,
            )
            row = {
                "turn_case_id": turn.turn_case_id,
                "session_case_id": turn.session_case_id,
                "split": turn.split,
                "family": turn.family,
                "released": True,
                "query_text": turn.query_text,
                "response_kind": trace.response_kind,
                "predicted_task_type": trace.predicted_task_type,
                "recommended_business_ids": list(trace.recommended_business_ids[:5]),
                "candidate_count": len(evaluation_scope),
                "evaluation_candidate_scope_source": scope_source,
                "previous_presented_business_ids": list(last_presented),
                "context_status": binding.context_status,
                "reference_mappings": list(binding.reference_mappings),
                "label_notes": list(binding.notes),
                **scored,
            }
            rows.append(row)
            explorers.append(
                _explorer_case(
                    initial=initial,
                    turn=turn,
                    trace=trace,
                    query_history=query_history,
                    previous_presented=reference_snapshots,
                    candidates=current_snapshots,
                    profile=profiles.get(_profile_key(initial.user_id, initial.cutoff_time)),
                    row=row,
                )
            )
            if trace.recommended_business_ids:
                last_presented = list(trace.recommended_business_ids)
            if trace.candidate_ranking:
                last_candidate_ranking = list(trace.candidate_ranking)

    metrics = _metrics(rows)
    cases_path = _write_jsonl(output / "behavior_cases_v3.jsonl", rows)
    explorer_path = _write_jsonl(output / "case_explorer_v3.jsonl", explorers)
    failures_path = _write_jsonl(
        output / "case_explorer_v3_failures.jsonl",
        [item for item in explorers if item["judge"]["outcome"] != "pass_top1"],
    )
    explorer_index_path = _write_text(
        output / "case_explorer_v3.md", _explorer_index(explorers)
    )
    metrics_path = _write_text(
        output / "metrics_v3.json",
        json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    summary_path = _write_text(output / "summary_v3.md", _summary(metrics))
    return ContextGroundedEvaluationResult(
        output,
        cases_path,
        explorer_path,
        failures_path,
        explorer_index_path,
        metrics_path,
        summary_path,
        metrics,
    )


def evaluation_candidate_scope(
    current_ranking: Sequence[str],
    previous_ranking: Sequence[str],
) -> tuple[list[str], Literal["current_turn", "carried_forward", "unavailable"]]:
    """Use the current scope, or the last real scope when a turn returns nothing."""

    if current_ranking:
        return list(current_ranking), "current_turn"
    if previous_ranking:
        return list(previous_ranking), "carried_forward"
    return [], "unavailable"


def _actual_reference(
    frozen_business_id: str | None,
    *,
    frozen_presentation: FrozenPresentation | None,
    previous_presented_business_ids: Sequence[str],
    snapshots: dict[str, PresentedBusinessSnapshot],
) -> tuple[PresentedBusinessSnapshot, dict[str, object]] | None:
    if frozen_business_id is None or frozen_presentation is None:
        return None
    frozen_ids = [item.business_id for item in frozen_presentation.presented_businesses]
    try:
        ordinal = frozen_ids.index(frozen_business_id)
        actual_id = previous_presented_business_ids[ordinal]
        actual = snapshots[actual_id]
    except (ValueError, IndexError, KeyError):
        return None
    return actual, {
        "ordinal": ordinal + 1,
        "frozen_business_id": frozen_business_id,
        "actual_business_id": actual_id,
    }


def _behavior_for_condition(
    delta: ExpectedConditionDelta,
    candidates: Sequence[PresentedBusinessSnapshot],
) -> BehaviorExpectation:
    acceptable: list[str] = []
    maximum: float | int | None = None
    category: str | None = None
    if delta.field == "price_level" and delta.value is not None:
        maximum = int(delta.value)
        acceptable = [
            item.business_id
            for item in candidates
            if item.price_level is not None and item.price_level <= maximum
        ]
    elif delta.field == "distance_km" and delta.value is not None:
        maximum = float(delta.value)
        acceptable = [
            item.business_id
            for item in candidates
            if item.distance_km is not None and item.distance_km <= maximum
        ]
    elif delta.field == "quiet_environment":
        acceptable = [item.business_id for item in candidates if item.noise_level == "quiet"]
    elif delta.field == "category" and delta.value is not None:
        category = str(delta.value)
        expected = category.casefold()
        includes = delta.operator != "excludes"
        acceptable = [
            item.business_id
            for item in candidates
            if (expected in {value.casefold() for value in item.categories}) == includes
        ]
    return BehaviorExpectation(
        kind="constraint_satisfaction",
        acceptable_business_ids=acceptable,
        required_category=category,
        maximum_value=maximum,
    )


def _condition_is_observable(delta: ExpectedConditionDelta) -> bool:
    return delta.field in {"price_level", "distance_km", "quiet_environment", "category"}


def _score_binding(
    binding: ContextualBehaviorBinding,
    *,
    recommendations: Sequence[str],
    ranking: Sequence[str],
) -> dict[str, object]:
    top5 = list(recommendations[:5])
    rank_by_id = {business_id: index for index, business_id in enumerate(ranking, start=1)}
    details: list[dict[str, object]] = []
    atomic_expected = atomic_hit1 = atomic_hit5 = 0
    rejection_expected = rejection_hits = 0
    acceptable_sets: list[set[str]] = []
    excluded_union: set[str] = set()
    for behavior in binding.behaviors:
        acceptable = set(behavior.acceptable_business_ids)
        excluded = set(behavior.excluded_business_ids)
        best_rank = min((rank_by_id[item] for item in acceptable if item in rank_by_id), default=None)
        if acceptable:
            acceptable_sets.append(acceptable)
            atomic_expected += 1
            atomic_hit1 += int(bool(top5) and top5[0] in acceptable)
            atomic_hit5 += int(bool(set(top5) & acceptable))
        if excluded:
            excluded_union.update(excluded)
            rejection_expected += 1
            rejection_hits += int(not (set(top5) & excluded))
        details.append(
            {
                "kind": behavior.kind,
                "baseline_business_id": behavior.baseline_business_id,
                "baseline_value": behavior.baseline_value,
                "acceptable_business_ids": sorted(acceptable),
                "acceptable_count": len(acceptable),
                "excluded_business_ids": sorted(excluded),
                "best_acceptable_rank": best_rank,
                "hit_at_1": bool(acceptable and top5 and top5[0] in acceptable),
                "hit_at_5": bool(acceptable and set(top5) & acceptable),
            }
        )
    joint = set.intersection(*acceptable_sets) if acceptable_sets else set()
    joint.difference_update(excluded_union)
    joint_best_rank = min((rank_by_id[item] for item in joint if item in rank_by_id), default=None)
    joint_expected = int(bool(acceptable_sets))
    joint_hit1 = int(bool(joint and top5 and top5[0] in joint))
    joint_hit5 = int(bool(joint and set(top5) & joint))
    outcome = _outcome(
        recommendations=top5,
        answer_count=len(joint),
        best_answer_rank=joint_best_rank,
        hit1=bool(joint_hit1),
        hit5=bool(joint_hit5),
        expected=bool(joint_expected),
    )
    return {
        "atomic_behavior_expected": atomic_expected,
        "atomic_behavior_at_1_hits": atomic_hit1,
        "atomic_behavior_at_5_hits": atomic_hit5,
        "joint_behavior_expected": joint_expected,
        "joint_behavior_at_1_hits": joint_hit1,
        "joint_behavior_at_5_hits": joint_hit5,
        "rejection_exclusion_expected": rejection_expected,
        "rejection_exclusion_hits": rejection_hits,
        "acceptable_business_ids": sorted(joint),
        "acceptable_business_count": len(joint),
        "best_acceptable_rank": joint_best_rank,
        "behavior_details": details,
        "outcome": outcome,
    }


def _outcome(
    *,
    recommendations: Sequence[str],
    answer_count: int,
    best_answer_rank: int | None,
    hit1: bool,
    hit5: bool,
    expected: bool,
) -> str:
    if not expected:
        return "not_scored"
    if not recommendations:
        return "no_recommendation"
    if answer_count == 0 or best_answer_rank is None:
        return "acceptable_absent_from_ranking"
    if hit1:
        return "pass_top1"
    if hit5:
        return "acceptable_in_top5_not_top1"
    if best_answer_rank <= 5:
        return "acceptable_in_candidate_top5_but_not_returned"
    return "acceptable_below_top5"


def _explorer_case(
    *,
    initial: MemoryBenchmarkInitialSession,
    turn: FrozenScriptedTurnV2,
    trace: AgentTurnTrace,
    query_history: Sequence[str],
    previous_presented: Sequence[PresentedBusinessSnapshot],
    candidates: Sequence[PresentedBusinessSnapshot],
    profile: dict[str, object] | None,
    row: dict[str, object],
) -> dict[str, object]:
    acceptable = set(row["acceptable_business_ids"])
    candidate_rows = [
        {**item.model_dump(mode="json"), "is_acceptable_answer": item.business_id in acceptable}
        for item in candidates[:20]
    ]
    answer_examples = [
        item.model_dump(mode="json")
        for item in candidates
        if item.business_id in acceptable
    ][:10]
    return {
        "turn_case_id": turn.turn_case_id,
        "session_case_id": turn.session_case_id,
        "split": turn.split,
        "family": turn.family,
        "user_id": initial.user_id,
        "cutoff_time": initial.cutoff_time.isoformat(),
        "user_profile": profile,
        "conversation": [
            {"turn_index": index, "query_text": query}
            for index, query in enumerate(query_history, start=1)
        ],
        "actual_previous_presentation": [
            item.model_dump(mode="json") for item in previous_presented
        ],
        "agent_flow": {
            "predicted_task_type": trace.predicted_task_type,
            "response_kind": trace.response_kind,
            "effective_request": trace.effective_request,
            "actions": [item.model_dump(mode="json") for item in trace.actions],
            "tools": [
                {
                    "tool_name": item.tool_name,
                    "status": item.status,
                    "latency_ms": item.latency_ms,
                }
                for item in trace.tool_calls
            ],
        },
        "candidate_top20": candidate_rows,
        "recommended_business_ids": row["recommended_business_ids"],
        "judge": {
            "label_scope": "current_turn_delta_on_actual_agent_context",
            "context_status": row["context_status"],
            "reference_mappings": row["reference_mappings"],
            "acceptable_business_ids": row["acceptable_business_ids"],
            "acceptable_business_count": row["acceptable_business_count"],
            "acceptable_answer_examples": answer_examples,
            "best_acceptable_rank": row["best_acceptable_rank"],
            "outcome": row["outcome"],
            "notes": row["label_notes"],
            "full_session_constraint_compliance_measured": False,
        },
    }


def _metrics(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    released = [item for item in rows if item["released"]]

    def ratio(numerator_key: str, denominator_key: str) -> float:
        denominator = sum(int(item[denominator_key]) for item in released)
        numerator = sum(int(item[numerator_key]) for item in released)
        return numerator / denominator if denominator else 1.0

    answer_counts = [
        int(item["acceptable_business_count"])
        for item in released
        if int(item["joint_behavior_expected"])
    ]
    outcomes: dict[str, int] = defaultdict(int)
    contexts: dict[str, int] = defaultdict(int)
    for item in released:
        outcomes[str(item["outcome"])] += 1
        contexts[str(item["context_status"])] += 1
    return {
        "schema_version": 3,
        "label_scope": "current_turn_delta_on_actual_agent_context",
        "full_session_constraint_compliance_measured": False,
        "turn_count": len(rows),
        "released_turn_count": len(released),
        "atomic_behavior_compliance_at_1": ratio(
            "atomic_behavior_at_1_hits", "atomic_behavior_expected"
        ),
        "atomic_behavior_compliance_at_5": ratio(
            "atomic_behavior_at_5_hits", "atomic_behavior_expected"
        ),
        "joint_delta_compliance_at_1": ratio(
            "joint_behavior_at_1_hits", "joint_behavior_expected"
        ),
        "joint_delta_compliance_at_5": ratio(
            "joint_behavior_at_5_hits", "joint_behavior_expected"
        ),
        "rejected_business_exclusion_rate": ratio(
            "rejection_exclusion_hits", "rejection_exclusion_expected"
        ),
        "context_status_counts": dict(sorted(contexts.items())),
        "outcome_counts": dict(sorted(outcomes.items())),
        "acceptable_answer_count": {
            "minimum": min(answer_counts, default=0),
            "median": median(answer_counts) if answer_counts else 0,
            "maximum": max(answer_counts, default=0),
            "mean": sum(answer_counts) / len(answer_counts) if answer_counts else 0,
        },
    }


def _unreleased_row(
    turn: FrozenScriptedTurnV2, previous_presented: Sequence[str]
) -> dict[str, object]:
    return {
        "turn_case_id": turn.turn_case_id,
        "session_case_id": turn.session_case_id,
        "split": turn.split,
        "family": turn.family,
        "released": False,
        "previous_presented_business_ids": list(previous_presented),
        "context_status": "unavailable",
        "atomic_behavior_expected": 0,
        "atomic_behavior_at_1_hits": 0,
        "atomic_behavior_at_5_hits": 0,
        "joint_behavior_expected": 0,
        "joint_behavior_at_1_hits": 0,
        "joint_behavior_at_5_hits": 0,
        "rejection_exclusion_expected": 0,
        "rejection_exclusion_hits": 0,
        "acceptable_business_count": 0,
        "outcome": "unreleased",
    }


def _load_profiles(
    sessions: Sequence[MemoryBenchmarkInitialSession],
    *,
    profile_snapshots_path: str | Path | None,
    preference_signals_path: str | Path | None,
) -> dict[tuple[str, str], dict[str, object]]:
    if profile_snapshots_path is None or not Path(profile_snapshots_path).is_file():
        return {}
    user_ids = sorted({item.user_id for item in sessions})
    table = pq.read_table(
        profile_snapshots_path,
        filters=[("user_id", "in", user_ids)],
    )
    profiles: dict[tuple[str, str], dict[str, object]] = {}
    profile_id_to_key: dict[str, tuple[str, str]] = {}
    for row in table.to_pylist():
        key = _profile_key(str(row["user_id"]), row["cutoff_time"])
        serialized = {key_: _json_value(value) for key_, value in row.items()}
        serialized["preference_signals"] = []
        profiles[key] = serialized
        profile_id_to_key[str(row["profile_id"])] = key
    if preference_signals_path is not None and Path(preference_signals_path).is_file():
        signal_table = pq.read_table(
            preference_signals_path,
            filters=[("user_id", "in", user_ids)],
        )
        for row in signal_table.to_pylist():
            key = profile_id_to_key.get(str(row["profile_id"]))
            if key in profiles:
                profiles[key]["preference_signals"].append(
                    {key_: _json_value(value) for key_, value in row.items()}
                )
    return profiles


def _profile_key(user_id: str, cutoff_time: object) -> tuple[str, str]:
    return user_id, _json_value(cutoff_time)


def _json_value(value: object) -> object:
    if hasattr(value, "isoformat"):
        return value.isoformat()  # type: ignore[union-attr]
    return value


def _snapshots(
    business_ids: Sequence[str],
    catalog: dict[str, dict[str, object]],
    initial: MemoryBenchmarkInitialSession,
) -> list[PresentedBusinessSnapshot]:
    return [
        _snapshot(
            catalog[business_id],
            business_id=business_id,
            rank=index,
            user_latitude=initial.user_latitude,
            user_longitude=initial.user_longitude,
        )
        for index, business_id in enumerate(business_ids, start=1)
        if business_id in catalog
    ]


def _empty_behavior(behavior: BehaviorExpectation) -> BehaviorExpectation:
    updates: dict[str, object] = {
        "acceptable_business_ids": [],
        "excluded_business_ids": [],
    }
    if behavior.kind in {"cheaper", "closer", "quieter"}:
        updates.update({"baseline_business_id": behavior.baseline_business_id, "baseline_value": behavior.baseline_value})
    return behavior.model_copy(update=updates)


def _relative_value(kind: str, reference: PresentedBusinessSnapshot) -> float | int | None:
    if kind == "cheaper":
        return reference.price_level
    if kind == "closer":
        return reference.distance_km
    return {"quiet": 0, "average": 1, "loud": 2, "very_loud": 3}.get(reference.noise_level)


def _combined_status(statuses: Sequence[ContextStatus]) -> ContextStatus:
    for value in ("unavailable", "rebound", "aligned", "not_required"):
        if value in statuses:
            return value  # type: ignore[return-value]
    return "not_required"


def _summary(metrics: dict[str, object]) -> str:
    answers = metrics["acceptable_answer_count"]
    assert isinstance(answers, dict)
    return "\n".join(
        [
            "# Context-grounded Agent Evaluator V3",
            "",
            "V3 将序号指代绑定到被评 Agent 上一轮真正展示的商家列表。",
            "当前只衡量本轮新增行为（delta），尚不宣称覆盖完整会话全部条件。",
            "",
            f"- Atomic Compliance@1 / @5: {float(metrics['atomic_behavior_compliance_at_1']):.2%} / {float(metrics['atomic_behavior_compliance_at_5']):.2%}",
            f"- Joint-delta Compliance@1 / @5: {float(metrics['joint_delta_compliance_at_1']):.2%} / {float(metrics['joint_delta_compliance_at_5']):.2%}",
            f"- 正确商家集合数量（min / median / max）: {answers['minimum']} / {answers['median']} / {answers['maximum']}",
            f"- 上下文绑定统计: {json.dumps(metrics['context_status_counts'], ensure_ascii=False, sort_keys=True)}",
            "",
        ]
    )


def _explorer_index(explorers: Sequence[dict[str, object]]) -> str:
    failures = [item for item in explorers if item["judge"]["outcome"] != "pass_top1"]
    lines = [
        "# Agent Case Explorer V3",
        "",
        "本索引用于快速定位错误；完整用户画像、动作、工具、Top-20 和正确答案示例位于 `case_explorer_v3_failures.jsonl`。",
        "",
        "| Turn | Family | Query | Response | Outcome | Correct count | Best rank |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for item in failures:
        judge = item["judge"]
        flow = item["agent_flow"]
        conversation = item["conversation"]
        query = str(conversation[-1]["query_text"]).replace("|", "\\|").replace("\n", " ")
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{str(item['turn_case_id'])[:10]}`",
                    str(item["family"]),
                    query,
                    str(flow["response_kind"]),
                    str(judge["outcome"]),
                    str(judge["acceptable_business_count"]),
                    str(judge["best_acceptable_rank"] or "-"),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _load(path: Path, model: type) -> tuple[object, ...]:
    return tuple(
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> Path:
    return _write_text(
        path,
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in rows),
    )


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(content, encoding="utf-8", newline="\n")
    partial.replace(path)
    return path
