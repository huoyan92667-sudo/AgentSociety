"""End-to-end Harness replay for the frozen multi-turn Benchmark V2."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from yelp_agent.agent_benchmark import VisibleAgentScenario
from yelp_agent.agent_evaluation import AgentScenarioRun, write_agent_scenario_runs
from yelp_agent.agent_harness import AgentHarness, UserTurnInput

from .schema import (
    FrozenScriptedTurnV2,
    FrozenTurnGroundTruthV2,
    MemoryBenchmarkInitialSession,
)


@dataclass(frozen=True, slots=True)
class AgentReplayV2Result:
    output_root: Path
    runs_path: Path
    cases_path: Path
    metrics_path: Path
    summary_path: Path
    metrics: dict[str, object]


def run_agent_replay_v2(
    harness: AgentHarness,
    *,
    benchmark_root: str | Path,
    output_root: str | Path,
    progress: Callable[[int, int], None] | None = None,
    limit: int | None = None,
) -> AgentReplayV2Result:
    root = Path(benchmark_root)
    output = Path(output_root)
    sessions = _load(root / "visible" / "initial_sessions.jsonl", MemoryBenchmarkInitialSession)
    turns = _load(root / "visible" / "scripted_turns.jsonl", FrozenScriptedTurnV2)
    truths = _load(root / "hidden" / "ground_truth.jsonl", FrozenTurnGroundTruthV2)
    truth_by_id = {item.turn_case_id: item for item in truths}
    turns_by_session: dict[str, list[FrozenScriptedTurnV2]] = defaultdict(list)
    for item in turns:
        turns_by_session[item.session_case_id].append(item)
    runs: list[AgentScenarioRun] = []
    rows: list[dict[str, object]] = []
    ordered = sorted(sessions, key=lambda item: item.session_case_id)
    if limit is not None:
        if limit < 1:
            raise ValueError("Agent replay limit must be positive")
        ordered = ordered[:limit]
    for session_index, initial in enumerate(ordered, start=1):
        result = harness.start(_scenario(initial))
        for turn in sorted(turns_by_session[initial.session_case_id], key=lambda item: item.turn_index):
            previous_run = result.run
            previous_trace = (
                previous_run.turns[-1]
                if previous_run is not None
                else result.session.turns[-1]
                if result.session.turns
                else None
            )
            trigger_correct = _trigger_observed(turn, previous_trace)
            truth = truth_by_id[turn.turn_case_id]
            if result.session.status == "awaiting_user":
                result = harness.resume(result.session, _user_turn(turn))
                release_mode = "resume"
            elif result.session.status == "completed":
                result = harness.follow_up(result.session, _user_turn(turn))
                release_mode = "follow_up"
            else:
                rows.append(_unreleased_row(turn, trigger_correct, result.session.status))
                continue
            trace = result.run.turns[-1] if result.run is not None else result.session.turns[-1]
            rows.append(_score_turn(turn, truth, trace, trigger_correct, release_mode))
        runs.append(result.run or _snapshot_awaiting_run(result.session))
        if session_index % 10 == 0:
            _write_jsonl(output / "checkpoint_behavior_cases.jsonl", rows)
            write_agent_scenario_runs(runs, output / "checkpoint_scenario_runs.jsonl")
        if progress is not None:
            progress(session_index, len(ordered))
    metrics = _metrics(rows, runs)
    runs_path = write_agent_scenario_runs(runs, output / "scenario_runs.jsonl")
    cases_path = _write_jsonl(output / "behavior_cases.jsonl", rows)
    metrics_path = _write_text(output / "metrics.json", json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    summary_path = _write_text(output / "summary.md", _summary(metrics))
    return AgentReplayV2Result(output, runs_path, cases_path, metrics_path, summary_path, metrics)


def _scenario(value: MemoryBenchmarkInitialSession) -> VisibleAgentScenario:
    return VisibleAgentScenario(
        scenario_id=value.session_case_id,
        split=value.split,
        language=value.language,
        user_id=value.user_id,
        session_id=value.session_id,
        cutoff_time=value.cutoff_time,
        query_text=value.query_text,
        user_latitude=value.user_latitude,
        user_longitude=value.user_longitude,
        referenced_business_ids=value.referenced_business_ids,
        generator_kind="deterministic",
    )


def _user_turn(value: FrozenScriptedTurnV2) -> UserTurnInput:
    latitude = value.state_updates.get("user_latitude")
    longitude = value.state_updates.get("user_longitude")
    return UserTurnInput(
        query_text=value.query_text,
        user_latitude=float(latitude) if latitude is not None else None,
        user_longitude=float(longitude) if longitude is not None else None,
    )


def _trigger_observed(value: FrozenScriptedTurnV2, trace: object | None) -> bool:
    if trace is None:
        return False
    expected = value.trigger_action
    actions = {item.action for item in getattr(trace, "actions", [])}
    return expected in actions


def _score_turn(value, truth, trace, trigger_correct: bool, release_mode: str) -> dict[str, object]:
    recommended = list(trace.recommended_business_ids[:5])
    behavior_expected = 0
    hit1 = 0
    hit5 = 0
    rejection_expected = 0
    rejection_hits = 0
    for behavior in truth.behaviors:
        acceptable = set(behavior.acceptable_business_ids)
        excluded = set(behavior.excluded_business_ids)
        if acceptable:
            behavior_expected += 1
            hit1 += int(bool(recommended) and recommended[0] in acceptable)
            hit5 += int(bool(set(recommended) & acceptable))
        if excluded:
            rejection_expected += 1
            rejection_hits += int(not (set(recommended) & excluded))
    return {
        "turn_case_id": value.turn_case_id,
        "session_case_id": value.session_case_id,
        "split": value.split,
        "family": value.family,
        "released": True,
        "release_mode": release_mode,
        "trigger_correct": trigger_correct,
        "response_kind": trace.response_kind,
        "predicted_task_type": trace.predicted_task_type,
        "recommended_business_ids": recommended,
        "candidate_count": len(trace.candidate_ranking),
        "valid_recommendations": set(recommended).issubset(trace.candidate_ranking),
        "behavior_expected": behavior_expected,
        "behavior_at_1_hits": hit1,
        "behavior_at_5_hits": hit5,
        "rejection_exclusion_expected": rejection_expected,
        "rejection_exclusion_hits": rejection_hits,
    }


def _unreleased_row(value, trigger_correct: bool, status: str) -> dict[str, object]:
    return {
        "turn_case_id": value.turn_case_id,
        "session_case_id": value.session_case_id,
        "split": value.split,
        "family": value.family,
        "released": False,
        "release_mode": "unavailable",
        "trigger_correct": trigger_correct,
        "response_kind": status,
        "predicted_task_type": "unknown",
        "recommended_business_ids": [],
        "candidate_count": 0,
        "valid_recommendations": True,
        "behavior_expected": 0,
        "behavior_at_1_hits": 0,
        "behavior_at_5_hits": 0,
        "rejection_exclusion_expected": 0,
        "rejection_exclusion_hits": 0,
    }


def _snapshot_awaiting_run(session: object) -> AgentScenarioRun:
    from yelp_agent.agent_harness import AgentSession

    state = AgentSession.model_validate(session)
    if state.status != "awaiting_user" or not state.turns:
        raise RuntimeError(
            f"session {state.scenario_id} produced neither a final run nor a visible question"
        )
    return AgentScenarioRun(
        scenario_id=state.scenario_id,
        agent_version=state.agent_version,
        turns=state.turns,
        fallback=False,
        latency_ms=state.elapsed_ms,
        input_tokens=state.input_tokens if state.token_usage_observed else None,
        output_tokens=state.output_tokens if state.token_usage_observed else None,
        cost_usd=state.cost_usd if state.cost_usd > 0 else None,
    )


def _metrics(rows: Sequence[dict[str, object]], runs: Sequence[AgentScenarioRun]) -> dict[str, object]:
    def ratio(key: str, denominator_key: str | None = None) -> float:
        denominator = len(rows) if denominator_key is None else sum(int(item[denominator_key]) for item in rows)
        numerator = sum(int(item[key]) for item in rows)
        return numerator / denominator if denominator else 1.0

    return {
        "schema_version": 2,
        "session_count": len(runs),
        "turn_count": len(rows),
        "released_turn_rate": ratio("released"),
        "trigger_accuracy": ratio("trigger_correct"),
        "valid_recommendation_rate": ratio("valid_recommendations"),
        "behavior_compliance_at_1": ratio("behavior_at_1_hits", "behavior_expected"),
        "behavior_compliance_at_5": ratio("behavior_at_5_hits", "behavior_expected"),
        "rejected_business_exclusion_rate": ratio("rejection_exclusion_hits", "rejection_exclusion_expected"),
        "fallback_session_rate": sum(item.fallback for item in runs) / len(runs),
        "mean_session_latency_ms": sum(item.latency_ms for item in runs) / len(runs),
        "input_tokens": sum(item.input_tokens or 0 for item in runs),
        "output_tokens": sum(item.output_tokens or 0 for item in runs),
    }


def _summary(metrics: dict[str, object]) -> str:
    return "\n".join([
        "# Session Memory Benchmark V2：完整 Agent Harness 重放",
        "",
        f"- 会话 / 后续轮次：{metrics['session_count']} / {metrics['turn_count']}",
        f"- 隐藏轮次触发准确率：{float(metrics['trigger_accuracy']):.2%}",
        f"- 推荐行为 Compliance@1 / @5：{float(metrics['behavior_compliance_at_1']):.2%} / {float(metrics['behavior_compliance_at_5']):.2%}",
        f"- 拒绝商家排除率：{float(metrics['rejected_business_exclusion_rate']):.2%}",
        f"- Fallback 会话率：{float(metrics['fallback_session_rate']):.2%}",
        "",
    ])


def _load(path: Path, model: type) -> tuple[object, ...]:
    return tuple(model.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> Path:
    return _write_text(path, "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in rows))


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(content, encoding="utf-8", newline="\n")
    partial.replace(path)
    return path
