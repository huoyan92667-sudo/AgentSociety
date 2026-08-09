"""Typed, deterministic file adapters for the Step 21 evaluator seam."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from yelp_agent.agent_benchmark import (
    load_evidence_labels,
    load_scenario_ground_truth,
    load_visible_scenarios,
)

from .evaluator import evaluate_agent_scenario_runs
from .schema import AgentEvaluationReport, AgentScenarioRun


def _atomic_write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.unlink(missing_ok=True)
    try:
        partial.write_text(content, encoding="utf-8", newline="\n")
        partial.replace(path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return path


def load_agent_scenario_runs(path: str | Path) -> tuple[AgentScenarioRun, ...]:
    """Load one run per scenario and reject malformed or duplicate output."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Agent run file does not exist: {source}")
    values: list[AgentScenarioRun] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            values.append(AgentScenarioRun.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"invalid Agent run at line {line_number}") from exc
    if not values:
        raise ValueError("Agent run file cannot be empty")
    ids = [item.scenario_id for item in values]
    if len(ids) != len(set(ids)):
        raise ValueError("Agent run file contains duplicate scenario IDs")
    return tuple(sorted(values, key=lambda item: item.scenario_id))


def write_agent_scenario_runs(
    runs: Sequence[AgentScenarioRun],
    path: str | Path,
) -> Path:
    """Publish runner output in stable scenario order for later hidden evaluation."""

    if not runs:
        raise ValueError("cannot write an empty Agent run file")
    ids = [item.scenario_id for item in runs]
    if len(ids) != len(set(ids)):
        raise ValueError("Agent scenario runs must be unique before writing")
    content = "".join(
        item.model_dump_json() + "\n"
        for item in sorted(runs, key=lambda item: item.scenario_id)
    )
    return _atomic_write_text(Path(path), content)


def write_agent_evaluation_report(
    report: AgentEvaluationReport,
    path: str | Path,
) -> Path:
    return _atomic_write_text(
        Path(path),
        report.model_dump_json(indent=2) + "\n",
    )


def evaluate_agent_scenario_files(
    runs_path: str | Path,
    *,
    benchmark_root: str | Path,
    output_path: str | Path,
) -> AgentEvaluationReport:
    """The single file-level interface used by future Rule and LLM Agents."""

    root = Path(benchmark_root)
    report = evaluate_agent_scenario_runs(
        load_agent_scenario_runs(runs_path),
        visible_scenarios=load_visible_scenarios(
            root / "visible" / "scenarios.jsonl"
        ),
        ground_truth=load_scenario_ground_truth(
            root / "hidden" / "ground_truth.jsonl"
        ),
        evidence_labels=load_evidence_labels(
            root / "hidden" / "evidence_labels.parquet"
        ),
    )
    write_agent_evaluation_report(report, output_path)
    return report
