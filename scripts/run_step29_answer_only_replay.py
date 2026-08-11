"""Replay only Step 29 answer composition over frozen Step 28 terminal claims.

This is an exact terminal-stage ablation: answer composition cannot change routing,
tools, candidates, rankings, or evidence retrieval, so expensive local models do
not need to be rerun. Hidden scripted turns are read only by this experiment
driver to reconstruct text that the original run had already released.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yelp_agent.agent_benchmark import (
    load_scenario_ground_truth,
    load_visible_scenarios,
)
from yelp_agent.agent_evaluation import (
    AgentScenarioRun,
    load_agent_scenario_runs,
    write_agent_scenario_runs,
)
from yelp_agent.controlled_llm import (
    AnswerCompositionInput,
    AnswerEvidenceItem,
    augment_runtime_metrics,
    build_controlled_llm_runtime,
    load_controlled_llm_config,
)
from yelp_agent.rule_router import (
    RuleAgentSourcePaths,
    replace_rule_agent_benchmark_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--base-root", type=Path, default=Path("runs/rule_agent_evidence_v1")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/controlled_llm_v1/agent/answer_only"),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    base_root = _resolve(root, args.base_root)
    output = _resolve(root, args.output_root)
    replacement = output / "terminal_replay"
    sources = RuleAgentSourcePaths.from_project_root(root)
    visible = {
        item.scenario_id: item
        for item in load_visible_scenarios(
            sources.benchmark_root / "visible" / "scenarios.jsonl"
        )
    }
    truth = {
        item.scenario_id: item
        for item in load_scenario_ground_truth(
            sources.benchmark_root / "hidden" / "ground_truth.jsonl"
        )
    }
    driver_rows = {
        str(item["scenario_id"]): item
        for item in _load_jsonl(base_root / "driver_results.jsonl")
    }
    base_runs = load_agent_scenario_runs(base_root / "scenario_runs.jsonl")
    config = load_controlled_llm_config(
        root / "configs" / "controlled_llm_answer_only.yaml"
    )
    changed: list[AgentScenarioRun] = []
    with build_controlled_llm_runtime(project_root=root, config=config) as runtime:
        for index, run in enumerate(base_runs, start=1):
            scenario = visible[run.scenario_id]
            hidden = truth[run.scenario_id]
            query_by_turn = {1: scenario.query_text}
            query_by_turn.update(
                {item.turn_index: item.query_text for item in hidden.scripted_user_turns}
            )
            turns = []
            call_latency = 0.0
            input_tokens = 0
            output_tokens = 0
            did_compose = False
            for turn in run.turns:
                if turn.response_kind not in {"grounded_answer", "uncertain_answer"} or not turn.claims:
                    turns.append(turn)
                    continue
                allowed = list(
                    dict.fromkeys(
                        [
                            *turn.candidate_ranking,
                            *turn.recommended_business_ids,
                            *(item.business_id for item in turn.claims if item.business_id),
                        ]
                    )
                )
                evidence = [
                    AnswerEvidenceItem(evidence_code=f"E{i}", claim=claim)
                    for i, claim in enumerate(turn.claims[: config.answer.maximum_evidence_items], 1)
                ]
                composed = runtime.answer_composer.compose(
                    AnswerCompositionInput(
                        context_id=run.scenario_id,
                        turn_index=turn.turn_index,
                        query_text=query_by_turn.get(turn.turn_index, scenario.query_text),
                        language=scenario.language,
                        task_type=turn.predicted_task_type,
                        response_kind=turn.response_kind,
                        allowed_business_ids=allowed,
                        evidence=evidence,
                        reported_conflict=turn.reported_conflict,
                        reported_evidence_recency=turn.reported_evidence_recency,
                        recommended_official_verification=(
                            turn.recommended_official_verification
                        ),
                    )
                )
                turns.append(turn.model_copy(update={"claims": composed.claims}))
                call_latency += composed.trace.latency_ms
                input_tokens += composed.trace.input_tokens or 0
                output_tokens += composed.trace.output_tokens or 0
                did_compose = True
            if did_compose:
                changed.append(
                    run.model_copy(
                        update={
                            "agent_version": config.agent_version,
                            "turns": turns,
                            "latency_ms": run.latency_ms + call_latency,
                            "input_tokens": (run.input_tokens or 0) + input_tokens,
                            "output_tokens": (run.output_tokens or 0) + output_tokens,
                        }
                    )
                )
            if index == 1 or index % 50 == 0 or index == len(base_runs):
                print(f"[{index}/{len(base_runs)}] replayed", flush=True)
        replacement.mkdir(parents=True, exist_ok=True)
        write_agent_scenario_runs(changed, replacement / "scenario_runs.jsonl")
        _write_jsonl(
            replacement / "driver_results.jsonl",
            [driver_rows[item.scenario_id] for item in changed],
        )
        result = replace_rule_agent_benchmark_outputs(
            base_root=base_root,
            replacement_root=replacement,
            benchmark_root=sources.benchmark_root,
            output_root=output,
        )
        calls_path, usage_path = runtime.ledger.write(output)
        augment_runtime_metrics(result.runtime_metrics_path, runtime.ledger.summary())
    print(f"changed_scenarios={len(changed)}")
    print(f"metrics={result.metrics_path}")
    print(f"calls={calls_path}")
    print(f"usage={usage_path}")


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "" if not rows else "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
