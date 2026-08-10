"""Run one real Yelp Agent scenario through the complete Step 25 chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yelp_agent.agent_benchmark import load_visible_scenarios
from yelp_agent.rule_router import RuleAgentSourcePaths, build_real_rule_agent_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--scenario-id",
        default="00292aa2559fe6f1e14a7f6ad0bfa7d66ed828da27ba9b8e6d0aa6aa07ad8d60",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    sources = RuleAgentSourcePaths.from_project_root(root)
    scenarios = load_visible_scenarios(
        sources.benchmark_root / "visible" / "scenarios.jsonl"
    )
    try:
        scenario = next(item for item in scenarios if item.scenario_id == args.scenario_id)
    except StopIteration:
        raise SystemExit(f"unknown scenario ID: {args.scenario_id}") from None
    with build_real_rule_agent_runtime(
        sources,
        embedding_config_path=root / "configs" / "embedding.yaml",
    ) as runtime:
        result = runtime.harness.start(scenario)
    run = result.run
    payload = {
        "scenario_id": scenario.scenario_id,
        "agent_version": result.session.agent_version,
        "status": result.session.status,
        "fallback_reason": result.session.fallback_reason,
        "semantic_call_count": result.session.semantic_call_count,
        "input_tokens": result.session.input_tokens,
        "cost_usd": result.session.cost_usd,
        "tool_names": (
            []
            if run is None
            else [call.tool_name for turn in run.turns for call in turn.tool_calls]
        ),
        "recommended_business_ids": (
            [] if run is None else run.turns[-1].recommended_business_ids
        ),
        "candidate_count": (
            0 if run is None else len(run.turns[-1].candidate_ranking)
        ),
    }
    output = args.output or root / "runs" / "semantic_embedding_v2" / "smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    partial.replace(output)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    print(f"output={output}")


if __name__ == "__main__":
    main()
