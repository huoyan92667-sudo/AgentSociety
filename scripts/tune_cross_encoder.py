"""Tune Step 26 prefix size and fusion beta on development labels only."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from yelp_agent.agent_benchmark import load_scenario_ground_truth, load_visible_scenarios
from yelp_agent.cross_encoder import (
    CachedCrossEncoderReranker,
    CrossEncoderPolicy,
    LocalQwenCrossEncoder,
    SqliteCrossEncoderCache,
    fuse_ranking_and_cross_encoder,
    load_cross_encoder_config,
)
from yelp_agent.data.temporal_view import BusinessRecord, TemporalDataView


class MappingBusinessReader:
    def __init__(self, records: dict[str, BusinessRecord]) -> None:
        self._records = records

    def business(self, business_id: str) -> BusinessRecord:
        return self._records[business_id]


def _read_runs(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                result[str(row["scenario_id"])] = row
    return result


def _final_ranking_turn(run: dict[str, Any]) -> dict[str, Any]:
    turns = run.get("turns") or []
    for turn in reversed(turns):
        if turn.get("candidate_ranking"):
            return turn
    return {}


def _metrics(rows: list[tuple[list[str], set[str]]]) -> dict[str, float]:
    count = len(rows)
    reciprocal = 0.0
    hits = {1: 0, 3: 0, 5: 0}
    for ranking, acceptable in rows:
        rank = next(
            (index for index, value in enumerate(ranking, 1) if value in acceptable),
            None,
        )
        if rank is not None:
            reciprocal += 1.0 / rank
            for cutoff in hits:
                hits[cutoff] += int(rank <= cutoff)
    return {
        "hr_at_1": hits[1] / count if count else 0.0,
        "hr_at_3": hits[3] / count if count else 0.0,
        "hr_at_5": hits[5] / count if count else 0.0,
        "mrr": reciprocal / count if count else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--step25-runs",
        type=Path,
        default=Path("runs/rule_agent_embedding_v2/development/scenario_runs.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/cross_encoder_v1/development_tuning.json"),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = load_cross_encoder_config(root / "configs" / "cross_encoder.yaml")
    runs = _read_runs(root / args.step25_runs)
    benchmark = root / "benchmarks" / "agent_scenarios_v1"
    visible = {
        item.scenario_id: item
        for item in load_visible_scenarios(benchmark / "visible" / "scenarios.jsonl")
        if item.split == "development"
    }
    truth = {
        item.scenario_id: item
        for item in load_scenario_ground_truth(benchmark / "hidden" / "ground_truth.jsonl")
        if item.scenario_id in visible
    }
    if not set(runs).issubset(visible):
        raise ValueError("Step-25 development runs contain non-development scenarios")
    businesses = TemporalDataView._load_businesses(  # noqa: SLF001 - offline CLI
        root / "data" / "processed" / "businesses.parquet"
    )
    environment = {
        "LOCAL_CROSS_ENCODER_MODEL_PATH": str(args.model_path),
        "LOCAL_CROSS_ENCODER_PYTHON": str(args.python_executable),
        "LOCAL_CROSS_ENCODER_DEVICE": args.device,
    }
    from yelp_agent.cross_encoder import load_local_cross_encoder_environment

    scorer = LocalQwenCrossEncoder.from_environment(
        config, load_local_cross_encoder_environment(environment)
    )
    cache = SqliteCrossEncoderCache(root / config.cache_relative_path)
    reranker = CachedCrossEncoderReranker(
        businesses=MappingBusinessReader(businesses),
        scorer=scorer,
        cache=cache,
        config=config,
    )
    cases: list[dict[str, Any]] = []
    try:
        ordered = sorted(runs)
        for index, scenario_id in enumerate(ordered, 1):
            run = runs[scenario_id]
            turn = _final_ranking_turn(run)
            ranking = [str(value) for value in turn.get("candidate_ranking") or []]
            acceptable = set(truth[scenario_id].acceptable_business_ids)
            if not ranking or not acceptable:
                continue
            turn_index = int(turn.get("turn_index") or 1)
            query_text = visible[scenario_id].query_text
            if turn_index > 1:
                scripted = truth[scenario_id].scripted_user_turns
                if turn_index - 2 < len(scripted):
                    query_text = scripted[turn_index - 2].query_text
            prefix = ranking[:20]
            result = reranker.rerank(
                query_text=query_text,
                business_ids=prefix,
                cutoff_time=visible[scenario_id].cutoff_time,
                usage_scope=f"tune:{scenario_id}",
            )
            scores = {item.business_id: item.relevance_score for item in result.matches}
            cases.append(
                {
                    "scenario_id": scenario_id,
                    "ranking": ranking,
                    "acceptable": acceptable,
                    "scores": scores,
                }
            )
            if index == 1 or index % 25 == 0 or index == len(ordered):
                print(f"[{index}/{len(ordered)}] development scenarios inspected", flush=True)
    finally:
        scorer.close()

    grid: list[dict[str, Any]] = []
    for candidate_limit in (5, 10, 20):
        for beta_tenth in range(11):
            beta = beta_tenth / 10.0
            evaluated = []
            for case in cases:
                prefix = case["ranking"][:candidate_limit]
                score_order = sorted(
                    prefix,
                    key=lambda business_id: (-case["scores"][business_id], business_id),
                )
                ranks = {
                    business_id: rank
                    for rank, business_id in enumerate(score_order, 1)
                }
                fused = fuse_ranking_and_cross_encoder(
                    case["ranking"], ranks, beta=beta
                )
                evaluated.append((fused, case["acceptable"]))
            grid.append(
                {
                    "candidate_limit": candidate_limit,
                    "fusion_beta": beta,
                    **_metrics(evaluated),
                }
            )
    grid.sort(
        key=lambda row: (
            -row["hr_at_5"],
            -row["mrr"],
            -row["hr_at_1"],
            row["candidate_limit"],
            row["fusion_beta"],
        )
    )
    selected = grid[0]
    policy = CrossEncoderPolicy(
        candidate_limit=int(selected["candidate_limit"]),
        fusion_beta=float(selected["fusion_beta"]),
        display_limit=5,
        development_scenario_count=len(cases),
        development_metrics={
            key: float(selected[key])
            for key in ("hr_at_1", "hr_at_3", "hr_at_5", "mrr")
        },
    )
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "split_used_for_selection": "development",
        "validation_labels_read": False,
        "ranking_case_count": len(cases),
        "model": scorer.model,
        "external_api_calls": 0,
        "selected_policy": policy.model_dump(),
        "grid": grid,
    }
    partial = output.with_name(output.name + ".partial")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    partial.replace(output)
    policy_path = root / config.policy_relative_path
    policy_partial = policy_path.with_name(policy_path.name + ".partial")
    policy_partial.write_text(
        json.dumps(policy.model_dump(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    policy_partial.replace(policy_path)
    print(json.dumps(policy.model_dump(), ensure_ascii=False, sort_keys=True))
    print(f"tuning={output}")
    print(f"frozen_policy={policy_path}")


if __name__ == "__main__":
    main()
