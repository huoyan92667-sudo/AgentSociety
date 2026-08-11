"""Read-only split evaluation for a frozen Review Retriever policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from yelp_agent.agent_benchmark import (
    load_evidence_labels,
    load_scenario_ground_truth,
    load_visible_scenarios,
)
from yelp_agent.agent_harness import RuleBasedRequestInterpreter

from .retriever import ReviewRetriever
from .tuning import _aggregate, _request, _score_result


def evaluate_frozen_review_retriever(
    retriever: ReviewRetriever,
    *,
    benchmark_root: str | Path,
    split: Literal["development", "validation"],
    output_root: str | Path,
) -> dict[str, object]:
    """Evaluate one frozen policy without exposing labels to the retriever."""

    root = Path(benchmark_root)
    visible = tuple(
        item
        for item in load_visible_scenarios(root / "visible" / "scenarios.jsonl")
        if item.split == split
    )
    visible_ids = {item.scenario_id for item in visible}
    truth = {
        item.scenario_id: item
        for item in load_scenario_ground_truth(root / "hidden" / "ground_truth.jsonl")
        if item.scenario_id in visible_ids
        and "retrieve_business_reviews" in item.required_actions
    }
    labels = [
        item
        for item in load_evidence_labels(root / "hidden" / "evidence_labels.parquet")
        if item.scenario_id in truth
        and item.source_type == "review"
        and item.relevance == "relevant"
    ]
    relevant: dict[str, set[str]] = {}
    for label in labels:
        relevant.setdefault(label.scenario_id, set()).add(str(label.review_id))
    interpreter = RuleBasedRequestInterpreter()
    rows = []
    for scenario in visible:
        if scenario.scenario_id not in truth:
            continue
        result = retriever.search(_request(scenario, interpreter))
        rows.append(
            _score_result(
                scenario,
                truth[scenario.scenario_id],
                relevant[scenario.scenario_id],
                result,
            )
        )
    metrics = _aggregate(rows)
    report: dict[str, object] = {
        "schema_version": 1,
        "split": split,
        "scenario_count": len(rows),
        "policy_frozen_before_evaluation": True,
        "validation_used_for_tuning": False,
        "metrics": metrics,
        "external_api_calls": 0,
        "billed_tokens": 0,
        "cost_cny": 0.0,
    }
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{split}_retrieval_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output / f"{split}_retrieval_runs.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    return report
