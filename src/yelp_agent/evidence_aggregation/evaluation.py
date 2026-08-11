"""Read-only evaluation for a Development-frozen evidence policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from yelp_agent.review_rag import ReviewRetriever

from .aggregator import EvidenceAggregator
from .tuning import _aggregate, _load_cases, _score


def evaluate_frozen_evidence_aggregator(
    aggregator: EvidenceAggregator,
    retriever: ReviewRetriever,
    *,
    benchmark_root: str | Path,
    split: Literal["development", "validation"],
    output_root: str | Path,
) -> dict[str, object]:
    cases = _load_cases(
        benchmark_root=benchmark_root,
        split=split,
        retriever=retriever,
    )
    rows = [_score(case, aggregator.aggregate(case.request)) for case in cases]
    report: dict[str, object] = {
        "schema_version": 1,
        "split": split,
        "scenario_count": len(rows),
        "policy": aggregator.policy.model_dump(mode="json"),
        "policy_frozen_before_evaluation": True,
        "validation_used_for_tuning": False,
        "metrics": _aggregate(rows),
        "external_api_calls": 0,
        "billed_tokens": 0,
        "cost_cny": 0.0,
    }
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{split}_aggregation_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output / f"{split}_aggregation_runs.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    return report
