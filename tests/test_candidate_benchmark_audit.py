from __future__ import annotations

from pathlib import Path

import pandas as pd

from synthetic_pipeline import run_synthetic_pipeline
from yelp_agent.config import load_config
from yelp_agent.evaluation.candidate_audit import (
    audit_20_candidate_benchmark,
    write_candidate_audit,
)


def _audit_pipeline(root: Path):
    return audit_20_candidate_benchmark(
        validation_tasks_path=(
            root / "task_dataset" / "tasks" / "validation_tasks.jsonl"
        ),
        test_tasks_path=root / "task_dataset" / "tasks" / "test_tasks.jsonl",
        ground_truth_path=(
            root
            / "task_dataset"
            / "ground_truth"
            / "candidate_ground_truth.parquet"
        ),
        provenance_path=(
            root
            / "task_dataset"
            / "audit"
            / "candidate_provenance.parquet"
        ),
        dropped_tasks_path=(
            root / "task_dataset" / "audit" / "dropped_tasks.parquet"
        ),
        businesses_path=root / "processed" / "businesses.parquet",
        reviews_path=root / "processed" / "reviews.parquet",
        interactions_path=root / "processed" / "interactions.parquet",
        histories_path=(
            root
            / "task_dataset"
            / "tasks"
            / "temporal_histories.parquet"
        ),
        config=load_config().data,
    )


def test_audits_the_complete_twenty_candidate_protocol(tmp_path: Path) -> None:
    pipeline = run_synthetic_pipeline(tmp_path / "pipeline")

    report = _audit_pipeline(pipeline.root)

    assert report.benchmark_name == "Current 20-Candidate Controlled Reranking"
    assert report.total_task_count == 2
    assert report.required_candidate_count == 20
    assert report.required_negative_count == 19
    assert report.splits["validation"].task_count == 1
    assert report.splits["legacy_test"].task_count == 1
    assert report.total_bucket_counts == {
        "preference": 6,
        "random": 6,
        "refill_random": 2,
        "refill_same_fine": 8,
        "same_fine": 16,
        "target": 2,
    }
    assert report.tasks_using_refill == 2
    assert report.invariant_violation_total == 0
    assert not any(report.invariant_violations.values())
    assert report.protocol_properties.target_conditioned_candidate_generation
    assert not report.protocol_properties.target_label_visible_to_ranker
    assert sum(report.target_position_counts.values()) == 2


def test_candidate_audit_is_byte_stable_and_reports_bucket_mislabeling(
    tmp_path: Path,
) -> None:
    pipeline = run_synthetic_pipeline(tmp_path / "pipeline")
    clean = _audit_pipeline(pipeline.root)
    output = tmp_path / "current_20_candidate_audit.json"

    write_candidate_audit(clean, output)
    first_bytes = output.read_bytes()
    write_candidate_audit(clean, output)
    assert output.read_bytes() == first_bytes

    provenance_path = (
        pipeline.root
        / "task_dataset"
        / "audit"
        / "candidate_provenance.parquet"
    )
    provenance = pd.read_parquet(provenance_path)
    row_index = provenance.index[provenance["source_bucket"] == "same_fine"][0]
    provenance.loc[row_index, "source_bucket"] = "related"
    provenance.to_parquet(provenance_path, index=False)

    corrupted = _audit_pipeline(pipeline.root)

    assert corrupted.invariant_violation_total > 0
    assert (
        corrupted.invariant_violations["related_bucket_rule_row_violations"]
        == 1
    )
