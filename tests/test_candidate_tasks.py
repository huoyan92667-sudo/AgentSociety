import json
from pathlib import Path

import pandas as pd

from yelp_agent.config import load_config
from yelp_agent.data.candidates import (
    CandidateBuildResult,
    build_candidate_tasks,
    write_candidate_build_report,
)
from yelp_agent.models import RecommendationTask


def business(business_id: str, categories: list[str]) -> dict:
    return {
        "business_id": business_id,
        "name": business_id,
        "address": "",
        "city": "Philadelphia",
        "state": "PA",
        "postal_code": "",
        "latitude": 0.0,
        "longitude": 0.0,
        "categories": categories,
        "attributes_json": "{}",
    }


def write_candidate_fixture(
    processed_root: Path,
    task_root: Path,
    businesses: list[dict],
    *,
    target_has_prior_review: bool = True,
) -> None:
    processed_root.mkdir()
    (task_root / "tasks").mkdir(parents=True)
    (task_root / "ground_truth").mkdir()
    pd.DataFrame(businesses).to_parquet(
        processed_root / "businesses.parquet",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "review_id": f"availability-{row['business_id']}",
                "business_id": row["business_id"],
                "stars": 4.0,
                "date": pd.Timestamp("2020-01-01"),
            }
            for row in businesses
            if target_has_prior_review or row["business_id"] != "target"
        ]
    ).to_parquet(processed_root / "reviews.parquet", index=False)
    pd.DataFrame(
        [
            {
                "review_id": "h1",
                "user_id": "user-1",
                "business_id": "history-preferred",
                "stars": 5.0,
                "date": pd.Timestamp("2020-01-02"),
            },
            {
                "review_id": "h2",
                "user_id": "user-1",
                "business_id": "history-low",
                "stars": 1.0,
                "date": pd.Timestamp("2020-01-03"),
            },
            {
                "review_id": "h3",
                "user_id": "user-1",
                "business_id": "history-other",
                "stars": 3.0,
                "date": pd.Timestamp("2020-01-04"),
            },
        ]
    ).to_parquet(processed_root / "interactions.parquet", index=False)
    pd.DataFrame(
        [
            {
                "task_id": "validation:user-1",
                "split": "validation",
                "user_id": "user-1",
                "cutoff_time": pd.Timestamp("2020-01-10"),
                "history_count": 3,
                "history_max_time": pd.Timestamp("2020-01-04"),
            }
        ]
    ).to_parquet(
        task_root / "tasks" / "temporal_contexts.parquet",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "task_id": "validation:user-1",
                "position": index,
                "review_id": review_id,
            }
            for index, review_id in enumerate(["h1", "h2", "h3"], start=1)
        ]
    ).to_parquet(
        task_root / "tasks" / "temporal_histories.parquet",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "task_id": "validation:user-1",
                "target_business_id": "target",
            }
        ]
    ).to_parquet(
        task_root / "ground_truth" / "ground_truth.parquet",
        index=False,
    )


def test_builds_twenty_candidates_from_the_four_negative_buckets(
    tmp_path: Path,
) -> None:
    processed_root = tmp_path / "processed"
    task_root = tmp_path / "task_dataset"
    businesses = [
        business("target", ["Nightlife", "Cocktail Bars"]),
        business("history-preferred", ["Restaurants", "Mexican"]),
        business("history-low", ["Restaurants", "Coffee & Tea"]),
        business("history-other", ["Beauty & Spas", "Hair Salons"]),
        *[
            business(f"same-{index}", ["Nightlife", "Cocktail Bars"])
            for index in range(8)
        ],
        *[
            business(f"related-{index}", ["Nightlife", "Dive Bars"])
            for index in range(5)
        ],
        *[
            business(f"preference-{index}", ["Restaurants", "Mexican"])
            for index in range(3)
        ],
        *[
            business(f"random-{index}", ["Shopping", "Books"])
            for index in range(3)
        ],
    ]
    write_candidate_fixture(processed_root, task_root, businesses)

    result = build_candidate_tasks(
        processed_root,
        task_root,
        load_config().data,
    )

    lines = Path(result.validation_tasks_path).read_text(
        encoding="utf-8"
    ).splitlines()
    task = RecommendationTask.model_validate_json(lines[0])
    provenance = pd.read_parquet(result.provenance_path)
    bucket_counts = provenance["source_bucket"].value_counts().to_dict()

    assert result.status == "written"
    assert result.input_tasks == 1
    assert result.retained_tasks == 1
    assert result.dropped_unavailable_targets == 0
    assert len(lines) == 1
    assert len(task.candidate_business_ids) == 20
    assert "target" in task.candidate_business_ids
    assert not {
        "history-preferred",
        "history-low",
        "history-other",
    }.intersection(task.candidate_business_ids)
    assert bucket_counts == {
        "same_fine": 8,
        "related": 5,
        "preference": 3,
        "random": 3,
        "target": 1,
    }
    assert Path(result.test_tasks_path).read_text(encoding="utf-8") == ""


def test_reuses_consistent_frozen_candidate_outputs(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    task_root = tmp_path / "task_dataset"
    businesses = [
        business("target", ["Nightlife", "Cocktail Bars"]),
        business("history-preferred", ["Restaurants", "Mexican"]),
        business("history-low", ["Restaurants", "Coffee & Tea"]),
        business("history-other", ["Beauty & Spas", "Hair Salons"]),
        *[
            business(f"negative-{index}", ["Shopping", "Books"])
            for index in range(19)
        ],
    ]
    write_candidate_fixture(processed_root, task_root, businesses)
    config = load_config().data
    first = build_candidate_tasks(processed_root, task_root, config)
    outputs = [
        Path(first.validation_tasks_path),
        Path(first.test_tasks_path),
        Path(first.ground_truth_path),
        Path(first.provenance_path),
        Path(first.dropped_tasks_path),
    ]
    mtimes = [path.stat().st_mtime_ns for path in outputs]

    second = build_candidate_tasks(processed_root, task_root, config)

    assert second.status == "skipped"
    assert second.input_tasks == 1
    assert second.retained_tasks == 1
    assert second.validation_tasks == 1
    assert second.test_tasks == 0
    assert second.tasks_using_refill == 1
    assert [path.stat().st_mtime_ns for path in outputs] == mtimes


def test_drops_target_that_did_not_exist_before_cutoff(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    task_root = tmp_path / "task_dataset"
    businesses = [
        business("target", ["Nightlife", "Cocktail Bars"]),
        business("history-preferred", ["Restaurants", "Mexican"]),
        business("history-low", ["Restaurants", "Coffee & Tea"]),
        business("history-other", ["Beauty & Spas", "Hair Salons"]),
        *[
            business(f"negative-{index}", ["Shopping", "Books"])
            for index in range(19)
        ],
    ]
    write_candidate_fixture(
        processed_root,
        task_root,
        businesses,
        target_has_prior_review=False,
    )

    result = build_candidate_tasks(
        processed_root,
        task_root,
        load_config().data,
    )

    dropped = pd.read_parquet(result.dropped_tasks_path)
    ground_truth = pd.read_parquet(result.ground_truth_path)
    assert result.input_tasks == 1
    assert result.retained_tasks == 0
    assert result.dropped_unavailable_targets == 1
    assert dropped.to_dict("records") == [
        {
            "task_id": "validation:user-1",
            "split": "validation",
            "reason": "target_unavailable_before_cutoff",
        }
    ]
    assert ground_truth.empty
    assert Path(result.validation_tasks_path).read_text(encoding="utf-8") == ""
    assert Path(result.test_tasks_path).read_text(encoding="utf-8") == ""


def test_future_reviews_cannot_change_existing_candidate_tasks(
    tmp_path: Path,
) -> None:
    processed_root = tmp_path / "processed"
    task_root = tmp_path / "task_dataset"
    businesses = [
        business("target", ["Nightlife", "Cocktail Bars"]),
        business("history-preferred", ["Restaurants", "Mexican"]),
        business("history-low", ["Restaurants", "Coffee & Tea"]),
        business("history-other", ["Beauty & Spas", "Hair Salons"]),
        *[
            business(f"same-{index}", ["Nightlife", "Cocktail Bars"])
            for index in range(8)
        ],
        *[
            business(f"related-{index}", ["Nightlife", "Dive Bars"])
            for index in range(5)
        ],
        *[
            business(f"preference-{index}", ["Restaurants", "Mexican"])
            for index in range(4)
        ],
        *[
            business(f"random-{index}", ["Shopping", "Books"])
            for index in range(3)
        ],
    ]
    write_candidate_fixture(processed_root, task_root, businesses)
    config = load_config().data
    first = build_candidate_tasks(processed_root, task_root, config)
    frozen_task = Path(first.validation_tasks_path).read_bytes()
    frozen_provenance = Path(first.provenance_path).read_bytes()

    reviews_path = processed_root / "reviews.parquet"
    reviews = pd.read_parquet(reviews_path)
    future_reviews = pd.DataFrame(
        [
            {
                "review_id": f"future-good-{index}",
                "business_id": "preference-3",
                "stars": 5.0,
                "date": pd.Timestamp("2030-01-01"),
            }
            for index in range(20)
        ]
        + [
            {
                "review_id": f"future-bad-{index}",
                "business_id": "preference-0",
                "stars": 1.0,
                "date": pd.Timestamp("2030-01-01"),
            }
            for index in range(20)
        ]
    )
    pd.concat([reviews, future_reviews], ignore_index=True).to_parquet(
        reviews_path,
        index=False,
    )

    second = build_candidate_tasks(
        processed_root,
        task_root,
        config,
        force=True,
    )

    assert Path(second.validation_tasks_path).read_bytes() == frozen_task
    assert Path(second.provenance_path).read_bytes() == frozen_provenance


def test_writes_candidate_build_audit_report(tmp_path: Path) -> None:
    report_path = tmp_path / "runs" / "candidate_build_report.json"
    result = CandidateBuildResult(
        status="written",
        validation_tasks_path="validation.jsonl",
        test_tasks_path="test.jsonl",
        ground_truth_path="ground_truth.parquet",
        provenance_path="provenance.parquet",
        dropped_tasks_path="dropped.parquet",
        input_tasks=10,
        retained_tasks=9,
        validation_tasks=4,
        test_tasks=5,
        dropped_unavailable_targets=1,
        tasks_using_refill=2,
        bucket_counts={"target": 9, "random": 27},
    )

    write_candidate_build_report(result, report_path)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["retained_tasks"] == 9
    assert payload["dropped_unavailable_targets"] == 1
    assert payload["tasks_using_refill"] == 2
