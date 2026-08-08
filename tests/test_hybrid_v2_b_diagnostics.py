from __future__ import annotations

from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq

from yelp_agent.learning_to_rank.lambdamart_diagnostics import (
    LambdaMARTDiagnosticSources,
    build_lambdamart_validation_diagnostics,
)


def test_diagnostics_compare_history_and_category_segments(tmp_path) -> None:
    cutoff = datetime(2020, 1, 10)
    paths = {
        name: tmp_path / f"{name}.parquet"
        for name in (
            "logistic",
            "lambdamart",
            "contexts",
            "truth",
            "reviews",
            "interactions",
            "businesses",
        )
    }
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"task_id": "t1", "business_id": "target1", "rank": 2},
                {"task_id": "t2", "business_id": "target2", "rank": 4},
            ]
        ),
        paths["logistic"],
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"task_id": "t1", "business_id": "target1", "rank": 1},
                {"task_id": "t2", "business_id": "target2", "rank": 2},
            ]
        ),
        paths["lambdamart"],
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "task_id": "t1",
                    "split": "validation",
                    "user_id": "u1",
                    "cutoff_time": cutoff,
                    "history_count": 10,
                },
                {
                    "task_id": "t2",
                    "split": "validation",
                    "user_id": "u2",
                    "cutoff_time": cutoff,
                    "history_count": 65,
                },
            ]
        ),
        paths["contexts"],
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"task_id": "t1", "target_business_id": "target1"},
                {"task_id": "t2", "target_business_id": "target2"},
            ]
        ),
        paths["truth"],
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"business_id": "target1", "date": datetime(2019, 1, 1)},
                {"business_id": "target2", "date": datetime(2019, 1, 1)},
            ]
        ),
        paths["reviews"],
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "user_id": "u1",
                    "business_id": "history1",
                    "date": datetime(2019, 1, 1),
                },
                {
                    "user_id": "u2",
                    "business_id": "history2",
                    "date": datetime(2019, 1, 1),
                },
            ]
        ),
        paths["interactions"],
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"business_id": "history1", "categories": ["Steakhouses"]},
                {"business_id": "target1", "categories": ["Steakhouses"]},
                {"business_id": "history2", "categories": ["Coffee & Tea"]},
                {"business_id": "target2", "categories": ["Museums"]},
            ]
        ),
        paths["businesses"],
    )

    report = build_lambdamart_validation_diagnostics(
        LambdaMARTDiagnosticSources(
            logistic_predictions=paths["logistic"],
            lambdamart_predictions=paths["lambdamart"],
            validation_contexts=paths["contexts"],
            validation_ground_truth=paths["truth"],
            reviews=paths["reviews"],
            interactions=paths["interactions"],
            businesses=paths["businesses"],
        ),
        bootstrap_samples=100,
        random_seed=42,
    )

    segments = {(item.dimension, item.value): item for item in report.segments}
    assert segments[("history_count", "8-15")].avg_hr_delta > 0
    assert segments[("history_count", "61+")].avg_hr_delta > 0
    assert segments[("category_familiarity", "seen")].avg_hr_delta > 0
    assert segments[("category_familiarity", "unseen")].avg_hr_delta > 0
    assert report.paired_user_avg_hr.probability_positive > 0.5
