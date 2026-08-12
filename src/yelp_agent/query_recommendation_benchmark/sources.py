"""Read only the temporal fields required to establish real positive anchors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb

from .selection import BehaviorAnchorCandidate


@dataclass(frozen=True, slots=True)
class QueryRecommendationSources:
    reviews: Path
    training_contexts: Path
    training_histories: Path
    training_ground_truth: Path
    validation_contexts: Path
    validation_histories: Path
    validation_ground_truth: Path

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> QueryRecommendationSources:
        root = Path(project_root).resolve()
        tasks = root / "data" / "task_dataset"
        return cls(
            reviews=root / "data" / "processed" / "reviews.parquet",
            training_contexts=tasks / "training" / "rolling_train_contexts.parquet",
            training_histories=tasks / "training" / "rolling_train_histories.parquet",
            training_ground_truth=(
                tasks / "ground_truth" / "rolling_train_ground_truth.parquet"
            ),
            validation_contexts=tasks / "tasks" / "temporal_contexts.parquet",
            validation_histories=tasks / "tasks" / "temporal_histories.parquet",
            validation_ground_truth=tasks / "ground_truth" / "ground_truth.parquet",
        )

    def validate(self) -> None:
        missing = [path for path in self.paths() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Query recommendation sources are incomplete:\n"
                + "\n".join(f"- {path}" for path in missing)
            )

    def paths(self) -> tuple[Path, ...]:
        return (
            self.reviews,
            self.training_contexts,
            self.training_histories,
            self.training_ground_truth,
            self.validation_contexts,
            self.validation_histories,
            self.validation_ground_truth,
        )


def _load_split(
    connection: duckdb.DuckDBPyConnection,
    *,
    reviews: Path,
    contexts: Path,
    histories: Path,
    truth: Path,
    source_split: str,
) -> list[BehaviorAnchorCandidate]:
    target_join = (
        "JOIN read_parquet(?) target ON target.review_id = truth.target_review_id"
        if source_split == "train"
        else "JOIN read_parquet(?) target ON target.user_id = context.user_id "
        "AND target.business_id = truth.target_business_id "
        "AND target.date = context.cutoff_time"
    )
    split_filter = "" if source_split == "train" else "WHERE context.split = 'validation'"
    query = f"""
        WITH history_rows AS (
            SELECT
                history.task_id,
                list(review.business_id ORDER BY history.position) AS history_business_ids
            FROM read_parquet(?) history
            JOIN read_parquet(?) review USING (review_id)
            GROUP BY history.task_id
        ), source AS (
            SELECT
                context.task_id AS source_task_id,
                context.user_id,
                context.cutoff_time,
                history_rows.history_business_ids,
                truth.target_business_id,
                target.review_id AS target_review_id,
                target.stars AS target_stars,
                target.date AS target_time,
                (
                    SELECT count(*)
                    FROM read_parquet(?) previous
                    WHERE previous.business_id = truth.target_business_id
                      AND previous.date < context.cutoff_time
                ) AS target_pre_cutoff_review_count
            FROM read_parquet(?) context
            JOIN read_parquet(?) truth USING (task_id)
            JOIN history_rows USING (task_id)
            {target_join}
            {split_filter}
        )
        SELECT * FROM source ORDER BY source_task_id
    """
    parameters = [
        str(histories),
        str(reviews),
        str(reviews),
        str(contexts),
        str(truth),
        str(reviews),
    ]
    rows = connection.execute(query, parameters).fetchall()
    return [
        BehaviorAnchorCandidate(
            source_task_id=str(row[0]),
            source_split=source_split,
            user_id=str(row[1]),
            cutoff_time=row[2],
            history_business_ids=tuple(str(value) for value in row[3]),
            target_business_id=str(row[4]),
            target_review_id=str(row[5]),
            target_stars=float(row[6]),
            target_time=row[7],
            target_pre_cutoff_review_count=int(row[8]),
        )
        for row in rows
    ]


def load_behavior_anchor_candidates(
    sources: QueryRecommendationSources,
) -> tuple[BehaviorAnchorCandidate, ...]:
    """Load train and validation anchors without selecting review text."""

    sources.validate()
    connection = duckdb.connect()
    try:
        values = _load_split(
            connection,
            reviews=sources.reviews,
            contexts=sources.training_contexts,
            histories=sources.training_histories,
            truth=sources.training_ground_truth,
            source_split="train",
        )
        values.extend(
            _load_split(
                connection,
                reviews=sources.reviews,
                contexts=sources.validation_contexts,
                histories=sources.validation_histories,
                truth=sources.validation_ground_truth,
                source_split="validation",
            )
        )
    finally:
        connection.close()
    return tuple(sorted(values, key=lambda item: (item.source_split, item.source_task_id)))
