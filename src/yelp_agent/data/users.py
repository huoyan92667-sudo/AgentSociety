"""Select eligible users and freeze their interaction histories."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field

from yelp_agent.config import DataConfig
from yelp_agent.data.reviews import REVIEW_SCHEMA
from yelp_agent.models import StrictModel


USER_SCHEMA = pa.schema(
    [
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        pa.field("yelping_since", pa.timestamp("us"), nullable=False),
        pa.field("interaction_count", pa.int64(), nullable=False),
        pa.field("distinct_business_count", pa.int64(), nullable=False),
        pa.field("distinct_rating_count", pa.int64(), nullable=False),
    ]
)


class UserPreprocessError(RuntimeError):
    """Raised when user selection cannot produce a consistent frozen dataset."""


class UserPreprocessResult(StrictModel):
    status: Literal["written", "skipped"]
    reviews_path: str
    raw_users_path: str
    users_output_path: str
    interactions_output_path: str
    source_users: int | None = Field(default=None, ge=0)
    eligible_users: int | None = Field(default=None, ge=0)
    selected_users: int = Field(ge=0)
    selected_interactions: int = Field(ge=0)


def _require_source(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _aggregate_eligible_users(
    reviews_path: Path,
    config: DataConfig,
) -> list[dict[str, object]]:
    try:
        with duckdb.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    user_id,
                    count(*)::BIGINT AS interaction_count,
                    count(DISTINCT business_id)::BIGINT
                        AS distinct_business_count,
                    count(DISTINCT stars)::BIGINT AS distinct_rating_count
                FROM read_parquet(?)
                GROUP BY user_id
                HAVING
                    count(*) >= ?
                    AND count(DISTINCT business_id) >= ?
                    AND count(DISTINCT stars) >= ?
                ORDER BY user_id
                """,
                [
                    str(reviews_path),
                    config.min_user_reviews,
                    config.min_distinct_businesses,
                    config.min_distinct_ratings,
                ],
            ).fetchall()
    except duckdb.Error as exc:
        raise UserPreprocessError(
            f"Could not aggregate eligible users from {reviews_path}: {exc}"
        ) from exc

    return [
        {
            "user_id": row[0],
            "interaction_count": row[1],
            "distinct_business_count": row[2],
            "distinct_rating_count": row[3],
        }
        for row in rows
    ]


def _validate_existing_outputs(
    users_path: Path,
    interactions_path: Path,
) -> tuple[int, int]:
    if not users_path.is_file() or not interactions_path.is_file():
        raise UserPreprocessError(
            "User preprocessing outputs are incomplete; both users.parquet and "
            "interactions.parquet must exist or be rebuilt with force=True"
        )
    try:
        users_file = pq.ParquetFile(users_path)
        interactions_file = pq.ParquetFile(interactions_path)
    except (OSError, pa.ArrowException) as exc:
        raise UserPreprocessError(
            "Existing user preprocessing output is unreadable; "
            "use force=True to rebuild both"
        ) from exc
    if not users_file.schema_arrow.equals(USER_SCHEMA, check_metadata=False):
        raise UserPreprocessError(
            f"Existing user Parquet has an unexpected schema: {users_path}"
        )
    if not interactions_file.schema_arrow.equals(
        REVIEW_SCHEMA, check_metadata=False
    ):
        raise UserPreprocessError(
            "Existing interaction Parquet has an unexpected schema: "
            f"{interactions_path}"
        )

    try:
        with duckdb.connect() as connection:
            mismatch_count = connection.execute(
                """
                WITH interaction_counts AS (
                    SELECT user_id, count(*)::BIGINT AS interaction_count
                    FROM read_parquet(?)
                    GROUP BY user_id
                )
                SELECT count(*)
                FROM read_parquet(?) AS users
                FULL OUTER JOIN interaction_counts USING (user_id)
                WHERE
                    users.user_id IS NULL
                    OR interaction_counts.user_id IS NULL
                    OR users.interaction_count
                        != interaction_counts.interaction_count
                """,
                [str(interactions_path), str(users_path)],
            ).fetchone()[0]
            duplicate_user_count = connection.execute(
                """
                SELECT count(*) - count(DISTINCT user_id)
                FROM read_parquet(?)
                """,
                [str(users_path)],
            ).fetchone()[0]
    except duckdb.Error as exc:
        raise UserPreprocessError(
            f"Could not validate existing user preprocessing outputs: {exc}"
        ) from exc
    if mismatch_count or duplicate_user_count:
        raise UserPreprocessError(
            "Existing users and interactions Parquet files are inconsistent; "
            "use force=True to rebuild both"
        )
    return users_file.metadata.num_rows, interactions_file.metadata.num_rows


def _sample_users(
    eligible: list[dict[str, object]],
    *,
    seed: int,
    limit: int,
) -> list[dict[str, object]]:
    if limit <= 0:
        raise ValueError("max_users must be greater than zero")

    def sampling_key(row: dict[str, object]) -> tuple[str, str]:
        user_id = str(row["user_id"])
        digest = hashlib.sha256(f"{seed}:{user_id}".encode("utf-8")).hexdigest()
        return digest, user_id

    return sorted(eligible, key=sampling_key)[:limit]


def _parse_yelping_since(value: object, *, line_number: int) -> datetime:
    if not isinstance(value, str):
        raise UserPreprocessError(
            f"Invalid 'yelping_since' for selected user at JSONL line {line_number}"
        )
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise UserPreprocessError(
            f"Invalid 'yelping_since' for selected user at JSONL line {line_number}"
        ) from exc


def _load_selected_user_rows(
    raw_users_path: Path,
    selected: list[dict[str, object]],
) -> tuple[list[dict[str, object]], int]:
    aggregates = {str(row["user_id"]): row for row in selected}
    selected_ids = set(aggregates)
    found: dict[str, dict[str, object]] = {}
    source_users = 0

    with raw_users_path.open(
        "r", encoding="utf-8", buffering=1024 * 1024
    ) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            source_users += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise UserPreprocessError(
                    f"Invalid JSON at user JSONL line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise UserPreprocessError(
                    f"Expected a JSON object at user JSONL line {line_number}"
                )
            user_id = record.get("user_id")
            if user_id not in selected_ids:
                continue
            if user_id in found:
                raise UserPreprocessError(
                    f"Duplicate selected user_id {user_id!r} "
                    f"at user JSONL line {line_number}"
                )
            name = record.get("name", "")
            if not isinstance(name, str):
                raise UserPreprocessError(
                    f"Invalid 'name' for selected user at JSONL line {line_number}"
                )
            aggregate = aggregates[str(user_id)]
            found[str(user_id)] = {
                "user_id": user_id,
                "name": name,
                "yelping_since": _parse_yelping_since(
                    record.get("yelping_since"),
                    line_number=line_number,
                ),
                "interaction_count": aggregate["interaction_count"],
                "distinct_business_count": aggregate["distinct_business_count"],
                "distinct_rating_count": aggregate["distinct_rating_count"],
            }

    missing = sorted(selected_ids.difference(found))
    if missing:
        preview = ", ".join(repr(user_id) for user_id in missing[:5])
        raise UserPreprocessError(
            f"Raw user JSONL is missing {len(missing)} selected user(s): {preview}"
        )
    return [found[user_id] for user_id in sorted(found)], source_users


def _load_selected_interactions(
    reviews_path: Path,
    selected_user_ids: list[str],
) -> pa.Table:
    try:
        with duckdb.connect() as connection:
            table = connection.execute(
                """
                SELECT
                    review_id,
                    user_id,
                    business_id,
                    stars,
                    useful,
                    funny,
                    cool,
                    text,
                    date
                FROM read_parquet(?)
                WHERE user_id IN (SELECT unnest(?))
                ORDER BY user_id, date, review_id
                """,
                [str(reviews_path), selected_user_ids],
            ).to_arrow_table()
    except duckdb.Error as exc:
        raise UserPreprocessError(
            f"Could not select interactions from {reviews_path}: {exc}"
        ) from exc
    try:
        return table.cast(REVIEW_SCHEMA)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
        raise UserPreprocessError(
            f"Selected interactions do not match the required schema: {exc}"
        ) from exc


def preprocess_users_and_interactions(
    reviews_path: str | Path,
    raw_users_path: str | Path,
    users_output_path: str | Path,
    interactions_output_path: str | Path,
    config: DataConfig,
    *,
    force: bool = False,
) -> UserPreprocessResult:
    """Select eligible users and atomically freeze users and interactions."""

    reviews_source = Path(reviews_path)
    raw_users_source = Path(raw_users_path)
    users_destination = Path(users_output_path)
    interactions_destination = Path(interactions_output_path)
    _require_source(reviews_source, "Review Parquet")
    _require_source(raw_users_source, "Raw user JSONL")

    if (users_destination.exists() or interactions_destination.exists()) and not force:
        selected_users, selected_interactions = _validate_existing_outputs(
            users_destination,
            interactions_destination,
        )
        return UserPreprocessResult(
            status="skipped",
            reviews_path=str(reviews_source),
            raw_users_path=str(raw_users_source),
            users_output_path=str(users_destination),
            interactions_output_path=str(interactions_destination),
            source_users=None,
            eligible_users=None,
            selected_users=selected_users,
            selected_interactions=selected_interactions,
        )

    eligible = _aggregate_eligible_users(reviews_source, config)
    selected = _sample_users(
        eligible,
        seed=config.random_seed,
        limit=config.max_users,
    )
    if not selected:
        raise UserPreprocessError(
            "No users satisfy the configured interaction thresholds"
        )

    user_rows, source_users = _load_selected_user_rows(raw_users_source, selected)
    selected_user_ids = [str(row["user_id"]) for row in selected]
    interactions = _load_selected_interactions(
        reviews_source,
        selected_user_ids,
    )
    expected_interactions = sum(int(row["interaction_count"]) for row in selected)
    if interactions.num_rows != expected_interactions:
        raise UserPreprocessError(
            "Selected interaction count does not match the user aggregates"
        )

    users_destination.parent.mkdir(parents=True, exist_ok=True)
    interactions_destination.parent.mkdir(parents=True, exist_ok=True)
    users_partial = users_destination.with_name(users_destination.name + ".partial")
    interactions_partial = interactions_destination.with_name(
        interactions_destination.name + ".partial"
    )
    users_partial.unlink(missing_ok=True)
    interactions_partial.unlink(missing_ok=True)

    try:
        pq.write_table(
            pa.Table.from_pylist(user_rows, schema=USER_SCHEMA),
            users_partial,
            compression="zstd",
        )
        pq.write_table(
            interactions,
            interactions_partial,
            compression="zstd",
            use_dictionary=["user_id", "business_id"],
            row_group_size=config.review_chunk_size,
        )
        os.replace(interactions_partial, interactions_destination)
        os.replace(users_partial, users_destination)
    except Exception:
        users_partial.unlink(missing_ok=True)
        interactions_partial.unlink(missing_ok=True)
        raise

    return UserPreprocessResult(
        status="written",
        reviews_path=str(reviews_source),
        raw_users_path=str(raw_users_source),
        users_output_path=str(users_destination),
        interactions_output_path=str(interactions_destination),
        source_users=source_users,
        eligible_users=len(eligible),
        selected_users=len(selected),
        selected_interactions=interactions.num_rows,
    )


def write_user_preprocess_report(
    result: UserPreprocessResult,
    output_path: str | Path,
) -> None:
    """Write a JSON summary without embedding raw user records."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        result.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
