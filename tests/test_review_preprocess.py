import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from yelp_agent.config import load_config
from yelp_agent.data.reviews import ReviewPreprocessError, preprocess_reviews


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def review(
    review_id: str,
    business_id: str,
    *,
    user_id: str = "user-1",
    stars: float = 4,
    date: str = "2020-01-01 12:00:00",
) -> dict:
    return {
        "review_id": review_id,
        "user_id": user_id,
        "business_id": business_id,
        "stars": stars,
        "useful": 1,
        "funny": 2,
        "cool": 3,
        "text": f"Review {review_id}",
        "date": date,
    }


def test_streams_only_target_business_reviews_to_typed_parquet(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "review.json"
    businesses_path = tmp_path / "businesses.parquet"
    output_path = tmp_path / "processed" / "reviews.parquet"
    pd.DataFrame({"business_id": ["business-1", "business-2"]}).to_parquet(
        businesses_path,
        index=False,
    )
    rows = [
        review("review-1", "business-1"),
        review("review-2", "outside"),
        review(
            "review-3",
            "business-2",
            user_id="user-2",
            stars=2,
            date="2021-02-03 04:05:06",
        ),
    ]
    write_jsonl(input_path, rows)
    config = load_config().data.model_copy(update={"review_chunk_size": 1})

    result = preprocess_reviews(
        input_path,
        businesses_path,
        output_path,
        config,
    )

    frame = pd.read_parquet(output_path)
    assert result.status == "written"
    assert result.source_rows == 3
    assert result.selected_rows == 2
    assert result.outside_target_businesses == 1
    assert set(frame["review_id"]) == {"review-1", "review-3"}
    assert list(frame.columns) == [
        "review_id",
        "user_id",
        "business_id",
        "stars",
        "useful",
        "funny",
        "cool",
        "text",
        "date",
    ]
    assert pd.api.types.is_datetime64_any_dtype(frame["date"])
    assert pq.ParquetFile(output_path).metadata.num_row_groups == 2


def test_invalid_json_removes_partial_output(tmp_path: Path) -> None:
    input_path = tmp_path / "review.json"
    businesses_path = tmp_path / "businesses.parquet"
    output_path = tmp_path / "reviews.parquet"
    pd.DataFrame({"business_id": ["business-1"]}).to_parquet(
        businesses_path,
        index=False,
    )
    input_path.write_text(
        json.dumps(review("review-1", "business-1")) + "\n{not-json}\n",
        encoding="utf-8",
    )

    with pytest.raises(ReviewPreprocessError, match=r"line 2"):
        preprocess_reviews(
            input_path,
            businesses_path,
            output_path,
            load_config().data,
        )

    assert not output_path.exists()
    assert not output_path.with_name("reviews.parquet.partial").exists()


def test_duplicate_selected_review_id_aborts_atomically(tmp_path: Path) -> None:
    input_path = tmp_path / "review.json"
    businesses_path = tmp_path / "businesses.parquet"
    output_path = tmp_path / "reviews.parquet"
    pd.DataFrame({"business_id": ["business-1"]}).to_parquet(
        businesses_path,
        index=False,
    )
    write_jsonl(
        input_path,
        [
            review("duplicate", "business-1"),
            review("duplicate", "business-1", user_id="user-2"),
        ],
    )

    with pytest.raises(ReviewPreprocessError, match="Duplicate selected review_id"):
        preprocess_reviews(
            input_path,
            businesses_path,
            output_path,
            load_config().data,
        )

    assert not output_path.exists()
    assert not output_path.with_name("reviews.parquet.partial").exists()


def test_existing_valid_output_is_reused_without_rescanning(tmp_path: Path) -> None:
    input_path = tmp_path / "review.json"
    businesses_path = tmp_path / "businesses.parquet"
    output_path = tmp_path / "reviews.parquet"
    pd.DataFrame({"business_id": ["business-1"]}).to_parquet(
        businesses_path,
        index=False,
    )
    write_jsonl(input_path, [review("review-1", "business-1")])
    config = load_config().data
    preprocess_reviews(input_path, businesses_path, output_path, config)
    initial_mtime = output_path.stat().st_mtime_ns
    input_path.write_text("{now-invalid-json}\n", encoding="utf-8")

    result = preprocess_reviews(input_path, businesses_path, output_path, config)

    assert result.status == "skipped"
    assert result.source_rows is None
    assert result.selected_rows == 1
    assert result.outside_target_businesses is None
    assert output_path.stat().st_mtime_ns == initial_mtime


def test_fractional_vote_count_is_rejected(tmp_path: Path) -> None:
    input_path = tmp_path / "review.json"
    businesses_path = tmp_path / "businesses.parquet"
    output_path = tmp_path / "reviews.parquet"
    pd.DataFrame({"business_id": ["business-1"]}).to_parquet(
        businesses_path,
        index=False,
    )
    row = review("review-1", "business-1")
    row["useful"] = 1.5
    write_jsonl(input_path, [row])

    with pytest.raises(ReviewPreprocessError, match="'useful'"):
        preprocess_reviews(
            input_path,
            businesses_path,
            output_path,
            load_config().data,
        )
