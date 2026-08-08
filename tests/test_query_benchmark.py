import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from yelp_agent.query import build_rule_based_request_parser
from yelp_agent.query.benchmark import (
    QueryBenchmarkCase,
    evaluate_request_parser,
    load_query_benchmark,
)


def test_benchmark_evaluates_structured_meaning_without_business_labels(
    tmp_path: Path,
) -> None:
    path = tmp_path / "queries.jsonl"
    rows = [
        {
            "case_id": "development-steak",
            "split": "development",
            "frame_family": "explicit-category-and-preference",
            "language": "zh-CN",
            "query_text": "想吃牛排，最好安静一点。",
            "expected_intent": "recommendation_request",
            "expected_conditions": [
                {
                    "field": "category",
                    "operator": "includes",
                    "value": "Steakhouses",
                    "importance": "strong",
                    "enforcement": "rank",
                },
                {
                    "field": "quiet_environment",
                    "operator": "prefer",
                    "value": True,
                    "importance": "preferred",
                    "enforcement": "rank",
                },
            ],
            "expected_party_size": None,
            "expected_missing_fields": [],
            "generator_kind": "human_seed",
            "generator_model": None,
            "generator_prompt_sha256": None,
            "uses_specific_business": False,
            "uses_future_review": False,
        },
        {
            "case_id": "validation-hard-category",
            "split": "validation",
            "frame_family": "exclusive-category-distance",
            "language": "zh-CN",
            "query_text": "只想吃日料，必须在3公里以内。",
            "expected_intent": "recommendation_request",
            "expected_conditions": [
                {
                    "field": "category",
                    "operator": "includes",
                    "value": "Japanese",
                    "importance": "mandatory",
                    "enforcement": "filter",
                },
                {
                    "field": "distance_km",
                    "operator": "less_than_or_equal",
                    "value": 3.0,
                    "importance": "mandatory",
                    "enforcement": "clarify",
                },
            ],
            "expected_party_size": None,
            "expected_missing_fields": ["user_location"],
            "generator_kind": "human_seed",
            "generator_model": None,
            "generator_prompt_sha256": None,
            "uses_specific_business": False,
            "uses_future_review": False,
        },
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    cases = load_query_benchmark(path)
    report = evaluate_request_parser(build_rule_based_request_parser(), cases)

    assert report.case_count == 2
    assert report.exact_match_rate == 1.0
    assert report.condition_precision == 1.0
    assert report.condition_recall == 1.0
    assert report.condition_f1 == 1.0
    assert report.failures == []


def test_benchmark_schema_forbids_target_business_leakage() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        QueryBenchmarkCase.model_validate(
            {
                "case_id": "leaky",
                "split": "validation",
                "frame_family": "bad",
                "language": "zh-CN",
                "query_text": "推荐那家店",
                "expected_intent": "recommendation_request",
                "expected_conditions": [],
                "expected_party_size": None,
                "expected_missing_fields": ["desired_category"],
                "generator_kind": "human_seed",
                "generator_model": None,
                "generator_prompt_sha256": None,
                "uses_specific_business": False,
                "uses_future_review": False,
                "target_business_id": "secret-target",
            }
        )
