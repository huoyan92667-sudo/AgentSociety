from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

from yelp_agent.recommendation_v2.review_evidence.offline_profiles import (
    prepare,
    server_judge,
)


def test_fixed_aspects_match_model_contract() -> None:
    definitions, contract = prepare._load_and_validate_definitions()

    assert len(definitions["aspects"]) == 14
    assert [item["id"] for item in definitions["aspects"]] == [
        item["id"] for item in contract["aspects"]
    ]
    assert len(prepare._dense_query_descriptors(definitions)) == 56
    assert len(prepare._keyword_query_descriptors(definitions)) == 28


def test_model_message_is_identical_in_shape_to_training_input() -> None:
    definitions, contract = prepare._load_and_validate_definitions()
    aspect = contract["aspects"][0]

    messages = prepare._messages(
        template=contract,
        aspect=aspect,
        review_text="The steak was excellent.",
    )
    model_input = json.loads(messages[1]["content"])

    assert messages[0]["content"] == contract["system_prompt"]
    assert set(model_input) == {
        "aspect_id",
        "definition",
        "relevance_scale",
        "strength_scale",
        "special_rules",
        "review_text",
    }
    assert "business_id" not in model_input
    assert "stars" not in model_input
    assert "source" not in model_input
    assert definitions["aspects"][0]["id"] == model_input["aspect_id"]


def test_numeric_strings_are_accepted_but_schema_stays_strict() -> None:
    assert server_judge._parse_label('{"relevance":"3","strength":"4"}') == (3, 4)
    assert server_judge._parse_label('{"relevance":"0","strength":null}') == (0, None)
    assert server_judge._parse_label('{"relevance":0,"strength":"0"}') == (None, None)
    assert server_judge._parse_label('{"relevance":3,"strength":4,"why":"x"}') == (
        None,
        None,
    )


def test_same_review_only_contributes_once_per_route_query() -> None:
    first = SimpleNamespace(
        review_id="r1",
        route_similarity=0.9,
        segment_id="r1:0",
        segment_index=0,
        segment_text="first",
    )
    second = SimpleNamespace(
        review_id="r1",
        route_similarity=0.8,
        segment_id="r1:1",
        segment_index=1,
        segment_text="second",
    )
    bucket: dict[str, dict[str, object]] = {}

    prepare._add_hit(
        bucket,
        first,
        route="dense",
        query_index=0,
        rank=1,
        contributes_to_rrf=True,
    )
    prepare._add_hit(
        bucket,
        second,
        route="dense",
        query_index=0,
        rank=2,
        contributes_to_rrf=False,
    )

    assert bucket["r1"]["rrf_score"] == 1 / 61
    assert len(bucket["r1"]["segments"]) == 2


def test_aggregate_keeps_degree_sufficiency_controversy_and_evidence() -> None:
    _, contract = prepare._load_and_validate_definitions()
    aspect_id = contract["aspects"][0]["id"]
    business_id = "b1"
    base = {
        "business_id": business_id,
        "business_name": "Example",
        "business_total_review_count": 100,
        "aspect_id": aspect_id,
        "aspect_name_zh": "菜品质量",
        "stars": 5.0,
        "useful": 1,
        "review_time": "2026-08-31T00:00:00+00:00",
        "full_review_text": "example",
    }
    inputs = [
        {**base, "sample_id": "s1", "review_id": "r1", "user_id": "u1"},
        {**base, "sample_id": "s2", "review_id": "r2", "user_id": "u2"},
    ]
    judgments = {
        "s1": {"sample_id": "s1", "relevance": 3, "strength": 4},
        "s2": {"sample_id": "s2", "relevance": 3, "strength": 0},
    }
    selection = {
        "businesses": [
            {
                "business_id": business_id,
                "name": "Example",
                "actual_review_count": 100,
            }
        ]
    }
    empty_count = {
        item["id"]: {
            "high_candidate_count": 0,
            "low_candidate_count": 0,
            "unique_candidate_count": 0,
            "high_limit_reached": False,
            "low_limit_reached": False,
        }
        for item in contract["aspects"]
    }
    empty_count[aspect_id] = {
        **empty_count[aspect_id],
        "high_candidate_count": 1,
        "low_candidate_count": 1,
        "unique_candidate_count": 2,
    }
    manifest = {
        "reference_time": datetime(2026, 9, 1, tzinfo=UTC).isoformat(),
        "counts_by_business_and_aspect": {business_id: empty_count},
    }
    args = SimpleNamespace(half_life_days=730.0, useful_alpha=0.2, useful_cap=1.2)

    result = server_judge._aggregate(
        inputs=inputs,
        judgments=judgments,
        selection=selection,
        contract=contract,
        prepare_manifest=manifest,
        args=args,
    )
    profile = result["business_profiles"][0]["aspects"][0]

    assert 0.49 < profile["degree"] < 0.51
    assert profile["model_related_review_count"] == 2
    assert profile["evidence_sufficiency"] > 0
    assert profile["controversy"] is None  # 有效样本量不足3时不冒充稳定争议值
    assert profile["strong_evidence_count"] == 2
    assert profile["unique_strong_user_count"] == 2
    assert profile["usable_for_ranking"] is False
    assert profile["ranking_degree"] is None
    assert "相关程度为2或3的评论少于3条" in profile["unusable_reasons"]
    assert len(profile["evidence"]["high_degree"]) == 1
    assert len(profile["evidence"]["low_degree"]) == 1
    assert profile["evidence"]["conditional_status"] == "not_available_in_current_model"
