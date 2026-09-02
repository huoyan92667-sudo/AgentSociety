from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from yelp_agent.recommendation_v2.business_aspect_profiles import (
    BusinessAspectProfileCatalog,
    builder,
)
from yelp_agent.recommendation_v2.schema import ASPECT_FIELDS


class _FakeBusinessFacts:
    def contains(self, business_id: str) -> bool:
        return business_id == "b1"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _source_profile() -> dict[str, object]:
    aspects = []
    for aspect_id in ASPECT_FIELDS:
        aspects.append(
            {
                "aspect_id": aspect_id,
                "aspect_name_zh": aspect_id,
                "degree": 0.75,
                "degree_0_to_100": 75.0,
                "degree_level": {
                    "code": "3",
                    "name_zh": "偏高",
                    "meaning": "测试含义",
                },
                "evidence_sufficiency": 0.8,
                "evidence_sufficiency_level": "充分",
                "controversy": 0.1,
                "controversy_level": "低",
                "business_total_review_count": 10,
                "retrieved_candidate_count": 3,
                "model_related_review_count": 3,
                "unique_evidence_user_count": 3,
                "strong_evidence_count": 3,
                "unique_strong_user_count": 3,
                "usable_for_ranking": True,
                "ranking_degree": 0.75,
                "unusable_reasons": [],
                "effective_sample_size": 3.0,
                "evidence_weight_sum": 2.0,
                "retrieval_limit_reached": {"high": False, "low": False},
                "evidence": {
                    "high_degree": [
                        {
                            "review_id": "r1",
                            "user_id": "u1",
                            "review_time": "2026-01-01T00:00:00+00:00",
                            "stars": 5.0,
                            "useful": 2,
                            "text": "A reusable complete review.",
                            "relevance": 3,
                            "strength": 3,
                            "evidence_weight": 0.9,
                        }
                    ],
                    "low_degree": [],
                    "middle_degree": [],
                    "conditional": [],
                    "conditional_status": "not_available_in_current_model",
                },
            }
        )
    return {
        "schema_version": "2.0",
        "business_profiles": [
            {
                "business": {
                    "business_id": "b1",
                    "name": "Example Restaurant",
                    "selection_index": 1,
                },
                "aspects": aspects,
            }
        ],
    }


def test_build_and_query_profiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "server_output"
    output = tmp_path / "profiles"
    source.mkdir()
    _write_json(source / "business_aspect_profiles.json", _source_profile())
    _write_json(
        source / "judge_manifest.json",
        {
            "input_count": 14,
            "valid_output_count": 14,
            "invalid_output_count": 0,
            "business_count": 1,
        },
    )
    _write_json(source / "invalid_outputs.json", [])
    monkeypatch.setattr(
        builder,
        "load_business_fact_catalog",
        lambda: _FakeBusinessFacts(),
    )

    manifest = builder.build_business_aspect_profiles(source, output)
    catalog = BusinessAspectProfileCatalog.from_directory(output)

    assert manifest.business_count == 1
    assert manifest.score_count == 14
    assert manifest.representative_review_count == 1
    assert manifest.evidence_count == 14
    assert catalog.contains("b1")
    assert catalog.supported_businesses()[0].name == "Example Restaurant"
    assert catalog.scores(["b1"], ["quiet_environment"])[0].ranking_degree == 0.75
    evidence = catalog.evidence(
        ["b1"],
        ["quiet_environment"],
        groups=("high_degree",),
        limit_per_group=1,
    )
    assert len(evidence) == 1
    assert evidence[0].text == "A reusable complete review."

    connection = sqlite3.connect(output / "business_aspect_profiles.sqlite3")
    try:
        assert connection.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 1
        assert (
            connection.execute("SELECT COUNT(*) FROM aspect_evidence").fetchone()[0]
            == 14
        )
    finally:
        connection.close()


def test_direction_comes_from_training_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "server_output"
    output = tmp_path / "profiles"
    source.mkdir()
    _write_json(source / "business_aspect_profiles.json", _source_profile())
    _write_json(
        source / "judge_manifest.json",
        {
            "input_count": 14,
            "valid_output_count": 14,
            "invalid_output_count": 0,
            "business_count": 1,
        },
    )
    _write_json(source / "invalid_outputs.json", [])
    monkeypatch.setattr(
        builder,
        "load_business_fact_catalog",
        lambda: _FakeBusinessFacts(),
    )
    builder.build_business_aspect_profiles(source, output)
    catalog = BusinessAspectProfileCatalog.from_directory(output)

    quiet = catalog.direction("quiet_environment")
    crowded = catalog.direction("crowded")
    queue_time = catalog.direction("queue_time")
    spiciness = catalog.direction("spiciness")

    assert "非常吵" in quiet.lower_value_means
    assert "非常安静" in quiet.higher_value_means
    assert "完全不拥挤" in crowded.lower_value_means
    assert "非常拥挤" in crowded.higher_value_means
    assert "无需等待" in queue_time.lower_value_means
    assert "超过60分钟" in queue_time.higher_value_means
    assert "完全不辣" in spiciness.lower_value_means
    assert "极辣" in spiciness.higher_value_means


def test_catalog_rejects_business_outside_supported_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "server_output"
    output = tmp_path / "profiles"
    source.mkdir()
    _write_json(source / "business_aspect_profiles.json", _source_profile())
    _write_json(
        source / "judge_manifest.json",
        {
            "input_count": 14,
            "valid_output_count": 14,
            "invalid_output_count": 0,
            "business_count": 1,
        },
    )
    _write_json(source / "invalid_outputs.json", [])
    monkeypatch.setattr(
        builder,
        "load_business_fact_catalog",
        lambda: _FakeBusinessFacts(),
    )
    builder.build_business_aspect_profiles(source, output)
    catalog = BusinessAspectProfileCatalog.from_directory(output)

    with pytest.raises(KeyError, match="unsupported restaurant"):
        catalog.scores(["outside"], ["food_quality"])
