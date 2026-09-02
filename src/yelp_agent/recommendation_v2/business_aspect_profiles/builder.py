"""把服务器输出转换成可按商家和软偏好快速查询的本地数据库。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from yelp_agent.recommendation_v2.business_facts import load_business_fact_catalog
from yelp_agent.recommendation_v2.schema import ASPECT_FIELDS

from .schema import (
    AspectDirection,
    BusinessAspectEvidence,
    BusinessAspectProfileManifest,
    BusinessAspectScore,
    SupportedBusiness,
)

_RECOMMENDATION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIRECTORY = Path(r"C:\Users\29072\Desktop\important\server_output")
DEFAULT_OUTPUT_ROOT = _RECOMMENDATION_ROOT / "data" / "business_aspect_profiles" / "v1"
_DIRECTION_SOURCE = (
    _RECOMMENDATION_ROOT
    / "review_evidence"
    / "training_data"
    / "teacher_input_templates.v1.json"
)

_DATABASE_SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE supported_businesses (
    business_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    selection_index INTEGER NOT NULL UNIQUE CHECK (selection_index > 0),
    business_json TEXT NOT NULL
) WITHOUT ROWID;
CREATE TABLE aspect_directions (
    aspect_id TEXT PRIMARY KEY,
    aspect_order INTEGER NOT NULL UNIQUE,
    name_zh TEXT NOT NULL,
    definition TEXT NOT NULL,
    lower_value_means TEXT NOT NULL,
    higher_value_means TEXT NOT NULL,
    strength_scale_json TEXT NOT NULL,
    special_rules_json TEXT NOT NULL
) WITHOUT ROWID;
CREATE TABLE aspect_scores (
    business_id TEXT NOT NULL,
    aspect_id TEXT NOT NULL,
    degree REAL,
    degree_0_to_100 REAL,
    degree_level_code TEXT NOT NULL,
    degree_level_name_zh TEXT NOT NULL,
    degree_level_meaning TEXT NOT NULL,
    evidence_sufficiency REAL NOT NULL,
    evidence_sufficiency_level TEXT NOT NULL,
    controversy REAL,
    controversy_level TEXT NOT NULL,
    business_total_review_count INTEGER NOT NULL,
    retrieved_candidate_count INTEGER NOT NULL,
    model_related_review_count INTEGER NOT NULL,
    unique_evidence_user_count INTEGER NOT NULL,
    strong_evidence_count INTEGER NOT NULL,
    unique_strong_user_count INTEGER NOT NULL,
    usable_for_ranking INTEGER NOT NULL CHECK (usable_for_ranking IN (0, 1)),
    ranking_degree REAL,
    unusable_reasons_json TEXT NOT NULL,
    effective_sample_size REAL NOT NULL,
    evidence_weight_sum REAL NOT NULL,
    high_retrieval_limit_reached INTEGER NOT NULL CHECK (high_retrieval_limit_reached IN (0, 1)),
    low_retrieval_limit_reached INTEGER NOT NULL CHECK (low_retrieval_limit_reached IN (0, 1)),
    PRIMARY KEY (business_id, aspect_id),
    FOREIGN KEY (business_id) REFERENCES supported_businesses (business_id),
    FOREIGN KEY (aspect_id) REFERENCES aspect_directions (aspect_id)
) WITHOUT ROWID;
CREATE INDEX aspect_scores_ranking_idx
    ON aspect_scores (aspect_id, usable_for_ranking, ranking_degree);
CREATE TABLE reviews (
    review_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    review_time TEXT NOT NULL,
    stars REAL NOT NULL,
    useful INTEGER NOT NULL,
    text TEXT NOT NULL
) WITHOUT ROWID;
CREATE TABLE aspect_evidence (
    business_id TEXT NOT NULL,
    aspect_id TEXT NOT NULL,
    evidence_group TEXT NOT NULL CHECK (evidence_group IN ('high_degree', 'low_degree', 'middle_degree')),
    evidence_rank INTEGER NOT NULL CHECK (evidence_rank > 0),
    review_id TEXT NOT NULL,
    relevance INTEGER NOT NULL CHECK (relevance BETWEEN 1 AND 3),
    strength INTEGER NOT NULL CHECK (strength BETWEEN 0 AND 4),
    evidence_weight REAL NOT NULL CHECK (evidence_weight > 0),
    PRIMARY KEY (business_id, aspect_id, review_id),
    FOREIGN KEY (business_id, aspect_id)
        REFERENCES aspect_scores (business_id, aspect_id),
    FOREIGN KEY (review_id) REFERENCES reviews (review_id)
) WITHOUT ROWID;
CREATE INDEX aspect_evidence_lookup_idx
    ON aspect_evidence (business_id, aspect_id, evidence_group, evidence_rank);
PRAGMA user_version = 1;
"""


class BusinessAspectProfileBuildError(RuntimeError):
    """服务器结果不完整或彼此矛盾时终止导入。"""


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"required source file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_directions() -> list[AspectDirection]:
    source = _read_json(_DIRECTION_SOURCE)
    raw_aspects = source.get("aspects", [])
    if [item.get("id") for item in raw_aspects] != list(ASPECT_FIELDS):
        raise BusinessAspectProfileBuildError(
            "training direction definitions do not match the unified 14 aspects"
        )
    return [
        AspectDirection(
            aspect_id=item["id"],
            name_zh=item["name_zh"],
            definition=item["definition"],
            lower_value_means=item["strength_scale"]["0"],
            higher_value_means=item["strength_scale"]["4"],
            strength_scale=item["strength_scale"],
            special_rules=item["special_rules"],
        )
        for item in raw_aspects
    ]


def _score_from_source(business_id: str, raw: dict[str, Any]) -> BusinessAspectScore:
    level = raw["degree_level"]
    retrieval = raw["retrieval_limit_reached"]
    return BusinessAspectScore(
        business_id=business_id,
        aspect_id=raw["aspect_id"],
        degree=raw["degree"],
        degree_0_to_100=raw["degree_0_to_100"],
        degree_level_code=level["code"],
        degree_level_name_zh=level["name_zh"],
        degree_level_meaning=level["meaning"],
        evidence_sufficiency=raw["evidence_sufficiency"],
        evidence_sufficiency_level=raw["evidence_sufficiency_level"],
        controversy=raw["controversy"],
        controversy_level=raw["controversy_level"],
        business_total_review_count=raw["business_total_review_count"],
        retrieved_candidate_count=raw["retrieved_candidate_count"],
        model_related_review_count=raw["model_related_review_count"],
        unique_evidence_user_count=raw["unique_evidence_user_count"],
        strong_evidence_count=raw["strong_evidence_count"],
        unique_strong_user_count=raw["unique_strong_user_count"],
        usable_for_ranking=raw["usable_for_ranking"],
        ranking_degree=raw["ranking_degree"],
        unusable_reasons=raw["unusable_reasons"],
        effective_sample_size=raw["effective_sample_size"],
        evidence_weight_sum=raw["evidence_weight_sum"],
        high_retrieval_limit_reached=retrieval["high"],
        low_retrieval_limit_reached=retrieval["low"],
    )


def _evidence_from_source(
    business_id: str,
    aspect_id: str,
    group: str,
    rank: int,
    raw: dict[str, Any],
) -> BusinessAspectEvidence:
    return BusinessAspectEvidence(
        business_id=business_id,
        aspect_id=aspect_id,
        evidence_group=group,
        evidence_rank=rank,
        review_id=raw["review_id"],
        user_id=raw["user_id"],
        review_time=raw["review_time"],
        stars=raw["stars"],
        useful=raw["useful"],
        text=raw["text"],
        relevance=raw["relevance"],
        strength=raw["strength"],
        evidence_weight=raw["evidence_weight"],
    )


def _insert_direction(
    connection: sqlite3.Connection,
    direction: AspectDirection,
    order: int,
) -> None:
    connection.execute(
        """INSERT INTO aspect_directions VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            direction.aspect_id,
            order,
            direction.name_zh,
            direction.definition,
            direction.lower_value_means,
            direction.higher_value_means,
            _json_text(direction.strength_scale),
            _json_text(direction.special_rules),
        ),
    )


def _insert_score(connection: sqlite3.Connection, score: BusinessAspectScore) -> None:
    connection.execute(
        """INSERT INTO aspect_scores VALUES (
        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )""",
        (
            score.business_id,
            score.aspect_id,
            score.degree,
            score.degree_0_to_100,
            score.degree_level_code,
            score.degree_level_name_zh,
            score.degree_level_meaning,
            score.evidence_sufficiency,
            score.evidence_sufficiency_level,
            score.controversy,
            score.controversy_level,
            score.business_total_review_count,
            score.retrieved_candidate_count,
            score.model_related_review_count,
            score.unique_evidence_user_count,
            score.strong_evidence_count,
            score.unique_strong_user_count,
            int(score.usable_for_ranking),
            score.ranking_degree,
            _json_text(score.unusable_reasons),
            score.effective_sample_size,
            score.evidence_weight_sum,
            int(score.high_retrieval_limit_reached),
            int(score.low_retrieval_limit_reached),
        ),
    )


def _insert_evidence(
    connection: sqlite3.Connection,
    evidence: BusinessAspectEvidence,
    reviews_seen: dict[str, tuple[Any, ...]],
) -> None:
    review_values = (
        evidence.user_id,
        evidence.review_time.isoformat(),
        evidence.stars,
        evidence.useful,
        evidence.text,
    )
    previous = reviews_seen.get(evidence.review_id)
    if previous is None:
        connection.execute(
            "INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?)",
            (evidence.review_id, *review_values),
        )
        reviews_seen[evidence.review_id] = review_values
    elif previous != review_values:
        raise BusinessAspectProfileBuildError(
            f"review metadata changed across aspect evidence: {evidence.review_id}"
        )
    connection.execute(
        """INSERT INTO aspect_evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            evidence.business_id,
            evidence.aspect_id,
            evidence.evidence_group,
            evidence.evidence_rank,
            evidence.review_id,
            evidence.relevance,
            evidence.strength,
            evidence.evidence_weight,
        ),
    )


def _build_database(
    path: Path,
    *,
    profiles: list[dict[str, Any]],
    directions: list[AspectDirection],
) -> tuple[int, int, int, Counter[str]]:
    path.unlink(missing_ok=True)
    connection = sqlite3.connect(path)
    usable_count = 0
    evidence_counts: Counter[str] = Counter()
    reviews_seen: dict[str, tuple[Any, ...]] = {}
    try:
        connection.executescript(_DATABASE_SCHEMA)
        with connection:
            for order, direction in enumerate(directions, start=1):
                _insert_direction(connection, direction, order)
            for selection_index, profile in enumerate(profiles, start=1):
                business = profile["business"]
                supported = SupportedBusiness(
                    business_id=business["business_id"],
                    name=business["name"],
                    selection_index=business.get("selection_index", selection_index),
                )
                connection.execute(
                    "INSERT INTO supported_businesses VALUES (?, ?, ?, ?)",
                    (
                        supported.business_id,
                        supported.name,
                        supported.selection_index,
                        _json_text(business),
                    ),
                )
                aspects = profile.get("aspects", [])
                if [item.get("aspect_id") for item in aspects] != list(ASPECT_FIELDS):
                    raise BusinessAspectProfileBuildError(
                        f"business does not contain the ordered 14 aspects: {supported.business_id}"
                    )
                for raw_score in aspects:
                    score = _score_from_source(supported.business_id, raw_score)
                    _insert_score(connection, score)
                    usable_count += int(score.usable_for_ranking)
                    for group in ("high_degree", "low_degree", "middle_degree"):
                        for rank, raw_evidence in enumerate(
                            raw_score["evidence"][group], start=1
                        ):
                            evidence = _evidence_from_source(
                                supported.business_id,
                                score.aspect_id,
                                group,
                                rank,
                                raw_evidence,
                            )
                            _insert_evidence(connection, evidence, reviews_seen)
                            evidence_counts[group] += 1
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise BusinessAspectProfileBuildError(
                f"generated SQLite database failed integrity check: {integrity}"
            )
        connection.execute("VACUUM")
    finally:
        connection.close()
    return (
        usable_count,
        len(reviews_seen),
        sum(evidence_counts.values()),
        evidence_counts,
    )


def build_business_aspect_profiles(
    source_directory: str | Path = DEFAULT_SOURCE_DIRECTORY,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> BusinessAspectProfileManifest:
    """生成500家支持清单、14项方向说明和带索引的分数/证据数据库。"""

    source = Path(source_directory).resolve()
    output = Path(output_root).resolve()
    profile_path = source / "business_aspect_profiles.json"
    judge_manifest_path = source / "judge_manifest.json"
    invalid_path = source / "invalid_outputs.json"
    profile_document = _read_json(profile_path)
    judge_manifest = _read_json(judge_manifest_path)
    invalid_outputs = _read_json(invalid_path)
    profiles = profile_document.get("business_profiles", [])
    directions = _load_directions()
    if len(profiles) != judge_manifest.get("business_count"):
        raise BusinessAspectProfileBuildError(
            "business count differs between profiles and judge manifest"
        )
    business_ids = [item["business"]["business_id"] for item in profiles]
    if len(business_ids) != len(set(business_ids)):
        raise BusinessAspectProfileBuildError("business profile IDs must be unique")
    fact_catalog = load_business_fact_catalog()
    missing_facts = [
        value for value in business_ids if not fact_catalog.contains(value)
    ]
    if missing_facts:
        raise BusinessAspectProfileBuildError(
            f"{len(missing_facts)} profile businesses are missing from business facts"
        )
    if len(invalid_outputs) != judge_manifest.get("invalid_output_count"):
        raise BusinessAspectProfileBuildError(
            "invalid output file count differs from judge manifest"
        )

    output.mkdir(parents=True, exist_ok=True)
    database_path = output / "business_aspect_profiles.sqlite3"
    directions_path = output / "aspect_directions.json"
    businesses_path = output / "supported_businesses.json"
    manifest_path = output / "manifest.json"
    temporary_database = output / "business_aspect_profiles.sqlite3.partial"
    temporary_directions = output / "aspect_directions.json.partial"
    temporary_businesses = output / "supported_businesses.json.partial"
    temporary_manifest = output / "manifest.json.partial"
    temporary_paths = (
        temporary_database,
        temporary_directions,
        temporary_businesses,
        temporary_manifest,
    )
    for path in temporary_paths:
        path.unlink(missing_ok=True)
    try:
        usable_count, review_count, evidence_count, evidence_counts = _build_database(
            temporary_database,
            profiles=profiles,
            directions=directions,
        )
        direction_document = {
            "schema_version": 1,
            "source": str(_DIRECTION_SOURCE.resolve()),
            "meaning": (
                "degree从0到1递增；lower_value_means对应0，"
                "higher_value_means对应1，在线排序不得重新猜测方向"
            ),
            "aspects": [item.model_dump(mode="json") for item in directions],
        }
        business_document = {
            "schema_version": 1,
            "business_count": len(profiles),
            "businesses": [item["business"] for item in profiles],
        }
        _write_json(temporary_directions, direction_document)
        _write_json(temporary_businesses, business_document)
        score_count = len(profiles) * len(directions)
        source_hashes = {
            "business_aspect_profiles": _sha256(profile_path),
            "judge_manifest": _sha256(judge_manifest_path),
            "invalid_outputs": _sha256(invalid_path),
            "direction_definitions": _sha256(_DIRECTION_SOURCE),
        }
        output_hashes = {
            "database": _sha256(temporary_database),
            "aspect_directions": _sha256(temporary_directions),
            "supported_businesses": _sha256(temporary_businesses),
        }
        manifest = BusinessAspectProfileManifest(
            created_at=datetime.now(UTC),
            source_directory=str(source),
            source_sha256=source_hashes,
            source_model_input_count=judge_manifest["input_count"],
            source_valid_output_count=judge_manifest["valid_output_count"],
            source_invalid_output_count=judge_manifest["invalid_output_count"],
            business_count=len(profiles),
            aspect_count=len(directions),
            score_count=score_count,
            usable_score_count=usable_count,
            unusable_score_count=score_count - usable_count,
            representative_review_count=review_count,
            evidence_count=evidence_count,
            evidence_counts_by_group=dict(evidence_counts),
            output_sha256=output_hashes,
        )
        _write_json(temporary_manifest, manifest.model_dump(mode="json"))
        os.replace(temporary_database, database_path)
        os.replace(temporary_directions, directions_path)
        os.replace(temporary_businesses, businesses_path)
        os.replace(temporary_manifest, manifest_path)
    except Exception:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        raise
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-directory", type=Path, default=DEFAULT_SOURCE_DIRECTORY
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = build_business_aspect_profiles(
        args.source_directory,
        args.output_root,
    )
    print(manifest.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
