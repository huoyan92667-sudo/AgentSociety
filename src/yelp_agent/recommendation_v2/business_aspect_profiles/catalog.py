"""隐藏SQLite和JSON细节，只向排序与证据工具提供小型查询接口。"""

from __future__ import annotations

import json
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any

from yelp_agent.recommendation_v2.schema import ASPECT_FIELDS, AspectField

from .builder import DEFAULT_OUTPUT_ROOT, _sha256
from .schema import (
    AspectDirection,
    BusinessAspectEvidence,
    BusinessAspectProfileManifest,
    BusinessAspectScore,
    EvidenceGroup,
    SupportedBusiness,
)


class BusinessAspectProfileCatalog:
    """读取500家支持范围、离线分数和代表性评论。"""

    def __init__(self, database_path: Path, manifest: BusinessAspectProfileManifest):
        self._database_path = database_path.resolve()
        self._manifest = manifest

    @classmethod
    def from_directory(cls, root: str | Path) -> BusinessAspectProfileCatalog:
        directory = Path(root)
        database_path = directory / "business_aspect_profiles.sqlite3"
        manifest_path = directory / "manifest.json"
        if not database_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError("business aspect profile data is incomplete")
        manifest = BusinessAspectProfileManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if _sha256(database_path) != manifest.output_sha256["database"]:
            raise ValueError("business aspect profile database hash does not match")
        return cls(database_path, manifest)

    @property
    def manifest(self) -> BusinessAspectProfileManifest:
        return self._manifest

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"file:{self._database_path.as_posix()}?mode=ro",
            uri=True,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def supported_businesses(self) -> tuple[SupportedBusiness, ...]:
        """返回固定选择顺序的全部可离线排序餐厅。"""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT business_id, name, selection_index "
                "FROM supported_businesses ORDER BY selection_index"
            ).fetchall()
        return tuple(SupportedBusiness.model_validate(dict(row)) for row in rows)

    def contains(self, business_id: str) -> bool:
        """判断餐厅是否属于当前500家支持范围。"""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM supported_businesses WHERE business_id = ?",
                (business_id,),
            ).fetchone()
        return row is not None

    def direction(self, aspect_id: AspectField) -> AspectDirection:
        """读取训练时固定的0到4方向和每档含义。"""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM aspect_directions WHERE aspect_id = ?",
                (aspect_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown aspect: {aspect_id}")
        return AspectDirection(
            aspect_id=row["aspect_id"],
            name_zh=row["name_zh"],
            definition=row["definition"],
            lower_value_means=row["lower_value_means"],
            higher_value_means=row["higher_value_means"],
            strength_scale=json.loads(row["strength_scale_json"]),
            special_rules=json.loads(row["special_rules_json"]),
        )

    def scores(
        self,
        business_ids: list[str] | tuple[str, ...],
        aspect_ids: list[AspectField] | tuple[AspectField, ...],
    ) -> tuple[BusinessAspectScore, ...]:
        """批量读取分数；未知商家直接报错，防止有分数和无分数混排。"""

        businesses = _unique_nonempty(business_ids, "business IDs")
        aspects = _unique_nonempty(aspect_ids, "aspect IDs")
        if not businesses or not aspects:
            return ()
        _validate_aspects(aspects)
        self._ensure_supported(businesses)
        placeholders_business = ",".join("?" for _ in businesses)
        placeholders_aspect = ",".join("?" for _ in aspects)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM aspect_scores WHERE business_id IN ({placeholders_business}) "
                f"AND aspect_id IN ({placeholders_aspect})",
                (*businesses, *aspects),
            ).fetchall()
        by_key = {
            (row["business_id"], row["aspect_id"]): _score_from_row(row) for row in rows
        }
        expected = [(business, aspect) for business in businesses for aspect in aspects]
        missing = [key for key in expected if key not in by_key]
        if missing:
            raise ValueError(
                f"profile database is missing {len(missing)} requested scores"
            )
        return tuple(by_key[key] for key in expected)

    def evidence(
        self,
        business_ids: list[str] | tuple[str, ...],
        aspect_ids: list[AspectField] | tuple[AspectField, ...],
        *,
        groups: tuple[EvidenceGroup, ...] = (
            "high_degree",
            "low_degree",
            "middle_degree",
        ),
        limit_per_group: int = 3,
    ) -> tuple[BusinessAspectEvidence, ...]:
        """批量读取代表性评论，并对每家每项每组执行明确上限。"""

        if limit_per_group < 1:
            raise ValueError("evidence limit must be positive")
        businesses = _unique_nonempty(business_ids, "business IDs")
        aspects = _unique_nonempty(aspect_ids, "aspect IDs")
        selected_groups = _unique_nonempty(groups, "evidence groups")
        if not businesses or not aspects or not selected_groups:
            return ()
        _validate_aspects(aspects)
        allowed_groups = {"high_degree", "low_degree", "middle_degree"}
        if any(value not in allowed_groups for value in selected_groups):
            raise ValueError("unknown evidence group")
        self._ensure_supported(businesses)
        business_marks = ",".join("?" for _ in businesses)
        aspect_marks = ",".join("?" for _ in aspects)
        group_marks = ",".join("?" for _ in selected_groups)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT e.*, r.user_id, r.review_time, r.stars, r.useful, r.text "
                "FROM aspect_evidence AS e JOIN reviews AS r USING (review_id) "
                f"WHERE e.business_id IN ({business_marks}) "
                f"AND e.aspect_id IN ({aspect_marks}) "
                f"AND e.evidence_group IN ({group_marks}) "
                "ORDER BY e.business_id, e.aspect_id, e.evidence_group, e.evidence_rank",
                (*businesses, *aspects, *selected_groups),
            ).fetchall()
        order = {
            (business, aspect, group): index
            for index, (business, aspect, group) in enumerate(
                (b, a, g) for b in businesses for a in aspects for g in selected_groups
            )
        }
        kept: list[BusinessAspectEvidence] = []
        counts: dict[tuple[str, str, str], int] = {}
        for row in sorted(
            rows,
            key=lambda item: (
                order[(item["business_id"], item["aspect_id"], item["evidence_group"])],
                item["evidence_rank"],
            ),
        ):
            key = (row["business_id"], row["aspect_id"], row["evidence_group"])
            if counts.get(key, 0) >= limit_per_group:
                continue
            kept.append(_evidence_from_row(row))
            counts[key] = counts.get(key, 0) + 1
        return tuple(kept)

    def _ensure_supported(self, business_ids: tuple[str, ...]) -> None:
        marks = ",".join("?" for _ in business_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT business_id FROM supported_businesses "
                f"WHERE business_id IN ({marks})",
                business_ids,
            ).fetchall()
        known = {row["business_id"] for row in rows}
        missing = [value for value in business_ids if value not in known]
        if missing:
            raise KeyError(f"unsupported restaurant business IDs: {', '.join(missing)}")


def _unique_nonempty(values: Any, name: str) -> tuple[Any, ...]:
    result = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError(f"{name} must be nonempty strings")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must be unique")
    return result


def _validate_aspects(values: tuple[str, ...]) -> None:
    unknown = [value for value in values if value not in ASPECT_FIELDS]
    if unknown:
        raise ValueError(f"unknown aspects: {', '.join(unknown)}")


def _score_from_row(row: sqlite3.Row) -> BusinessAspectScore:
    return BusinessAspectScore(
        business_id=row["business_id"],
        aspect_id=row["aspect_id"],
        degree=row["degree"],
        degree_0_to_100=row["degree_0_to_100"],
        degree_level_code=row["degree_level_code"],
        degree_level_name_zh=row["degree_level_name_zh"],
        degree_level_meaning=row["degree_level_meaning"],
        evidence_sufficiency=row["evidence_sufficiency"],
        evidence_sufficiency_level=row["evidence_sufficiency_level"],
        controversy=row["controversy"],
        controversy_level=row["controversy_level"],
        business_total_review_count=row["business_total_review_count"],
        retrieved_candidate_count=row["retrieved_candidate_count"],
        model_related_review_count=row["model_related_review_count"],
        unique_evidence_user_count=row["unique_evidence_user_count"],
        strong_evidence_count=row["strong_evidence_count"],
        unique_strong_user_count=row["unique_strong_user_count"],
        usable_for_ranking=bool(row["usable_for_ranking"]),
        ranking_degree=row["ranking_degree"],
        unusable_reasons=json.loads(row["unusable_reasons_json"]),
        effective_sample_size=row["effective_sample_size"],
        evidence_weight_sum=row["evidence_weight_sum"],
        high_retrieval_limit_reached=bool(row["high_retrieval_limit_reached"]),
        low_retrieval_limit_reached=bool(row["low_retrieval_limit_reached"]),
    )


def _evidence_from_row(row: sqlite3.Row) -> BusinessAspectEvidence:
    return BusinessAspectEvidence(
        business_id=row["business_id"],
        aspect_id=row["aspect_id"],
        evidence_group=row["evidence_group"],
        evidence_rank=row["evidence_rank"],
        review_id=row["review_id"],
        user_id=row["user_id"],
        review_time=row["review_time"],
        stars=row["stars"],
        useful=row["useful"],
        text=row["text"],
        relevance=row["relevance"],
        strength=row["strength"],
        evidence_weight=row["evidence_weight"],
    )


@lru_cache(maxsize=1)
def load_business_aspect_profile_catalog() -> BusinessAspectProfileCatalog:
    """从 recommendation_v2 固定位置加载共享离线画像。"""

    return BusinessAspectProfileCatalog.from_directory(DEFAULT_OUTPUT_ROOT)
