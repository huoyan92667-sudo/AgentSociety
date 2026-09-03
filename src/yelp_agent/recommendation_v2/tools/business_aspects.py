"""按商家编号读取已经离线计算好的14种评论特征和真实证据。"""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

from pydantic import Field, field_validator

from yelp_agent.models import StrictModel
from yelp_agent.recommendation_v2.business_aspect_profiles import (
    AspectDirection,
    BusinessAspectEvidence,
    BusinessAspectProfileCatalog,
    BusinessAspectScore,
)
from yelp_agent.recommendation_v2.schema import AspectField


class BusinessAspectEvidenceQuery(StrictModel):
    """一次查询少量商家和特征，防止代表性评论淹没主模型上下文。"""

    business_ids: list[str] = Field(min_length=1, max_length=5)
    aspect_ids: list[AspectField] = Field(min_length=1, max_length=5)
    evidence_limit_per_group: int = Field(default=2, ge=1, le=3)

    @field_validator("business_ids", "aspect_ids")
    @classmethod
    def unique_values(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("business IDs and aspect IDs must be unique")
        return values


class BusinessAspectAssessment(StrictModel):
    """一个商家的一项客观特征、证据充分程度、争议和代表性评论。"""

    business_id: str = Field(min_length=1)
    aspect_id: AspectField
    direction: AspectDirection
    score: BusinessAspectScore
    high_degree_evidence: list[BusinessAspectEvidence] = Field(default_factory=list)
    low_degree_evidence: list[BusinessAspectEvidence] = Field(default_factory=list)
    middle_degree_evidence: list[BusinessAspectEvidence] = Field(default_factory=list)


class BusinessAspectEvidenceObservation(StrictModel):
    """明确区分有离线画像和需要改走实时评论检索的商家。"""

    status: Literal["found", "partial", "not_found"]
    assessments: list[BusinessAspectAssessment] = Field(default_factory=list)
    unsupported_business_ids: list[str] = Field(default_factory=list, max_length=5)


class BusinessAspectEvidenceTool:
    """只读取离线结果，不判断用户是否应该使用这项能力。"""

    name = "lookup_business_aspect_evidence"

    def __init__(self, catalog: BusinessAspectProfileCatalog) -> None:
        self._catalog = catalog

    def execute(
        self,
        query: BusinessAspectEvidenceQuery,
    ) -> BusinessAspectEvidenceObservation:
        supported = [
            business_id
            for business_id in query.business_ids
            if self._catalog.contains(business_id)
        ]
        unsupported = [
            business_id
            for business_id in query.business_ids
            if business_id not in supported
        ]
        if not supported:
            return BusinessAspectEvidenceObservation(
                status="not_found",
                unsupported_business_ids=unsupported,
            )

        scores = self._catalog.scores(supported, query.aspect_ids)
        evidence = self._catalog.evidence(
            supported,
            query.aspect_ids,
            limit_per_group=query.evidence_limit_per_group,
        )
        evidence_by_key: dict[
            tuple[str, str, str], list[BusinessAspectEvidence]
        ] = defaultdict(list)
        for item in evidence:
            evidence_by_key[
                (item.business_id, item.aspect_id, item.evidence_group)
            ].append(item)

        assessments = [
            BusinessAspectAssessment(
                business_id=score.business_id,
                aspect_id=score.aspect_id,
                direction=self._catalog.direction(score.aspect_id),
                score=score,
                high_degree_evidence=evidence_by_key.get(
                    (score.business_id, score.aspect_id, "high_degree"), []
                ),
                low_degree_evidence=evidence_by_key.get(
                    (score.business_id, score.aspect_id, "low_degree"), []
                ),
                middle_degree_evidence=evidence_by_key.get(
                    (score.business_id, score.aspect_id, "middle_degree"), []
                ),
            )
            for score in scores
        ]
        return BusinessAspectEvidenceObservation(
            status="partial" if unsupported else "found",
            assessments=assessments,
            unsupported_business_ids=unsupported,
        )
