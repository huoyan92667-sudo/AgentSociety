"""在指定商家范围内，按主模型给出的任意问题查找真实评论证据。"""

from __future__ import annotations

from datetime import datetime
from time import perf_counter
from typing import Literal

from pydantic import Field

from yelp_agent.models import StrictModel
from yelp_agent.recommendation_v2.schema import OpenRequirement, RequirementBasis

from .descriptions import PreferenceDescriptionBuilder
from .retrieval import ReviewEvidenceRetriever
from .schema import (
    BusinessPreferenceEvidence,
    PreferenceSearchDescription,
    ReviewRetrievalMetrics,
)
from .scoring import EvidenceScoringConfig, aggregate_business_evidence


class DirectReviewEvidenceFinding(StrictModel):
    """某个自然语言问题在某家商家下得到的正反评论与聚合结果。"""

    business_id: str = Field(min_length=1)
    requirement_id: str = Field(min_length=1)
    requirement_text: str = Field(min_length=1)
    assessment: BusinessPreferenceEvidence


class DirectReviewEvidenceResult(StrictModel):
    """实时评论查询的完整、可持久化结果。"""

    status: Literal["success", "description_failure", "retrieval_failure"]
    requirements: list[PreferenceSearchDescription] = Field(default_factory=list)
    findings: list[DirectReviewEvidenceFinding] = Field(default_factory=list)
    retrieval_metrics: ReviewRetrievalMetrics | None = None
    model_call_count: int = Field(default=0, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float = Field(ge=0)
    failure_reason: str | None = Field(default=None, max_length=500)


class DirectReviewEvidenceSearch:
    """把一句证据需求扩写、混合检索并聚合；不负责选择是否应该查评论。"""

    def __init__(
        self,
        *,
        description_builder: PreferenceDescriptionBuilder,
        retriever: ReviewEvidenceRetriever,
        scoring_config: EvidenceScoringConfig | None = None,
    ) -> None:
        self._description_builder = description_builder
        self._retriever = retriever
        self._scoring_config = scoring_config or EvidenceScoringConfig()
        if (
            self._scoring_config.acceptance_threshold
            != retriever.acceptance_threshold
        ):
            raise ValueError("retrieval and scoring acceptance thresholds must match")

    def search(
        self,
        *,
        business_ids: list[str],
        evidence_queries: list[str],
        user_query_text: str,
        reference_time: datetime,
    ) -> DirectReviewEvidenceResult:
        """只搜索主模型已经选定的商家和证据问题。"""

        started = perf_counter()
        businesses = list(dict.fromkeys(business_ids))
        queries = [" ".join(item.split()) for item in evidence_queries]
        if not businesses or len(businesses) != len(business_ids):
            raise ValueError("business IDs must be nonempty and unique")
        if not queries or any(not item for item in queries):
            raise ValueError("evidence queries must be nonempty")
        if len(queries) != len(set(queries)):
            raise ValueError("evidence queries must be unique")

        # 这里的 OpenRequirement 只是复用已经验证过的长尾描述生成入口。
        # 用户意图已经由主模型确定，内部不会再次判断该查事实还是查评论。
        open_requirements = [
            OpenRequirement(
                key=f"direct_review.q{index}",
                text=text,
                behavior="prefer",
                priority=index,
                controlling_source="current_query",
                sources=[
                    RequirementBasis(
                        source="current_query",
                        text=text,
                        turn_index=1,
                        preference_strength=75,
                    )
                ],
            )
            for index, text in enumerate(queries, start=1)
        ]
        descriptions = self._description_builder.build(
            [],
            open_requirements,
            query_text=user_query_text,
        )
        call = descriptions.call
        usage = {
            "model_call_count": 1 if call is not None else 0,
            "input_tokens": None if call is None else call.input_tokens,
            "output_tokens": None if call is None else call.output_tokens,
        }
        if descriptions.failure_reason is not None:
            return DirectReviewEvidenceResult(
                status="description_failure",
                requirements=descriptions.descriptions,
                latency_ms=(perf_counter() - started) * 1000,
                failure_reason=descriptions.failure_reason,
                **usage,
            )

        try:
            retrieval = self._retriever.retrieve_many(
                descriptions.descriptions,
                businesses,
                cutoff_time=reference_time,
            )
            findings = [
                DirectReviewEvidenceFinding(
                    business_id=business_id,
                    requirement_id=requirement.requirement_id,
                    requirement_text=requirement.requirement_text,
                    assessment=aggregate_business_evidence(
                        requirement,
                        business_id,
                        retrieval.by_requirement[requirement.requirement_id].get(
                            business_id, []
                        ),
                        reference_time=reference_time,
                        config=self._scoring_config,
                    ),
                )
                for requirement in descriptions.descriptions
                for business_id in businesses
            ]
        except Exception as exc:  # noqa: BLE001
            return DirectReviewEvidenceResult(
                status="retrieval_failure",
                requirements=descriptions.descriptions,
                latency_ms=(perf_counter() - started) * 1000,
                failure_reason=f"review evidence loading failed: {exc}",
                **usage,
            )
        return DirectReviewEvidenceResult(
            status="success",
            requirements=descriptions.descriptions,
            findings=findings,
            retrieval_metrics=retrieval.metrics,
            latency_ms=(perf_counter() - started) * 1000,
            **usage,
        )
