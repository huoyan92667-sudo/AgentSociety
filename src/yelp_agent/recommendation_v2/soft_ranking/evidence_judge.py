"""大模型逐条判断评论方向，程序再计算满足程度和证据层级。"""

from __future__ import annotations

import json
from typing import Protocol, Self

from pydantic import Field, ValidationError, model_validator

from yelp_agent.agent.llm import LLMCallResult, LLMMessage
from yelp_agent.models import StrictModel
from yelp_agent.recommendation_v2.schema import SoftPreference

from .schema import (
    EvidenceLevel,
    PreferenceEvidenceAssessment,
    RetrievedReview,
    ReviewVerdict,
    SelectedEvidence,
)


class EvidenceGenerator(Protocol):
    """评论判断只依赖项目统一的大模型调用接口。"""

    def generate(self, messages: list[LLMMessage]) -> LLMCallResult: ...


class ReviewEvidenceProposal(StrictModel):
    """大模型对一条真实评论的方向判断。"""

    review_id: str = Field(min_length=1)
    verdict: ReviewVerdict
    reason: str = Field(min_length=1, max_length=300)


class BusinessEvidenceProposal(StrictModel):
    """大模型必须逐条判断一家餐厅收到的所有候选评论。"""

    business_id: str = Field(min_length=1)
    reviews: list[ReviewEvidenceProposal] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_reviews(self) -> Self:
        ids = [item.review_id for item in self.reviews]
        if len(ids) != len(set(ids)):
            raise ValueError("each review can only be judged once")
        return self


class EvidenceJudgeProposal(StrictModel):
    """一次模型调用中所有餐厅的逐评论判断。"""

    assessments: list[BusinessEvidenceProposal]


class EvidenceJudgeResult(StrictModel):
    """模型轨迹和程序根据证据权重算出的最终判断。"""

    call: LLMCallResult
    raw_json: str | None = None
    assessments: list[PreferenceEvidenceAssessment] = Field(default_factory=list)
    failure_reason: str | None = None


class ReviewEvidenceJudge:
    """大模型只判评论方向，不允许它直接决定商家层级和排序。"""

    def __init__(self, generator: EvidenceGenerator) -> None:
        self._generator = generator

    def judge(
        self,
        preference: SoftPreference,
        reviews_by_business: dict[str, list[RetrievedReview]],
    ) -> EvidenceJudgeResult:
        """逐条判断后，用固定公式计算每家餐厅的满足程度。"""

        call = self._generator.generate(self._messages(preference, reviews_by_business))
        if call.status != "success" or call.content is None:
            return EvidenceJudgeResult(
                call=call,
                failure_reason=call.failure_reason or call.status,
            )
        try:
            proposal = EvidenceJudgeProposal.model_validate_json(call.content)
            assessments = self._materialize(
                preference,
                proposal,
                reviews_by_business,
            )
        except (ValidationError, ValueError, TypeError) as exc:
            return EvidenceJudgeResult(
                call=call,
                raw_json=call.content,
                failure_reason=f"invalid_evidence_output: {exc}",
            )
        return EvidenceJudgeResult(
            call=call,
            raw_json=call.content,
            assessments=assessments,
        )

    @staticmethod
    def _messages(
        preference: SoftPreference,
        reviews_by_business: dict[str, list[RetrievedReview]],
    ) -> list[LLMMessage]:
        payload = {
            "user_preference": {
                "field": preference.field,
                "direction": preference.direction,
                "target_value": preference.target_value,
                "priority": preference.priority,
                "plain_meaning": [item.text for item in preference.sources],
            },
            "business_reviews": {
                business_id: [item.model_dump(mode="json") for item in reviews]
                for business_id, reviews in reviews_by_business.items()
            },
        }
        return [
            LLMMessage(
                role="system",
                content=(
                    "你负责逐条阅读完整原评论，判断每条评论对用户当前偏好究竟是支持、反对、"
                    "有条件支持，还是无关。向量相似度只说明这条评论可能相关，不能代替你的语义判断；"
                    "星级也不能代替对当前偏好的判断。positive表示评论支持用户想要的方向，"
                    "negative表示评论明确反对用户想要的方向，conditional表示只在特定时间、菜品、"
                    "座位或其他条件下成立，或一条评论同时包含明显正反信息；irrelevant表示并未真正谈到该要求。"
                    "否定、转折、指代和上下文必须按整条review_text理解。"
                    "你不能决定餐厅属于哪个层级，不能给餐厅打总分，也不能编造或改写评论。"
                    "每家餐厅给了哪些review_id，就必须不重不漏地逐条返回哪些review_id；"
                    "没有候选评论的餐厅也要返回空reviews。reason用简短中文说明依据。"
                    "必须返回严格JSON对象，形状为"
                    '{"assessments":[{"business_id":"...","reviews":['
                    '{"review_id":"...","verdict":"positive|negative|conditional|irrelevant",'
                    '"reason":"简短中文依据"}]}]}'
                ),
            ),
            LLMMessage(
                role="user",
                content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            ),
        ]

    @staticmethod
    def _materialize(
        preference: SoftPreference,
        proposal: EvidenceJudgeProposal,
        reviews_by_business: dict[str, list[RetrievedReview]],
    ) -> list[PreferenceEvidenceAssessment]:
        expected_business_ids = list(reviews_by_business)
        proposed_business_ids = [item.business_id for item in proposal.assessments]
        if len(proposed_business_ids) != len(set(proposed_business_ids)) or set(
            proposed_business_ids
        ) != set(expected_business_ids):
            raise ValueError("model must assess every scoped business exactly once")
        by_business = {item.business_id: item for item in proposal.assessments}
        preference_weight = (preference.preference_strength / 100.0) * (
            0.75 ** (preference.priority - 1)
        )
        output: list[PreferenceEvidenceAssessment] = []
        for business_id in expected_business_ids:
            reviews = reviews_by_business[business_id]
            review_index = {item.review_id: item for item in reviews}
            judgments = by_business[business_id].reviews
            judgment_ids = [item.review_id for item in judgments]
            if set(judgment_ids) != set(review_index) or len(judgment_ids) != len(
                review_index
            ):
                raise ValueError("model must judge every retrieved review exactly once")

            selected: dict[str, list[SelectedEvidence]] = {
                "positive": [],
                "negative": [],
                "conditional": [],
            }
            totals = {"positive": 0.0, "negative": 0.0, "conditional": 0.0}
            counts = {"positive": 0, "negative": 0, "conditional": 0}
            for judgment in judgments:
                if judgment.verdict == "irrelevant":
                    continue
                review = review_index[judgment.review_id]
                # 向量只衡量相关程度，评论最终是正是反完全采用大模型的逐条判断。
                evidence_weight = max(review.positive_weight, review.negative_weight)
                totals[judgment.verdict] += evidence_weight
                counts[judgment.verdict] += 1
                selected[judgment.verdict].append(
                    SelectedEvidence(
                        role=judgment.verdict,
                        review_id=review.review_id,
                        review_time=review.review_time,
                        stars=review.stars,
                        review_text=review.review_text,
                        evidence_weight=evidence_weight,
                        judgment_reason=judgment.reason,
                    )
                )

            accepted_count = sum(counts.values())
            raw_score = (totals["positive"] + 0.5 * totals["conditional"] + 1.0) / (
                sum(totals.values()) + 2.0
            )
            if accepted_count < 2:
                level: EvidenceLevel = "unknown"
                score = 0.5
            elif raw_score >= 0.75 and counts["positive"] >= 2:
                level = "clearly_satisfies"
                score = raw_score
            elif raw_score <= 0.25 and counts["negative"] >= 2:
                level = "clearly_contradicts"
                score = raw_score
            else:
                level = "conditionally_satisfies"
                score = raw_score
            reason = (
                f"大模型逐条确认正面{counts['positive']}条、反面{counts['negative']}条、"
                f"条件性{counts['conditional']}条；程序按正面{totals['positive']:.3f}、"
                f"反面{totals['negative']:.3f}、条件性{totals['conditional']:.3f}计算。"
            )
            for values in selected.values():
                values.sort(key=lambda item: (-item.evidence_weight, item.review_id))
            output.append(
                PreferenceEvidenceAssessment(
                    business_id=business_id,
                    level=level,
                    satisfaction_score=score,
                    preference_weight=preference_weight,
                    reason=reason,
                    positive_weight=totals["positive"],
                    negative_weight=totals["negative"],
                    conditional_weight=totals["conditional"],
                    accepted_evidence_count=accepted_count,
                    positive_evidence=selected["positive"][:2],
                    negative_evidence=selected["negative"][:2],
                    conditional_evidence=selected["conditional"][:2],
                    retrieved_review_count=len(reviews),
                )
            )
        return output
