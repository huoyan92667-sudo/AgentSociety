"""单问题全量评论标注和召回对比共同使用的固定数据结构。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import Field, model_validator

from yelp_agent.models import StrictModel

type ReviewLabelKind = Literal[
    "positive",
    "negative",
    "mixed",
    "ambiguous",
    "irrelevant",
]
type EvidenceGrade = Literal["direct", "supporting", "weak", "none"]


class QuestionProposal(StrictModel):
    """GLM 只把程序给出的真实范围改写成一句自然问题。"""

    query_text: str = Field(min_length=10, max_length=300)
    evidence_requirement: str = Field(min_length=2, max_length=200)
    positive_definition: str = Field(min_length=5, max_length=500)
    negative_definition: str = Field(min_length=5, max_length=500)


class ReviewAnnotationRecord(StrictModel):
    """交给标注模型的一条完整原评论；不提供星级和现有相似度。"""

    review_id: str = Field(min_length=1)
    business_id: str = Field(min_length=1)
    review_time: datetime
    review_text: str = Field(min_length=1, max_length=5000)


class ReviewEvidenceLabel(StrictModel):
    """一条评论相对当前问题的完整人工替代标签。"""

    review_id: str = Field(min_length=1)
    business_id: str = Field(min_length=1)
    label: ReviewLabelKind
    evidence_grade: EvidenceGrade
    positive_spans: list[str] = Field(max_length=2)
    negative_spans: list[str] = Field(max_length=2)
    ambiguous_span: str | None
    condition_text: str | None = Field(max_length=300)

    @model_validator(mode="before")
    @classmethod
    def normalize_non_directional_grade(cls, value: object) -> object:
        """模糊和无关评论不参与正反证据程度，程序固定补成 none。"""

        if isinstance(value, dict) and value.get("label") in {
            "ambiguous",
            "irrelevant",
        }:
            return {**value, "evidence_grade": "none"}
        return value

    @model_validator(mode="after")
    def validate_evidence_shape(self) -> Self:
        if any(not value or len(value) > 500 for value in self.positive_spans):
            raise ValueError("positive evidence spans must contain 1-500 characters")
        if any(not value or len(value) > 500 for value in self.negative_spans):
            raise ValueError("negative evidence spans must contain 1-500 characters")
        if self.ambiguous_span is not None and not 1 <= len(self.ambiguous_span) <= 500:
            raise ValueError("ambiguous evidence span must contain 1-500 characters")

        if self.label == "positive":
            valid = bool(self.positive_spans) and not self.negative_spans and self.ambiguous_span is None
        elif self.label == "negative":
            valid = bool(self.negative_spans) and not self.positive_spans and self.ambiguous_span is None
        elif self.label == "mixed":
            valid = bool(self.positive_spans) and bool(self.negative_spans) and self.ambiguous_span is None
        elif self.label == "ambiguous":
            valid = not self.positive_spans and not self.negative_spans and self.ambiguous_span is not None
        else:
            valid = not self.positive_spans and not self.negative_spans and self.ambiguous_span is None
        if not valid:
            raise ValueError("review label and evidence fields do not agree")

        if self.label in {"ambiguous", "irrelevant"}:
            if self.evidence_grade != "none":
                raise ValueError("ambiguous and irrelevant reviews use evidence_grade=none")
        elif self.evidence_grade == "none":
            raise ValueError("directional evidence requires a non-none grade")
        return self


class ReviewAnnotationBatch(StrictModel):
    """一次 GLM 调用必须逐条返回本批所有评论。"""

    batch_id: str = Field(min_length=1, max_length=100)
    case_id: str = Field(min_length=1, max_length=100)
    labels: list[ReviewEvidenceLabel] = Field(min_length=1)


class ClaudeWorkerTrace(StrictModel):
    """记录一次 Claude Code/GLM 调用的真实耗时和词元。"""

    model: str = Field(min_length=1)
    duration_ms: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    thinking_tokens: int | None = Field(default=None, ge=0)
    reported_cost_usd: float | None = Field(default=None, ge=0)
    stderr_excerpt: str | None = Field(default=None, max_length=1000)


class LabeledReview(StrictModel):
    """合并后的正确答案保留原评论事实和模型标签。"""

    review: ReviewAnnotationRecord
    label: ReviewEvidenceLabel
    batch_id: str = Field(min_length=1)


class RecallComparison(StrictModel):
    """当前召回相对全量标签的命中、漏召回和错误方向。"""

    total_review_count: int = Field(ge=0)
    direct_relevant_count: int = Field(ge=0)
    positive_direct_count: int = Field(ge=0)
    negative_direct_count: int = Field(ge=0)
    retrieved_direct_count: int = Field(ge=0)
    direct_recall: float | None = Field(default=None, ge=0, le=1)
    positive_direct_recall: float | None = Field(default=None, ge=0, le=1)
    negative_direct_recall: float | None = Field(default=None, ge=0, le=1)
    strict_relevant_count: int = Field(ge=0)
    positive_relevant_count: int = Field(ge=0)
    negative_relevant_count: int = Field(ge=0)
    retrieved_review_count: int = Field(ge=0)
    retrieved_strict_relevant_count: int = Field(ge=0)
    strict_recall: float | None = Field(default=None, ge=0, le=1)
    positive_recall: float | None = Field(default=None, ge=0, le=1)
    negative_recall: float | None = Field(default=None, ge=0, le=1)
    direction_correct_count: int = Field(ge=0)
    wrong_direction_count: int = Field(ge=0)
    false_directional_evidence_count: int = Field(ge=0)
    missed_review_ids: list[str]
    wrong_direction_review_ids: list[str]
    false_directional_review_ids: list[str]


class SingleCaseBenchmarkReport(StrictModel):
    """一个问题从生成、硬筛、全量标注到召回对比的完整结果。"""

    case_id: str = Field(min_length=1)
    question: QuestionProposal
    fusion_status: str = Field(min_length=1)
    fusion_model: str | None = None
    fusion_failure_reason: str | None = None
    state_source: Literal["four_source_fusion", "benchmark_fixed_scope_fallback"]
    search_center: dict[str, object] | None
    hard_constraints: list[dict[str, object]]
    default_constraints: list[dict[str, object]]
    review_search_descriptions: list[dict[str, object]]
    hard_filtered_business_count: int = Field(ge=0)
    hard_filtered_businesses: list[dict[str, object]]
    exhaustive_review_count: int = Field(ge=0)
    annotation_batch_count: int = Field(ge=0)
    annotation_label_counts: dict[str, int]
    question_trace: ClaudeWorkerTrace
    fusion_latency_ms: float = Field(ge=0)
    fusion_input_tokens: int | None = Field(default=None, ge=0)
    fusion_output_tokens: int | None = Field(default=None, ge=0)
    annotation_model_call_count: int = Field(ge=0)
    annotation_total_input_tokens: int = Field(ge=0)
    annotation_total_output_tokens: int = Field(ge=0)
    annotation_total_thinking_tokens: int = Field(ge=0)
    annotation_api_time_ms_sum: int = Field(ge=0)
    annotation_reported_cost_usd: float = Field(ge=0)
    annotation_traces: list[ClaudeWorkerTrace]
    retrieval_latency_ms: float = Field(ge=0)
    comparison: RecallComparison
    output_root: str = Field(min_length=1)
