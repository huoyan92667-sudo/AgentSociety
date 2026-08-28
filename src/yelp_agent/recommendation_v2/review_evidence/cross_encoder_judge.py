"""用本地小模型分两步判断评论是否相关，以及它支持哪一侧。"""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter
from typing import Literal

from pydantic import Field

from yelp_agent.cross_encoder.scorer import PairScorer
from yelp_agent.models import StrictModel

from .schema import PreferenceSearchDescription, ReviewSimilarityCandidate


type JudgedEvidenceLabel = Literal[
    "positive",
    "negative",
    "mixed",
    "ambiguous",
    "irrelevant",
]


REVIEW_EVIDENCE_INSTRUCTION = (
    "Given an evidence criterion in Query and a Yelp review passage in Document, "
    "judge whether the passage itself contains evidence satisfying the criterion. "
    "Respect negation, comparisons, and which restaurant the statement refers to. "
    "Keyword overlap, a restaurant name, or general praise alone is not evidence."
)


class CrossEncoderJudgmentConfig(StrictModel):
    """把三个模型分数变成最终标签的暂定实验门槛。"""

    relevance_threshold: float = Field(default=0.50, ge=0, le=1)
    support_threshold: float = Field(default=0.50, ge=0, le=1)
    direction_margin: float = Field(default=0.10, ge=0, le=1)


class ReviewEvidenceJudgment(StrictModel):
    """一条候选评论的三项原始分数和可解释判定。"""

    review_id: str = Field(min_length=1)
    business_id: str = Field(min_length=1)
    matched_segment_text: str = Field(min_length=1)
    relevance_score: float = Field(ge=0, le=1)
    positive_support_score: float | None = Field(default=None, ge=0, le=1)
    negative_support_score: float | None = Field(default=None, ge=0, le=1)
    label: JudgedEvidenceLabel


class CrossEncoderJudgmentMetrics(StrictModel):
    """只统计本地小模型判断，不把前面的评论召回时间混进来。"""

    candidate_count: int = Field(ge=0)
    relevant_candidate_count: int = Field(ge=0)
    scored_pair_count: int = Field(ge=0)
    scorer_call_count: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    truncated_pair_count: int = Field(ge=0)
    provider_latency_ms: float = Field(ge=0)
    wall_latency_ms: float = Field(ge=0)


class CrossEncoderJudgmentBatch(StrictModel):
    """调用方只需拿一批候选进来，得到判定和完整耗时。"""

    judgments: list[ReviewEvidenceJudgment]
    metrics: CrossEncoderJudgmentMetrics


class _ScoreAccumulator:
    """在三轮批量判断之间累计真实模型用量。"""

    def __init__(self) -> None:
        self.scored_pair_count = 0
        self.scorer_call_count = 0
        self.input_tokens = 0
        self.truncated_pair_count = 0
        self.provider_latency_ms = 0.0

    def add(
        self,
        *,
        pair_count: int,
        input_tokens: int,
        truncated_pair_count: int,
        provider_latency_ms: float,
    ) -> None:
        self.scored_pair_count += pair_count
        self.scorer_call_count += 1
        self.input_tokens += input_tokens
        self.truncated_pair_count += truncated_pair_count
        self.provider_latency_ms += provider_latency_ms


class CrossEncoderReviewEvidenceJudge:
    """先拦掉无关评论，再区分正面、反面、混合和确实不清楚。"""

    def __init__(
        self,
        scorer: PairScorer,
        *,
        config: CrossEncoderJudgmentConfig | None = None,
    ) -> None:
        self._scorer = scorer
        self.config = config or CrossEncoderJudgmentConfig()

    def judge(
        self,
        requirement: PreferenceSearchDescription,
        candidates: Sequence[ReviewSimilarityCandidate],
    ) -> CrossEncoderJudgmentBatch:
        """判断一批评论；第二步只处理第一步认为相关的片段。"""

        started = perf_counter()
        usage = _ScoreAccumulator()
        documents = [candidate.matched_segment_text for candidate in candidates]
        relevance_scores = self._score_many(
            _relevance_query(requirement), documents, usage
        )
        relevant_indexes = [
            index
            for index, score in enumerate(relevance_scores)
            if score >= self.config.relevance_threshold
        ]
        relevant_documents = [documents[index] for index in relevant_indexes]
        positive_scores = self._score_many(
            _positive_query(requirement), relevant_documents, usage
        )
        negative_scores = self._score_many(
            _negative_query(requirement), relevant_documents, usage
        )
        positive_by_index = dict(zip(relevant_indexes, positive_scores, strict=True))
        negative_by_index = dict(zip(relevant_indexes, negative_scores, strict=True))

        judgments: list[ReviewEvidenceJudgment] = []
        for index, candidate in enumerate(candidates):
            relevance = relevance_scores[index]
            positive = positive_by_index.get(index)
            negative = negative_by_index.get(index)
            label = self._label(relevance, positive, negative)
            judgments.append(
                ReviewEvidenceJudgment(
                    review_id=candidate.review_id,
                    business_id=candidate.business_id,
                    matched_segment_text=candidate.matched_segment_text,
                    relevance_score=relevance,
                    positive_support_score=positive,
                    negative_support_score=negative,
                    label=label,
                )
            )

        return CrossEncoderJudgmentBatch(
            judgments=judgments,
            metrics=CrossEncoderJudgmentMetrics(
                candidate_count=len(candidates),
                relevant_candidate_count=len(relevant_indexes),
                scored_pair_count=usage.scored_pair_count,
                scorer_call_count=usage.scorer_call_count,
                input_tokens=usage.input_tokens,
                truncated_pair_count=usage.truncated_pair_count,
                provider_latency_ms=usage.provider_latency_ms,
                wall_latency_ms=(perf_counter() - started) * 1000,
            ),
        )

    def _score_many(
        self,
        query: str,
        documents: Sequence[str],
        usage: _ScoreAccumulator,
    ) -> list[float]:
        scores: list[float] = []
        for offset in range(0, len(documents), self._scorer.batch_size):
            batch = documents[offset : offset + self._scorer.batch_size]
            if not batch:
                continue
            result = self._scorer.score(query, batch)
            scores.extend(result.scores)
            usage.add(
                pair_count=len(batch),
                input_tokens=result.input_tokens,
                truncated_pair_count=result.truncated_pair_count,
                provider_latency_ms=result.latency_ms,
            )
        return scores

    def _label(
        self,
        relevance: float,
        positive: float | None,
        negative: float | None,
    ) -> JudgedEvidenceLabel:
        if relevance < self.config.relevance_threshold:
            return "irrelevant"
        assert positive is not None and negative is not None
        positive_passes = positive >= self.config.support_threshold
        negative_passes = negative >= self.config.support_threshold
        if positive_passes and negative_passes:
            if abs(positive - negative) < self.config.direction_margin:
                return "mixed"
            return "positive" if positive > negative else "negative"
        if positive_passes and positive - negative >= self.config.direction_margin:
            return "positive"
        if negative_passes and negative - positive >= self.config.direction_margin:
            return "negative"
        return "ambiguous"


def _relevance_query(requirement: PreferenceSearchDescription) -> str:
    positive = "; ".join(requirement.positive_descriptions)
    negative = "; ".join(requirement.negative_descriptions)
    return (
        f"Evidence topic: {requirement.requirement_text}\n"
        f"Examples supporting the topic: {positive}\n"
        f"Examples opposing the topic: {negative}\n"
        "Does the review passage discuss this topic with concrete evidence on either "
        "side? General food quality or merely naming the cuisine does not count."
    )


def _positive_query(requirement: PreferenceSearchDescription) -> str:
    return (
        f"Requirement: {requirement.requirement_text}\n"
        "Judge only whether this review passage provides evidence that the reviewed "
        "restaurant satisfies the requirement. Supporting meanings include: "
        + "; ".join(requirement.positive_descriptions)
    )


def _negative_query(requirement: PreferenceSearchDescription) -> str:
    return (
        f"Requirement: {requirement.requirement_text}\n"
        "Judge only whether this review passage provides evidence that the reviewed "
        "restaurant fails the requirement. Opposing meanings include: "
        + "; ".join(requirement.negative_descriptions)
    )
