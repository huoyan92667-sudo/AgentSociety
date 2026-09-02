"""固定提示词只约束证据边界，不限制大模型的自然表达形式。"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Protocol

from pydantic import Field

from yelp_agent.agent.llm import LLMCallResult, LLMMessage, OpenAICompatibleLLM
from yelp_agent.config import AgentConfig
from yelp_agent.models import StrictModel
from yelp_agent.recommendation_v2.business_facts import (
    BusinessFact,
    is_open_at,
    parse_visit_time,
)
from yelp_agent.recommendation_v2.review_evidence import ReviewEvidenceRankingResult
from yelp_agent.recommendation_v2.review_evidence.schema import (
    BusinessPreferenceEvidence,
    PreferenceSearchDescription,
    RankedReviewEvidence,
)
from yelp_agent.recommendation_v2.schema import UnifiedRecommendationState

PROMPT_VERSION = "personalized-recommendation-answer-v1"

_SYSTEM_PROMPT = """
你负责把已经完成硬过滤和证据排序的餐厅，写成给用户看的中文推荐回答。实际可能少于五家。

你可以自然组织语言，不需要返回 JSON，也不必机械填写固定栏目。但必须遵守：
1. 围绕 current_query 回答，不能只泛泛介绍餐厅。
2. 商家顺序已经确定，不得偷偷重排、删除或加入商家。
3. 只能使用提供的商家事实和真实评论。不得编造菜品、价格、距离、营业时间、服务或用户偏好。
4. “整体食物不错”不能改写成某道具体菜很好；只有评论直接支持时才能下具体菜品结论。
5. 一条评论同时包含优点、缺点或适用条件时，必须保留完整意思。
6. 有重要反面证据时要告诉用户；没有直接证据时明确说证据不足，不能硬夸。
7. 结合软偏好的先后顺序和 preference_assessments 中的通俗档位解释个性化原因，但不要暴露内部字段名、相似度、公式和计算过程。证据不足或争议高时必须降低结论强度。
8. 每家控制在一小段：核心推荐理由、最相关的真实证据、必要的风险或条件。
9. 营业时间来自历史 Yelp 数据；如果提到，只能表述为“数据记录显示”，不能声称实时准确。
10. top_count 是本轮真实商家数。只按已有 rank 从1介绍到 top_count，介绍完最后一家直接结束。严禁补写不存在的名次、道歉段落、额外商家，也禁止在末尾再次比较或重新排序。
11. visit_context 是程序按到店时间算好的结果。星期、当天营业时段和是否营业必须原样使用，禁止自己从日期推算星期，禁止拿其他星期的营业时间代替。
12. straight_line_distance_km 是经纬度直线距离，不是步行、驾车或路线距离。只能说“直线距离约多少”，不能改写成步行可达或步行多少公里。
13. 禁止使用模型自身知道的餐厅背景。除非商家事实或所给评论直接写明，否则不能说“知名、连锁、老字号、核心区、最佳位置”。
14. 评论里的外送经历只能说明外送，不能推导堂食普遍如何。只有一条评论时不能写“普遍、多次、多条评论都认为”。
15. 正反证据混合时必须保留风险，不能新造“最稳妥、口味有保障、位置最佳”等更强结论。
16. 输出必须由现有商家的编号段落组成。最后一家风险写完就立刻停止；禁止另写“综合来看、总体来说、如果更看重、前几家、后几家”等总结或二次比较段落。
""".strip()


class AnswerGenerator(Protocol):
    def generate(self, messages: list[LLMMessage]) -> LLMCallResult: ...


class RecommendationAnswer(StrictModel):
    """自然回答和程序已知的证据编号分开保存。"""

    status: str
    text: str | None = Field(default=None, max_length=20000)
    prompt_version: str = PROMPT_VERSION
    selected_review_ids_by_business: dict[str, list[str]] = Field(
        default_factory=dict
    )
    model: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float = Field(ge=0)
    failure_reason: str | None = Field(default=None, max_length=500)


class RecommendationAnswerSynthesizer:
    """每家最多两条完整评论，一次调用生成本轮全部推荐。"""

    def __init__(self, generator: AnswerGenerator) -> None:
        self._generator = generator

    def synthesize(
        self,
        *,
        query_text: str,
        state: UnifiedRecommendationState,
        ranking: ReviewEvidenceRankingResult,
        on_delta: Callable[[str], None] | None = None,
    ) -> RecommendationAnswer:
        selected_by_business: dict[str, list[RankedReviewEvidence]] = {}
        businesses: list[dict[str, object]] = []
        requirement_by_id = {
            item.requirement_id: item for item in ranking.requirements
        }
        for ranked in ranking.ranking:
            selected = _select_business_evidence(ranked.preference_evidence)
            business_id = ranked.business.business_id
            selected_by_business[business_id] = selected
            # 完整的一周营业表既浪费上下文，也容易让模型读错星期。程序只把
            # 本轮真正到店那一天和已经计算好的营业结论交给最终总结。
            business_facts = _answer_business_facts(ranked.business)
            businesses.append(
                {
                    "rank": ranked.final_rank,
                    "business": business_facts,
                    "straight_line_distance_km": ranked.distance_km,
                    "visit_context": _visit_context(state, ranked.business),
                    # 只告诉回答模型通俗档位，不暴露精确内部计算分，避免它
                    # 根据小数自行重新排序。真实顺序仍以程序给出的rank为准。
                    "preference_assessments": [
                        _answer_assessment(item, requirement_by_id)
                        for item in ranked.preference_evidence
                    ],
                    "evidence": [
                        {
                            "review_id": item.review_id,
                            "role": item.role,
                            "review_time": item.review_time.isoformat(),
                            "stars": item.stars,
                            # 完整评论仍保存在排序结果文件中。最终回答只读取
                            # 命中内容及相邻句，既保留语境也避免传入整篇长文。
                            "review_context": _review_context(item),
                            "supports_requirement": _requirement_for_review(
                                item.review_id,
                                ranked.preference_evidence,
                                requirement_by_id,
                            ),
                        }
                        for item in selected
                    ],
                }
            )
        selected_ids = {
            business_id: [item.review_id for item in evidence]
            for business_id, evidence in selected_by_business.items()
        }
        payload = {
            "current_query": query_text,
            "top_count": len(businesses),
            "active_hard_constraints": [
                {
                    "field": item.field,
                    "operator": item.operator,
                    "value": item.value,
                }
                for item in state.hard_constraints
            ],
            "ordered_soft_preferences": [
                {
                    "field": item.field,
                    "direction": item.direction,
                    "target_value": item.target_value,
                    "priority": item.priority,
                }
                for item in state.soft_preferences
            ],
            "scene": (
                None
                if state.scene is None
                else {
                    "kind": state.scene.kind,
                    "custom_label": state.scene.custom_label,
                }
            ),
            "search_center": (
                None
                if state.search_center is None
                else {
                    "label": state.search_center.label,
                    # 搜索中心本身只保存计算起点；距离上限属于硬条件，
                    # 已经在 active_hard_constraints 中单独提供给回答模型。
                    "latitude": state.search_center.location.latitude,
                    "longitude": state.search_center.location.longitude,
                }
            ),
            "top5": businesses,
        }
        messages = [
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            ),
        ]
        stream = getattr(self._generator, "stream", None)
        if on_delta is not None and callable(stream):
            call = stream(messages, on_delta)
        else:
            call = self._generator.generate(messages)
            # 旧生成器没有逐片能力时仍维持回调语义，但真实运行时不会走这里。
            if on_delta is not None and call.status == "success" and call.content:
                on_delta(call.content)
        if call.status != "success" or call.content is None:
            return RecommendationAnswer(
                status="failure",
                selected_review_ids_by_business=selected_ids,
                model=call.model,
                input_tokens=call.input_tokens,
                output_tokens=call.output_tokens,
                latency_ms=call.latency_ms,
                failure_reason=call.failure_reason or call.status,
            )
        return RecommendationAnswer(
            status="success",
            text=call.content.strip(),
            selected_review_ids_by_business=selected_ids,
            model=call.model,
            input_tokens=call.input_tokens,
            output_tokens=call.output_tokens,
        latency_ms=call.latency_ms,
    )


def _visit_context(
    state: UnifiedRecommendationState,
    business: BusinessFact,
) -> dict[str, object] | None:
    """把到店日期、正确星期和当天营业记录算好，避免模型自己心算出错。"""

    visit_values = {
        str(item.value)
        for item in [*state.hard_constraints, *state.default_constraints]
        if item.field == "open_at"
    }
    if len(visit_values) != 1:
        return None
    visit_time = parse_visit_time(next(iter(visit_values)))
    weekday_fields = (
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    )
    weekday_labels = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")
    recorded_hours = (
        None
        if business.weekly_hours is None
        else getattr(business.weekly_hours, weekday_fields[visit_time.weekday()])
    )
    return {
        "local_datetime": visit_time.isoformat(timespec="minutes"),
        "weekday": weekday_labels[visit_time.weekday()],
        "recorded_hours_for_visit_day": recorded_hours,
        "recorded_open_at_visit_time": is_open_at(business, visit_time),
        "source": "historical_yelp_weekly_hours",
    }


def _select_business_evidence(
    assessments: list[BusinessPreferenceEvidence],
) -> list[RankedReviewEvidence]:
    """每家只给一条高优先正面和一条重要风险，避免20条长评论撑爆上下文。"""

    selected: list[RankedReviewEvidence] = []
    seen: set[str] = set()
    # assessments 已按融合后的偏好顺序生成，所以先出现的证据对应更重要要求。
    for assessment in assessments:
        positives = assessment.positive_evidence
        if positives:
            _append_unique(selected, seen, positives[0])
            break
    # 反面证据不能因为节省词元被吞掉；优先保留最高优先要求的风险。
    for assessment in assessments:
        negatives = assessment.negative_evidence
        if negatives:
            _append_unique(selected, seen, negatives[0])
            break
    # 某一方向完全没有证据时，用剩余最强证据补到两条，但绝不超过两条。
    remaining = sorted(
        [
            item
            for assessment in assessments
            for item in [
                *assessment.positive_evidence,
                *assessment.negative_evidence,
            ]
            if item.review_id not in seen
        ],
        key=lambda item: (-item.evidence_weight, item.review_id),
    )
    for item in remaining:
        if len(selected) >= 2:
            break
        _append_unique(selected, seen, item)
    return selected[:2]


def _answer_assessment(
    assessment: BusinessPreferenceEvidence,
    requirement_by_id: dict[str, PreferenceSearchDescription],
) -> dict[str, object]:
    """把内部数值压成回答可用的档位，并保留证据不足与争议提醒。"""

    requirement = requirement_by_id.get(assessment.requirement_id)
    level = assessment.satisfaction_level
    if level is None:
        if assessment.recalled_review_count == 0:
            level = "证据不足"
        elif assessment.evidence_score >= 0.8:
            level = "明确满足"
        elif assessment.evidence_score >= 0.6:
            level = "比较满足"
        elif assessment.evidence_score > 0.4:
            level = "一般"
        elif assessment.evidence_score > 0.2:
            level = "比较不满足"
        else:
            level = "明确不满足"
    return {
        "requirement": (
            assessment.requirement_id
            if requirement is None
            else requirement.requirement_text
        ),
        "satisfaction_level": level,
        "evidence_sufficiency_level": assessment.evidence_sufficiency_level,
        "controversy_level": assessment.controversy_level,
    }


def _review_context(
    item: RankedReviewEvidence,
    *,
    maximum_characters: int = 1200,
) -> str:
    """截取命中句及前后各一句；短评论直接保留全文。"""

    full_text = item.review_text.strip()
    if len(full_text) <= maximum_characters:
        return full_text
    sentences = [
        value.strip()
        for value in re.split(r"(?<=[.!?。！？])\s+", full_text)
        if value.strip()
    ]
    if len(sentences) <= 3:
        return full_text[:maximum_characters].rstrip()

    matched = item.matched_segment_text.strip()
    matched_lower = matched.casefold()
    match_index = next(
        (
            index
            for index, sentence in enumerate(sentences)
            if sentence.casefold() in matched_lower
            or matched_lower in sentence.casefold()
        ),
        None,
    )
    if match_index is None:
        matched_terms = set(re.findall(r"\w+", matched_lower))
        match_index = max(
            range(len(sentences)),
            key=lambda index: len(
                matched_terms
                & set(re.findall(r"\w+", sentences[index].casefold()))
            ),
        )
    start = max(0, match_index - 1)
    end = min(len(sentences), match_index + 2)
    context = " ".join(sentences[start:end])
    return context[:maximum_characters].rstrip()


def _answer_business_facts(business: BusinessFact) -> dict[str, object]:
    """只把本轮推荐理由会使用的商家事实交给最终总结模型。"""

    return {
        "business_id": business.business_id,
        "name": business.name,
        "address": business.address,
        "city": business.city,
        "state": business.state,
        "postal_code": business.postal_code,
        "categories": business.categories,
        "price_level": business.price_level,
        "price_lower_usd": business.price_lower_usd,
        "price_upper_usd": business.price_upper_usd,
        "rating": business.rating,
        "review_count": business.review_count,
    }


def _append_unique(
    selected: list[RankedReviewEvidence],
    seen: set[str],
    item: RankedReviewEvidence,
) -> None:
    if item.review_id not in seen:
        selected.append(item)
        seen.add(item.review_id)


def _requirement_for_review(
    review_id: str,
    assessments: list[BusinessPreferenceEvidence],
    requirement_by_id: dict[str, PreferenceSearchDescription],
) -> dict[str, object] | None:
    for assessment in assessments:
        evidence = [
            *assessment.positive_evidence,
            *assessment.negative_evidence,
        ]
        if any(item.review_id == review_id for item in evidence):
            requirement = requirement_by_id.get(assessment.requirement_id)
            if requirement is None:
                return None
            preference = requirement.preference
            return {
                "text": requirement.requirement_text,
                "field": None if preference is None else preference.field,
                "priority": requirement.priority,
            }
    return None


def build_recommendation_answer_synthesizer() -> RecommendationAnswerSynthesizer:
    generator = OpenAICompatibleLLM.from_environment(
        AgentConfig(
            enabled=True,
            temperature=0.0,
            timeout_seconds=120,
            max_retries=2,
            max_tokens=3000,
            response_format_json=False,
            thinking="disabled",
        )
    )
    return RecommendationAnswerSynthesizer(generator)
