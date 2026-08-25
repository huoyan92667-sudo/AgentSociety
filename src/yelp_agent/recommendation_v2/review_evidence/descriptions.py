"""用当前问题改写评论检索说法，同时保持融合后的偏好含义不变。"""

from __future__ import annotations

import json
from typing import Protocol

from pydantic import Field, ValidationError

from yelp_agent.agent.llm import LLMCallResult, LLMMessage
from yelp_agent.models import StrictModel
from yelp_agent.recommendation_v2.review_features.definitions import (
    preference_semantic_anchors,
)
from yelp_agent.recommendation_v2.schema import (
    ASPECT_FIELDS,
    OpenRequirement,
    SoftPreference,
)

from .schema import PreferenceSearchDescription


class DescriptionGenerator(Protocol):
    """长尾描述生成只依赖一次结构化大模型调用。"""

    def generate(self, messages: list[LLMMessage]) -> LLMCallResult: ...


class _SearchDescriptionItem(StrictModel):
    requirement_id: str = Field(min_length=1, max_length=200)
    positive_descriptions: list[str] = Field(min_length=2, max_length=3)
    negative_descriptions: list[str] = Field(min_length=2, max_length=3)


class _SearchDescriptionProposal(StrictModel):
    items: list[_SearchDescriptionItem]


class DescriptionBuildResult(StrictModel):
    """固定描述和一次长尾生成合并后的结果及模型用量。"""

    descriptions: list[PreferenceSearchDescription]
    call: LLMCallResult | None = None
    raw_json: str | None = None
    failure_reason: str | None = None


class PreferenceDescriptionBuilder:
    """让检索说法知道用户正在找什么，但不让模型改动偏好与顺序。"""

    def __init__(self, generator: DescriptionGenerator) -> None:
        self._generator = generator

    def build(
        self,
        preferences: list[SoftPreference],
        open_requirements: list[OpenRequirement],
        *,
        query_text: str | None = None,
    ) -> DescriptionBuildResult:
        fixed = [
            self._fixed_description(item)
            for item in sorted(preferences, key=lambda value: value.priority)
            if item.field in ASPECT_FIELDS
        ]
        long_tail = [
            item
            for item in sorted(
                open_requirements,
                key=lambda value: value.priority or 10_000,
            )
            if item.behavior in {"prefer", "avoid"}
        ]
        # 没有当前问题时保留原来的固定锚点，方便离线工具和旧调用方使用。
        # 在线推荐一定传 query_text，因此固定偏好也会经过一次上下文改写。
        if not long_tail and not query_text:
            return DescriptionBuildResult(descriptions=fixed)

        call = self._generator.generate(
            self._messages(query_text or "", fixed, long_tail)
        )
        if call.status != "success" or call.content is None:
            return DescriptionBuildResult(
                descriptions=fixed,
                call=call,
                failure_reason=call.failure_reason or "description_generation_failed",
            )
        try:
            proposal = _SearchDescriptionProposal.model_validate_json(call.content)
            expanded = self._validate_and_materialize(fixed, long_tail, proposal)
        except (ValidationError, ValueError) as exc:
            return DescriptionBuildResult(
                descriptions=fixed,
                call=call,
                raw_json=call.content,
                failure_reason=f"invalid search descriptions: {exc}",
            )
        return DescriptionBuildResult(
            descriptions=expanded,
            call=call,
            raw_json=call.content,
        )

    @staticmethod
    def _fixed_description(preference: SoftPreference) -> PreferenceSearchDescription:
        anchors = preference_semantic_anchors(
            preference.field,  # type: ignore[arg-type]
            preference.direction,
        )
        return PreferenceSearchDescription(
            requirement_id=preference.key,
            requirement_text=preference.key,
            kind="fixed_aspect",
            priority=preference.priority,
            preference_strength=preference.preference_strength,
            positive_descriptions=anchors.satisfying,
            negative_descriptions=anchors.contradicting,
            preference=preference,
        )

    @staticmethod
    def _messages(
        query_text: str,
        fixed: list[PreferenceSearchDescription],
        long_tail: list[OpenRequirement],
    ) -> list[LLMMessage]:
        items = [
            {
                "requirement_id": item.requirement_id,
                "kind": "fixed_aspect",
                "field": item.preference.field if item.preference is not None else None,
                "direction": (
                    item.preference.direction if item.preference is not None else None
                ),
                "base_positive_descriptions": item.positive_descriptions,
                "base_negative_descriptions": item.negative_descriptions,
            }
            for item in fixed
        ]
        items.extend(
            {
                "requirement_id": item.key,
                "kind": "long_tail",
                "user_text": item.text,
                "behavior": item.behavior,
            }
            for item in long_tail
        )
        system = """
你只负责为英文 Yelp 评论生成语义检索说法，不推荐商家，也不判断评论真假。
你会同时看到用户当前问题和已经融合好的软偏好。一次处理全部要求，每项输出2到3条英文正向描述和2到3条英文反向描述。
正向描述表示商家满足用户要求时评论可能表达的意思；反向描述表示商家违反用户要求时评论可能表达的意思。
若 behavior=avoid，正向描述应表达成功避开该问题，反向描述应表达出现了用户想避开的情况。
固定特征已经给出基础含义，你不能改变特征方向。当前问题中的具体菜品或菜系会改变特征含义时，应把它写进检索说法。例如用户想吃牛排且特征是菜品质量，应该检索牛排肉质、味道和熟度，而不是宽泛的“所有食物都很好”。
环境、拥挤、停车、服务等商家整体特征不需要生硬地绑定菜品名称，继续表达餐厅层面的真实含义。
描述应具体、互补、适合向量检索，不得编造商家、评论或数值。
只返回严格 JSON：
{"items":[{"requirement_id":"原编号","positive_descriptions":["...","..."],"negative_descriptions":["...","..."]}]}
""".strip()
        return [
            LLMMessage(role="system", content=system),
            LLMMessage(
                role="user",
                content=json.dumps(
                    {"query_text": query_text, "items": items},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            ),
        ]

    @staticmethod
    def _validate_and_materialize(
        fixed: list[PreferenceSearchDescription],
        long_tail: list[OpenRequirement],
        proposal: _SearchDescriptionProposal,
    ) -> list[PreferenceSearchDescription]:
        expected_ids = {
            *(item.requirement_id for item in fixed),
            *(item.key for item in long_tail),
        }
        received = {item.requirement_id: item for item in proposal.items}
        if set(received) != expected_ids or len(received) != len(proposal.items):
            raise ValueError("model must return every requirement exactly once")
        result = [
            item.model_copy(
                update={
                    "positive_descriptions": received[
                        item.requirement_id
                    ].positive_descriptions,
                    "negative_descriptions": received[
                        item.requirement_id
                    ].negative_descriptions,
                },
                deep=True,
            )
            for item in fixed
        ]
        for requirement in long_tail:
            item = received[requirement.key]
            strengths = [
                basis.preference_strength
                for basis in requirement.sources
                if basis.preference_strength is not None
            ]
            result.append(
                PreferenceSearchDescription(
                    requirement_id=requirement.key,
                    requirement_text=requirement.text,
                    kind="long_tail",
                    priority=requirement.priority or 100,
                    preference_strength=max(strengths, default=75),
                    positive_descriptions=item.positive_descriptions,
                    negative_descriptions=item.negative_descriptions,
                )
            )
        return result
