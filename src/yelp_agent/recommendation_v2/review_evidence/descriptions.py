"""固定14种偏好直接读固定说法；只有长尾偏好才调用一次大模型。"""

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


class _LongTailItem(StrictModel):
    requirement_id: str = Field(min_length=1, max_length=200)
    positive_descriptions: list[str] = Field(min_length=2, max_length=3)
    negative_descriptions: list[str] = Field(min_length=2, max_length=3)


class _LongTailProposal(StrictModel):
    items: list[_LongTailItem]


class DescriptionBuildResult(StrictModel):
    """固定描述和一次长尾生成合并后的结果及模型用量。"""

    descriptions: list[PreferenceSearchDescription]
    call: LLMCallResult | None = None
    raw_json: str | None = None
    failure_reason: str | None = None


class PreferenceDescriptionBuilder:
    """把融合后的软要求变成检索说法，不让大模型改写固定14种含义。"""

    def __init__(self, generator: DescriptionGenerator) -> None:
        self._generator = generator

    def build(
        self,
        preferences: list[SoftPreference],
        open_requirements: list[OpenRequirement],
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
        if not long_tail:
            return DescriptionBuildResult(descriptions=fixed)

        call = self._generator.generate(self._messages(long_tail))
        if call.status != "success" or call.content is None:
            return DescriptionBuildResult(
                descriptions=fixed,
                call=call,
                failure_reason=call.failure_reason or "long_tail_description_failed",
            )
        try:
            proposal = _LongTailProposal.model_validate_json(call.content)
            expanded = self._validate_and_materialize(long_tail, proposal)
        except (ValidationError, ValueError) as exc:
            return DescriptionBuildResult(
                descriptions=fixed,
                call=call,
                raw_json=call.content,
                failure_reason=f"invalid long-tail descriptions: {exc}",
            )
        return DescriptionBuildResult(
            descriptions=[*fixed, *expanded],
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
    def _messages(requirements: list[OpenRequirement]) -> list[LLMMessage]:
        payload = [
            {
                "requirement_id": item.key,
                "user_text": item.text,
                "behavior": item.behavior,
            }
            for item in requirements
        ]
        system = """
你只负责为英文 Yelp 评论生成语义检索说法，不推荐商家，也不判断评论真假。
一次处理输入中的全部用户要求。每个要求输出2到3条英文正向描述和2到3条英文反向描述。
正向描述表示商家满足用户要求时评论可能表达的意思；反向描述表示商家违反用户要求时评论可能表达的意思。
若 behavior=avoid，正向描述应表达成功避开该问题，反向描述应表达出现了用户想避开的情况。
描述应具体、互补、适合向量检索，不得编造商家、评论或数值。
只返回严格 JSON：
{"items":[{"requirement_id":"原编号","positive_descriptions":["...","..."],"negative_descriptions":["...","..."]}]}
""".strip()
        return [
            LLMMessage(role="system", content=system),
            LLMMessage(
                role="user",
                content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            ),
        ]

    @staticmethod
    def _validate_and_materialize(
        requirements: list[OpenRequirement],
        proposal: _LongTailProposal,
    ) -> list[PreferenceSearchDescription]:
        expected = {item.key: item for item in requirements}
        received = {item.requirement_id: item for item in proposal.items}
        if set(received) != set(expected) or len(received) != len(proposal.items):
            raise ValueError("model must return each long-tail requirement exactly once")
        result: list[PreferenceSearchDescription] = []
        for requirement in requirements:
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
