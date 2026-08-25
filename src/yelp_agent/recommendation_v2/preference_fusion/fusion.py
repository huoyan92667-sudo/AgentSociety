"""让大模型理解多轮需求，再由程序补齐成唯一的完整推荐状态。"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol, Self, cast

from pydantic import Field, ValidationError, model_validator

from yelp_agent.agent.llm import LLMCallResult, LLMMessage
from yelp_agent.models import StrictModel
from yelp_agent.recommendation_v2.business_facts import (
    CATALOG_TIME_ZONE,
    catalog_local_time,
)
from yelp_agent.recommendation_v2.category_catalog import load_fixed_category_catalog
from yelp_agent.recommendation_v2.preference_fusion.profile_adapter import (
    ProfilePreferenceSet,
)
from yelp_agent.recommendation_v2.schema import (
    ASPECT_FIELDS,
    BaselineSceneKind,
    BusinessReference,
    ConstraintDerivation,
    DefaultConstraint,
    GeoPoint,
    HardConstraint,
    HardConstraintField,
    HardOperator,
    OpenRequirement,
    PreferenceDirection,
    PreferenceMemoryItem,
    PreferenceStrength,
    RequirementBasis,
    RequirementField,
    RequirementValue,
    SceneBaseline,
    SceneKind,
    SceneSelection,
    SearchCenter,
    SoftPreference,
    SourceKind,
    UnifiedRecommendationState,
    merchant_feature_for,
    requirement_unit_for,
)
from yelp_agent.recommendation_v2.tools.business_facts import (
    BusinessFactsObservation,
    BusinessFactsQuery,
    BusinessFactsTool,
)
from yelp_agent.recommendation_v2.tools.history_business import (
    HistoryBusinessFact,
    HistoryBusinessFactTool,
    HistoryFactObservation,
    HistoryFactQuery,
)

PROMPT_VERSION = "recommendation-v2-compact-tool-fusion-v1"
MAX_TOOL_CALLS = 4

_SOURCE_PRIORITY: dict[SourceKind, int] = {
    "current_query": 4,
    "session": 3,
    "user_profile": 2,
    "scene": 1,
    "system_default": 0,
}
_CHOICE_FIELDS: set[RequirementField] = {"category"}
_DIRECTIONAL_FIELDS: set[RequirementField] = {
    "distance_km",
    *ASPECT_FIELDS,
}
_BOOLEAN_FIELDS: set[RequirementField] = {
    "accepts_reservations",
    "delivery",
    "takeout",
    "outdoor_seating",
    "good_for_kids",
    "good_for_groups",
    "wheelchair_accessible",
    "dogs_allowed",
    "parking_available",
}

type SemanticRelation = Literal["same", "conflict", "shadow", "independent"]


class RecommendationSnapshot(StrictModel):
    """一轮推荐的紧凑记忆；完整评论继续留在评论库中。"""

    state_revision: int = Field(ge=1)
    ordered_business_ids: list[str] = Field(min_length=1, max_length=5)
    evidence_review_ids_by_business: dict[str, list[str]] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_evidence_businesses(self) -> Self:
        if len(self.ordered_business_ids) != len(set(self.ordered_business_ids)):
            raise ValueError("snapshot business IDs must be unique")
        if not set(self.evidence_review_ids_by_business) <= set(
            self.ordered_business_ids
        ):
            raise ValueError("snapshot evidence must belong to a shown business")
        if any(
            len(values) != len(set(values))
            for values in self.evidence_review_ids_by_business.values()
        ):
            raise ValueError("snapshot review IDs must be unique per business")
        return self


class ConversationHistoryTurn(StrictModel):
    """一轮原始对话，以及这一轮真正展示给用户的商家快照。"""

    turn_index: int = Field(ge=1)
    user_message: str = Field(min_length=1, max_length=4000)
    assistant_message: str | None = Field(default=None, max_length=10000)
    presented_businesses: list[BusinessReference] = Field(
        default_factory=list,
        max_length=100,
    )
    recommendation_snapshot: RecommendationSnapshot | None = None

    @model_validator(mode="after")
    def validate_business_turns(self) -> Self:
        """展示商家的轮次必须和所属对话轮次一致。"""

        if any(
            item.presented_turn_index != self.turn_index
            for item in self.presented_businesses
        ):
            raise ValueError("presented businesses must belong to their history turn")
        if self.recommendation_snapshot is not None:
            ordered_ids = [item.business_id for item in self.presented_businesses]
            if self.recommendation_snapshot.ordered_business_ids != ordered_ids:
                raise ValueError(
                    "recommendation snapshot must match presented business order"
                )
        return self


class CompactSceneSelection(StrictModel):
    """大模型只填写识别出的场景和用户原话依据。"""

    kind: SceneKind
    custom_label: str | None = Field(default=None, min_length=1, max_length=100)
    evidence_text: str = Field(min_length=1, max_length=500)
    evidence_turn_index: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_custom_label(self) -> Self:
        if (self.kind == "custom") != (self.custom_label is not None):
            raise ValueError("only a custom scene uses custom_label")
        return self


class CompactSearchCenter(StrictModel):
    """大模型从用户地点原话得到的近似搜索中心和合理搜索半径。"""

    label: str = Field(min_length=1, max_length=200)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    radius_km: float = Field(gt=0, le=30)
    evidence_text: str = Field(min_length=1, max_length=500)
    evidence_turn_index: int = Field(ge=1)


class CompactHardRequirement(StrictModel):
    """大模型理解出的硬条件，不让它重复填写固定技术字段。"""

    field: HardConstraintField
    operator: HardOperator
    value: RequirementValue
    evidence_text: str = Field(min_length=1, max_length=500)
    evidence_turn_index: int = Field(ge=1)
    supporting_fact_ids: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_fact_ids(self) -> Self:
        if len(self.supporting_fact_ids) != len(set(self.supporting_fact_ids)):
            raise ValueError("supporting fact IDs must be unique")
        return self


class CompactSoftRequirement(StrictModel):
    """大模型理解出的排序要求，只保留语义和先后顺序。"""

    field: RequirementField
    direction: PreferenceDirection
    target_value: RequirementValue | None = None
    priority: int = Field(ge=1, le=100)
    evidence_text: str = Field(min_length=1, max_length=500)
    evidence_turn_index: int = Field(ge=1)
    supporting_fact_ids: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_fact_ids(self) -> Self:
        if len(self.supporting_fact_ids) != len(set(self.supporting_fact_ids)):
            raise ValueError("supporting fact IDs must be unique")
        return self


class CompactOpenRequirement(StrictModel):
    """暂时不能放进现有硬筛选或软排序字段的用户要求。"""

    text: str = Field(min_length=1, max_length=500)
    behavior: Literal["must_have", "prefer", "avoid"]
    priority: int | None = Field(default=None, ge=1, le=100)
    evidence_text: str = Field(min_length=1, max_length=500)
    evidence_turn_index: int = Field(ge=1)
    supporting_fact_ids: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_open_requirement(self) -> Self:
        if (self.behavior == "must_have") == (self.priority is not None):
            raise ValueError(
                "must-have items have no rank; prefer/avoid items require one"
            )
        if len(self.supporting_fact_ids) != len(set(self.supporting_fact_ids)):
            raise ValueError("supporting fact IDs must be unique")
        return self


class PreferenceFusionProposal(StrictModel):
    """大模型最终只输出当前仍然有效的对话需求。"""

    scene: CompactSceneSelection | None = None
    search_center: CompactSearchCenter | None = None
    hard_constraints: list[CompactHardRequirement] = Field(
        default_factory=list,
        max_length=50,
    )
    soft_preferences: list[CompactSoftRequirement] = Field(
        default_factory=list,
        max_length=100,
    )
    open_requirements: list[CompactOpenRequirement] = Field(
        default_factory=list,
        max_length=30,
    )

    @model_validator(mode="after")
    def validate_dialogue_priority(self) -> Self:
        """对话软偏好必须给出唯一连续顺序，禁止用相同名次糊弄先后。"""

        priorities = sorted(item.priority for item in self.soft_preferences)
        if priorities != list(range(1, len(priorities) + 1)):
            raise ValueError(
                "dialogue soft-preference priorities must be unique and contiguous"
            )
        return self


class PreferenceFusionToolCall(StrictModel):
    """大模型在缺少历史商家事实时主动发起的工具调用。"""

    action: Literal["lookup_history_business"]
    arguments: HistoryFactQuery


class BusinessFactsFusionToolCall(StrictModel):
    """大模型拿到历史商家编号后，按需读取新版完整商家属性。"""

    action: Literal["lookup_business_facts"]
    arguments: BusinessFactsQuery


class PreferenceCandidate(StrictModel):
    """程序补齐后的单一来源软偏好，供固定优先级裁决使用。"""

    candidate_id: str = Field(min_length=1, max_length=200)
    source: SourceKind
    preference: SoftPreference

    @model_validator(mode="after")
    def validate_atomic_source(self) -> Self:
        if self.preference.controlling_source != self.source:
            raise ValueError("candidate preference must be controlled by its source")
        if any(item.source != self.source for item in self.preference.sources):
            raise ValueError("candidate preference must contain one source")
        return self


class PreferenceFusionRequest(StrictModel):
    """本轮融合所需输入；场景和画像只在首次或变化时传入。"""

    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    turn_index: int = Field(ge=1)
    query_text: str = Field(min_length=1, max_length=4000)
    request_time: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
    previous_state: UnifiedRecommendationState | None = None
    conversation_history: list[ConversationHistoryTurn] = Field(
        default_factory=list,
        max_length=100,
    )
    scene_baseline: SceneBaseline | None = None
    profile_preferences: ProfilePreferenceSet | None = None
    user_location: GeoPoint | None = None

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if self.previous_state is not None:
            if self.previous_state.user_id != self.user_id:
                raise ValueError("previous state user must match fusion user")
            if self.previous_state.session_id != self.session_id:
                raise ValueError("previous state session must match fusion session")
            if self.turn_index <= self.previous_state.turn_index:
                raise ValueError("current turn must be later than previous state")
        if (
            self.profile_preferences is not None
            and self.profile_preferences.user_id != self.user_id
        ):
            raise ValueError("profile user must match fusion user")
        history_turns = [item.turn_index for item in self.conversation_history]
        if len(history_turns) != len(set(history_turns)):
            raise ValueError("conversation history turns must be unique")
        if any(item >= self.turn_index for item in history_turns):
            raise ValueError("conversation history must precede the current turn")
        return self


class PreferenceFusionAttempt(StrictModel):
    """保存最终结果、工具查询记录和可安全暴露的失败原因。"""

    status: Literal["success", "provider_failure", "invalid_output"]
    raw_json: str | None = Field(default=None, max_length=100000)
    state: UnifiedRecommendationState | None = None
    failure_reason: str | None = Field(default=None, max_length=500)
    model: str | None = None
    latency_ms: float = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    model_call_count: int = Field(default=0, ge=0, le=MAX_TOOL_CALLS + 1)
    tool_observations: list[HistoryFactObservation | BusinessFactsObservation] = Field(
        default_factory=list,
        max_length=MAX_TOOL_CALLS,
    )

    @model_validator(mode="after")
    def validate_attempt(self) -> Self:
        if self.status == "success":
            if self.state is None or self.failure_reason is not None:
                raise ValueError("successful fusion requires state and no failure")
        elif self.state is not None or self.failure_reason is None:
            raise ValueError("failed fusion requires a reason and no state")
        return self


class ChatGenerator(Protocol):
    """融合模块需要的最小大模型调用能力。"""

    def generate(self, messages: list[LLMMessage]) -> LLMCallResult:
        """根据当前消息返回一个严格 JSON 对象。"""


@dataclass
class _ModelLoopResult:
    """大模型查询工具直到给出最终需求后的内部结果。"""

    proposal: PreferenceFusionProposal | None
    raw_json: str | None
    status: Literal["success", "provider_failure", "invalid_output"]
    failure_reason: str | None
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    call_count: int
    observations: list[HistoryFactObservation | BusinessFactsObservation]
    facts: dict[str, HistoryBusinessFact]


@dataclass
class _ActiveGroup:
    """固定优先级裁决后的一条生效候选及同义支持项。"""

    winner: PreferenceCandidate
    supporters: list[PreferenceCandidate] = field(default_factory=list)


@dataclass
class _CandidateDecision:
    """程序对一条来源候选作出的可追踪裁决。"""

    candidate: PreferenceCandidate
    status: Literal["active", "supporting", "suppressed", "shadowed"]
    reason: str
    controller_candidate_id: str | None = None
    hard_constraint_keys: list[str] = field(default_factory=list)


class PreferenceFusion:
    """让模型理解和查事实，让程序负责固定补齐与四来源裁决。"""

    def __init__(
        self,
        generator: ChatGenerator,
        *,
        history_tool: HistoryBusinessFactTool | None = None,
        business_tool: BusinessFactsTool | None = None,
    ) -> None:
        self._generator = generator
        self._history_tool = history_tool or HistoryBusinessFactTool()
        self._business_tool = business_tool

    def fuse(self, request: PreferenceFusionRequest) -> PreferenceFusionAttempt:
        """返回本轮唯一完整状态；任一步失败都不产生半份状态。"""

        started = time.perf_counter()
        result = self._run_model_loop(request)
        latency_ms = (time.perf_counter() - started) * 1000
        if result.status != "success" or result.proposal is None:
            return PreferenceFusionAttempt(
                status=result.status,
                raw_json=result.raw_json,
                failure_reason=result.failure_reason or result.status,
                model=result.model,
                latency_ms=latency_ms,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                model_call_count=result.call_count,
                tool_observations=result.observations,
            )
        try:
            _validate_model_understanding(request, result.proposal, result.facts)
            state = _materialize_state(request, result.proposal, result.facts)
        except (TypeError, ValueError, ValidationError) as exc:
            return PreferenceFusionAttempt(
                status="invalid_output",
                raw_json=result.raw_json,
                failure_reason=_safe_reason(exc),
                model=result.model,
                latency_ms=latency_ms,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                model_call_count=result.call_count,
                tool_observations=result.observations,
            )
        return PreferenceFusionAttempt(
            status="success",
            raw_json=result.raw_json,
            state=state,
            model=result.model,
            latency_ms=latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            model_call_count=result.call_count,
            tool_observations=result.observations,
        )

    def _run_model_loop(self, request: PreferenceFusionRequest) -> _ModelLoopResult:
        """允许大模型按需查询历史事实，直到返回精简后的最终需求。"""

        messages = _messages(
            request,
            business_tool_available=self._business_tool is not None,
        )
        businesses = _all_business_references(request)
        observations: list[HistoryFactObservation | BusinessFactsObservation] = []
        facts: dict[str, HistoryBusinessFact] = {}
        seen_calls: set[str] = set()
        calls: list[LLMCallResult] = []
        last_raw_json: str | None = None

        for _ in range(MAX_TOOL_CALLS + 1):
            call = self._generator.generate(messages)
            calls.append(call)
            if call.status != "success" or call.content is None:
                return _loop_result(
                    status="provider_failure",
                    reason=call.failure_reason or "provider_failure",
                    calls=calls,
                    observations=observations,
                    facts=facts,
                    raw_json=last_raw_json,
                )
            raw_json = _strip_code_fence(call.content)
            last_raw_json = raw_json
            try:
                payload = json.loads(raw_json)
                if not isinstance(payload, dict):
                    raise TypeError("model output must be a JSON object")
                if payload.get("action") == self._history_tool.name:
                    if len(observations) >= MAX_TOOL_CALLS:
                        raise ValueError("model exceeded the history tool call limit")
                    tool_call = PreferenceFusionToolCall.model_validate(payload)
                    call_signature = tool_call.arguments.model_dump_json()
                    if call_signature in seen_calls:
                        raise ValueError("model repeated the same history fact query")
                    seen_calls.add(call_signature)
                    observation = self._history_tool.execute(
                        tool_call.arguments,
                        businesses,
                    )
                    observations.append(observation)
                    if observation.fact is not None:
                        facts[observation.fact.fact_id] = observation.fact
                    messages.extend(
                        [
                            LLMMessage(role="assistant", content=raw_json),
                            LLMMessage(
                                role="user",
                                content=json.dumps(
                                    {
                                        "tool_observation": observation.model_dump(
                                            mode="json"
                                        ),
                                        "instruction": (
                                            "根据查询结果继续；如果仍缺事实可再调用工具，"
                                            "否则输出最终精简需求。"
                                        ),
                                    },
                                    ensure_ascii=False,
                                    sort_keys=True,
                                ),
                            ),
                        ]
                    )
                    continue
                if (
                    payload.get("action") == BusinessFactsTool.name
                    and self._business_tool is not None
                ):
                    if len(observations) >= MAX_TOOL_CALLS:
                        raise ValueError("model exceeded the business tool call limit")
                    tool_call = BusinessFactsFusionToolCall.model_validate(payload)
                    visible_ids = {
                        item.business_id for item in businesses
                    }
                    if not set(tool_call.arguments.business_ids) <= visible_ids:
                        raise ValueError(
                            "business facts may only be queried for visible history"
                        )
                    call_signature = tool_call.arguments.model_dump_json()
                    if call_signature in seen_calls:
                        raise ValueError("model repeated the same business fact query")
                    seen_calls.add(call_signature)
                    observation = self._business_tool.execute(tool_call.arguments)
                    observations.append(observation)
                    messages.extend(
                        [
                            LLMMessage(role="assistant", content=raw_json),
                            LLMMessage(
                                role="user",
                                content=json.dumps(
                                    {
                                        "tool_observation": observation.model_dump(
                                            mode="json"
                                        ),
                                        "instruction": (
                                            "根据真实商家属性继续；信息足够时输出最终精简需求。"
                                        ),
                                    },
                                    ensure_ascii=False,
                                    sort_keys=True,
                                ),
                            ),
                        ]
                    )
                    continue
                proposal = PreferenceFusionProposal.model_validate(payload)
            except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as exc:
                return _loop_result(
                    status="invalid_output",
                    reason=_safe_reason(exc),
                    calls=calls,
                    observations=observations,
                    facts=facts,
                    raw_json=raw_json,
                )
            return _loop_result(
                status="success",
                reason=None,
                calls=calls,
                observations=observations,
                facts=facts,
                raw_json=raw_json,
                proposal=proposal,
            )

        return _loop_result(
            status="invalid_output",
            reason="model did not produce a final proposal",
            calls=calls,
            observations=observations,
            facts=facts,
            raw_json=last_raw_json,
        )


def _messages(
    request: PreferenceFusionRequest,
    *,
    business_tool_available: bool = False,
) -> list[LLMMessage]:
    """给模型完整语境，但把商家详细事实留给可核验工具查询。"""

    system = """你负责把餐厅推荐的多轮对话整理成“当前仍然有效的对话需求”。
你会看到本轮用户原话，以及多轮时仍然有效的历史对话需求和历史展示商家。

最重要的输出边界：
- hard_constraints、soft_preferences、open_requirements 只能整理本轮用户原话和历史 user_message 中的要求。
- 场景基准和长期画像由程序在你输出后直接加入并处理冲突，不会交给你抄写。
- evidence_text 绝对不能取自 scene_baseline、profile_preferences、assistant_message 或程序说明。
- 你只需要输出识别到的 scene 种类；不要输出该场景自带的默认条件和软偏好。

你的职责：
1. 理解用户是在新增、修改、放宽、收紧、删除还是重新排序要求。
2. 最终只输出当前仍有效的对话场景、硬条件、软偏好和未结构化要求；被取消的不要输出。
3. 软偏好只填写先后顺序，不填写强度。用户同一句话明确说“最重要、其次、先……再……”时，priority 必须体现这个顺序。
4. 不复制场景默认值、画像偏好、编号、单位、数据库字段名或来源名称，程序会统一补齐。
5. 不推荐商家，不生成商家分数，不猜商家事实。

地点和到店时间：
- 用户说出城市、区域、商圈、地标或详细地址时，填写 search_center。latitude/longitude 是该地点的大致中心；radius_km 是这个地点合理的附近范围。社区或商圈一般使用1到3公里，城市级地点可以更大。
- search_center 只取自用户地点原话。用户没说新地点时保持 null，程序会沿用上一轮地点或用户定位。
- 用户明确说“今天晚上九点、周六中午、现在”等到店时间时，输出 hard_constraints：field=open_at、operator=equals、value 为目录当地时区的 ISO 日期时间，例如 2026-08-25T21:00:00。
- 用户没有说到店时间时不要输出 open_at；程序会自动使用请求发生时刻。

怎样把常见原话放入现有字段：
- “想吃川菜/日料”等菜系要求：hard_constraints，field=category，operator=any_of，value 使用数据库类别名称列表。对话中的菜系决定候选范围，不允许放进 soft_preferences。
- “辣一点/清淡一点”：field=spiciness，direction=higher/lower。
- “安静/别太吵”：field=quiet_environment，direction=higher。
- “近一点”：field=distance_km，direction=lower。
- “五公里以内”：hard_constraints，field=distance_km，operator=less_than_or_equal，value=5。
- “今晚九点还营业”：hard_constraints，field=open_at，operator=equals，value 使用 ISO 日期时间字符串。
- “价格二档左右”：soft_preferences，field=price_level，direction=closer_to，target_value=2。
- “地道、正宗、本地人才会去”等无法由现有14种特征直接表达的要求：open_requirements，behavior=prefer，保留用户原话供评论检索。
- 只有现有硬条件字段和软偏好字段确实无法表达时，才放 open_requirements；不要把川菜、辣度、安静、距离、价格放进开放要求。

顺序和开放要求格式：
- soft_preferences 的 priority 必须从1开始、不能重复、不能跳号。“最重要”必须排在其他软偏好前面。
- open_requirements 中 behavior=must_have 时 priority 必须是 null；behavior=prefer 或 avoid 时 priority 必须是数字。

必须遵守的排序例子：
- 用户说“想吃川菜，辣一点，安静最重要”，category 必须成为硬条件；软排序中 quiet_environment priority=1，spiciness priority=2。不能把菜系混入软排序，也不能按词语出现顺序把安静排到后面。
- 用户说“先保证安静，在安静程度相近时再选距离近的”，正确顺序是 quiet_environment priority=1，distance_km priority=2。
- 类别目标只能从 allowed_dining_categories 原样复制，例如川菜选择 ["Szechuan"]、日料选择 ["Japanese"]，不要填写中文类别名，也不要自己翻译或创造类别。

历史商家事实查询：
- 当用户用“第几家、那家、上一家、和某家一样”等指代，而且你需要距离、价格、类别、位置或评论特征才能形成要求时，先输出 lookup_history_business 工具调用。
- 只有 available_tool 不为 null，并且 visible history 中确实列出了用户所指的历史商家时才能调用。没有历史展示商家，或者用户没有指代某家商家时，绝对不要调用。
- 由你根据语义决定是否调用、查哪一轮哪一家、需要哪些字段。程序不会匹配任何固定句式。
- 调用时 position 和 business_id 必须且只能填写一个非 null 值：用户说“第几家”就填写 position；明确商家编号时才填写 business_id。
- 工具调用格式严格遵守 tool_call_schema。
- 使用工具事实形成要求时，把返回的 fact_id 放进 supporting_fact_ids。不得编造 fact_id。
- 找不到事实时，不要猜数值；可以把仍然有效但无法结构化的原话放进 open_requirements。
- 需要查询历史商家的地址、评分、评论数、预订、外卖、停车或营业时间等完整属性时，先用历史工具确认商家编号，再调用 lookup_business_facts。只能查询 visible history 中出现的商家编号。

证据规则：
- evidence_text 必须逐字取自对应轮次的用户原话。
- evidence_turn_index 等于当前轮表示本轮明确要求；更早轮表示仍然有效的会话要求。
- 最终输出是完整的当前对话状态，不是只输出本轮变化。

直接输出一个 JSON 对象。需要查询时只输出工具调用；信息齐全时只输出 final_output_schema 对应的精简需求，不要解释或输出思考过程。"""
    payload = {
        "turn_index": request.turn_index,
        "query_text": request.query_text,
        "current_time_in_catalog_timezone": catalog_local_time(
            request.request_time
        ).isoformat(),
        "catalog_timezone": CATALOG_TIME_ZONE,
        "previous_state": _visible_previous_state(request.previous_state),
        "conversation_history": _visible_history(request.conversation_history),
        "available_tool": _history_tool_description(request),
        "available_business_tool": _business_tool_description(
            request,
            enabled=business_tool_available,
        ),
        "allowed_dining_categories": list(
            load_fixed_category_catalog().model_options()
        ),
        "final_output_schema": PreferenceFusionProposal.model_json_schema(),
    }
    return [
        LLMMessage(role="system", content=system),
        LLMMessage(
            role="user",
            content=json.dumps(payload, ensure_ascii=False, sort_keys=True),
        ),
    ]


def _history_tool_description(
    request: PreferenceFusionRequest,
) -> dict[str, object] | None:
    """没有历史展示商家时不向模型宣传一个必然查不到的工具。"""

    if not _all_business_references(request):
        return None
    return {
        "name": HistoryBusinessFactTool.name,
        "description": "按历史展示轮次与位置，或按商家编号，读取真实商家快照字段。",
        "tool_call_schema": PreferenceFusionToolCall.model_json_schema(),
    }


def _business_tool_description(
    request: PreferenceFusionRequest,
    *,
    enabled: bool,
) -> dict[str, object] | None:
    """只有模型确实看得到历史商家编号时才开放完整属性查询。"""

    if not enabled or not _all_business_references(request):
        return None
    return {
        "name": BusinessFactsTool.name,
        "description": "按历史商家编号读取地址、评分、价格、服务属性和每周营业时间。",
        "tool_call_schema": BusinessFactsFusionToolCall.model_json_schema(),
    }


def _visible_previous_state(
    state: UnifiedRecommendationState | None,
) -> dict[str, object] | None:
    """只给模型上一轮对话需求，画像、场景默认和商家事实由程序管理。"""

    if state is None:
        return None
    soft_preferences: list[dict[str, object]] = []
    for preference in state.soft_preferences:
        if preference.controlling_source not in {"current_query", "session"}:
            continue
        serialized = preference.model_dump(mode="json")
        serialized["sources"] = [
            basis.model_dump(mode="json")
            for basis in preference.sources
            if basis.source in {"current_query", "session"}
        ]
        soft_preferences.append(serialized)
    return {
        "revision": state.revision,
        "turn_index": state.turn_index,
        "latest_query_text": state.latest_query_text,
        "scene": None if state.scene is None else state.scene.model_dump(mode="json"),
        "search_center": (
            None
            if state.search_center is None
            else state.search_center.model_dump(mode="json")
        ),
        "hard_constraints": [
            item.model_dump(mode="json") for item in state.hard_constraints
        ],
        "soft_preferences": soft_preferences,
        "open_requirements": [
            item.model_dump(mode="json") for item in state.open_requirements
        ],
    }


def _visible_history(
    turns: list[ConversationHistoryTurn],
) -> list[dict[str, object]]:
    """历史中保留对话和商家身份，不直接暴露距离、价格等工具事实。"""

    return [
        {
            "turn_index": turn.turn_index,
            "user_message": turn.user_message,
            "assistant_message": turn.assistant_message,
            "presented_businesses": [
                {
                    "presented_turn_index": item.presented_turn_index,
                    "position": item.position,
                    "business_id": item.business_id,
                    "business_name": item.business_name,
                }
                for item in turn.presented_businesses
            ],
        }
        for turn in turns
    ]


def _loop_result(
    *,
    status: Literal["success", "provider_failure", "invalid_output"],
    reason: str | None,
    calls: list[LLMCallResult],
    observations: list[HistoryFactObservation | BusinessFactsObservation],
    facts: dict[str, HistoryBusinessFact],
    raw_json: str | None,
    proposal: PreferenceFusionProposal | None = None,
) -> _ModelLoopResult:
    """汇总多次模型调用的用量，避免工具循环丢失统计。"""

    return _ModelLoopResult(
        proposal=proposal,
        raw_json=raw_json,
        status=status,
        failure_reason=reason,
        model=next((item.model for item in reversed(calls) if item.model), None),
        input_tokens=_sum_optional(item.input_tokens for item in calls),
        output_tokens=_sum_optional(item.output_tokens for item in calls),
        call_count=len(calls),
        observations=list(observations),
        facts=dict(facts),
    )


def _sum_optional(values: Iterable[int | None]) -> int | None:
    """有用量信息时求和，供应商完全不返回用量时保留空值。"""

    materialized = [value for value in values if value]
    return sum(materialized) if materialized else None


def _validate_model_understanding(
    request: PreferenceFusionRequest,
    proposal: PreferenceFusionProposal,
    facts: dict[str, HistoryBusinessFact],
) -> None:
    """核对用户原话和工具事实引用，不用代码重新解释语言。"""

    requirements: list[
        CompactHardRequirement | CompactSoftRequirement | CompactOpenRequirement
    ] = [
        *proposal.hard_constraints,
        *proposal.soft_preferences,
        *proposal.open_requirements,
    ]
    for item in requirements:
        _validate_dialogue_evidence(
            request,
            item.evidence_text,
            item.evidence_turn_index,
        )
        unknown_facts = set(item.supporting_fact_ids) - set(facts)
        if unknown_facts:
            raise ValueError("requirement references a fact not returned by a tool")
    if any(item.field == "category" for item in proposal.soft_preferences):
        raise ValueError("dialogue category requirements must be hard constraints")
    category_catalog = load_fixed_category_catalog()
    for item in proposal.hard_constraints:
        if item.field != "category":
            continue
        values = item.value
        if not isinstance(values, list) or any(
            not category_catalog.is_selectable(value) for value in values
        ):
            raise ValueError(
                "category hard constraints must use the supplied Yelp category list"
            )
    if proposal.scene is not None:
        _validate_dialogue_evidence(
            request,
            proposal.scene.evidence_text,
            proposal.scene.evidence_turn_index,
        )
    if proposal.search_center is not None:
        _validate_dialogue_evidence(
            request,
            proposal.search_center.evidence_text,
            proposal.search_center.evidence_turn_index,
        )


def _validate_dialogue_evidence(
    request: PreferenceFusionRequest,
    text: str,
    turn_index: int,
) -> None:
    """每条需求都必须能回到用户某一轮真正说过的原话。"""

    if turn_index == request.turn_index:
        if text not in request.query_text:
            raise ValueError("current-query evidence must appear in the current message")
        return
    if turn_index >= request.turn_index:
        raise ValueError("session evidence must come from an earlier turn")
    known_texts = _known_user_texts(request).get(turn_index, [])
    if not any(text in known for known in known_texts):
        raise ValueError("session evidence must appear in recorded user history")


def _known_user_texts(request: PreferenceFusionRequest) -> dict[int, list[str]]:
    """收集用户原话及上一状态中已经核验过的用户依据。"""

    values: dict[int, list[str]] = {}
    for turn in request.conversation_history:
        values.setdefault(turn.turn_index, []).append(turn.user_message)
    previous = request.previous_state
    if previous is None:
        return values
    values.setdefault(previous.turn_index, []).append(previous.latest_query_text)
    requirements = [
        *previous.hard_constraints,
        *previous.soft_preferences,
        *previous.open_requirements,
    ]
    bases = [basis for item in requirements for basis in item.sources]
    bases.extend(
        basis
        for memory in previous.preference_memory
        for basis in memory.preference.sources
    )
    if previous.scene is not None:
        bases.append(previous.scene.basis)
    for basis in bases:
        if basis.source in {"current_query", "session"} and basis.turn_index is not None:
            values.setdefault(basis.turn_index, []).append(basis.text)
    return values


def _source_for_turn(request: PreferenceFusionRequest, turn_index: int) -> SourceKind:
    """来源不让模型填写，由证据轮次机械确定。"""

    if turn_index == request.turn_index:
        return "current_query"
    if turn_index < request.turn_index:
        return "session"
    raise ValueError("evidence turn cannot be in the future")


def _basis(
    request: PreferenceFusionRequest,
    *,
    text: str,
    turn_index: int,
    strength: PreferenceStrength | None = None,
) -> RequirementBasis:
    """把精简证据补齐为统一状态使用的标准来源依据。"""

    return RequirementBasis(
        source=_source_for_turn(request, turn_index),
        text=text,
        turn_index=turn_index,
        preference_strength=strength,
    )


def _stable_suffix(payload: object, length: int = 12) -> str:
    """用需求语义生成稳定短编号，避免依赖模型随意起名。"""

    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:length]


def _hard_group(operator: HardOperator) -> str:
    """把比较符归到同一个覆盖槽位。"""

    if operator in {"less_than", "less_than_or_equal"}:
        return "max"
    if operator in {"greater_than", "greater_than_or_equal"}:
        return "min"
    if operator == "any_of":
        return "include"
    if operator == "none_of":
        return "exclude"
    return "exact"


def _materialize_hard_requirement(
    request: PreferenceFusionRequest,
    compact: CompactHardRequirement,
    facts: dict[str, HistoryBusinessFact],
) -> HardConstraint:
    """补齐硬条件的编号、单位、商家字段、来源和可核验推导。"""

    basis = _basis(
        request,
        text=compact.evidence_text,
        turn_index=compact.evidence_turn_index,
    )
    suffix = _stable_suffix(
        [compact.field, compact.operator, compact.value, compact.evidence_turn_index]
    )
    return HardConstraint(
        key=f"hard.{compact.field}.{_hard_group(compact.operator)}.{suffix}",
        field=compact.field,
        operator=compact.operator,
        value=compact.value,
        unit=requirement_unit_for(compact.field),
        merchant_feature=merchant_feature_for(compact.field),
        controlling_source=cast(
            Literal["current_query", "session"],
            basis.source,
        ),
        sources=[basis],
        derivation=_distance_derivation(compact, facts),
    )


def _distance_derivation(
    compact: CompactHardRequirement,
    facts: dict[str, HistoryBusinessFact],
) -> ConstraintDerivation | None:
    """当距离阈值正好来自工具事实时，机械保存其商家来源。"""

    if compact.field != "distance_km" or not isinstance(compact.value, (int, float)):
        return None
    for fact_id in compact.supporting_fact_ids:
        fact = facts[fact_id]
        if fact.distance_km is not None and float(compact.value) == fact.distance_km:
            return ConstraintDerivation(
                kind="reference_business_distance",
                reference_business_id=fact.business_id,
                source_value=fact.distance_km,
                source_unit="kilometer",
            )
    return None


def _materialize_dialogue_candidate(
    request: PreferenceFusionRequest,
    compact: CompactSoftRequirement,
) -> PreferenceCandidate:
    """把一条精简软偏好补齐为单一对话来源候选。"""

    source = _source_for_turn(request, compact.evidence_turn_index)
    strength = _dialogue_strength(request, compact)
    basis = _basis(
        request,
        text=compact.evidence_text,
        turn_index=compact.evidence_turn_index,
        strength=strength,
    )
    semantic = [
        compact.field,
        compact.direction,
        compact.target_value,
        compact.evidence_turn_index,
        compact.evidence_text,
    ]
    suffix = _stable_suffix(semantic)
    preference = SoftPreference(
        key=f"dialogue.soft.{compact.field}.{suffix}",
        field=compact.field,
        direction=compact.direction,
        target_value=compact.target_value,
        preference_strength=strength,
        priority=compact.priority,
        merchant_feature=merchant_feature_for(compact.field),
        controlling_source=source,
        sources=[basis],
    )
    return PreferenceCandidate(
        candidate_id=f"dialogue:{compact.evidence_turn_index}:{suffix}",
        source=source,
        preference=preference,
    )


def _dialogue_strength(
    request: PreferenceFusionRequest,
    compact: CompactSoftRequirement,
) -> PreferenceStrength:
    """本轮明确要求统一为100；历史要求沿用上一状态，不让模型猜数值。"""

    if compact.evidence_turn_index == request.turn_index:
        return 100
    previous = request.previous_state
    if previous is not None:
        remembered = [
            item.preference
            for item in previous.preference_memory
            if item.source in {"current_query", "session"}
        ]
        remembered.extend(
            item
            for item in previous.soft_preferences
            if item.controlling_source in {"current_query", "session"}
        )
        for preference in remembered:
            if (
                preference.field == compact.field
                and preference.direction == compact.direction
                and preference.target_value == compact.target_value
                and any(
                    basis.turn_index == compact.evidence_turn_index
                    and basis.text == compact.evidence_text
                    for basis in preference.sources
                )
            ):
                return preference.preference_strength
    # 没有上一状态可继承时，它仍是用户历史原话，而不是画像或场景推测。
    return 100


def _persistent_candidates(
    request: PreferenceFusionRequest,
    scene: SceneSelection | None,
) -> tuple[list[DefaultConstraint], list[PreferenceCandidate]]:
    """场景和画像由程序从可信输入恢复，不要求模型重新抄写。"""

    defaults, scene_preferences = _scene_requirements(request, scene)
    profile_preferences = _profile_requirements(request)
    candidates = [
        PreferenceCandidate(
            candidate_id=f"scene:{scene.kind}:{item.key}",
            source="scene",
            preference=item.model_copy(deep=True),
        )
        for item in scene_preferences
    ]
    candidates.extend(
        PreferenceCandidate(
            candidate_id=f"profile:{item.key}",
            source="user_profile",
            preference=item.model_copy(deep=True),
        )
        for item in profile_preferences
    )
    return defaults, candidates


def _scene_requirements(
    request: PreferenceFusionRequest,
    scene: SceneSelection | None,
) -> tuple[list[DefaultConstraint], list[SoftPreference]]:
    """取得当前场景对应基准；没有场景就明确返回空。"""

    if scene is None:
        return [], []
    if (
        request.scene_baseline is not None
        and request.scene_baseline.scene == scene.kind
    ):
        return (
            [item.model_copy(deep=True) for item in request.scene_baseline.default_constraints],
            [item.model_copy(deep=True) for item in request.scene_baseline.soft_preferences],
        )
    previous = request.previous_state
    if previous is not None and previous.scene is not None and previous.scene.kind == scene.kind:
        return (
            [
                item.model_copy(deep=True)
                for item in previous.default_constraints
                if item.controlling_source == "scene"
            ],
            [
                item.preference.model_copy(deep=True)
                for item in previous.preference_memory
                if item.source == "scene"
            ],
        )
    if scene.kind == "custom":
        return [], []

    # 六个内置场景已经是项目基准；场景中途变化时也能直接取到对应版本。
    from yelp_agent.recommendation_v2.scenes import get_scene_baseline

    baseline = get_scene_baseline(cast(BaselineSceneKind, scene.kind))
    return (
        [item.model_copy(deep=True) for item in baseline.default_constraints],
        [item.model_copy(deep=True) for item in baseline.soft_preferences],
    )


def _profile_requirements(request: PreferenceFusionRequest) -> list[SoftPreference]:
    """首次使用画像输入，后续轮次从上一状态完整恢复画像候选。"""

    if request.profile_preferences is not None:
        return [
            item.model_copy(deep=True)
            for item in request.profile_preferences.soft_preferences
        ]
    if request.previous_state is None:
        return []
    return [
        item.preference.model_copy(deep=True)
        for item in request.previous_state.preference_memory
        if item.source == "user_profile"
    ]


def _materialize_scene(
    request: PreferenceFusionRequest,
    compact: CompactSceneSelection | None,
) -> SceneSelection | None:
    """根据场景证据轮次补齐来源。"""

    if compact is None:
        return None
    return SceneSelection(
        kind=compact.kind,
        custom_label=compact.custom_label,
        basis=_basis(
            request,
            text=compact.evidence_text,
            turn_index=compact.evidence_turn_index,
        ),
    )


def _materialize_search_center(
    request: PreferenceFusionRequest,
    compact: CompactSearchCenter | None,
) -> SearchCenter | None:
    """新地点覆盖旧地点；本轮没说地点时沿用上一轮搜索中心。"""

    if compact is not None:
        return SearchCenter(
            kind="named_place",
            label=compact.label,
            location=GeoPoint(
                latitude=compact.latitude,
                longitude=compact.longitude,
            ),
        )
    previous = request.previous_state
    if previous is None or previous.search_center is None:
        return None
    return previous.search_center.model_copy(deep=True)


def _materialize_open_requirements(
    request: PreferenceFusionRequest,
    compact_items: list[CompactOpenRequirement],
) -> list[OpenRequirement]:
    """补齐无法结构化要求的稳定编号与来源。"""

    result: list[OpenRequirement] = []
    for item in compact_items:
        basis = _basis(
            request,
            text=item.evidence_text,
            turn_index=item.evidence_turn_index,
        )
        suffix = _stable_suffix(
            [item.text, item.behavior, item.evidence_turn_index]
        )
        result.append(
            OpenRequirement(
                key=f"open.{item.behavior}.{suffix}",
                text=item.text,
                behavior=item.behavior,
                priority=item.priority,
                controlling_source=cast(
                    Literal["current_query", "session"],
                    basis.source,
                ),
                sources=[basis],
            )
        )
    return result


def _hard_slot(constraint: HardConstraint) -> tuple[object, ...]:
    """把同字段硬条件划分为最大值、最小值、包含、排除和精确值。"""

    return constraint.field, _hard_group(constraint.operator)


def _resolve_hard_constraints(
    constraints: list[HardConstraint],
) -> list[HardConstraint]:
    """同一硬条件槽位固定由当前问题覆盖历史会话。"""

    grouped: dict[tuple[object, ...], list[HardConstraint]] = {}
    for item in constraints:
        grouped.setdefault(_hard_slot(item), []).append(item)
    result: list[HardConstraint] = []
    for slot, values in grouped.items():
        values.sort(
            key=lambda item: (-_SOURCE_PRIORITY[item.controlling_source], item.key)
        )
        highest = _SOURCE_PRIORITY[values[0].controlling_source]
        same_level = [
            item
            for item in values
            if _SOURCE_PRIORITY[item.controlling_source] == highest
        ]
        if len(same_level) != 1:
            raise ValueError(f"multiple hard constraints compete for slot {slot}")
        result.append(values[0].model_copy(deep=True))
    result.sort(key=lambda item: (item.field, item.key))
    return result


def _resolve_default_constraints(
    defaults: list[DefaultConstraint],
    hard_constraints: list[HardConstraint],
) -> list[DefaultConstraint]:
    """明确硬条件出现后，移除同字段较弱的场景默认条件。"""

    explicit_fields = {item.field for item in hard_constraints}
    return sorted(
        [
            item.model_copy(deep=True)
            for item in defaults
            if item.field not in explicit_fields
        ],
        key=lambda item: (item.field, item.key),
    )


def _target_set(preference: SoftPreference) -> set[str]:
    """读取选择型偏好的目标集合。"""

    if not isinstance(preference.target_value, list):
        raise TypeError("choice preference requires a list target")
    return set(preference.target_value)


def _soft_signature(preference: SoftPreference) -> tuple[object, ...]:
    """用字段、方向和目标判断语义，不用模型生成的编号判断。"""

    return (
        preference.field,
        preference.direction,
        json.dumps(preference.target_value, ensure_ascii=False, sort_keys=True),
    )


def _semantic_relation(
    controller: PreferenceCandidate,
    candidate: PreferenceCandidate,
) -> SemanticRelation:
    """判断两条规范化偏好是同义、冲突、遮蔽还是独立。"""

    left = controller.preference
    right = candidate.preference
    if left.field != right.field:
        return "independent"
    if left.field in _DIRECTIONAL_FIELDS:
        return "same" if left.direction == right.direction else "conflict"
    if left.field in _CHOICE_FIELDS:
        if _target_set(left) == _target_set(right):
            return "same" if left.direction == right.direction else "conflict"
        if (
            _SOURCE_PRIORITY[controller.source]
            > _SOURCE_PRIORITY[candidate.source]
            and left.direction == "match"
        ):
            return "shadow"
        return "independent"
    return "same" if _soft_signature(left) == _soft_signature(right) else "conflict"


def _numeric_satisfies(operator: str, hard_value: float, target: float) -> bool:
    """检查明确软目标是否落在硬条件允许范围内。"""

    operations = {
        "equals": target == hard_value,
        "less_than": target < hard_value,
        "less_than_or_equal": target <= hard_value,
        "greater_than": target > hard_value,
        "greater_than_or_equal": target >= hard_value,
    }
    if operator not in operations:
        raise ValueError(f"unsupported numeric operator: {operator}")
    return operations[operator]


def _hard_soft_compatible(
    hard: HardConstraint,
    preference: SoftPreference,
) -> bool:
    """硬条件控制边界；边界内仍可保留不冲突的排序偏好。"""

    if hard.field != preference.field:
        return True
    if hard.field == "distance_km":
        return True
    if hard.field == "price_level":
        if preference.direction != "closer_to":
            return True
        target = preference.target_value
        if not isinstance(target, (int, float)) or isinstance(target, bool):
            return False
        return _numeric_satisfies(
            hard.operator,
            float(cast(int | float, hard.value)),
            float(target),
        )
    if hard.field in _CHOICE_FIELDS:
        targets = _target_set(preference)
        hard_targets = set(cast(list[str], hard.value))
        if preference.direction == "match":
            if hard.operator in {"any_of", "all_of"}:
                return bool(targets.intersection(hard_targets))
            return not targets.intersection(hard_targets)
        if preference.direction == "avoid":
            if hard.operator in {"any_of", "all_of"}:
                return not hard_targets.issubset(targets)
            return True
    if hard.field in _BOOLEAN_FIELDS:
        target = cast(bool, preference.target_value)
        return (
            target == hard.value
            if preference.direction == "match"
            else target != hard.value
        )
    return True


def _candidate_sort_key(candidate: PreferenceCandidate) -> tuple[object, ...]:
    """跨来源固定分层，同来源使用大模型或基准给出的内部顺序。"""

    return (
        -_SOURCE_PRIORITY[candidate.source],
        candidate.preference.priority,
        -candidate.preference.preference_strength,
        candidate.candidate_id,
    )


def _resolve_preferences(
    hard_constraints: list[HardConstraint],
    candidates: list[PreferenceCandidate],
) -> tuple[list[_ActiveGroup], list[_CandidateDecision]]:
    """代码只执行固定来源优先级、硬软兼容和同义合并。"""

    groups: list[_ActiveGroup] = []
    decisions: list[_CandidateDecision] = []
    for candidate in sorted(candidates, key=_candidate_sort_key):
        hard_conflicts = [
            item
            for item in hard_constraints
            if not _hard_soft_compatible(item, candidate.preference)
        ]
        if hard_conflicts:
            decisions.append(
                _CandidateDecision(
                    candidate=candidate,
                    status="suppressed",
                    reason="与当前生效硬条件冲突",
                    hard_constraint_keys=[item.key for item in hard_conflicts],
                )
            )
            continue
        for group in groups:
            relation = _semantic_relation(group.winner, candidate)
            if relation == "independent":
                continue
            if relation == "same":
                group.supporters.append(candidate)
                decisions.append(
                    _CandidateDecision(
                        candidate=candidate,
                        status="supporting",
                        reason="与控制偏好表达相同意思",
                        controller_candidate_id=group.winner.candidate_id,
                    )
                )
            elif relation == "shadow":
                decisions.append(
                    _CandidateDecision(
                        candidate=candidate,
                        status="shadowed",
                        reason="更高来源已经明确选择同类目标",
                        controller_candidate_id=group.winner.candidate_id,
                    )
                )
            else:
                decisions.append(
                    _CandidateDecision(
                        candidate=candidate,
                        status="suppressed",
                        reason="与更高来源或同来源更高顺序偏好冲突",
                        controller_candidate_id=group.winner.candidate_id,
                    )
                )
            break
        else:
            groups.append(_ActiveGroup(winner=candidate))
            decisions.append(
                _CandidateDecision(
                    candidate=candidate,
                    status="active",
                    reason="当前偏好领域中控制权最高",
                )
            )
    groups.sort(key=lambda item: _candidate_sort_key(item.winner))
    return groups, decisions


def _merge_bases(bases: list[RequirementBasis]) -> list[RequirementBasis]:
    """去重并按固定来源顺序保存支持依据。"""

    unique: dict[tuple[object, ...], RequirementBasis] = {}
    for basis in bases:
        signature = (
            basis.source,
            basis.turn_index,
            basis.text,
            basis.profile_score,
            basis.profile_last_confirmed,
        )
        unique[signature] = basis.model_copy(deep=True)
    return sorted(
        unique.values(),
        key=lambda item: (
            -_SOURCE_PRIORITY[item.source],
            -(item.turn_index or 0),
            item.text,
        ),
    )


def _materialize_preference(
    group: _ActiveGroup,
    priority: int,
) -> SoftPreference:
    """把同义来源合并成一条最终生效偏好。"""

    sources = _merge_bases(
        [
            basis
            for candidate in [group.winner, *group.supporters]
            for basis in candidate.preference.sources
        ]
    )
    controller = group.winner.source
    strength = cast(
        PreferenceStrength,
        max(
            basis.preference_strength
            for basis in sources
            if basis.source == controller and basis.preference_strength is not None
        ),
    )
    base = group.winner.preference
    return SoftPreference(
        key=base.key,
        field=base.field,
        direction=base.direction,
        target_value=base.target_value,
        preference_strength=strength,
        priority=priority,
        merchant_feature=base.merchant_feature,
        controlling_source=controller,
        sources=sources,
    )


def _materialize_memory(
    decisions: list[_CandidateDecision],
) -> list[PreferenceMemoryItem]:
    """把所有来源候选及裁决结果完整保存到下一轮状态。"""

    return [
        PreferenceMemoryItem(
            candidate_id=item.candidate.candidate_id,
            source=item.candidate.source,
            preference=item.candidate.preference.model_copy(deep=True),
            status=item.status,
            reason=item.reason,
            controller_candidate_id=item.controller_candidate_id,
            hard_constraint_keys=item.hard_constraint_keys,
        )
        for item in sorted(
            decisions,
            key=lambda value: _candidate_sort_key(value.candidate),
        )
    ]


def _all_business_references(
    request: PreferenceFusionRequest,
) -> list[BusinessReference]:
    """把上一状态和历史记录中的商家快照合并成工具与最终状态的事实源。"""

    values: dict[tuple[int, int], BusinessReference] = {}
    if request.previous_state is not None:
        for item in request.previous_state.referenced_businesses:
            values[(item.presented_turn_index, item.position)] = item.model_copy(
                deep=True
            )
    for turn in request.conversation_history:
        for item in turn.presented_businesses:
            values[(item.presented_turn_index, item.position)] = item.model_copy(
                deep=True
            )
    return [values[key] for key in sorted(values)]


def _normalize_ranked_requirements(
    soft_preferences: list[SoftPreference],
    open_requirements: list[OpenRequirement],
) -> tuple[list[SoftPreference], list[OpenRequirement]]:
    """显式长尾偏好和普通软偏好遵守同一来源优先级与连续顺序。"""

    ranked_open = [item for item in open_requirements if item.priority is not None]
    must_have = [item for item in open_requirements if item.priority is None]
    combined: list[tuple[str, SoftPreference | OpenRequirement]] = [
        ("soft", item) for item in soft_preferences
    ]
    combined.extend(("open", item) for item in ranked_open)
    combined.sort(
        key=lambda pair: (
            -_SOURCE_PRIORITY[pair[1].controlling_source],
            pair[1].priority or 10_000,
            0 if pair[0] == "soft" else 1,
            pair[1].key,
        )
    )
    normalized_soft: list[SoftPreference] = []
    normalized_open: list[OpenRequirement] = []
    for priority, (kind, item) in enumerate(combined, start=1):
        if kind == "soft":
            normalized_soft.append(
                item.model_copy(update={"priority": priority}, deep=True)  # type: ignore[union-attr]
            )
        else:
            normalized_open.append(
                item.model_copy(update={"priority": priority}, deep=True)  # type: ignore[union-attr]
            )
    normalized_open.extend(item.model_copy(deep=True) for item in must_have)
    return normalized_soft, normalized_open


def _materialize_state(
    request: PreferenceFusionRequest,
    proposal: PreferenceFusionProposal,
    facts: dict[str, HistoryBusinessFact],
) -> UnifiedRecommendationState:
    """补齐固定字段、执行四来源规则，生成下游只读的完整状态。"""

    scene = _materialize_scene(request, proposal.scene)
    compact_hard = list(proposal.hard_constraints)
    if (
        proposal.search_center is not None
        and not any(item.field == "distance_km" for item in compact_hard)
    ):
        # 地点只有坐标还不能限制候选范围。大模型同时给出地点尺度，程序把它
        # 补成真正可执行的距离硬条件，后续仍走同一个数据库过滤入口。
        compact_hard.append(
            CompactHardRequirement(
                field="distance_km",
                operator="less_than_or_equal",
                value=proposal.search_center.radius_km,
                evidence_text=proposal.search_center.evidence_text,
                evidence_turn_index=proposal.search_center.evidence_turn_index,
            )
        )
    hard_constraints = _resolve_hard_constraints(
        [
            _materialize_hard_requirement(request, item, facts)
            for item in compact_hard
        ]
    )
    defaults, persistent = _persistent_candidates(request, scene)
    default_constraints = _resolve_default_constraints(defaults, hard_constraints)
    dialogue = [
        _materialize_dialogue_candidate(request, item)
        for item in proposal.soft_preferences
    ]
    groups, decisions = _resolve_preferences(
        hard_constraints,
        [*dialogue, *persistent],
    )
    soft_preferences = [
        _materialize_preference(group, priority)
        for priority, group in enumerate(groups, start=1)
    ]
    open_requirements = _materialize_open_requirements(
        request,
        proposal.open_requirements,
    )
    soft_preferences, open_requirements = _normalize_ranked_requirements(
        soft_preferences,
        open_requirements,
    )
    previous = request.previous_state
    user_location = (
        request.user_location
        if request.user_location is not None
        else (None if previous is None else previous.user_location)
    )
    search_center = _materialize_search_center(request, proposal.search_center)
    return UnifiedRecommendationState(
        user_id=request.user_id,
        session_id=request.session_id,
        revision=1 if previous is None else previous.revision + 1,
        turn_index=request.turn_index,
        latest_query_text=request.query_text,
        user_location=user_location,
        search_center=search_center,
        scene=scene,
        hard_constraints=hard_constraints,
        default_constraints=default_constraints,
        soft_preferences=soft_preferences,
        preference_memory=_materialize_memory(decisions),
        open_requirements=open_requirements,
        referenced_businesses=_all_business_references(request),
    )


def _strip_code_fence(value: str) -> str:
    """兼容模型偶尔在 JSON 外包一层代码块。"""

    text = value.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return text


def _safe_reason(exc: Exception) -> str:
    """只暴露简短结构错误，不记录模型的隐藏推理。"""

    if isinstance(exc, ValidationError) and exc.errors():
        error = exc.errors(include_url=False)[0]
        location = ".".join(str(item) for item in error.get("loc", ())) or "root"
        return f"schema_validation_failed:{location}:{error.get('type', 'invalid')}"
    if isinstance(exc, json.JSONDecodeError):
        return "invalid_json"
    return str(exc)[:500] or "invalid_output"
