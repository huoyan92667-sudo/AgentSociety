"""调用大模型把当前用户问题编译成统一要求，不执行商家推荐。"""

from __future__ import annotations

import json
from typing import Literal, Protocol, Self

from pydantic import Field, ValidationError, model_validator

from yelp_agent.agent.llm import LLMCallResult, LLMMessage
from yelp_agent.models import StrictModel
from yelp_agent.recommendation_v2.schema import (
    BaselineSceneKind,
    HardConstraint,
    HardConstraintField,
    HardOperator,
    MerchantFeature,
    PreferenceDirection,
    PreferenceStrength,
    RequirementBasis,
    RequirementField,
    RequirementUnit,
    RequirementValue,
    SceneSelection,
    SoftPreference,
)

PROMPT_VERSION = "recommendation-v2-query-compiler-v1"

_FIELD_FEATURE: dict[RequirementField, MerchantFeature] = {
    "category": "categories",
    "distance_km": "coordinates",
    "price_level": "price_level",
    "business_id": "business_id",
    "area": "area",
    "food_quality": "food_quality",
    "service": "service",
    "price_value": "price_value",
    "quiet_environment": "quiet_environment",
    "crowded": "crowded",
    "queue_time": "queue_time",
    "portion_size": "portion_size",
    "parking": "parking",
    "pet_friendly": "pet_friendly",
    "family_friendly": "family_friendly",
    "date_suitable": "date_suitable",
    "group_suitable": "group_suitable",
    "spiciness": "spiciness",
    "cleanliness": "cleanliness",
}
_HARD_FIELD_UNIT: dict[HardConstraintField, RequirementUnit] = {
    "category": "category",
    "distance_km": "kilometer",
    "price_level": "price_level",
    "business_id": "business_id",
    "area": "area",
}


class QueryCompilerRequest(StrictModel):
    """交给问题编译器的一条当前用户问题。"""

    query_text: str = Field(min_length=1, max_length=2000)
    turn_index: int = Field(default=1, ge=1)


class ExtractedScene(StrictModel):
    """大模型从当前问题中识别出的一个场景。"""

    kind: BaselineSceneKind
    evidence_span: str = Field(min_length=1, max_length=500)


class ExtractedHardConstraint(StrictModel):
    """大模型抽取的一条用户明确结构化硬条件。"""

    field: HardConstraintField
    operator: HardOperator
    value: RequirementValue = Field(union_mode="left_to_right")
    evidence_span: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        """硬条件只接受可直接核对的字段和值形状。"""

        if self.field in {"category", "business_id", "area"}:
            if self.operator not in {"any_of", "none_of"}:
                raise ValueError("collection hard fields require any_of or none_of")
            if not isinstance(self.value, list) or not self.value:
                raise ValueError("collection hard fields require a nonempty list")
        elif not isinstance(self.value, (int, float)) or isinstance(self.value, bool):
            raise ValueError("distance and price hard fields require a number")
        if self.field == "price_level" and (
            not isinstance(self.value, int) or not 1 <= self.value <= 4
        ):
            raise ValueError("price level must be an integer from 1 to 4")
        if self.field == "distance_km" and float(self.value) < 0:
            raise ValueError("distance cannot be negative")
        return self


class ExtractedSoftPreference(StrictModel):
    """大模型抽取的一条当前软偏好及其排序位置。"""

    field: RequirementField
    direction: PreferenceDirection
    target_value: RequirementValue | None = Field(
        default=None,
        union_mode="left_to_right",
    )
    preference_strength: PreferenceStrength
    priority: int = Field(ge=1, le=100)
    evidence_span: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_direction(self) -> Self:
        """明确数值方向，避免把“越近越好”误写成接近某个未知距离。"""

        needs_target = self.direction in {"closer_to", "match", "avoid"}
        if needs_target != (self.target_value is not None):
            raise ValueError("closer_to, match, and avoid require target_value")
        if self.field == "distance_km" and self.direction != "lower":
            raise ValueError("nearer distance must use lower with a null target")
        return self


class QueryCompilerProposal(StrictModel):
    """大模型必须直接返回的 JSON 形状。"""

    scene: ExtractedScene | None = None
    hard_constraints: list[ExtractedHardConstraint] = Field(
        default_factory=list,
        max_length=20,
    )
    soft_preferences: list[ExtractedSoftPreference] = Field(
        default_factory=list,
        max_length=20,
    )

    @model_validator(mode="after")
    def validate_proposal(self) -> Self:
        """保证软偏好顺序连续，并阻止同一条件被重复输出。"""

        priorities = sorted(item.priority for item in self.soft_preferences)
        if priorities != list(range(1, len(priorities) + 1)):
            raise ValueError("soft preference priorities must be contiguous from 1")
        hard_keys = [
            (item.field, item.operator, json.dumps(item.value, ensure_ascii=False))
            for item in self.hard_constraints
        ]
        if len(hard_keys) != len(set(hard_keys)):
            raise ValueError("hard constraints must be unique")
        soft_fields = [item.field for item in self.soft_preferences]
        if len(soft_fields) != len(set(soft_fields)):
            raise ValueError("one query may rank each soft field only once")
        return self


class CompiledQuery(StrictModel):
    """大模型结果通过检查并补齐执行字段后的最终当前问题。"""

    schema_version: Literal["1.0.0"] = "1.0.0"
    query_text: str = Field(min_length=1, max_length=2000)
    turn_index: int = Field(ge=1)
    scene: SceneSelection | None = None
    hard_constraints: list[HardConstraint] = Field(default_factory=list, max_length=20)
    soft_preferences: list[SoftPreference] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def validate_compiled_query(self) -> Self:
        """保证最终条件键唯一，排序位置从1开始连续。"""

        keys = [item.key for item in self.hard_constraints]
        keys.extend(item.key for item in self.soft_preferences)
        if len(keys) != len(set(keys)):
            raise ValueError("compiled query requirement keys must be unique")
        priorities = sorted(item.priority for item in self.soft_preferences)
        if priorities != list(range(1, len(priorities) + 1)):
            raise ValueError("compiled preference priorities must be contiguous from 1")
        return self


class QueryCompilerAttempt(StrictModel):
    """一次模型调用的可检查结果，保留模型原始 JSON 和最终结果。"""

    status: Literal["success", "provider_failure", "invalid_output"]
    raw_json: str | None = Field(default=None, max_length=30000)
    compiled_query: CompiledQuery | None = None
    failure_reason: str | None = Field(default=None, max_length=500)
    model: str | None = None
    latency_ms: float = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_attempt(self) -> Self:
        """成功必须有最终结果，失败必须说明原因。"""

        if self.status == "success":
            if self.compiled_query is None or self.failure_reason is not None:
                raise ValueError(
                    "successful compilation requires output and no failure"
                )
        elif self.compiled_query is not None or self.failure_reason is None:
            raise ValueError("failed compilation requires a reason and no output")
        return self


class ChatGenerator(Protocol):
    """问题编译器所需的最小模型调用能力，测试时可以替换。"""

    def generate(self, messages: list[LLMMessage]) -> LLMCallResult: ...


class QueryCompiler:
    """让大模型只理解当前问题，再把结果转换成统一结构。"""

    def __init__(self, generator: ChatGenerator) -> None:
        self._generator = generator

    def compile(self, request: QueryCompilerRequest) -> QueryCompilerAttempt:
        """调用一次大模型，检查 JSON，并补齐可执行字段。"""

        call = self._generator.generate(_messages(request))
        if call.status != "success" or call.content is None:
            return QueryCompilerAttempt(
                status="provider_failure",
                failure_reason=call.failure_reason or "provider_failure",
                model=call.model,
                latency_ms=call.latency_ms,
                input_tokens=call.input_tokens,
                output_tokens=call.output_tokens,
            )

        raw_json = _strip_code_fence(call.content)
        try:
            raw_payload = json.loads(raw_json)
            proposal = QueryCompilerProposal.model_validate(raw_payload)
            _validate_evidence_spans(request.query_text, proposal)
            compiled = _to_compiled_query(request, proposal)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            return QueryCompilerAttempt(
                status="invalid_output",
                raw_json=raw_json,
                failure_reason=_safe_validation_reason(exc),
                model=call.model,
                latency_ms=call.latency_ms,
                input_tokens=call.input_tokens,
                output_tokens=call.output_tokens,
            )
        return QueryCompilerAttempt(
            status="success",
            raw_json=raw_json,
            compiled_query=compiled,
            model=call.model,
            latency_ms=call.latency_ms,
            input_tokens=call.input_tokens,
            output_tokens=call.output_tokens,
        )


def _messages(request: QueryCompilerRequest) -> list[LLMMessage]:
    """构造只允许抽取要求、不允许推荐商家的提示内容。"""

    system = """你是餐厅推荐系统的问题编译器，只负责理解当前用户问题。
你只能抽取：场景、用户明确说出的结构化硬条件、当前软偏好、软偏好优先级。
绝对不要推荐商家，不要输出商家名称，不要解释推理过程，不要加入场景默认值。

场景只能是 casual、date、business、friends、family、solo 之一；用户没有表达场景时必须返回 null，不能猜成 casual。
硬条件只能使用 category、distance_km、price_level、business_id、area。
安静、服务、性价比、拥挤、排队、分量、停车、宠物、家庭、约会、多人、辣度、干净、食物质量等评论归纳特征只能放入 soft_preferences。
用户说“想吃日料/火锅”等明确菜系时，category 使用 any_of；用户说“不要快餐”时使用 none_of。
距离和价格中的“不超过/以内”使用 less_than_or_equal。
只抽取用户当前明确表达的软偏好，不要根据场景自行补充。
“距离近”必须输出 field=distance_km、direction=lower、target_value=null；closer_to 只用于接近一个明确目标值，例如价格接近二档。
软偏好 priority 从1开始连续；“最重要/第一”强度100，“其次/第二”强度75，“第三”强度50，普通弱偏好强度25或50。
evidence_span 必须逐字复制用户问题中的连续原文，不能改写。
严格按照给定结构返回一个 JSON 对象；可选值未知时用 null，没有项目时用空数组。"""
    user = json.dumps(
        {
            "query_text": request.query_text,
            "output_schema": QueryCompilerProposal.model_json_schema(),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return [
        LLMMessage(role="system", content=system),
        LLMMessage(role="user", content=user),
    ]


def _validate_evidence_spans(
    query_text: str,
    proposal: QueryCompilerProposal,
) -> None:
    """确认大模型给出的依据确实逐字存在于当前问题中。"""

    spans = [] if proposal.scene is None else [proposal.scene.evidence_span]
    spans.extend(item.evidence_span for item in proposal.hard_constraints)
    spans.extend(item.evidence_span for item in proposal.soft_preferences)
    missing = [span for span in spans if span not in query_text]
    if missing:
        raise ValueError("evidence_span_not_in_query")


def _hard_key(item: ExtractedHardConstraint) -> str:
    """根据字段和比较方式生成稳定键，方便后续修改同一条要求。"""

    if item.field in {"category", "business_id", "area"}:
        suffix = "include" if item.operator == "any_of" else "exclude"
    elif item.operator in {"less_than", "less_than_or_equal"}:
        suffix = "max"
    elif item.operator in {"greater_than", "greater_than_or_equal"}:
        suffix = "min"
    else:
        suffix = "exact"
    return f"{item.field}.{suffix}"


def _soft_key(item: ExtractedSoftPreference) -> str:
    """为同一软偏好生成固定键，后续对话可以直接定位并修改。"""

    if item.field == "distance_km" and item.direction == "lower":
        return "distance.near"
    suffix = "avoid" if item.direction in {"avoid", "lower"} else "prefer"
    return f"{item.field}.{suffix}"


def _to_compiled_query(
    request: QueryCompilerRequest,
    proposal: QueryCompilerProposal,
) -> CompiledQuery:
    """把大模型的精简 JSON 补齐成统一结构，数值和语义不在这里改写。"""

    scene = (
        None
        if proposal.scene is None
        else SceneSelection(
            kind=proposal.scene.kind,
            basis=RequirementBasis(
                source="current_query",
                text=proposal.scene.evidence_span,
                turn_index=request.turn_index,
            ),
        )
    )
    hard_constraints = [
        HardConstraint(
            key=_hard_key(item),
            field=item.field,
            operator=item.operator,
            value=item.value,
            unit=_HARD_FIELD_UNIT[item.field],
            merchant_feature=_FIELD_FEATURE[item.field],
            controlling_source="current_query",
            sources=[
                RequirementBasis(
                    source="current_query",
                    text=item.evidence_span,
                    turn_index=request.turn_index,
                )
            ],
        )
        for item in proposal.hard_constraints
    ]
    soft_preferences = [
        SoftPreference(
            key=_soft_key(item),
            field=item.field,
            direction=item.direction,
            target_value=item.target_value,
            preference_strength=item.preference_strength,
            priority=item.priority,
            merchant_feature=_FIELD_FEATURE[item.field],
            controlling_source="current_query",
            sources=[
                RequirementBasis(
                    source="current_query",
                    text=item.evidence_span,
                    turn_index=request.turn_index,
                    preference_strength=item.preference_strength,
                )
            ],
        )
        for item in proposal.soft_preferences
    ]
    return CompiledQuery(
        query_text=request.query_text,
        turn_index=request.turn_index,
        scene=scene,
        hard_constraints=hard_constraints,
        soft_preferences=soft_preferences,
    )


def _strip_code_fence(value: str) -> str:
    """兼容模型偶尔在 JSON 外包一层代码块的情况。"""

    text = value.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return text


def _safe_validation_reason(exc: Exception) -> str:
    """只保留安全、简短的结构错误信息，不记录模型推理内容。"""

    if isinstance(exc, ValidationError) and exc.errors():
        error = exc.errors(include_url=False)[0]
        location = ".".join(str(item) for item in error.get("loc", ())) or "root"
        return f"schema_validation_failed:{location}:{error.get('type', 'invalid')}"
    if isinstance(exc, json.JSONDecodeError):
        return "invalid_json"
    return str(exc)[:500] or "invalid_output"
