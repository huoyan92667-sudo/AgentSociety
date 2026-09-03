"""不依赖餐厅、旅游或酒店字段的通用记忆结构。"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, field_validator, model_validator

from yelp_agent.models import StrictModel


class EntityReference(StrictModel):
    """当前对话涉及的一个真实对象；编号由工具结果提供。"""

    entity_type: str = Field(min_length=1, max_length=64)
    entity_id: str = Field(min_length=1, max_length=256)
    display_name: str = Field(min_length=1, max_length=300)
    source_turn_id: str | None = Field(default=None, min_length=1, max_length=64)


class RankedEntityReference(EntityReference):
    """最近一次列表结果中的位置，用于解析“第三家、第二天”等指代。"""

    position: int = Field(ge=1, le=10)


class ResultSetReference(StrictModel):
    """一次展示给用户的有序结果，最多保留十项。"""

    result_type: str = Field(min_length=1, max_length=128)
    items: list[RankedEntityReference] = Field(min_length=1, max_length=10)
    source_turn_id: str = Field(min_length=1, max_length=64)
    result_id: str | None = Field(default=None, min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_unique_positions_and_entities(self) -> ResultSetReference:
        positions = [item.position for item in self.items]
        identities = [(item.entity_type, item.entity_id) for item in self.items]
        if len(positions) != len(set(positions)):
            raise ValueError("result-set positions must be unique")
        if len(identities) != len(set(identities)):
            raise ValueError("result-set entities must be unique")
        return self


class DomainStateReference(StrictModel):
    """指向餐厅、旅游等业务自己的最新状态，不复制状态正文。"""

    domain: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]{1,63}$")
    revision: int = Field(ge=1)


class ToolMemoryUpdate(StrictModel):
    """工具根据真实结果提交的机械记忆更新，不包含模型推测。"""

    focused_entities: list[EntityReference] = Field(default_factory=list, max_length=10)
    result_sets: list[ResultSetReference] = Field(default_factory=list, max_length=3)
    domain_state_refs: list[DomainStateReference] = Field(
        default_factory=list,
        max_length=10,
    )


class WorkingMemory(StrictModel):
    """每个会话只保存一份最新工作记忆。"""

    session_id: str = Field(min_length=1, max_length=64)
    version: int = Field(default=1, ge=1)
    focused_entities: list[EntityReference] = Field(default_factory=list, max_length=10)
    recent_result_sets: list[ResultSetReference] = Field(
        default_factory=list,
        max_length=3,
    )
    domain_state_refs: list[DomainStateReference] = Field(
        default_factory=list,
        max_length=10,
    )
    pending_question: str | None = Field(default=None, min_length=1, max_length=1000)
    # 摘要边界以会话内轮次编号为准；时间只保留给人查看和旧数据兼容。
    summarized_through_turn_index: int | None = Field(default=None, ge=1)
    summarized_through: datetime | None = None
    updated_at: datetime


class ConversationEpisodeDraft(StrictModel):
    """总结模型生成正文；程序另外填写真实轮次和对象编号。"""

    session_id: str = Field(min_length=1, max_length=64)
    user_id: str = Field(min_length=1, max_length=128)
    topic: str = Field(min_length=1, max_length=300)
    summary: str = Field(min_length=1, max_length=4000)
    decisions: list[str] = Field(default_factory=list, max_length=20)
    unresolved_questions: list[str] = Field(default_factory=list, max_length=20)
    entities: list[EntityReference] = Field(default_factory=list, max_length=20)
    result_ids: list[str] = Field(default_factory=list, max_length=30)
    source_turn_ids: list[str] = Field(min_length=1, max_length=20)
    source_start_turn_index: int | None = Field(default=None, ge=1)
    source_end_turn_index: int | None = Field(default=None, ge=1)
    source_started_at: datetime
    source_ended_at: datetime

    @field_validator("decisions", "unresolved_questions", "result_ids")
    @classmethod
    def unique_text_values(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))

    @model_validator(mode="after")
    def require_valid_time_range(self) -> ConversationEpisodeDraft:
        if self.source_ended_at < self.source_started_at:
            raise ValueError("episode end cannot precede start")
        if len(self.source_turn_ids) != len(set(self.source_turn_ids)):
            raise ValueError("episode turn IDs must be unique")
        indexes = (self.source_start_turn_index, self.source_end_turn_index)
        if (indexes[0] is None) != (indexes[1] is None):
            raise ValueError("episode turn-index range must be complete")
        if (
            indexes[0] is not None
            and indexes[1] is not None
            and indexes[1] < indexes[0]
        ):
            raise ValueError("episode turn-index end cannot precede start")
        return self


class ConversationEpisode(ConversationEpisodeDraft):
    """可按用户、会话、对象和文字检索的一段旧对话总结。"""

    episode_id: str = Field(min_length=1, max_length=64)
    created_at: datetime
