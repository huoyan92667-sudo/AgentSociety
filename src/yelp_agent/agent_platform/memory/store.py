"""当前工作记忆和旧对话片段共同遵守的持久化接口。"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from .models import (
    ConversationEpisode,
    ConversationEpisodeDraft,
    ToolMemoryUpdate,
    WorkingMemory,
)


class ConversationMemoryStore(Protocol):
    """上下文管理只依赖这组通用能力，不理解任何餐厅字段。"""

    async def get_working_memory(self, session_id: str) -> WorkingMemory | None: ...

    async def apply_tool_memory_update(
        self,
        *,
        session_id: str,
        update: ToolMemoryUpdate,
        now: datetime,
    ) -> WorkingMemory: ...

    async def set_pending_question(
        self,
        *,
        session_id: str,
        question: str | None,
        now: datetime,
    ) -> WorkingMemory: ...

    async def save_episode(
        self,
        value: ConversationEpisodeDraft,
        *,
        now: datetime,
    ) -> ConversationEpisode: ...

    async def list_recent_episodes(
        self,
        *,
        session_id: str,
        limit: int,
    ) -> list[ConversationEpisode]: ...

    async def search_episodes(
        self,
        *,
        user_id: str,
        query: str,
        session_id: str | None,
        limit: int,
    ) -> list[ConversationEpisode]: ...
