"""由主模型按需调用的旧对话查询工具。"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import Field

from yelp_agent.models import StrictModel

from ..persistence.store import RuntimePersistence
from ..runtime.schema import FinalAnswerAction
from ..session.events import SessionEvent
from ..tools.definition import ToolDefinition, ToolExecutionContext
from ..tools.result import ToolBodyResult


class SearchConversationMemoryArguments(StrictModel):
    """模型用用户当前关心的意思搜索旧话题，必要时可取回原始问答。"""

    query: str = Field(min_length=1, max_length=500)
    scope: Literal["current_session", "all_user_sessions"] = "current_session"
    limit: int = Field(default=3, ge=1, le=5)
    include_original_dialogue: bool = False


class _SearchConversationMemoryHandler:
    def __init__(self, store: RuntimePersistence) -> None:
        self._store = store

    async def __call__(
        self,
        arguments: SearchConversationMemoryArguments,
        context: ToolExecutionContext,
    ) -> ToolBodyResult:
        episodes = await self._store.search_episodes(
            user_id=context.user_id,
            query=arguments.query,
            session_id=(
                context.session_id
                if arguments.scope == "current_session"
                else None
            ),
            limit=arguments.limit,
        )
        matches: list[dict[str, object]] = []
        for episode in episodes:
            item: dict[str, object] = {
                "episode_id": episode.episode_id,
                "session_id": episode.session_id,
                "topic": episode.topic,
                "summary": episode.summary,
                "decisions": episode.decisions,
                "unresolved_questions": episode.unresolved_questions,
                "entities": [
                    entity.model_dump(mode="json") for entity in episode.entities
                ],
                "source_time": {
                    "start": episode.source_started_at.isoformat(),
                    "end": episode.source_ended_at.isoformat(),
                },
            }
            if arguments.include_original_dialogue:
                item["original_dialogue"] = await self._original_dialogue(
                    episode.session_id,
                    episode.source_turn_ids,
                )
            matches.append(item)
        payload = {
            "status": "found" if matches else "not_found",
            "matches": matches,
            "instruction": (
                "这些是旧对话记录，不是外部世界的最新事实；涉及商家现状时仍应调用查询工具。"
            ),
        }
        rendered = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return ToolBodyResult(value=payload, model_content=rendered)

    async def _original_dialogue(
        self,
        session_id: str,
        turn_ids: list[str],
    ) -> list[dict[str, str]]:
        """只恢复用户原话和最终回答，不把旧工具调用链重新塞给模型。"""

        dialogue: list[dict[str, str]] = []
        for turn_id in turn_ids:
            events = await self._store.list_turn_events(
                session_id=session_id,
                turn_id=turn_id,
            )
            user_text = _user_message(events)
            answer = _final_answer(events)
            if user_text:
                dialogue.append({"role": "user", "content": user_text})
            if answer:
                dialogue.append({"role": "assistant", "content": answer})
        return dialogue


def build_conversation_memory_tool(store: RuntimePersistence) -> ToolDefinition:
    """提供一个通用工具；餐厅、旅游、闲聊都可以使用同一套旧记忆。"""

    return ToolDefinition(
        name="search_conversation_memory",
        description=(
            "当用户明确提到更早的话题、以前的选择或已经不在当前上下文中的内容时，"
            "搜索该用户的旧对话摘要。默认只查当前会话；确实需要跨会话回忆时再查全部会话。"
            "如果摘要不足以确认原话，可把 include_original_dialogue 设为 true。"
            "近期上下文已经包含的信息不要重复查询。"
        ),
        input_model=SearchConversationMemoryArguments,
        handler=_SearchConversationMemoryHandler(store),
        timeout_seconds=20.0,
        max_retries=1,
        read_only=True,
        concurrency_safe=True,
    )


def _user_message(events: list[SessionEvent]) -> str | None:
    for event in events:
        if event.type == "user/message":
            content = event.payload.get("content")
            if isinstance(content, str) and content:
                return content
    return None


def _final_answer(events: list[SessionEvent]) -> str | None:
    for event in reversed(events):
        if event.type != "assistant/message":
            continue
        action = event.payload.get("action")
        if not isinstance(action, dict) or action.get("type") != "final_answer":
            continue
        try:
            return FinalAnswerAction.model_validate(action).answer
        except ValueError:
            return None
    return None
