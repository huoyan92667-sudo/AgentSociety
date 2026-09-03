"""从完整存档中挑选本轮真正需要交给模型的有限上下文。"""

from __future__ import annotations

import json

from ..memory.models import ConversationEpisode, WorkingMemory
from ..persistence.store import RuntimePersistence
from ..session.events import SessionEvent, SessionRecord
from .schema import ContextStats, ModelMessage, ModelRequest, ToolCall, ToolSchema


class ContextBuilder:
    """只保留当前工具链、尚未总结的近期对话和少量旧话题总结。"""

    def __init__(
        self,
        system_prompt: str,
        *,
        recent_turn_limit: int = 6,
        recent_episode_limit: int = 2,
    ) -> None:
        if not system_prompt.strip():
            raise ValueError("system prompt cannot be blank")
        if recent_turn_limit < 1 or recent_episode_limit < 0:
            raise ValueError("context history limits are invalid")
        self._system_prompt = system_prompt.strip()
        self._recent_turn_limit = recent_turn_limit
        self._recent_episode_limit = recent_episode_limit

    async def build(
        self,
        *,
        store: RuntimePersistence,
        session: SessionRecord,
        turn_id: str,
        step_index: int,
        tools: list[ToolSchema],
    ) -> ModelRequest:
        memory = await store.get_working_memory(session.session_id)
        completed_turns = await store.list_recent_turns_after(
            session_id=session.session_id,
            after_turn_index=(
                None if memory is None else memory.summarized_through_turn_index
            ),
            limit=self._recent_turn_limit,
            exclude_turn_id=turn_id,
        )
        current_events = await store.list_turn_events(
            session_id=session.session_id,
            turn_id=turn_id,
        )
        episodes = await store.list_recent_episodes(
            session_id=session.session_id,
            limit=self._recent_episode_limit,
        )

        messages: list[ModelMessage] = []
        current_messages: list[ModelMessage] = []
        source_event_seqs: list[int] = []
        for turn in completed_turns:
            messages.append(ModelMessage(role="user", content=turn.user_message))
            if turn.answer:
                messages.append(ModelMessage(role="assistant", content=turn.answer))
        for event in current_events:
            message = self._event_to_message(event)
            if message is None:
                continue
            messages.append(message)
            current_messages.append(message)
            source_event_seqs.append(event.seq)
        if not messages:
            raise ValueError("cannot call the model without a user-visible message")

        memory_text = self._memory_text(memory)
        episode_text = self._episode_text(episodes)
        system_prompt = self._system_prompt
        if memory_text:
            system_prompt += "\n\n当前会话中由真实工具结果维护的记忆：\n" + memory_text
        if episode_text:
            system_prompt += "\n\n已经结束的旧话题总结：\n" + episode_text

        completed_chars = sum(
            len(item.user_message) + len(item.answer or "") for item in completed_turns
        )
        current_chars = sum(len(item.content or "") for item in current_messages)
        return ModelRequest(
            session_id=session.session_id,
            turn_id=turn_id,
            step_index=step_index,
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            source_event_seqs=source_event_seqs,
            context_stats=ContextStats(
                working_memory_chars=len(memory_text),
                episode_summary_chars=len(episode_text),
                completed_turn_chars=completed_chars,
                current_turn_chars=current_chars,
                included_completed_turns=len(completed_turns),
                included_episodes=len(episodes),
            ),
        )

    @staticmethod
    def _memory_text(memory: WorkingMemory | None) -> str:
        if memory is None:
            return ""
        payload = memory.model_dump(
            mode="json",
            exclude={
                "session_id",
                "version",
                "updated_at",
                "summarized_through",
                "summarized_through_turn_index",
            },
        )
        if not any(payload.values()):
            return ""
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _episode_text(episodes: list[ConversationEpisode]) -> str:
        values = [
            {
                "topic": item.topic,
                "summary": item.summary,
                "decisions": item.decisions,
                "unresolved_questions": item.unresolved_questions,
                "entities": [entity.model_dump(mode="json") for entity in item.entities],
            }
            for item in reversed(episodes)
        ]
        return (
            ""
            if not values
            else json.dumps(values, ensure_ascii=False, separators=(",", ":"))
        )

    @staticmethod
    def _event_to_message(event: SessionEvent) -> ModelMessage | None:
        if event.type == "user/message":
            return ModelMessage(role="user", content=str(event.payload["content"]))

        if event.type == "assistant/message":
            action = event.payload.get("action")
            if not isinstance(action, dict):
                return None
            action_type = action.get("type")
            if action_type == "final_answer":
                return ModelMessage(role="assistant", content=str(action["answer"]))
            if action_type == "ask_user":
                return ModelMessage(role="assistant", content=str(action["question"]))
            if action_type == "tool_calls":
                raw_calls = action.get("calls", [])
                return ModelMessage(
                    role="assistant",
                    tool_calls=[ToolCall.model_validate(item) for item in raw_calls],
                )

        if event.type == "tool/result":
            raw_result = event.payload.get("result")
            if not isinstance(raw_result, dict):
                return None
            content = raw_result.get("model_content")
            if raw_result.get("model_content_from_value") is True:
                value = raw_result.get("value")
                content = (
                    "工具执行成功。"
                    if value is None
                    else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                )
            return ModelMessage(
                role="tool",
                content=str(content or "工具没有返回内容。"),
                tool_call_id=str(raw_result["call_id"]),
                tool_name=str(raw_result["tool_name"]),
            )
        return None
