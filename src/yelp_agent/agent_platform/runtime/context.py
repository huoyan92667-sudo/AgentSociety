"""从只追加会话事件中还原本轮真正交给模型的历史。"""

from __future__ import annotations

from ..session.events import SessionEvent, SessionRecord
from ..session.store import SessionStore
from .schema import ModelMessage, ModelRequest, ToolCall, ToolSchema


class ContextBuilder:
    """第一版保留完整历史；后续可在这里加入摘要和按需取回。"""

    def __init__(self, system_prompt: str) -> None:
        if not system_prompt.strip():
            raise ValueError("system prompt cannot be blank")
        self._system_prompt = system_prompt.strip()

    async def build(
        self,
        *,
        store: SessionStore,
        session: SessionRecord,
        turn_id: str,
        step_index: int,
        tools: list[ToolSchema],
    ) -> ModelRequest:
        events = await store.list_events(session.session_id)
        messages: list[ModelMessage] = []
        source_event_seqs: list[int] = []
        for event in events:
            message = self._event_to_message(event)
            if message is None:
                continue
            messages.append(message)
            source_event_seqs.append(event.seq)
        if not messages:
            raise ValueError("cannot call the model without a user-visible message")
        return ModelRequest(
            session_id=session.session_id,
            turn_id=turn_id,
            step_index=step_index,
            system_prompt=self._system_prompt,
            messages=messages,
            tools=tools,
            source_event_seqs=source_event_seqs,
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
            return ModelMessage(
                role="tool",
                content=str(raw_result.get("model_content") or "工具没有返回内容。"),
                tool_call_id=str(raw_result["call_id"]),
                tool_name=str(raw_result["tool_name"]),
            )
        return None
