"""把已经结束的旧对话压成可检索短总结，近期原话继续保留。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import Field

from yelp_agent.models import StrictModel

from ..llm.adapter import LanguageModel
from ..persistence.schema import TurnRecord
from ..persistence.store import RuntimePersistence
from ..runtime.schema import (
    FinalAnswerAction,
    ModelMessage,
    ModelRequest,
    ModelResponse,
)
from ..session.events import SessionRecord
from .models import ConversationEpisode, ConversationEpisodeDraft, WorkingMemory


class _SummaryText(StrictModel):
    """总结模型只填写语义正文，真实编号由程序从数据库补入。"""

    topic: str = Field(min_length=1, max_length=300)
    summary: str = Field(min_length=1, max_length=4000)
    decisions: list[str] = Field(default_factory=list, max_length=20)
    unresolved_questions: list[str] = Field(default_factory=list, max_length=20)


@dataclass(slots=True)
class ConversationCompactionAttempt:
    """压缩失败不影响用户本轮，只留下可观测错误。"""

    response: ModelResponse | None = None
    episode: ConversationEpisode | None = None
    error_code: str | None = None


class ConversationCompactor:
    """保留最近原始轮次，把更老且已结束的轮次交给模型总结。"""

    def __init__(
        self,
        model: LanguageModel,
        *,
        recent_turns_to_keep: int = 4,
        compact_after_turns: int = 6,
        compact_after_chars: int = 12_000,
        max_batch_turns: int = 8,
    ) -> None:
        if recent_turns_to_keep < 1:
            raise ValueError("recent_turns_to_keep must be positive")
        if compact_after_turns <= recent_turns_to_keep:
            raise ValueError("compact_after_turns must exceed the kept turn count")
        if compact_after_chars < 1 or max_batch_turns < 1:
            raise ValueError("compaction limits must be positive")
        self._model = model
        self._recent_turns_to_keep = recent_turns_to_keep
        self._compact_after_turns = compact_after_turns
        self._compact_after_chars = compact_after_chars
        self._max_batch_turns = max_batch_turns

    async def compact_if_needed(
        self,
        *,
        store: RuntimePersistence,
        session: SessionRecord,
        current_turn_id: str,
    ) -> ConversationCompactionAttempt:
        """达到轮数或字符门槛才调用模型；任何失败都保留原始对话兜底。"""

        memory = await store.get_working_memory(session.session_id)
        turns = await store.list_completed_turns_after(
            session_id=session.session_id,
            after_turn_index=(
                None if memory is None else memory.summarized_through_turn_index
            ),
            # 每次最多读取一个小窗口，避免压缩器自己造成大查询。
            limit=self._max_batch_turns + self._recent_turns_to_keep,
        )
        turns = [item for item in turns if item.turn_id != current_turn_id]
        batch = self._select_batch(turns)
        if not batch:
            return ConversationCompactionAttempt()

        request = self._request(session, current_turn_id, batch)
        response: ModelResponse | None = None
        try:
            response = await self._model.generate(request)
            summary = self._parse_response(response)
            latest_memory = await store.get_working_memory(session.session_id)
            draft = self._draft(
                session=session,
                turns=batch,
                summary=summary,
                memory=latest_memory,
            )
            episode = await store.save_episode(draft, now=datetime.now(UTC))
        # 对话压缩只是减少后续上下文，不能让它的异常击穿当前用户请求。
        except Exception as exc:  # noqa: BLE001
            return ConversationCompactionAttempt(
                response=response,
                error_code=f"{type(exc).__name__}",
            )
        return ConversationCompactionAttempt(response=response, episode=episode)

    def _select_batch(self, turns: list[TurnRecord]) -> list[TurnRecord]:
        if len(turns) <= self._recent_turns_to_keep:
            return []
        total_chars = sum(
            len(item.user_message) + len(item.answer or "") for item in turns
        )
        if (
            len(turns) < self._compact_after_turns
            and total_chars < self._compact_after_chars
        ):
            return []
        count = min(
            len(turns) - self._recent_turns_to_keep,
            self._max_batch_turns,
        )
        return turns[:count]

    @staticmethod
    def _request(
        session: SessionRecord,
        current_turn_id: str,
        turns: list[TurnRecord],
    ) -> ModelRequest:
        dialogue = [
            {
                "turn_id": item.turn_id,
                "user": item.user_message,
                "assistant": item.answer,
            }
            for item in turns
        ]
        prompt = (
            "把下面已经结束的旧对话压成一段事实摘要。不要回答其中的问题，"
            "不要添加原文没有的信息。保留用户目标、已经作出的选择、关键结论、"
            "仍未解决的问题和可用于以后指代的名称。只输出一个 JSON 对象，字段固定为："
            "topic（短标题）、summary（完整但精炼的摘要）、decisions（字符串数组）、"
            "unresolved_questions（字符串数组）。不要输出代码围栏或额外文字。\n"
            + json.dumps(dialogue, ensure_ascii=False, separators=(",", ":"))
        )
        return ModelRequest(
            session_id=session.session_id,
            turn_id=current_turn_id,
            step_index=1,
            system_prompt="你只负责忠实压缩已经结束的旧对话。",
            messages=[ModelMessage(role="user", content=prompt)],
            tools=[],
        )

    @staticmethod
    def _parse_response(response: ModelResponse) -> _SummaryText:
        if not isinstance(response.action, FinalAnswerAction):
            raise TypeError("summary model attempted to call a tool")
        text = response.action.answer.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return _SummaryText.model_validate_json(text)

    @staticmethod
    def _draft(
        *,
        session: SessionRecord,
        turns: list[TurnRecord],
        summary: _SummaryText,
        memory: WorkingMemory | None,
    ) -> ConversationEpisodeDraft:
        turn_ids = {item.turn_id for item in turns}
        entities = []
        result_ids: list[str] = []
        if memory is not None:
            entities = [
                item
                for item in memory.focused_entities
                if item.source_turn_id in turn_ids
            ]
            result_ids = [
                item.result_id
                for item in memory.recent_result_sets
                if item.source_turn_id in turn_ids and item.result_id is not None
            ]
        ended_at = turns[-1].ended_at
        if ended_at is None:
            raise ValueError("cannot summarize an unfinished turn")
        return ConversationEpisodeDraft(
            session_id=session.session_id,
            user_id=session.user_id,
            topic=summary.topic,
            summary=summary.summary,
            decisions=summary.decisions,
            unresolved_questions=summary.unresolved_questions,
            entities=entities,
            result_ids=result_ids,
            source_turn_ids=[item.turn_id for item in turns],
            source_start_turn_index=turns[0].turn_index,
            source_end_turn_index=turns[-1].turn_index,
            source_started_at=turns[0].started_at,
            source_ended_at=ended_at,
        )
