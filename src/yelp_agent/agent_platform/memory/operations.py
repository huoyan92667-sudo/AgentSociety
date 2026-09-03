"""记忆合并和旧话题排序的确定性规则。"""

from __future__ import annotations

import re
from datetime import datetime

from .models import ConversationEpisode, ToolMemoryUpdate, WorkingMemory


def apply_tool_update(
    current: WorkingMemory | None,
    *,
    session_id: str,
    update: ToolMemoryUpdate,
    now: datetime,
) -> WorkingMemory:
    """同一对象保留最新信息；最近结果集和业务状态按类型覆盖。"""

    existing = current or WorkingMemory(session_id=session_id, updated_at=now)
    focused = {
        (item.entity_type, item.entity_id): item
        for item in existing.focused_entities
    }
    for item in update.focused_entities:
        focused[(item.entity_type, item.entity_id)] = item

    result_sets = {
        item.result_type: item for item in existing.recent_result_sets
    }
    for item in update.result_sets:
        result_sets[item.result_type] = item

    domain_states = {
        item.domain: item for item in existing.domain_state_refs
    }
    for item in update.domain_state_refs:
        domain_states[item.domain] = item

    return existing.model_copy(
        update={
            "version": existing.version + (1 if current is not None else 0),
            "focused_entities": list(focused.values())[-10:],
            "recent_result_sets": list(result_sets.values())[-3:],
            "domain_state_refs": list(domain_states.values())[-10:],
            "pending_question": None,
            "updated_at": now,
        },
        deep=True,
    )


def with_pending_question(
    current: WorkingMemory | None,
    *,
    session_id: str,
    question: str | None,
    now: datetime,
) -> WorkingMemory:
    """用户补充信息后清空旧追问；需要追问时只保留最新一条。"""

    existing = current or WorkingMemory(session_id=session_id, updated_at=now)
    return existing.model_copy(
        update={
            "version": existing.version + (1 if current is not None else 0),
            "pending_question": question,
            "updated_at": now,
        },
        deep=True,
    )


def rank_episode_matches(
    episodes: list[ConversationEpisode],
    query: str,
    *,
    limit: int,
) -> list[ConversationEpisode]:
    """先匹配完整短语，再用字符二元组覆盖中文无空格查询。"""

    wanted = _normalize(query)
    wanted_pairs = _character_pairs(wanted)
    scored: list[tuple[float, datetime, ConversationEpisode]] = []
    for item in episodes:
        searchable = _normalize(
            " ".join(
                [
                    item.topic,
                    item.summary,
                    *item.decisions,
                    *item.unresolved_questions,
                    *(entity.display_name for entity in item.entities),
                ]
            )
        )
        if not searchable:
            continue
        exact = 1.0 if wanted and wanted in searchable else 0.0
        pairs = _character_pairs(searchable)
        overlap = (
            len(wanted_pairs & pairs) / len(wanted_pairs)
            if wanted_pairs
            else 0.0
        )
        score = exact * 2.0 + overlap
        if score > 0:
            scored.append((score, item.created_at, item))
    scored.sort(key=lambda value: (value[0], value[1]), reverse=True)
    return [item.model_copy(deep=True) for _, _, item in scored[:limit]]


def _normalize(value: str) -> str:
    return "".join(re.findall(r"\w+", value.casefold(), flags=re.UNICODE))


def _character_pairs(value: str) -> set[str]:
    if len(value) < 2:
        return {value} if value else set()
    return {value[index : index + 2] for index in range(len(value) - 1)}
