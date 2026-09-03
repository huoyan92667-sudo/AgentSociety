"""一个用户轮次内反复调用模型与工具的通用循环。"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

from ..llm.adapter import LanguageModel
from ..memory.compaction import ConversationCompactor
from ..observability.usage import TurnUsageTracker
from ..persistence.schema import LLMCallRecord, RecoveryReport, TurnStatus
from ..persistence.store import RuntimePersistence
from ..session.events import SessionRecord
from ..tools.definition import ToolExecutionContext
from ..tools.pipeline import ToolPipeline
from ..tools.registry import ToolRegistry
from ..tools.result import ToolResult
from .context import ContextBuilder
from .schema import (
    AgentLimits,
    AgentTurnInput,
    AgentTurnResult,
    AskUserAction,
    FinalAnswerAction,
    ModelResponse,
    ToolCallsAction,
)


@dataclass(slots=True)
class _LoopState:
    """只在当前轮次内存在的计数，不属于任何业务领域状态。"""

    turn_id: str
    step_count: int = 0
    step_open: bool = False
    used_tools: list[str] = field(default_factory=list)


class AgentLoop:
    """领域无关的主循环；它只理解模型动作、工具结果和运行上限。"""

    def __init__(
        self,
        *,
        model: LanguageModel,
        session_store: RuntimePersistence,
        registry: ToolRegistry,
        pipeline: ToolPipeline,
        context_builder: ContextBuilder,
        limits: AgentLimits,
        conversation_compactor: ConversationCompactor | None = None,
    ) -> None:
        self._model = model
        self._store = session_store
        self._registry = registry
        self._pipeline = pipeline
        self._context_builder = context_builder
        self._limits = limits
        self._conversation_compactor = conversation_compactor

    async def run(
        self,
        value: AgentTurnInput,
        *,
        on_answer_delta: Callable[[str], None] | None = None,
    ) -> AgentTurnResult:
        """执行一轮对话；所有可恢复事实都会按发生顺序写入会话事件。"""

        opened = await self._store.get_or_create(
            session_id=value.session_id,
            user_id=value.user_id,
            now=self._now(),
        )
        session = opened.session
        state = _LoopState(turn_id=uuid4().hex)
        usage = TurnUsageTracker()
        await self._store.begin_turn(
            session_id=session.session_id,
            turn_id=state.turn_id,
            user_message=value.message,
            request_time=value.request_time,
            now=self._now(),
        )
        # 新消息意味着用户已经有机会回应上一条追问；本轮若仍缺信息，
        # AskUserAction 会再次写入最新问题。
        await self._store.set_pending_question(
            session_id=session.session_id,
            question=None,
            now=self._now(),
        )

        if self._conversation_compactor is not None:
            try:
                compaction = await self._conversation_compactor.compact_if_needed(
                    store=self._store,
                    session=session,
                    current_turn_id=state.turn_id,
                )
            # 旧对话压缩失败不应阻断当前用户问题；原始轮次仍在数据库中。
            except Exception:  # noqa: BLE001
                compaction = None
            if compaction is not None and compaction.response is not None:
                usage.record_model(compaction.response)
                await self._store.record_llm_call(
                    LLMCallRecord(
                        llm_call_id=uuid4().hex,
                        session_id=session.session_id,
                        turn_id=state.turn_id,
                        step_index=1,
                        purpose="conversation_summary",
                        provider=compaction.response.provider,
                        model=compaction.response.model,
                        status="success",
                        input_tokens=compaction.response.usage.input_tokens,
                        output_tokens=compaction.response.usage.output_tokens,
                        latency_ms=compaction.response.latency_ms,
                        provider_request_id=(
                            compaction.response.provider_request_id
                        ),
                        error_code=compaction.error_code,
                        created_at=self._now(),
                    )
                )

        try:
            return await asyncio.wait_for(
                self._run_steps(
                    value=value,
                    session=session,
                    state=state,
                    usage=usage,
                    on_answer_delta=on_answer_delta,
                ),
                timeout=self._limits.timeout_seconds,
            )
        except TimeoutError:
            return await self._finish_runtime_failure(
                session=session,
                state=state,
                usage=usage,
                status="limit_reached",
                error_code="turn_timeout",
                answer="本轮执行时间达到系统上限，请稍后重试。",
            )
        # 主循环是最后一道故障隔离层，不能让任意模型或工具适配异常击穿调用方。
        except Exception as exc:  # noqa: BLE001
            return await self._finish_runtime_failure(
                session=session,
                state=state,
                usage=usage,
                status="failed",
                error_code=f"runtime_error:{type(exc).__name__}",
                answer="本轮处理失败，请稍后重试。",
            )

    async def recover_interrupted(self, *, now: datetime) -> RecoveryReport:
        """把上次进程异常退出所遗留的运行中轮次标记为中断。"""

        return await self._store.recover_interrupted_turns(now=now)

    async def _run_steps(
        self,
        *,
        value: AgentTurnInput,
        session: SessionRecord,
        state: _LoopState,
        usage: TurnUsageTracker,
        on_answer_delta: Callable[[str], None] | None,
    ) -> AgentTurnResult:
        for step_index in range(1, self._limits.max_steps + 1):
            state.step_count = step_index
            state.step_open = True
            await self._append(
                session,
                "step/start",
                {},
                turn_id=state.turn_id,
                step_index=step_index,
            )

            tool_schemas = self._registry.schemas()
            request = await self._context_builder.build(
                store=self._store,
                session=session,
                turn_id=state.turn_id,
                step_index=step_index,
                tools=tool_schemas,
            )
            await self._append(
                session,
                "model/request",
                {
                    # 固定说明由当前程序版本提供；事件只保存校验值，避免每一步
                    # 重复写同一大段正文。恢复时若校验值变化，应放弃自动续跑。
                    "system_prompt_sha256": hashlib.sha256(
                        request.system_prompt.encode("utf-8")
                    ).hexdigest(),
                    "source_event_seqs": request.source_event_seqs,
                    "tool_names": [item.name for item in tool_schemas],
                    "context_stats": (
                        None
                        if request.context_stats is None
                        else request.context_stats.model_dump(mode="json")
                    ),
                },
                turn_id=state.turn_id,
                step_index=step_index,
            )

            response = await self._model.generate(request)
            usage.record_model(response)
            await self._append_assistant_response(
                session,
                state=state,
                response=response,
            )

            if isinstance(response.action, FinalAnswerAction):
                await self._close_step(session, state, "completed")
                return await self._finish(
                    session=session,
                    state=state,
                    usage=usage,
                    status="completed",
                    answer=response.action.answer,
                )

            if isinstance(response.action, AskUserAction):
                await self._store.set_pending_question(
                    session_id=session.session_id,
                    question=response.action.question,
                    now=self._now(),
                )
                await self._close_step(session, state, "awaiting_user")
                return await self._finish(
                    session=session,
                    state=state,
                    usage=usage,
                    status="awaiting_user",
                    answer=response.action.question,
                )

            if not isinstance(response.action, ToolCallsAction):
                raise TypeError("unsupported model action")

            calls = response.action.calls
            exceeds_tool_budget = (
                usage.tool_call_count + len(calls) > self._limits.max_tool_calls
            )
            exceeds_token_budget = usage.total_tokens > self._limits.max_total_tokens
            if exceeds_tool_budget or exceeds_token_budget:
                reason = (
                    "本轮工具调用次数达到系统上限。"
                    if exceeds_tool_budget
                    else "本轮模型消耗达到系统上限。"
                )
                await self._record_unexecuted_calls(
                    session=session,
                    state=state,
                    calls=calls,
                    reason=reason,
                )
                await self._close_step(session, state, "limit_reached")
                return await self._finish_runtime_failure(
                    session=session,
                    state=state,
                    usage=usage,
                    status="limit_reached",
                    error_code=(
                        "tool_budget_exceeded"
                        if exceeds_tool_budget
                        else "token_budget_exceeded"
                    ),
                    answer=reason,
                    step_already_closed=True,
                )

            for call in calls:
                state.used_tools.append(call.tool_name)
                await self._append(
                    session,
                    "tool/call",
                    {"call": call.model_dump(mode="json")},
                    turn_id=state.turn_id,
                    step_index=step_index,
                )
                context = ToolExecutionContext(
                    user_id=value.user_id,
                    session_id=value.session_id,
                    turn_id=state.turn_id,
                    step_index=step_index,
                    call_id=call.call_id,
                    request_time=value.request_time,
                    # 当前原话由运行时直接注入，餐饮工具无需让模型再抄写一遍，
                    # 避免“地道川菜”等关键要求在工具参数里被改写丢失。
                    current_user_message=value.message,
                    # 只存在于本轮进程内，不写数据库，也不会提供给大模型。
                    metadata=(
                        {}
                        if on_answer_delta is None
                        else {"answer_delta_callback": on_answer_delta}
                    ),
                )
                result = await self._pipeline.execute(call, context=context)
                usage.record_tool(result)
                await self._append(
                    session,
                    "tool/result",
                    {"result": self._tool_result_event_payload(result)},
                    turn_id=state.turn_id,
                    step_index=step_index,
                )
                if result.status == "success" and result.memory_update is not None:
                    await self._store.apply_tool_memory_update(
                        session_id=session.session_id,
                        update=result.memory_update,
                        now=self._now(),
                    )
            if len(calls) == 1 and result.terminal_answer is not None:
                answer = result.terminal_answer
                # 某些高层工具已经完成证据约束下的最终总结。此时直接结束，
                # 避免外层模型再次改写排序、删掉风险或增加没有证据的结论。
                await self._append(
                    session,
                    "assistant/message",
                    {
                        "action": {"type": "final_answer", "answer": answer},
                        "model": f"tool:{calls[0].tool_name}",
                        "provider": "tool_terminal",
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                        "latency_ms": 0.0,
                        "provider_request_id": None,
                    },
                    turn_id=state.turn_id,
                    step_index=step_index,
                )
                await self._close_step(session, state, "completed")
                return await self._finish(
                    session=session,
                    state=state,
                    usage=usage,
                    status="completed",
                    answer=answer,
                )
            await self._close_step(session, state, "continue")

        return await self._finish_runtime_failure(
            session=session,
            state=state,
            usage=usage,
            status="limit_reached",
            error_code="step_budget_exceeded",
            answer="本轮思考步骤达到系统上限，请缩小问题范围后重试。",
        )

    async def _append_assistant_response(
        self,
        session: SessionRecord,
        *,
        state: _LoopState,
        response: ModelResponse,
    ) -> None:
        event_payload = {
            "action": response.action.model_dump(mode="json"),
            "model": response.model,
            "provider": response.provider,
            "usage": response.usage.model_dump(mode="json"),
            "latency_ms": response.latency_ms,
            "provider_request_id": response.provider_request_id,
        }
        await self._store.record_model_response(
            record=LLMCallRecord(
                llm_call_id=uuid4().hex,
                session_id=session.session_id,
                turn_id=state.turn_id,
                step_index=state.step_count,
                purpose="agent_step",
                provider=response.provider,
                model=response.model,
                status="success",
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_ms=response.latency_ms,
                provider_request_id=response.provider_request_id,
                created_at=self._now(),
            ),
            event_payload=event_payload,
            now=self._now(),
        )

    async def _record_unexecuted_calls(
        self,
        *,
        session: SessionRecord,
        state: _LoopState,
        calls: list,
        reason: str,
    ) -> None:
        """即使预算拒绝执行，也记录成配对的调用和结果，保持历史合法。"""

        for call in calls:
            state.used_tools.append(call.tool_name)
            await self._append(
                session,
                "tool/call",
                {"call": call.model_dump(mode="json")},
                turn_id=state.turn_id,
                step_index=state.step_count,
            )
            result = ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                status="denied",
                model_content=f"工具 {call.tool_name} 未执行：{reason}",
                error_code="turn_budget_exceeded",
                error_message=reason,
            )
            await self._append(
                session,
                "tool/result",
                {"result": result.model_dump(mode="json")},
                turn_id=state.turn_id,
                step_index=state.step_count,
            )

    async def _close_step(
        self,
        session: SessionRecord,
        state: _LoopState,
        outcome: str,
    ) -> None:
        if not state.step_open:
            return
        await self._append(
            session,
            "step/end",
            {"outcome": outcome},
            turn_id=state.turn_id,
            step_index=state.step_count,
        )
        state.step_open = False

    async def _finish_runtime_failure(
        self,
        *,
        session: SessionRecord,
        state: _LoopState,
        usage: TurnUsageTracker,
        status: str,
        error_code: str,
        answer: str,
        step_already_closed: bool = False,
    ) -> AgentTurnResult:
        if not step_already_closed:
            await self._close_step(session, state, status)
        # 运行时生成的安全回答也必须进入历史，否则下一轮会缺少真实上下文。
        await self._append(
            session,
            "assistant/message",
            {
                "action": {
                    "type": "final_answer",
                    "answer": answer,
                },
                "model": "agent_runtime",
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "latency_ms": 0.0,
                "provider_request_id": None,
            },
            turn_id=state.turn_id,
            step_index=state.step_count or None,
        )
        return await self._finish(
            session=session,
            state=state,
            usage=usage,
            status=status,
            answer=answer,
            error_code=error_code,
        )

    async def _finish(
        self,
        *,
        session: SessionRecord,
        state: _LoopState,
        usage: TurnUsageTracker,
        status: str,
        answer: str,
        error_code: str | None = None,
    ) -> AgentTurnResult:
        usage_snapshot = usage.snapshot()
        session_status = {
            "completed": "idle",
            "awaiting_user": "awaiting_user",
            "failed": "failed",
            "limit_reached": "idle",
        }[status]
        await self._store.finish_turn(
            session_id=session.session_id,
            turn_id=state.turn_id,
            turn_status=cast(TurnStatus, status),
            session_status=session_status,
            answer=answer,
            error_code=error_code,
            step_count=state.step_count,
            used_tools=list(state.used_tools),
            usage=usage_snapshot,
            now=self._now(),
        )
        return AgentTurnResult(
            session_id=session.session_id,
            turn_id=state.turn_id,
            status=status,
            answer=answer,
            step_count=state.step_count,
            used_tools=list(state.used_tools),
            usage=usage_snapshot,
            error_code=error_code,
        )

    async def _append(
        self,
        session: SessionRecord,
        event_type: str,
        payload: dict,
        *,
        turn_id: str | None = None,
        step_index: int | None = None,
    ) -> None:
        await self._store.append_event(
            session_id=session.session_id,
            event_type=event_type,
            payload=payload,
            now=self._now(),
            turn_id=turn_id,
            step_index=step_index,
        )

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)

    @staticmethod
    def _tool_result_event_payload(result: ToolResult) -> dict:
        """正式值和模型正文相同时只存一份，读取上下文时原样恢复。"""

        payload = result.model_dump(mode="json")
        payload.pop("terminal_answer", None)
        canonical_value = (
            "工具执行成功。"
            if result.value is None
            else json.dumps(
                result.value,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        if result.model_content == canonical_value:
            payload.pop("model_content", None)
            payload["model_content_from_value"] = True
        return payload
