"""新 Agent 框架对应用层提供的单一入口。"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import UTC, datetime

from ..llm.adapter import LanguageModel
from ..memory.compaction import ConversationCompactor
from ..persistence.hooks import PersistLargeToolResultHook
from ..persistence.schema import RecoveryReport
from ..persistence.store import ResultStore, RuntimePersistence
from ..tools.definition import ToolDefinition
from ..tools.hooks import ToolPipelineHooks
from ..tools.pipeline import ToolPipeline
from ..tools.registry import ToolRegistry
from .context import ContextBuilder
from .loop import AgentLoop
from .schema import AgentLimits, AgentTurnInput, AgentTurnResult

DEFAULT_SYSTEM_PROMPT = """
你是一个通用生活助手。你可以直接回答，也可以根据用户目标选择工具。
不要假装已经查询过工具中才有的数据。调用工具后，请阅读真实结果再继续判断。
如果缺少完成任务必需的信息，明确向用户提问。
""".strip()


class AgentRuntime:
    """隐藏会话、上下文、模型和工具流水线的完整运行入口。"""

    def __init__(
        self,
        *,
        model: LanguageModel,
        session_store: RuntimePersistence,
        result_store: ResultStore | None = None,
        tools: Iterable[ToolDefinition] = (),
        hooks: ToolPipelineHooks | None = None,
        limits: AgentLimits | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_model_tool_result_chars: int = 8_000,
        large_tool_result_threshold_bytes: int = 64 * 1024,
        memory_summary_model: LanguageModel | None = None,
    ) -> None:
        registry = ToolRegistry(tools)
        configured_hooks = ToolPipelineHooks(
            pre_execute=list(hooks.pre_execute) if hooks else [],
            guards=list(hooks.guards) if hooks else [],
            execute_wrappers=list(hooks.execute_wrappers) if hooks else [],
            post_execute=list(hooks.post_execute) if hooks else [],
            result_observers=list(hooks.result_observers) if hooks else [],
        )
        if result_store is not None:
            configured_hooks.post_execute.append(
                PersistLargeToolResultHook(
                    result_store,
                    threshold_bytes=large_tool_result_threshold_bytes,
                )
            )
        pipeline = ToolPipeline(
            registry,
            hooks=configured_hooks,
            max_model_content_chars=max_model_tool_result_chars,
        )
        self._loop = AgentLoop(
            model=model,
            session_store=session_store,
            registry=registry,
            pipeline=pipeline,
            context_builder=ContextBuilder(system_prompt),
            limits=limits or AgentLimits(),
            conversation_compactor=(
                None
                if memory_summary_model is None
                else ConversationCompactor(memory_summary_model)
            ),
        )

    async def handle(
        self,
        value: AgentTurnInput,
        *,
        on_answer_delta: Callable[[str], None] | None = None,
    ) -> AgentTurnResult:
        """处理一条用户消息，直到回答、追问、失败或达到运行上限。"""

        return await self._loop.run(value, on_answer_delta=on_answer_delta)

    async def recover_interrupted(self) -> RecoveryReport:
        """应用启动后，关闭上次进程遗留的未完成轮次。"""

        return await self._loop.recover_interrupted(now=datetime.now(UTC))
