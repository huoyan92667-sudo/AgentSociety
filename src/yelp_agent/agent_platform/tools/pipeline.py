"""所有工具都必须经过的执行前、执行中和执行后流水线。"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from time import perf_counter
from typing import Any

from pydantic import BaseModel, ValidationError

from ..runtime.schema import ToolCall
from .definition import ToolDefinition, ToolExecution, ToolExecutionContext
from .hooks import ToolPipelineHooks
from .registry import ToolRegistry
from .result import ToolBodyResult, ToolResult


class _OutputValidationError(RuntimeError):
    pass


logger = logging.getLogger(__name__)


class ToolPipeline:
    """隐藏参数检查、策略、超时、重试、结果整理和观察等通用复杂度。"""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        hooks: ToolPipelineHooks | None = None,
        max_model_content_chars: int = 8_000,
    ) -> None:
        if max_model_content_chars < 200:
            raise ValueError("max_model_content_chars must be at least 200")
        self._registry = registry
        self._hooks = hooks or ToolPipelineHooks()
        self._max_model_content_chars = max_model_content_chars

    async def execute(
        self,
        call: ToolCall,
        *,
        context: ToolExecutionContext,
        allowed_names: set[str] | None = None,
    ) -> ToolResult:
        """执行一次工具调用并始终返回结构化结果，不把工具异常泄漏到主循环。"""

        started_at = perf_counter()
        definition = self._registry.get(
            call.tool_name,
            allowed_names=allowed_names,
        )
        if definition is None:
            return self._standalone_failure(
                call,
                status="denied",
                error_code="tool_not_available",
                error_message="当前步骤没有提供这个工具。",
                started_at=started_at,
            )

        parsed, validation_error = self._parse_arguments(definition, call.arguments)
        if parsed is None:
            return self._standalone_failure(
                call,
                status="invalid_arguments",
                error_code="invalid_arguments",
                error_message=(
                    "工具参数不符合要求：" + (validation_error or "缺少或错误的字段。")
                ),
                started_at=started_at,
            )

        execution = ToolExecution(
            call=call,
            definition=definition,
            arguments=parsed,
            context=context,
        )

        # 执行前处理可以规范参数，但每次替换后都会重新执行结构验证。
        try:
            for hook in self._hooks.pre_execute:
                decision = await self._await_if_needed(hook(execution))
                if decision.action == "deny":
                    return await self._finalize(
                        execution,
                        self._failure_for_execution(
                            execution,
                            status="denied",
                            error_code="pre_execute_denied",
                            error_message=decision.reason or "工具调用被拒绝。",
                        ),
                        started_at=started_at,
                    )
                if decision.action == "ask_user":
                    return await self._finalize(
                        execution,
                        self._failure_for_execution(
                            execution,
                            status="needs_user_input",
                            error_code="needs_user_input",
                            error_message=decision.reason or "需要用户补充信息。",
                        ),
                        started_at=started_at,
                    )
                if decision.arguments is not None:
                    parsed, validation_error = self._parse_arguments(
                        definition,
                        decision.arguments,
                    )
                    if parsed is None:
                        return await self._finalize(
                            execution,
                            self._failure_for_execution(
                                execution,
                                status="invalid_arguments",
                                error_code="invalid_pre_execute_arguments",
                                error_message=(
                                    validation_error or "执行前处理生成了无效参数。"
                                ),
                            ),
                            started_at=started_at,
                        )
                    execution = ToolExecution(
                        call=call.model_copy(update={"arguments": decision.arguments}),
                        definition=definition,
                        arguments=parsed,
                        context=context,
                    )
        # 扩展由应用方提供，流水线必须把任意扩展异常收敛成结构化失败。
        except Exception as exc:  # noqa: BLE001
            return await self._finalize(
                execution,
                self._failure_for_execution(
                    execution,
                    status="error",
                    error_code="pre_execute_failed",
                    error_message=f"执行前处理失败：{type(exc).__name__}",
                ),
                started_at=started_at,
            )

        # 最终检查只有“拒绝”这一种有效结果，任何后续处理都不能重新放行。
        try:
            for guard in self._hooks.guards:
                denial_reason = await self._await_if_needed(guard(execution))
                if denial_reason:
                    return await self._finalize(
                        execution,
                        self._failure_for_execution(
                            execution,
                            status="denied",
                            error_code="guard_denied",
                            error_message=str(denial_reason),
                        ),
                        started_at=started_at,
                    )
        except Exception as exc:  # noqa: BLE001
            return await self._finalize(
                execution,
                self._failure_for_execution(
                    execution,
                    status="error",
                    error_code="guard_failed",
                    error_message=f"最终检查失败：{type(exc).__name__}",
                ),
                started_at=started_at,
            )

        execution_started_at = perf_counter()
        attempt_count = 0

        async def dispatch() -> ToolBodyResult:
            nonlocal attempt_count
            maximum_attempts = 1 + definition.max_retries
            last_error: Exception | None = None
            for _ in range(maximum_attempts):
                attempt_count += 1
                try:
                    raw = definition.handler(execution.arguments, execution.context)
                    raw = await asyncio.wait_for(
                        self._await_if_needed(raw),
                        timeout=definition.timeout_seconds,
                    )
                    body = self._normalize_body(raw)
                    return self._validate_output(definition, body)
                except _OutputValidationError:
                    raise
                except Exception as exc:
                    last_error = exc
                    if attempt_count >= maximum_attempts:
                        raise
            raise AssertionError("tool retry loop exited unexpectedly") from last_error

        call_next = dispatch
        # 后注册的包装器位于更内层；每一层都可以在调用前后增加行为。
        for wrapper in reversed(self._hooks.execute_wrappers):
            inner = call_next

            async def wrapped(
                wrapper=wrapper,
                inner=inner,
            ) -> ToolBodyResult:
                raw = wrapper(execution, inner)
                return self._normalize_body(await self._await_if_needed(raw))

            call_next = wrapped

        try:
            body = await call_next()
            # 执行包装器也可能替换结果，所以离开执行阶段时再统一验证一次。
            body = self._validate_output(definition, self._normalize_body(body))
        except TimeoutError:
            provisional = self._failure_for_execution(
                execution,
                status="timeout",
                error_code="tool_timeout",
                error_message="工具执行超时。",
                attempt_count=attempt_count,
            )
        except _OutputValidationError as exc:
            provisional = self._failure_for_execution(
                execution,
                status="error",
                error_code="invalid_tool_output",
                error_message=str(exc),
                attempt_count=attempt_count,
            )
        except Exception as exc:  # noqa: BLE001
            provisional = self._failure_for_execution(
                execution,
                status="error",
                error_code="tool_execution_failed",
                error_message=f"工具执行失败：{type(exc).__name__}",
                attempt_count=attempt_count,
            )
        else:
            provisional = self._success_for_execution(
                execution,
                body,
                attempt_count=attempt_count,
            )

        provisional = provisional.model_copy(
            update={
                "execution_latency_ms": (perf_counter() - execution_started_at) * 1000.0
            }
        )
        return await self._finalize(
            execution,
            provisional,
            started_at=started_at,
        )

    async def _finalize(
        self,
        execution: ToolExecution,
        provisional: ToolResult,
        *,
        started_at: float,
    ) -> ToolResult:
        """依次执行结果处理，随后通知只读观察者。"""

        result = provisional
        try:
            for hook in self._hooks.post_execute:
                updated = await self._await_if_needed(hook(execution, result))
                if (
                    updated.call_id != execution.call.call_id
                    or updated.tool_name != execution.call.tool_name
                ):
                    raise ValueError("post-execute hooks cannot change tool identity")
                result = updated
        except Exception as exc:  # noqa: BLE001
            result = self._failure_for_execution(
                execution,
                status="error",
                error_code="post_execute_failed",
                error_message=f"执行后处理失败：{type(exc).__name__}",
                attempt_count=result.attempt_count,
            )

        try:
            finalized_value = self._json_value(result.value)
            finalized_content = result.model_content
            finalized_warnings = list(result.warnings)
            if len(finalized_content) > self._max_model_content_chars:
                finalized_content = (
                    finalized_content[: self._max_model_content_chars].rstrip()
                    + "\n[结果已截断]"
                )
                finalized_warnings.append("model_content_truncated")
            result = result.model_copy(
                update={
                    "value": finalized_value,
                    "model_content": finalized_content,
                    "warnings": finalized_warnings,
                    "total_latency_ms": (perf_counter() - started_at) * 1000.0,
                }
            )
        except _OutputValidationError:
            result = self._failure_for_execution(
                execution,
                status="error",
                error_code="invalid_final_tool_result",
                error_message="执行后结果无法保存为 JSON。",
                attempt_count=result.attempt_count,
            ).model_copy(
                update={"total_latency_ms": (perf_counter() - started_at) * 1000.0}
            )
        for observer in self._hooks.result_observers:
            try:
                await self._await_if_needed(observer(execution, result))
            except Exception:
                # 统计或界面通知失败不能反过来改变已经确定的工具结果。
                logger.exception("tool result observer failed")
        return result

    def _success_for_execution(
        self,
        execution: ToolExecution,
        body: ToolBodyResult,
        *,
        attempt_count: int,
    ) -> ToolResult:
        value = self._json_value(body.value)
        content = body.model_content or self._default_model_content(value)
        warnings = list(body.warnings)
        return ToolResult(
            call_id=execution.call.call_id,
            tool_name=execution.call.tool_name,
            status="success",
            value=value,
            model_content=content,
            warnings=warnings,
            artifact_id=body.artifact_id,
            attempt_count=attempt_count,
            nested_model_usage=body.nested_model_usage,
            nested_model_calls=body.nested_model_calls,
            terminal_answer=body.terminal_answer,
            memory_update=body.memory_update,
        )

    @staticmethod
    def _failure_for_execution(
        execution: ToolExecution,
        *,
        status: str,
        error_code: str,
        error_message: str,
        attempt_count: int = 0,
    ) -> ToolResult:
        return ToolResult(
            call_id=execution.call.call_id,
            tool_name=execution.call.tool_name,
            status=status,
            model_content=(
                f"工具 {execution.call.tool_name} 未成功执行：{error_message}"
            ),
            error_code=error_code,
            error_message=error_message,
            attempt_count=attempt_count,
        )

    def _standalone_failure(
        self,
        call: ToolCall,
        *,
        status: str,
        error_code: str,
        error_message: str,
        started_at: float,
    ) -> ToolResult:
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            status=status,
            model_content=f"工具 {call.tool_name} 未成功执行：{error_message}",
            error_code=error_code,
            error_message=error_message,
            total_latency_ms=(perf_counter() - started_at) * 1000.0,
        )

    @staticmethod
    def _parse_arguments(
        definition: ToolDefinition,
        arguments: dict[str, Any],
    ) -> tuple[BaseModel | None, str | None]:
        try:
            json.dumps(arguments, ensure_ascii=False)
            return definition.input_model.model_validate(arguments), None
        except ValidationError as exc:
            details = exc.errors(include_url=False)
            return None, json.dumps(details, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            return None, str(exc)

    @staticmethod
    def _normalize_body(raw: Any) -> ToolBodyResult:
        if isinstance(raw, ToolBodyResult):
            return raw
        return ToolBodyResult(value=raw)

    def _validate_output(
        self,
        definition: ToolDefinition,
        body: ToolBodyResult,
    ) -> ToolBodyResult:
        if definition.output_model is None:
            self._json_value(body.value)
            return body
        try:
            source = (
                body.value.model_dump(mode="json")
                if isinstance(body.value, BaseModel)
                else body.value
            )
            validated = definition.output_model.model_validate(source)
        except ValidationError as exc:
            raise _OutputValidationError(
                "工具返回内容不符合声明："
                + json.dumps(exc.errors(include_url=False), ensure_ascii=False)
            ) from exc
        return body.model_copy(update={"value": validated})

    @staticmethod
    def _json_value(value: Any) -> Any:
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
        try:
            return json.loads(json.dumps(value, ensure_ascii=False))
        except (TypeError, ValueError) as exc:
            raise _OutputValidationError("工具返回内容必须能够保存为 JSON。") from exc

    @staticmethod
    def _default_model_content(value: Any) -> str:
        if value is None:
            return "工具执行成功。"
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    async def _await_if_needed(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value
