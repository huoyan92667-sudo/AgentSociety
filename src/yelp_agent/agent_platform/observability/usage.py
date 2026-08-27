"""一轮对话内的模型、工具、时间消耗统计。"""

from __future__ import annotations

from time import perf_counter

from ..runtime.schema import ModelResponse, TurnUsage
from ..tools.result import ToolResult


class TurnUsageTracker:
    """集中统计，避免每个工具重复编写相同计数逻辑。"""

    def __init__(self) -> None:
        self._started_at = perf_counter()
        self._model_calls = 0
        self._tool_calls = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._model_latency_ms = 0.0
        self._tool_latency_ms = 0.0

    def record_model(self, response: ModelResponse) -> None:
        self._model_calls += 1
        self._input_tokens += response.usage.input_tokens
        self._output_tokens += response.usage.output_tokens
        self._model_latency_ms += response.latency_ms

    def record_tool(self, result: ToolResult) -> None:
        self._tool_calls += 1
        self._model_calls += result.nested_model_calls
        self._input_tokens += result.nested_model_usage.input_tokens
        self._output_tokens += result.nested_model_usage.output_tokens
        self._tool_latency_ms += result.total_latency_ms

    @property
    def tool_call_count(self) -> int:
        return self._tool_calls

    @property
    def total_tokens(self) -> int:
        return self._input_tokens + self._output_tokens

    def snapshot(self) -> TurnUsage:
        return TurnUsage(
            model_calls=self._model_calls,
            tool_calls=self._tool_calls,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            model_latency_ms=self._model_latency_ms,
            tool_latency_ms=self._tool_latency_ms,
            total_latency_ms=(perf_counter() - self._started_at) * 1000.0,
        )
