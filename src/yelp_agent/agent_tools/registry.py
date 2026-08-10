"""Deep registry module: validate, execute, and normalize every Agent tool."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import json
from time import perf_counter
from typing import Mapping, Protocol

from pydantic import BaseModel, ValidationError

from yelp_agent.agent_benchmark.schema import AgentAction
from yelp_agent.agent_evaluation.schema import ToolKind

from .config import AgentToolRuntimeConfig
from .errors import PermanentToolError, RetryableToolError
from .schema import (
    ToolAvailability,
    ToolDescriptor,
    ToolExecutionContext,
    ToolObservation,
)


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    version: str
    kind: ToolKind
    allowed_actions: tuple[AgentAction, ...]
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    public_summary: str
    availability: ToolAvailability | Mapping[str, object] = field(
        default_factory=lambda: ToolAvailability(available=True)
    )
    resolves_information_gaps: tuple[str, ...] = ()
    resolves_uncertainties: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    cache_scope: str = "none"
    empty_result_behavior: str = "observe_no_result"
    max_attempts: int = 1
    timeout_ms: float = 30_000
    estimated_cost_usd: float = 0.0
    fallback_policy: str = "safe_hybrid_v2"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "availability",
            ToolAvailability.model_validate(self.availability),
        )
        if self.cache_scope not in {"none", "request"}:
            raise ValueError("cache_scope must be none or request")
        if self.empty_result_behavior not in {"observe_no_result", "error"}:
            raise ValueError("invalid empty_result_behavior")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")

    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name,
            version=self.version,
            kind=self.kind,
            allowed_actions=list(self.allowed_actions),
            public_summary=self.public_summary,
            availability=self.availability,
            input_schema=self.input_model.model_json_schema(),
            output_schema=self.output_model.model_json_schema(),
            preconditions=list(self.preconditions),
            resolves_information_gaps=list(self.resolves_information_gaps),
            resolves_uncertainties=list(self.resolves_uncertainties),
            cache_scope=self.cache_scope,
            empty_result_behavior=self.empty_result_behavior,
            max_attempts=self.max_attempts,
            timeout_ms=self.timeout_ms,
            estimated_cost_usd=self.estimated_cost_usd,
            fallback_policy=self.fallback_policy,
        )


class AgentTool(Protocol):
    definition: ToolDefinition

    def run(
        self,
        arguments: BaseModel,
        context: ToolExecutionContext,
    ) -> ToolObservation: ...


@dataclass(frozen=True, slots=True)
class UnavailableTool:
    """Catalog placeholder that makes future capability gaps explicit."""

    definition: ToolDefinition

    def __post_init__(self) -> None:
        if self.definition.availability.available:
            raise ValueError("UnavailableTool requires unavailable metadata")

    def run(
        self,
        arguments: BaseModel,
        context: ToolExecutionContext,
    ) -> ToolObservation:
        del arguments, context
        return ToolObservation.error(
            tool_name=self.definition.name,
            status="unavailable",
            error_code="TOOL_NOT_IMPLEMENTED",
            warning=self.definition.availability.reason or "tool is unavailable",
        )


class AgentToolRegistry:
    """The only public execution seam used by the Harness and tests."""

    def __init__(
        self,
        tools: list[AgentTool] | tuple[AgentTool, ...],
        *,
        runtime_config: AgentToolRuntimeConfig | None = None,
    ) -> None:
        self._runtime_config = runtime_config or AgentToolRuntimeConfig()
        self._tools: dict[str, AgentTool] = {}
        self._cache: OrderedDict[str, ToolObservation] = OrderedDict()
        for tool in tools:
            name = tool.definition.name
            if name in self._tools:
                raise ValueError(f"duplicate Agent tool: {name}")
            self._tools[name] = tool

    def describe(self, tool_name: str) -> ToolDescriptor:
        try:
            return self._tools[tool_name].definition.descriptor()
        except KeyError as exc:
            raise KeyError(f"unknown Agent tool: {tool_name}") from exc

    def list_tools(self) -> tuple[ToolDescriptor, ...]:
        """Return a deterministic catalog without exposing implementations."""

        return tuple(
            self._tools[name].definition.descriptor() for name in sorted(self._tools)
        )

    def available_tools(
        self,
        context: ToolExecutionContext,
    ) -> tuple[ToolDescriptor, ...]:
        return tuple(
            item
            for item in self.list_tools()
            if item.availability.available and context.action in item.allowed_actions
        )

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, object],
        context: ToolExecutionContext,
    ) -> ToolObservation:
        try:
            tool = self._tools[tool_name]
        except KeyError:
            return ToolObservation.error(
                tool_name=tool_name,
                status="permanent_error",
                error_code="UNKNOWN_TOOL",
                warning="requested tool is not registered",
            )
        if context.action not in tool.definition.allowed_actions:
            return ToolObservation.error(
                tool_name=tool_name,
                status="permanent_error",
                error_code="ACTION_NOT_ALLOWED",
                warning=(
                    f"{tool_name} cannot execute for action {context.action}"
                ),
            )
        if not tool.definition.availability.available:
            return ToolObservation.error(
                tool_name=tool_name,
                status="unavailable",
                error_code="TOOL_NOT_IMPLEMENTED",
                warning=(
                    tool.definition.availability.reason or "tool is unavailable"
                ),
            )
        try:
            parsed = tool.definition.input_model.model_validate(arguments)
        except ValidationError:
            return ToolObservation.error(
                tool_name=tool_name,
                status="permanent_error",
                error_code="INVALID_ARGUMENTS",
                warning="tool arguments failed schema validation",
            )
        parsed_values = parsed.model_dump()
        requested_ids: set[str] = set()
        if isinstance(parsed_values.get("business_id"), str):
            requested_ids.add(str(parsed_values["business_id"]))
        if isinstance(parsed_values.get("business_ids"), list):
            requested_ids.update(str(value) for value in parsed_values["business_ids"])
        if (
            requested_ids
            and context.business_scope_known
            and not requested_ids.issubset(context.business_scope)
        ):
            return ToolObservation.error(
                tool_name=tool_name,
                status="permanent_error",
                error_code="BUSINESS_OUT_OF_SCOPE",
                warning="requested business is outside the current candidate scope",
            )
        cache_key = self._cache_key(tool.definition, parsed, context)
        if cache_key is not None and cache_key in self._cache:
            cached = self._cache.pop(cache_key)
            self._cache[cache_key] = cached
            return cached.model_copy(update={"cache_hit": True})
        total_latency_ms = 0.0
        observation: ToolObservation | None = None
        attempt_limit = min(
            tool.definition.max_attempts,
            self._runtime_config.max_retry_attempts,
        )
        timeout_ms = min(
            tool.definition.timeout_ms,
            self._runtime_config.max_timeout_ms,
        )
        for attempt in range(1, attempt_limit + 1):
            started = perf_counter()
            try:
                observation = ToolObservation.model_validate(
                    tool.run(parsed, context)
                )
            except ValidationError:
                observation = ToolObservation.error(
                    tool_name=tool_name,
                    status="permanent_error",
                    error_code="MALFORMED_TOOL_OUTPUT",
                    warning="tool did not return a valid ToolObservation",
                )
            except (TimeoutError, RetryableToolError) as exc:
                observation = ToolObservation.error(
                    tool_name=tool_name,
                    status="retryable_error",
                    error_code=(
                        "TOOL_TIMEOUT" if isinstance(exc, TimeoutError) else "TEMPORARY_FAILURE"
                    ),
                    warning=type(exc).__name__,
                )
            except PermanentToolError as exc:
                observation = ToolObservation.error(
                    tool_name=tool_name,
                    status="permanent_error",
                    error_code="TOOL_REJECTED",
                    warning=str(exc) or type(exc).__name__,
                )
            except Exception as exc:
                observation = ToolObservation.error(
                    tool_name=tool_name,
                    status="permanent_error",
                    error_code="TOOL_EXCEPTION",
                    warning=type(exc).__name__,
                )
            elapsed_ms = (perf_counter() - started) * 1000.0
            total_latency_ms += elapsed_ms
            if elapsed_ms > timeout_ms:
                observation = ToolObservation.error(
                    tool_name=tool_name,
                    status="retryable_error",
                    error_code="TOOL_TIMEOUT",
                    warning="tool exceeded its declared timeout",
                )
            observation = observation.model_copy(
                update={
                    "attempt_count": attempt,
                    "latency_ms": max(observation.latency_ms, total_latency_ms),
                }
            )
            if observation.status != "retryable_error":
                break
        assert observation is not None
        if observation.tool_name != tool_name:
            return ToolObservation.error(
                tool_name=tool_name,
                status="permanent_error",
                error_code="MALFORMED_TOOL_OUTPUT",
                warning="tool observation name does not match requested tool",
            ).model_copy(
                update={
                    "attempt_count": observation.attempt_count,
                    "latency_ms": observation.latency_ms,
                }
            )
        if observation.status in {"success", "partial", "no_result"}:
            try:
                validated = tool.definition.output_model.model_validate(
                    observation.data
                )
            except ValidationError:
                return ToolObservation.error(
                    tool_name=tool_name,
                    status="permanent_error",
                    error_code="MALFORMED_TOOL_OUTPUT",
                    warning="tool result failed schema validation",
                ).model_copy(
                    update={
                        "attempt_count": observation.attempt_count,
                        "latency_ms": observation.latency_ms,
                    }
                )
            observation = observation.model_copy(update={"data": validated.model_dump()})
        if cache_key is not None and observation.status in {
            "success",
            "partial",
            "no_result",
        }:
            self._cache[cache_key] = observation.model_copy(update={"cache_hit": False})
            while len(self._cache) > self._runtime_config.cache_max_entries:
                self._cache.popitem(last=False)
        return observation

    @staticmethod
    def _cache_key(
        definition: ToolDefinition,
        arguments: BaseModel,
        context: ToolExecutionContext,
    ) -> str | None:
        if definition.cache_scope == "none":
            return None
        payload = json.dumps(
            {
                "tool": definition.name,
                "version": definition.version,
                "request_id": context.request_id,
                "user_id": context.user_id,
                "cutoff_time": context.cutoff_time.isoformat(),
                "arguments": arguments.model_dump(mode="json"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
