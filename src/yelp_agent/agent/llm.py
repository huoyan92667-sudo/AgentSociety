"""OpenAI-compatible chat client with deterministic no-LLM fallback."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any, Literal, Protocol

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
)
from pydantic import Field

from yelp_agent.config import (
    AgentConfig,
    LLMEnvironment,
    load_llm_environment,
)
from yelp_agent.models import StrictModel


class LLMMessage(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class LLMAttemptTrace(StrictModel):
    """One sanitized provider attempt within a logical LLM call."""

    attempt_index: int = Field(ge=1)
    status: Literal["success", "failure"]
    latency_ms: float = Field(ge=0)
    failure_reason: str | None = None
    retryable: bool
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    provider_request_id: str | None = None
    usage_unknown: bool = False


class LLMCallResult(StrictModel):
    status: Literal["success", "disabled", "failure"]
    content: str | None = None
    model: str | None = None
    latency_ms: float = Field(ge=0)
    attempt_count: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    failure_reason: str | None = None
    attempts: list[LLMAttemptTrace] = Field(default_factory=list)
    observed_total_tokens: int | None = Field(default=None, ge=0)
    unknown_usage_attempts: int = Field(default=0, ge=0)


class LLMTransportResponse(StrictModel):
    content: str
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    provider_request_id: str | None = None


_FAILURE_REASONS = {
    "timeout",
    "rate_limit",
    "api_connection",
    "server_error",
    "authentication_error",
    "permission_error",
    "bad_request",
    "api_error",
}


class LLMTransportError(RuntimeError):
    """Provider-neutral error containing no response body or secret."""

    def __init__(self, reason: str, *, retryable: bool) -> None:
        if reason not in _FAILURE_REASONS:
            raise ValueError(f"unsupported LLM failure reason: {reason!r}")
        super().__init__(reason)
        self.reason = reason
        self.retryable = retryable


class _ChatTransport(Protocol):
    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        timeout_seconds: float,
    ) -> LLMTransportResponse: ...


class OpenAIChatTransport:
    """Adapt the OpenAI Python SDK to provider-neutral response models."""

    def __init__(self, client: Any) -> None:
        self._client = client

    @classmethod
    def from_settings(
        cls,
        config: AgentConfig,
        environment: LLMEnvironment,
    ) -> "OpenAIChatTransport":
        if not environment.llm_enabled or environment.api_key is None:
            raise ValueError("cannot create an OpenAI transport without LLM config")
        client_options: dict[str, Any] = {
            "api_key": environment.api_key.get_secret_value(),
            "max_retries": 0,
            "timeout": float(config.timeout_seconds),
        }
        if environment.base_url is not None:
            client_options["base_url"] = environment.base_url
        return cls(OpenAI(**client_options))

    @staticmethod
    def _translate_error(exc: OpenAIError) -> LLMTransportError:
        if isinstance(exc, APITimeoutError):
            return LLMTransportError("timeout", retryable=True)
        if isinstance(exc, APIConnectionError):
            return LLMTransportError("api_connection", retryable=True)
        if isinstance(exc, RateLimitError):
            return LLMTransportError("rate_limit", retryable=True)
        if isinstance(exc, AuthenticationError):
            return LLMTransportError("authentication_error", retryable=False)
        if isinstance(exc, PermissionDeniedError):
            return LLMTransportError("permission_error", retryable=False)
        if isinstance(exc, BadRequestError):
            return LLMTransportError("bad_request", retryable=False)
        if isinstance(exc, APIStatusError):
            retryable = exc.status_code in {408, 409} or exc.status_code >= 500
            return LLMTransportError(
                "server_error" if retryable else "api_error",
                retryable=retryable,
            )
        return LLMTransportError("api_error", retryable=False)

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        timeout_seconds: float,
    ) -> LLMTransportResponse:
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout_seconds,
            )
        except OpenAIError as exc:
            raise self._translate_error(exc) from None

        content = ""
        if response.choices:
            raw_content = response.choices[0].message.content
            content = raw_content if isinstance(raw_content, str) else ""
        usage = getattr(response, "usage", None)
        raw_request_id = getattr(response, "_request_id", None)
        if raw_request_id is None:
            raw_request_id = getattr(response, "request_id", None)
        return LLMTransportResponse(
            content=content,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            provider_request_id=(
                raw_request_id if isinstance(raw_request_id, str) else None
            ),
        )


class OpenAICompatibleLLM:
    """Hide provider setup, retries, usage extraction, and safe failures."""

    def __init__(
        self,
        config: AgentConfig,
        environment: LLMEnvironment,
        *,
        transport: _ChatTransport | None = None,
    ) -> None:
        self._config = config
        self._environment = environment
        self._transport = transport
        if (
            self._transport is None
            and self._config.enabled
            and self._environment.llm_enabled
        ):
            self._transport = OpenAIChatTransport.from_settings(
                self._config,
                self._environment,
            )

    @classmethod
    def from_environment(
        cls,
        config: AgentConfig,
        *,
        environment: Mapping[str, str] | None = None,
        transport: _ChatTransport | None = None,
    ) -> "OpenAICompatibleLLM":
        return cls(
            config,
            load_llm_environment(environment),
            transport=transport,
        )

    def generate(self, messages: Sequence[LLMMessage]) -> LLMCallResult:
        """Generate one chat response or return a structured safe failure."""

        if not messages:
            raise ValueError("messages cannot be empty")
        if not self._config.enabled or not self._environment.llm_enabled:
            return LLMCallResult(
                status="disabled",
                content=None,
                model=self._environment.model,
                latency_ms=0.0,
                attempt_count=0,
                failure_reason="llm_disabled",
            )
        if self._transport is None:
            raise AssertionError("enabled LLM has no transport")
        started_at = perf_counter()
        request_messages = [message.model_dump() for message in messages]
        maximum_attempts = 1 + max(0, self._config.max_retries)
        attempts: list[LLMAttemptTrace] = []
        for attempt_count in range(1, maximum_attempts + 1):
            attempt_started_at = perf_counter()
            try:
                response = self._transport.complete(
                    model=self._environment.model,
                    messages=request_messages,
                    temperature=float(self._config.temperature),
                    timeout_seconds=float(self._config.timeout_seconds),
                )
            except LLMTransportError as exc:
                attempts.append(
                    LLMAttemptTrace(
                        attempt_index=attempt_count,
                        status="failure",
                        latency_ms=(
                            perf_counter() - attempt_started_at
                        )
                        * 1000.0,
                        failure_reason=exc.reason,
                        retryable=exc.retryable,
                        usage_unknown=True,
                    )
                )
                if exc.retryable and attempt_count < maximum_attempts:
                    continue
                return LLMCallResult(
                    status="failure",
                    content=None,
                    model=self._environment.model,
                    latency_ms=(perf_counter() - started_at) * 1000.0,
                    attempt_count=attempt_count,
                    failure_reason=exc.reason,
                    attempts=attempts,
                    observed_total_tokens=_observed_total_tokens(attempts),
                    unknown_usage_attempts=_unknown_usage_attempts(attempts),
                )
            except Exception:
                attempts.append(
                    LLMAttemptTrace(
                        attempt_index=attempt_count,
                        status="failure",
                        latency_ms=(
                            perf_counter() - attempt_started_at
                        )
                        * 1000.0,
                        failure_reason="api_error",
                        retryable=False,
                        usage_unknown=True,
                    )
                )
                return LLMCallResult(
                    status="failure",
                    content=None,
                    model=self._environment.model,
                    latency_ms=(perf_counter() - started_at) * 1000.0,
                    attempt_count=attempt_count,
                    failure_reason="api_error",
                    attempts=attempts,
                    observed_total_tokens=_observed_total_tokens(attempts),
                    unknown_usage_attempts=_unknown_usage_attempts(attempts),
                )
            if not response.content.strip():
                attempts.append(
                    LLMAttemptTrace(
                        attempt_index=attempt_count,
                        status="failure",
                        latency_ms=(
                            perf_counter() - attempt_started_at
                        )
                        * 1000.0,
                        failure_reason="empty_response",
                        retryable=False,
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens,
                        total_tokens=response.total_tokens,
                        provider_request_id=response.provider_request_id,
                        usage_unknown=response.total_tokens is None,
                    )
                )
                return LLMCallResult(
                    status="failure",
                    content=None,
                    model=self._environment.model,
                    latency_ms=(perf_counter() - started_at) * 1000.0,
                    attempt_count=attempt_count,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    total_tokens=response.total_tokens,
                    failure_reason="empty_response",
                    attempts=attempts,
                    observed_total_tokens=_observed_total_tokens(attempts),
                    unknown_usage_attempts=_unknown_usage_attempts(attempts),
                )
            attempts.append(
                LLMAttemptTrace(
                    attempt_index=attempt_count,
                    status="success",
                    latency_ms=(perf_counter() - attempt_started_at) * 1000.0,
                    failure_reason=None,
                    retryable=False,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    total_tokens=response.total_tokens,
                    provider_request_id=response.provider_request_id,
                    usage_unknown=response.total_tokens is None,
                )
            )
            return LLMCallResult(
                status="success",
                content=response.content,
                model=self._environment.model,
                latency_ms=(perf_counter() - started_at) * 1000.0,
                attempt_count=attempt_count,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                total_tokens=response.total_tokens,
                failure_reason=None,
                attempts=attempts,
                observed_total_tokens=_observed_total_tokens(attempts),
                unknown_usage_attempts=_unknown_usage_attempts(attempts),
            )
        raise AssertionError("LLM retry loop exited without a result")


def _observed_total_tokens(attempts: Sequence[LLMAttemptTrace]) -> int | None:
    observed = [
        attempt.total_tokens
        for attempt in attempts
        if attempt.total_tokens is not None
    ]
    return sum(observed) if observed else None


def _unknown_usage_attempts(attempts: Sequence[LLMAttemptTrace]) -> int:
    return sum(attempt.usage_unknown for attempt in attempts)
