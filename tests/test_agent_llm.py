from __future__ import annotations

from types import SimpleNamespace

from yelp_agent.agent.llm import (
    LLMMessage,
    LLMTransportError,
    LLMTransportResponse,
    OpenAIChatTransport,
    OpenAICompatibleLLM,
)
from yelp_agent.config import AgentConfig


def _agent_config() -> AgentConfig:
    return AgentConfig(
        enabled=True,
        temperature=0,
        timeout_seconds=30,
        max_retries=2,
    )


class FailIfCalledTransport:
    def __init__(self) -> None:
        self.call_count = 0

    def complete(self, **kwargs: object) -> object:
        self.call_count += 1
        raise AssertionError("disabled LLM must not call its transport")


def test_missing_key_uses_no_llm_mode_without_sending_a_request() -> None:
    transport = FailIfCalledTransport()
    llm = OpenAICompatibleLLM.from_environment(
        _agent_config(),
        environment={
            "OPENAI_BASE_URL": "https://api.deepseek.com",
            "OPENAI_MODEL": "deepseek-v4-flash",
        },
        transport=transport,
    )

    result = llm.generate([LLMMessage(role="user", content="rerank")])

    assert result.status == "disabled"
    assert result.failure_reason == "llm_disabled"
    assert result.attempt_count == 0
    assert result.content is None
    assert result.model == "deepseek-v4-flash"
    assert transport.call_count == 0


class RecordingTransport:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def complete(self, **kwargs: object) -> LLMTransportResponse:
        self.requests.append(kwargs)
        return LLMTransportResponse(
            content='{"ranking": ["business-1"]}',
            input_tokens=120,
            output_tokens=30,
            total_tokens=150,
            provider_request_id="request-success-1",
        )


class StreamingTransport(RecordingTransport):
    def stream(self, **kwargs: object) -> LLMTransportResponse:
        self.requests.append(kwargs)
        on_delta = kwargs["on_delta"]
        on_delta("第一段")
        on_delta("第二段")
        return LLMTransportResponse(
            content="第一段第二段",
            input_tokens=80,
            output_tokens=12,
            total_tokens=92,
            provider_request_id="stream-request-1",
        )


def test_stream_forwards_each_delta_and_returns_complete_usage() -> None:
    transport = StreamingTransport()
    llm = OpenAICompatibleLLM.from_environment(
        _agent_config(),
        environment={
            "OPENAI_API_KEY": "secret-test-key",
            "OPENAI_MODEL": "deepseek-v4-flash",
        },
        transport=transport,
    )
    deltas: list[str] = []

    result = llm.stream(
        [LLMMessage(role="user", content="生成推荐")],
        deltas.append,
    )

    assert deltas == ["第一段", "第二段"]
    assert result.status == "success"
    assert result.content == "第一段第二段"
    assert result.input_tokens == 80
    assert result.output_tokens == 12


def test_successful_call_forwards_deterministic_settings_and_usage() -> None:
    transport = RecordingTransport()
    llm = OpenAICompatibleLLM.from_environment(
        _agent_config(),
        environment={
            "OPENAI_API_KEY": "secret-test-key",
            "OPENAI_BASE_URL": "https://api.deepseek.com",
            "OPENAI_MODEL": "deepseek-v4-flash",
        },
        transport=transport,
    )
    messages = [
        LLMMessage(role="system", content="Return JSON only."),
        LLMMessage(role="user", content="Rerank these businesses."),
    ]

    result = llm.generate(messages)

    assert result.status == "success"
    assert result.content == '{"ranking": ["business-1"]}'
    assert result.model == "deepseek-v4-flash"
    assert result.attempt_count == 1
    assert result.input_tokens == 120
    assert result.output_tokens == 30
    assert result.total_tokens == 150
    assert result.failure_reason is None
    assert result.latency_ms >= 0
    assert result.observed_total_tokens == 150
    assert result.unknown_usage_attempts == 0
    assert len(result.attempts) == 1
    attempt = result.attempts[0]
    assert attempt.attempt_index == 1
    assert attempt.status == "success"
    assert attempt.failure_reason is None
    assert attempt.retryable is False
    assert attempt.total_tokens == 150
    assert attempt.provider_request_id == "request-success-1"
    assert attempt.usage_unknown is False
    assert attempt.latency_ms >= 0
    assert transport.requests == [
        {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": "Rerank these businesses."},
            ],
            "temperature": 0.0,
            "timeout_seconds": 30.0,
        }
    ]


def test_structured_non_thinking_request_options_are_forwarded() -> None:
    transport = RecordingTransport()
    config = AgentConfig(
        enabled=True,
        temperature=0,
        timeout_seconds=90,
        max_retries=1,
        max_tokens=2000,
        response_format_json=True,
        thinking="disabled",
    )
    llm = OpenAICompatibleLLM.from_environment(
        config,
        environment={
            "OPENAI_API_KEY": "secret-test-key",
            "OPENAI_MODEL": "deepseek-v4-flash",
        },
        transport=transport,
    )

    llm.generate([LLMMessage(role="user", content="Return JSON.")])

    assert transport.requests[0]["max_tokens"] == 2000
    assert transport.requests[0]["response_format_json"] is True
    assert transport.requests[0]["thinking"] == "disabled"


class TwiceTimeoutThenSuccessTransport:
    def __init__(self) -> None:
        self.call_count = 0

    def complete(self, **kwargs: object) -> LLMTransportResponse:
        self.call_count += 1
        if self.call_count <= 2:
            raise LLMTransportError("timeout", retryable=True)
        return LLMTransportResponse(
            content='{"ranking": ["business-2"]}',
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            provider_request_id="request-success-3",
        )


def test_retryable_failures_are_retried_at_most_configured_times() -> None:
    transport = TwiceTimeoutThenSuccessTransport()
    llm = OpenAICompatibleLLM.from_environment(
        _agent_config(),
        environment={
            "OPENAI_API_KEY": "secret-test-key",
            "OPENAI_BASE_URL": "https://api.deepseek.com",
            "OPENAI_MODEL": "deepseek-v4-flash",
        },
        transport=transport,
    )

    result = llm.generate([LLMMessage(role="user", content="rerank")])

    assert result.status == "success"
    assert result.attempt_count == 3
    assert transport.call_count == 3
    assert [attempt.status for attempt in result.attempts] == [
        "failure",
        "failure",
        "success",
    ]
    assert [attempt.failure_reason for attempt in result.attempts] == [
        "timeout",
        "timeout",
        None,
    ]
    assert [attempt.retryable for attempt in result.attempts] == [
        True,
        True,
        False,
    ]
    assert all(attempt.latency_ms >= 0 for attempt in result.attempts)
    assert result.attempts[0].usage_unknown is True
    assert result.attempts[1].usage_unknown is True
    assert result.attempts[2].usage_unknown is False
    assert result.attempts[2].provider_request_id == "request-success-3"
    assert result.observed_total_tokens == 15
    assert result.unknown_usage_attempts == 2


class EmptyResponseTransport:
    def complete(self, **kwargs: object) -> LLMTransportResponse:
        return LLMTransportResponse(
            content="   ",
            input_tokens=20,
            output_tokens=0,
            total_tokens=20,
        )


def test_empty_model_content_is_a_structured_failure() -> None:
    llm = OpenAICompatibleLLM.from_environment(
        _agent_config(),
        environment={
            "OPENAI_API_KEY": "secret-test-key",
            "OPENAI_BASE_URL": "https://api.deepseek.com",
            "OPENAI_MODEL": "deepseek-v4-flash",
        },
        transport=EmptyResponseTransport(),
    )

    result = llm.generate([LLMMessage(role="user", content="rerank")])

    assert result.status == "failure"
    assert result.failure_reason == "empty_response"
    assert result.content is None
    assert result.attempt_count == 1
    assert result.total_tokens == 20
    assert result.observed_total_tokens == 20
    assert result.unknown_usage_attempts == 0
    assert len(result.attempts) == 1
    assert result.attempts[0].status == "failure"
    assert result.attempts[0].failure_reason == "empty_response"
    assert result.attempts[0].usage_unknown is False


class FakeOpenAICompletions:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        return SimpleNamespace(
            _request_id="provider-request-50",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"ranking": []}')
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=40,
                completion_tokens=10,
                total_tokens=50,
            ),
        )


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeOpenAICompletions())


def test_openai_adapter_converts_sdk_response_without_provider_objects() -> None:
    client = FakeOpenAIClient()
    transport = OpenAIChatTransport(client)

    result = transport.complete(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": "rerank"}],
        temperature=0.0,
        timeout_seconds=30.0,
    )

    assert result == LLMTransportResponse(
        content='{"ranking": []}',
        input_tokens=40,
        output_tokens=10,
        total_tokens=50,
        provider_request_id="provider-request-50",
    )
    assert client.chat.completions.requests == [
        {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "rerank"}],
            "temperature": 0.0,
            "timeout": 30.0,
        }
    ]


def test_openai_adapter_maps_deepseek_structured_request_options() -> None:
    client = FakeOpenAIClient()
    transport = OpenAIChatTransport(client)

    transport.complete(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": "Return JSON."}],
        temperature=0.0,
        timeout_seconds=90.0,
        max_tokens=2000,
        response_format_json=True,
        thinking="disabled",
    )

    assert client.chat.completions.requests == [
        {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "Return JSON."}],
            "temperature": 0.0,
            "timeout": 90.0,
            "max_tokens": 2000,
            "response_format": {"type": "json_object"},
            "extra_body": {"thinking": {"type": "disabled"}},
        }
    ]


class AlwaysTimeoutTransport:
    def __init__(self) -> None:
        self.call_count = 0

    def complete(self, **kwargs: object) -> LLMTransportResponse:
        self.call_count += 1
        raise LLMTransportError("timeout", retryable=True)


def test_retry_exhaustion_returns_failure_instead_of_raising() -> None:
    transport = AlwaysTimeoutTransport()
    llm = OpenAICompatibleLLM.from_environment(
        _agent_config(),
        environment={
            "OPENAI_API_KEY": "secret-test-key",
            "OPENAI_MODEL": "deepseek-v4-flash",
        },
        transport=transport,
    )

    result = llm.generate([LLMMessage(role="user", content="rerank")])

    assert result.status == "failure"
    assert result.failure_reason == "timeout"
    assert result.attempt_count == 3
    assert transport.call_count == 3
    assert len(result.attempts) == 3
    assert all(attempt.status == "failure" for attempt in result.attempts)
    assert all(
        attempt.failure_reason == "timeout" for attempt in result.attempts
    )
    assert all(attempt.retryable is True for attempt in result.attempts)
    assert result.observed_total_tokens is None
    assert result.unknown_usage_attempts == 3


class LeakyUnexpectedErrorTransport:
    def complete(self, **kwargs: object) -> LLMTransportResponse:
        raise RuntimeError("Authorization: Bearer secret-test-key")


def test_unexpected_error_text_and_api_key_are_not_returned() -> None:
    llm = OpenAICompatibleLLM.from_environment(
        _agent_config(),
        environment={
            "OPENAI_API_KEY": "secret-test-key",
            "OPENAI_MODEL": "deepseek-v4-flash",
        },
        transport=LeakyUnexpectedErrorTransport(),
    )

    result = llm.generate([LLMMessage(role="user", content="rerank")])
    serialized = result.model_dump_json()

    assert result.status == "failure"
    assert result.failure_reason == "api_error"
    assert result.attempt_count == 1
    assert len(result.attempts) == 1
    assert result.attempts[0].failure_reason == "api_error"
    assert result.attempts[0].retryable is False
    assert result.unknown_usage_attempts == 1
    assert "secret-test-key" not in serialized
    assert "Authorization" not in serialized
