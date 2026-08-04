from __future__ import annotations

from collections.abc import Callable
import json

import pytest

from yelp_agent.agent.llm import LLMAttemptTrace, LLMCallResult, LLMMessage
from yelp_agent.models import Prediction, RecommendationTask
from yelp_agent.protocols import Ranker
from yelp_agent.rankers.agent_ranker import AgentRanker

from test_agent_prompt import _prompt_fixture


class FixedHybridRanker:
    def __init__(self, ranking: list[str]) -> None:
        self.ranking = ranking

    def rank(self, task: RecommendationTask) -> Prediction:
        return Prediction(
            task_id=task.task_id,
            ranking=self.ranking,
            latency_ms=1.0,
            fallback=False,
            metadata={
                "method": "hybrid",
                "weights": {
                    "category": 0.0,
                    "text": 0.5,
                    "quality": 0.1,
                    "location": 0.4,
                },
                "llm_attempted": False,
            },
        )


class FixedToolSession:
    def __init__(self, history: object, profile: object, hybrid: object, details: object):
        self._history = history
        self._profile = profile
        self._hybrid = hybrid
        self._details = details
        self.calls: list[str] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def get_user_history(self, *args: object, **kwargs: object) -> object:
        self.calls.append("history")
        return self._history

    def get_user_profile(self, *args: object, **kwargs: object) -> object:
        self.calls.append("profile")
        return self._profile

    def get_hybrid_ranking(self, *args: object, **kwargs: object) -> object:
        self.calls.append("hybrid")
        return self._hybrid

    def get_business_details(
        self,
        business_ids: list[str],
        *args: object,
        **kwargs: object,
    ) -> object:
        self.calls.append("details")
        assert business_ids == self._hybrid.ranking[:8]
        return self._details


class FixedToolbox:
    def __init__(self, session: FixedToolSession) -> None:
        self.session = session

    def for_task(self, task: RecommendationTask) -> FixedToolSession:
        return self.session


class SuccessfulLLM:
    def __init__(self, reranking: list[str]) -> None:
        self._reranking = reranking
        self.requests: list[list[LLMMessage]] = []

    def generate(self, messages: list[LLMMessage]) -> LLMCallResult:
        self.requests.append(messages)
        return LLMCallResult(
            status="success",
            content=json.dumps(
                {
                    "ranking": self._reranking,
                    "reason": "Personal taste favors these candidates.",
                }
            ),
            model="deepseek-v4-flash",
            latency_ms=10.0,
            attempt_count=1,
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            attempts=[
                LLMAttemptTrace(
                    attempt_index=1,
                    status="success",
                    latency_ms=10.0,
                    failure_reason=None,
                    retryable=False,
                    input_tokens=100,
                    output_tokens=20,
                    total_tokens=120,
                    provider_request_id="fake-request-1",
                    usage_unknown=False,
                )
            ],
            observed_total_tokens=120,
            unknown_usage_attempts=0,
        )


class DisabledLLM:
    def generate(self, messages: list[LLMMessage]) -> LLMCallResult:
        return LLMCallResult(
            status="disabled",
            content=None,
            model="deepseek-v4-flash",
            latency_ms=0.0,
            attempt_count=0,
            failure_reason="llm_disabled",
        )


class NonJsonLLM:
    def generate(self, messages: list[LLMMessage]) -> LLMCallResult:
        return LLMCallResult(
            status="success",
            content="I recommend the first restaurant.",
            model="deepseek-v4-flash",
            latency_ms=5.0,
            attempt_count=1,
            input_tokens=80,
            output_tokens=8,
            total_tokens=88,
        )


class RaisingLLM:
    def generate(self, messages: list[LLMMessage]) -> LLMCallResult:
        raise RuntimeError("Authorization: Bearer secret-live-key")


class TimeoutLLM:
    def generate(self, messages: list[LLMMessage]) -> LLMCallResult:
        return LLMCallResult(
            status="failure",
            content=None,
            model="deepseek-v4-flash",
            latency_ms=30_000.0,
            attempt_count=3,
            failure_reason="timeout",
            attempts=[
                LLMAttemptTrace(
                    attempt_index=index,
                    status="failure",
                    latency_ms=10_000.0,
                    failure_reason="timeout",
                    retryable=True,
                    usage_unknown=True,
                )
                for index in range(1, 4)
            ],
            unknown_usage_attempts=3,
        )


class FailingProfileSession(FixedToolSession):
    def get_user_profile(self, *args: object, **kwargs: object) -> object:
        self.calls.append("profile")
        raise RuntimeError("corrupt profile details must stay private")


class NeverCalledLLM:
    def generate(self, messages: list[LLMMessage]) -> LLMCallResult:
        raise AssertionError("LLM must not run after a tool failure")


class FixedContentLLM:
    def __init__(self, content: str) -> None:
        self._content = content

    def generate(self, messages: list[LLMMessage]) -> LLMCallResult:
        return LLMCallResult(
            status="success",
            content=self._content,
            model="deepseek-v4-flash",
            latency_ms=2.0,
            attempt_count=1,
            input_tokens=50,
            output_tokens=10,
            total_tokens=60,
        )


def test_agent_ranker_reranks_top_eight_and_records_safe_trace() -> None:
    task, history, profile, hybrid, details = _prompt_fixture()
    session = FixedToolSession(history, profile, hybrid, details)
    reranked_top = list(reversed(hybrid.ranking[:8]))
    llm = SuccessfulLLM(reranked_top)
    ranker = AgentRanker(
        hybrid_ranker=FixedHybridRanker(hybrid.ranking),
        toolbox=FixedToolbox(session),
        llm=llm,
    )

    prediction = ranker.rank(task)

    assert isinstance(ranker, Ranker)
    assert prediction.ranking == [*reranked_top, *hybrid.ranking[8:]]
    assert prediction.fallback is False
    assert prediction.tool_calls == 4
    assert prediction.llm_tokens == 120
    assert prediction.metadata["method"] == "hybrid_agent"
    assert prediction.metadata["llm_attempted"] is True
    assert prediction.metadata["attempt_count"] == 1
    assert session.calls == ["history", "profile", "hybrid", "details"]
    assert len(llm.requests) == 1

    trace = ranker.trace_for(task.task_id)
    assert trace.task_id == task.task_id
    assert trace.representative_review_ids == [
        "review-positive",
        "review-negative",
    ]
    assert trace.hybrid_ranking == hybrid.ranking
    assert trace.top_k_before == hybrid.ranking[:8]
    assert trace.top_k_after == reranked_top
    assert trace.tool_calls == 4
    assert trace.llm_reason == "Personal taste favors these candidates."
    assert trace.llm_latency_ms == 10.0
    assert trace.observed_total_tokens == 120
    assert trace.unknown_usage_attempts == 0
    assert len(trace.llm_attempts) == 1
    assert trace.llm_attempts[0].provider_request_id == "fake-request-1"
    assert trace.timing is not None
    assert trace.timing.hybrid_ms >= 0
    assert trace.timing.tools_ms >= 0
    assert trace.timing.prompt_ms >= 0
    assert trace.timing.llm_ms >= 0
    assert trace.timing.parse_merge_ms >= 0
    assert trace.timing.total_ms == prediction.latency_ms
    assert trace.fallback is False
    assert trace.fallback_reason is None


def test_disabled_llm_returns_complete_hybrid_without_counting_a_failure() -> None:
    task, history, profile, hybrid, details = _prompt_fixture()
    session = FixedToolSession(history, profile, hybrid, details)
    ranker = AgentRanker(
        hybrid_ranker=FixedHybridRanker(hybrid.ranking),
        toolbox=FixedToolbox(session),
        llm=DisabledLLM(),
    )

    prediction = ranker.rank(task)

    assert prediction.ranking == hybrid.ranking
    assert prediction.fallback is True
    assert prediction.fallback_reason == "llm_disabled"
    assert prediction.metadata["llm_attempted"] is False
    assert prediction.tool_calls == 4
    assert prediction.llm_tokens is None
    trace = ranker.trace_for(task.task_id)
    assert trace.llm_status == "disabled"
    assert trace.top_k_after == hybrid.ranking[:8]
    assert trace.fallback is True
    assert trace.fallback_reason == "llm_disabled"


def test_invalid_llm_output_falls_back_to_the_entire_hybrid_ranking() -> None:
    task, history, profile, hybrid, details = _prompt_fixture()
    session = FixedToolSession(history, profile, hybrid, details)
    ranker = AgentRanker(
        hybrid_ranker=FixedHybridRanker(hybrid.ranking),
        toolbox=FixedToolbox(session),
        llm=NonJsonLLM(),
    )

    prediction = ranker.rank(task)

    assert prediction.ranking == hybrid.ranking
    assert prediction.fallback is True
    assert prediction.fallback_reason == "non_json"
    assert prediction.metadata["llm_attempted"] is True
    assert prediction.llm_tokens == 88
    trace = ranker.trace_for(task.task_id)
    assert trace.llm_status == "success"
    assert trace.top_k_after == hybrid.ranking[:8]
    assert trace.total_tokens == 88
    assert trace.fallback_reason == "non_json"


def test_unexpected_llm_exception_is_sanitized_and_falls_back() -> None:
    task, history, profile, hybrid, details = _prompt_fixture()
    session = FixedToolSession(history, profile, hybrid, details)
    ranker = AgentRanker(
        hybrid_ranker=FixedHybridRanker(hybrid.ranking),
        toolbox=FixedToolbox(session),
        llm=RaisingLLM(),
    )

    prediction = ranker.rank(task)
    serialized = prediction.model_dump_json()

    assert prediction.ranking == hybrid.ranking
    assert prediction.fallback is True
    assert prediction.fallback_reason == "llm_error"
    assert prediction.metadata["llm_attempted"] is True
    assert "secret-live-key" not in serialized
    assert "Authorization" not in serialized
    assert ranker.trace_for(task.task_id).fallback_reason == "llm_error"


def test_api_failure_preserves_attempt_metadata_for_failure_rate() -> None:
    task, history, profile, hybrid, details = _prompt_fixture()
    session = FixedToolSession(history, profile, hybrid, details)
    ranker = AgentRanker(
        hybrid_ranker=FixedHybridRanker(hybrid.ranking),
        toolbox=FixedToolbox(session),
        llm=TimeoutLLM(),
    )

    prediction = ranker.rank(task)

    assert prediction.ranking == hybrid.ranking
    assert prediction.fallback_reason == "timeout"
    assert prediction.metadata["llm_attempted"] is True
    assert prediction.metadata["attempt_count"] == 3
    trace = ranker.trace_for(task.task_id)
    assert trace.attempt_count == 3
    assert trace.model == "deepseek-v4-flash"
    assert len(trace.llm_attempts) == 3
    assert trace.unknown_usage_attempts == 3
    assert all(
        attempt.failure_reason == "timeout"
        for attempt in trace.llm_attempts
    )


def test_tool_failure_falls_back_before_any_llm_request() -> None:
    task, history, profile, hybrid, details = _prompt_fixture()
    session = FailingProfileSession(history, profile, hybrid, details)
    ranker = AgentRanker(
        hybrid_ranker=FixedHybridRanker(hybrid.ranking),
        toolbox=FixedToolbox(session),
        llm=NeverCalledLLM(),
    )

    prediction = ranker.rank(task)

    assert prediction.ranking == hybrid.ranking
    assert prediction.fallback is True
    assert prediction.fallback_reason == "tool_error"
    assert prediction.metadata["llm_attempted"] is False
    assert prediction.tool_calls == 2
    trace = ranker.trace_for(task.task_id)
    assert trace.llm_status == "not_called"
    assert trace.profile is None
    assert trace.tool_calls == 2
    assert trace.fallback_reason == "tool_error"


@pytest.mark.parametrize(
    "content_factory, expected_reason",
    [
        (lambda top: "", "empty_response"),
        (
            lambda top: json.dumps({"ranking": [*top[:7], top[0]]}),
            "duplicate_id",
        ),
        (
            lambda top: json.dumps({"ranking": top[:7]}),
            "missing_id",
        ),
        (
            lambda top: json.dumps(
                {"ranking": [*top[:7], "outside-candidate"]}
            ),
            "unknown_id",
        ),
    ],
)
def test_every_invalid_top_eight_shape_returns_full_hybrid(
    content_factory: Callable[[list[str]], str],
    expected_reason: str,
) -> None:
    task, history, profile, hybrid, details = _prompt_fixture()
    session = FixedToolSession(history, profile, hybrid, details)
    content = content_factory(hybrid.ranking[:8])
    ranker = AgentRanker(
        hybrid_ranker=FixedHybridRanker(hybrid.ranking),
        toolbox=FixedToolbox(session),
        llm=FixedContentLLM(content),
    )

    prediction = ranker.rank(task)

    assert prediction.ranking == hybrid.ranking
    assert prediction.fallback_reason == expected_reason
    assert prediction.metadata["llm_attempted"] is True
