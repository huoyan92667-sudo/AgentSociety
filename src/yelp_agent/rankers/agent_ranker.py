"""Hybrid-first LLM Agent ranker with complete deterministic fallback."""

from __future__ import annotations

from time import perf_counter
from typing import Literal, Protocol, Sequence

from pydantic import Field

from yelp_agent.agent.llm import LLMAttemptTrace, LLMCallResult, LLMMessage
from yelp_agent.agent.parser import (
    AgentResponseError,
    TOP_K_TO_RERANK,
    merge_reranked_top_k,
    parse_rerank_response,
)
from yelp_agent.agent.prompt import build_rerank_prompt
from yelp_agent.agent.tools import (
    BusinessDetailsResult,
    HybridRankingResult,
    TaskAgentTools,
    UserHistoryResult,
)
from yelp_agent.models import (
    Prediction,
    RecommendationTask,
    ScoreBreakdown,
    StrictModel,
    UserProfile,
)


class AgentTimingBreakdown(StrictModel):
    hybrid_ms: float = Field(ge=0)
    tools_ms: float = Field(ge=0)
    prompt_ms: float = Field(ge=0)
    llm_ms: float = Field(ge=0)
    parse_merge_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)


class AgentTrace(StrictModel):
    task_id: str = Field(min_length=1)
    profile: UserProfile | None = None
    representative_review_ids: list[str]
    hybrid_ranking: list[str]
    hybrid_score_breakdowns: dict[str, ScoreBreakdown]
    top_k_before: list[str]
    top_k_after: list[str]
    llm_status: Literal["success", "disabled", "failure", "not_called"]
    model: str | None = None
    attempt_count: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    observed_total_tokens: int | None = Field(default=None, ge=0)
    unknown_usage_attempts: int = Field(default=0, ge=0)
    llm_latency_ms: float | None = Field(default=None, ge=0)
    llm_attempts: list[LLMAttemptTrace] = Field(default_factory=list)
    tool_calls: int = Field(ge=0)
    llm_reason: str | None = None
    fallback: bool
    fallback_reason: str | None = None
    latency_ms: float = Field(ge=0)
    timing: AgentTimingBreakdown | None = None


def _empty_timing_state() -> dict[str, float]:
    return {
        "hybrid_ms": 0.0,
        "tools_ms": 0.0,
        "prompt_ms": 0.0,
        "llm_ms": 0.0,
        "parse_merge_ms": 0.0,
    }


def _timing_breakdown(
    timing: dict[str, float],
    *,
    total_ms: float,
) -> AgentTimingBreakdown:
    return AgentTimingBreakdown(
        hybrid_ms=timing["hybrid_ms"],
        tools_ms=timing["tools_ms"],
        prompt_ms=timing["prompt_ms"],
        llm_ms=timing["llm_ms"],
        parse_merge_ms=timing["parse_merge_ms"],
        total_ms=total_ms,
    )


class _HybridRanker(Protocol):
    def rank(self, task: RecommendationTask) -> Prediction: ...


class _Toolbox(Protocol):
    def for_task(self, task: RecommendationTask) -> TaskAgentTools: ...


class _LLM(Protocol):
    def generate(self, messages: Sequence[LLMMessage]) -> LLMCallResult: ...


class AgentRanker:
    """Rerank Hybrid Top-8 with an LLM while preserving a safe baseline."""

    def __init__(
        self,
        *,
        hybrid_ranker: _HybridRanker,
        toolbox: _Toolbox,
        llm: _LLM,
    ) -> None:
        self._hybrid_ranker = hybrid_ranker
        self._toolbox = toolbox
        self._llm = llm
        self._traces: dict[str, AgentTrace] = {}

    def trace_for(self, task_id: str) -> AgentTrace:
        """Return the trace produced by the latest rank call for one task."""

        try:
            return self._traces[task_id]
        except KeyError as exc:
            raise KeyError(f"No Agent trace exists for {task_id!r}") from exc

    def _fallback_prediction(
        self,
        *,
        task: RecommendationTask,
        hybrid_prediction: Prediction,
        session: TaskAgentTools | None,
        history: UserHistoryResult | None,
        profile: UserProfile | None,
        hybrid: HybridRankingResult | None,
        llm_result: LLMCallResult | None,
        fallback_reason: str,
        started_at: float,
        timing: dict[str, float],
    ) -> Prediction:
        latency_ms = (perf_counter() - started_at) * 1000.0
        llm_attempted = (
            llm_result is not None and llm_result.status != "disabled"
        )
        tool_calls = session.call_count if session is not None else 0
        hybrid_ranking = (
            hybrid.ranking
            if hybrid is not None
            else hybrid_prediction.ranking
        )
        score_breakdowns = (
            hybrid.score_breakdowns if hybrid is not None else {}
        )
        trace = AgentTrace(
            task_id=task.task_id,
            profile=profile,
            representative_review_ids=[
                review.review_id
                for review in [
                    *(history.recent_positive if history is not None else []),
                    *(history.recent_negative if history is not None else []),
                ]
            ],
            hybrid_ranking=hybrid_ranking,
            hybrid_score_breakdowns=score_breakdowns,
            top_k_before=hybrid_ranking[:TOP_K_TO_RERANK],
            top_k_after=hybrid_ranking[:TOP_K_TO_RERANK],
            llm_status=(
                llm_result.status if llm_result is not None else "not_called"
            ),
            model=llm_result.model if llm_result is not None else None,
            attempt_count=(
                llm_result.attempt_count if llm_result is not None else 0
            ),
            input_tokens=(
                llm_result.input_tokens if llm_result is not None else None
            ),
            output_tokens=(
                llm_result.output_tokens if llm_result is not None else None
            ),
            total_tokens=(
                llm_result.total_tokens if llm_result is not None else None
            ),
            observed_total_tokens=(
                llm_result.observed_total_tokens
                if llm_result is not None
                else None
            ),
            unknown_usage_attempts=(
                llm_result.unknown_usage_attempts
                if llm_result is not None
                else 0
            ),
            llm_latency_ms=(
                llm_result.latency_ms if llm_result is not None else None
            ),
            llm_attempts=(
                llm_result.attempts if llm_result is not None else []
            ),
            tool_calls=tool_calls,
            llm_reason=None,
            fallback=True,
            fallback_reason=fallback_reason,
            latency_ms=latency_ms,
            timing=_timing_breakdown(timing, total_ms=latency_ms),
        )
        self._traces[task.task_id] = trace
        metadata = {
            "method": "hybrid_agent",
            "llm_attempted": llm_attempted,
            "llm_status": (
                llm_result.status if llm_result is not None else "not_called"
            ),
            "model": llm_result.model if llm_result is not None else None,
            "attempt_count": (
                llm_result.attempt_count if llm_result is not None else 0
            ),
            "llm_latency_ms": (
                llm_result.latency_ms if llm_result is not None else None
            ),
            "observed_total_tokens": (
                llm_result.observed_total_tokens
                if llm_result is not None
                else None
            ),
            "unknown_usage_attempts": (
                llm_result.unknown_usage_attempts
                if llm_result is not None
                else 0
            ),
            "top_k_reranked": 0,
        }
        if "weights" in hybrid_prediction.metadata:
            metadata["hybrid_weights"] = hybrid_prediction.metadata["weights"]
        return Prediction(
            task_id=task.task_id,
            ranking=hybrid_prediction.ranking,
            latency_ms=latency_ms,
            fallback=True,
            fallback_reason=fallback_reason,
            tool_calls=tool_calls,
            llm_tokens=(
                llm_result.total_tokens if llm_result is not None else None
            ),
            metadata=metadata,
        )

    def rank(self, task: RecommendationTask) -> Prediction:
        started_at = perf_counter()
        timing = _empty_timing_state()
        hybrid_started_at = perf_counter()
        hybrid_prediction = self._hybrid_ranker.rank(task)
        timing["hybrid_ms"] = (
            perf_counter() - hybrid_started_at
        ) * 1000.0
        if (
            hybrid_prediction.task_id != task.task_id
            or set(hybrid_prediction.ranking)
            != set(task.candidate_business_ids)
        ):
            raise ValueError("Hybrid fallback does not match the Agent task")

        session: TaskAgentTools | None = None
        history: UserHistoryResult | None = None
        profile: UserProfile | None = None
        hybrid: HybridRankingResult | None = None
        tools_started_at = perf_counter()
        try:
            session = self._toolbox.for_task(task)
            history = session.get_user_history(
                task.user_id,
                task.cutoff_time,
            )
            profile = session.get_user_profile(
                task.user_id,
                task.cutoff_time,
            )
            hybrid = session.get_hybrid_ranking(
                task.user_id,
                task.candidate_business_ids,
                task.cutoff_time,
            )
            if hybrid.ranking != hybrid_prediction.ranking:
                raise ValueError("Hybrid tool ranking disagrees with fallback")
            details: BusinessDetailsResult = session.get_business_details(
                hybrid.ranking[:TOP_K_TO_RERANK],
                task.cutoff_time,
            )
        except Exception:
            timing["tools_ms"] = (
                perf_counter() - tools_started_at
            ) * 1000.0
            return self._fallback_prediction(
                task=task,
                hybrid_prediction=hybrid_prediction,
                session=session,
                history=history,
                profile=profile,
                hybrid=hybrid,
                llm_result=None,
                fallback_reason="tool_error",
                started_at=started_at,
                timing=timing,
            )
        timing["tools_ms"] = (perf_counter() - tools_started_at) * 1000.0
        prompt_started_at = perf_counter()
        try:
            messages = build_rerank_prompt(
                task=task,
                history=history,
                profile=profile,
                hybrid=hybrid,
                business_details=details,
            )
        except Exception:
            timing["prompt_ms"] = (
                perf_counter() - prompt_started_at
            ) * 1000.0
            return self._fallback_prediction(
                task=task,
                hybrid_prediction=hybrid_prediction,
                session=session,
                history=history,
                profile=profile,
                hybrid=hybrid,
                llm_result=None,
                fallback_reason="prompt_error",
                started_at=started_at,
                timing=timing,
            )
        timing["prompt_ms"] = (perf_counter() - prompt_started_at) * 1000.0
        llm_started_at = perf_counter()
        try:
            llm_result = self._llm.generate(messages)
        except Exception:
            llm_latency_ms = (perf_counter() - llm_started_at) * 1000.0
            llm_result = LLMCallResult(
                status="failure",
                content=None,
                model=None,
                latency_ms=llm_latency_ms,
                attempt_count=1,
                failure_reason="llm_error",
                attempts=[
                    LLMAttemptTrace(
                        attempt_index=1,
                        status="failure",
                        latency_ms=llm_latency_ms,
                        failure_reason="llm_error",
                        retryable=False,
                        usage_unknown=True,
                    )
                ],
                unknown_usage_attempts=1,
            )
        timing["llm_ms"] = (perf_counter() - llm_started_at) * 1000.0
        if llm_result.status != "success" or llm_result.content is None:
            return self._fallback_prediction(
                task=task,
                hybrid_prediction=hybrid_prediction,
                session=session,
                history=history,
                profile=profile,
                hybrid=hybrid,
                llm_result=llm_result,
                fallback_reason=(
                    llm_result.failure_reason
                    or (
                        "llm_disabled"
                        if llm_result.status == "disabled"
                        else "llm_failure"
                    )
                ),
                started_at=started_at,
                timing=timing,
            )
        parse_started_at = perf_counter()
        try:
            parsed = parse_rerank_response(
                llm_result.content,
                expected_business_ids=hybrid.ranking[:TOP_K_TO_RERANK],
            )
            final_ranking = merge_reranked_top_k(parsed, hybrid.ranking)
        except AgentResponseError as exc:
            timing["parse_merge_ms"] = (
                perf_counter() - parse_started_at
            ) * 1000.0
            return self._fallback_prediction(
                task=task,
                hybrid_prediction=hybrid_prediction,
                session=session,
                history=history,
                profile=profile,
                hybrid=hybrid,
                llm_result=llm_result,
                fallback_reason=exc.reason,
                started_at=started_at,
                timing=timing,
            )
        timing["parse_merge_ms"] = (
            perf_counter() - parse_started_at
        ) * 1000.0
        latency_ms = (perf_counter() - started_at) * 1000.0
        trace = AgentTrace(
            task_id=task.task_id,
            profile=profile,
            representative_review_ids=[
                review.review_id
                for review in [
                    *history.recent_positive,
                    *history.recent_negative,
                ]
            ],
            hybrid_ranking=hybrid.ranking,
            hybrid_score_breakdowns=hybrid.score_breakdowns,
            top_k_before=hybrid.ranking[:TOP_K_TO_RERANK],
            top_k_after=parsed.ranking,
            llm_status=llm_result.status,
            model=llm_result.model,
            attempt_count=llm_result.attempt_count,
            input_tokens=llm_result.input_tokens,
            output_tokens=llm_result.output_tokens,
            total_tokens=llm_result.total_tokens,
            observed_total_tokens=llm_result.observed_total_tokens,
            unknown_usage_attempts=llm_result.unknown_usage_attempts,
            llm_latency_ms=llm_result.latency_ms,
            llm_attempts=llm_result.attempts,
            tool_calls=session.call_count,
            llm_reason=parsed.reason,
            fallback=False,
            fallback_reason=None,
            latency_ms=latency_ms,
            timing=_timing_breakdown(timing, total_ms=latency_ms),
        )
        self._traces[task.task_id] = trace
        metadata = {
            "method": "hybrid_agent",
            "llm_attempted": True,
            "llm_status": llm_result.status,
            "model": llm_result.model,
            "attempt_count": llm_result.attempt_count,
            "llm_latency_ms": llm_result.latency_ms,
            "observed_total_tokens": llm_result.observed_total_tokens,
            "unknown_usage_attempts": llm_result.unknown_usage_attempts,
            "top_k_reranked": TOP_K_TO_RERANK,
        }
        if "weights" in hybrid_prediction.metadata:
            metadata["hybrid_weights"] = hybrid_prediction.metadata["weights"]
        return Prediction(
            task_id=task.task_id,
            ranking=final_ranking,
            latency_ms=latency_ms,
            fallback=False,
            tool_calls=session.call_count,
            llm_tokens=llm_result.total_tokens,
            metadata=metadata,
        )
