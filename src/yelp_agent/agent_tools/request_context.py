"""One seam for reading the executable request from an Agent tool context."""

from __future__ import annotations

from yelp_agent.query import RecommendationRequest
from yelp_agent.session_memory.effective_request import effective_request_from_snapshot

from .schema import ToolExecutionContext


def request_from_tool_context(context: ToolExecutionContext) -> RecommendationRequest:
    effective = effective_request_from_snapshot(context.state_snapshot)
    if effective is not None:
        return effective.request
    raw = context.state_snapshot.get("request")
    if not isinstance(raw, dict):
        raise ValueError("current structured request is absent from Agent state")
    return RecommendationRequest.model_validate(
        {
            key: value
            for key, value in raw.items()
            if key in RecommendationRequest.model_fields
        }
    )


def query_text_from_tool_context(context: ToolExecutionContext) -> str:
    effective = effective_request_from_snapshot(context.state_snapshot)
    if effective is not None:
        text = effective.request.query_text
    else:
        raw = context.state_snapshot.get("request")
        # Older semantic-tool callers intentionally expose only query_text rather
        # than a complete RecommendationRequest.  Keep that narrow, read-only
        # contract while making the compiled Session request authoritative when
        # it is present.
        if isinstance(raw, dict) and isinstance(raw.get("query_text"), str):
            text = raw["query_text"]
        else:
            text = request_from_tool_context(context).query_text
    if not text.strip():
        raise ValueError("effective request query text is empty")
    return text


def rejected_business_ids_from_tool_context(
    context: ToolExecutionContext,
) -> set[str]:
    effective = effective_request_from_snapshot(context.state_snapshot)
    if effective is not None:
        return set(effective.rejected_business_ids)
    memory = context.state_snapshot.get("memory_context")
    if not isinstance(memory, dict):
        return set()
    values = memory.get("rejected_business_ids")
    if not isinstance(values, list):
        return set()
    return {str(value) for value in values if isinstance(value, str)}
