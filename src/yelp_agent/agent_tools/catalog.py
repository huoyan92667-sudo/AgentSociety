"""Construction of the frozen Step 23 tool catalog."""

from __future__ import annotations

from .adapters import (
    ApplyConstraintsTool,
    BusinessProfileCandidateReader,
    CompareBusinessesTool,
    ComputeEmbeddingMatchTool,
    ExpandCandidatesTool,
    GetBusinessDetailsTool,
    GetBusinessProfileTool,
    GetHybridRankingTool,
    GetSessionMemoryTool,
    GetUserProfileTool,
)
from .config import AgentToolRuntimeConfig
from .registry import AgentToolRegistry, ToolDefinition, UnavailableTool
from .schema import ToolAvailability
from .tool_schemas import EmptyToolInput


def _future_tool(
    *,
    name: str,
    step: int,
    kind: str,
    actions: tuple[str, ...],
    summary: str,
) -> UnavailableTool:
    return UnavailableTool(
        ToolDefinition(
            name=name,
            version=f"planned-step-{step}",
            kind=kind,  # type: ignore[arg-type]
            allowed_actions=actions,  # type: ignore[arg-type]
            input_model=EmptyToolInput,
            output_model=EmptyToolInput,
            public_summary=summary,
            availability=ToolAvailability(
                available=False,
                reason=f"planned_for_step_{step}",
            ),
        )
    )


def build_step23_tool_registry(
    *,
    user_profiles: object,
    business_profiles: object,
    retriever: object,
    history_reader: object,
    hybrid_ranking: object,
    embedding_match: object | None = None,
    runtime_config: AgentToolRuntimeConfig | None = None,
) -> AgentToolRegistry:
    """Build the complete catalog while leaving future tools explicitly disabled."""

    candidate_reader = BusinessProfileCandidateReader(business_profiles)  # type: ignore[arg-type]
    return AgentToolRegistry(
        [
            GetSessionMemoryTool(),
            GetUserProfileTool(user_profiles),  # type: ignore[arg-type]
            ExpandCandidatesTool(retriever, history_reader),  # type: ignore[arg-type]
            ApplyConstraintsTool(candidate_reader),
            GetHybridRankingTool(hybrid_ranking),  # type: ignore[arg-type]
            GetBusinessDetailsTool(business_profiles),  # type: ignore[arg-type]
            GetBusinessProfileTool(business_profiles),  # type: ignore[arg-type]
            CompareBusinessesTool(candidate_reader),
            (
                ComputeEmbeddingMatchTool(embedding_match)  # type: ignore[arg-type]
                if embedding_match is not None
                else _future_tool(
                    name="COMPUTE_EMBEDDING_MATCH",
                    step=25,
                    kind="semantic",
                    actions=("rank_candidates",),
                    summary="Compute semantic request-candidate similarity.",
                )
            ),
            _future_tool(
                name="COMPUTE_CROSS_ENCODER_MATCH",
                step=26,
                kind="semantic",
                actions=("rank_candidates",),
                summary="Rerank a small candidate set with a cross-encoder.",
            ),
            _future_tool(
                name="SEARCH_BUSINESS_REVIEWS",
                step=27,
                kind="review_rag",
                actions=("retrieve_business_reviews",),
                summary="Search time-safe reviews for one locked business.",
            ),
            _future_tool(
                name="AGGREGATE_REVIEW_EVIDENCE",
                step=28,
                kind="review_rag",
                actions=("retrieve_business_reviews",),
                summary="Aggregate supporting and conflicting review evidence.",
            ),
            _future_tool(
                name="ASSESS_LLM_SEMANTICS",
                step=29,
                kind="semantic",
                actions=(
                    "rank_candidates",
                    "get_business_details",
                    "compare_candidates",
                ),
                summary="Resolve only low-confidence semantic interpretation cases.",
            ),
        ],
        runtime_config=runtime_config,
    )
