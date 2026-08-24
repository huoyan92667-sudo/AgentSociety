"""只接收用户、会话和问题的新版推荐入口。"""

from yelp_agent.recommendation_v2.workflow.recommendation import (
    RecommendationInput,
    RecommendationTurnResult,
    RecommendationWorkflow,
    build_recommendation_workflow,
)

__all__ = [
    "RecommendationInput",
    "RecommendationTurnResult",
    "RecommendationWorkflow",
    "build_recommendation_workflow",
]
