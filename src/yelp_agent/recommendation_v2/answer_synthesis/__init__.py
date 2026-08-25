"""把已经排好的 Top5 和真实评论整理成面向用户的自然回答。"""

from .synthesizer import (
    PROMPT_VERSION,
    RecommendationAnswer,
    RecommendationAnswerSynthesizer,
    build_recommendation_answer_synthesizer,
)

__all__ = [
    "PROMPT_VERSION",
    "RecommendationAnswer",
    "RecommendationAnswerSynthesizer",
    "build_recommendation_answer_synthesizer",
]
