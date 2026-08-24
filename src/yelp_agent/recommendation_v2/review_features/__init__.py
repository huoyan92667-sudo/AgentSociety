"""评论特征候选召回：先找可能相关的完整评论，不在这里判断正负和打分。"""

from .builder import build_keyword_review_candidates
from .definitions import AspectRecallDefinition, build_aspect_recall_definitions
from .keyword_recall import KeywordAspectMatcher
from .merge import merge_review_candidates
from .schema import (
    CANDIDATE_ASPECT_SCHEMA,
    CANDIDATE_REVIEW_SCHEMA,
    CandidateRecallBuildResult,
    CandidateRecallManifest,
    KeywordRecallBuildResult,
    KeywordRecallManifest,
    SemanticRecallBuildResult,
    SemanticRecallManifest,
)
from .semantic_recall import build_semantic_review_candidates

__all__ = [
    "CANDIDATE_ASPECT_SCHEMA",
    "CANDIDATE_REVIEW_SCHEMA",
    "AspectRecallDefinition",
    "CandidateRecallBuildResult",
    "CandidateRecallManifest",
    "KeywordAspectMatcher",
    "KeywordRecallBuildResult",
    "KeywordRecallManifest",
    "SemanticRecallBuildResult",
    "SemanticRecallManifest",
    "build_aspect_recall_definitions",
    "build_keyword_review_candidates",
    "build_semantic_review_candidates",
    "merge_review_candidates",
]
