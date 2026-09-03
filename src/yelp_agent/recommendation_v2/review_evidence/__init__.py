"""基于真实评论证据的新版软偏好排序。"""

from .builder import (
    DEFAULT_EVIDENCE_ROOT,
    ReviewEvidenceIndexBuildResult,
    build_review_evidence_index,
    build_review_segment_source,
)
from .direct_search import (
    DirectReviewEvidenceFinding,
    DirectReviewEvidenceResult,
    DirectReviewEvidenceSearch,
)
from .ranker import ReviewEvidenceRanker
from .runtime import (
    ReviewEvidenceCapabilities,
    build_review_evidence_capabilities,
    build_review_evidence_ranker,
)
from .schema import ReviewEvidenceRankingResult
from .segmenter import OverlapSegmentConfig, segment_review_with_overlap

__all__ = [
    "DEFAULT_EVIDENCE_ROOT",
    "DirectReviewEvidenceFinding",
    "DirectReviewEvidenceResult",
    "DirectReviewEvidenceSearch",
    "OverlapSegmentConfig",
    "ReviewEvidenceCapabilities",
    "ReviewEvidenceIndexBuildResult",
    "ReviewEvidenceRanker",
    "ReviewEvidenceRankingResult",
    "build_review_evidence_capabilities",
    "build_review_evidence_index",
    "build_review_evidence_ranker",
    "build_review_segment_source",
    "segment_review_with_overlap",
]
