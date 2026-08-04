from yelp_agent.features.category import CategoryTaskFeatures
from yelp_agent.features.hybrid import HybridFeatureStore, HybridWeights
from yelp_agent.features.location import (
    LocationBusinessScore,
    LocationTaskFeatures,
)
from yelp_agent.features.quality import BusinessQuality
from yelp_agent.features.text import TextBusinessScore, TextTaskFeatures
from yelp_agent.models import LocationCenter, RecommendationTask, UserProfile
from yelp_agent.protocols import Ranker
from yelp_agent.rankers.hybrid_ranker import HybridRanker


class FixedCategoryStore:
    def features_for(self, task: RecommendationTask) -> CategoryTaskFeatures:
        return CategoryTaskFeatures(
            profile=UserProfile(
                user_id=task.user_id,
                history_count=3,
                average_rating=4.0,
                rating_distribution={
                    "1": 1,
                    "2": 0,
                    "3": 0,
                    "4": 1,
                    "5": 1,
                },
                preferred_categories={"Noodles": 0.9},
                disliked_categories={},
            ),
            category_scores={
                business_id: 1.0 if business_id == "business-00" else 0.0
                for business_id in task.candidate_business_ids
            },
        )


class FixedTextStore:
    def features_for(self, task: RecommendationTask) -> TextTaskFeatures:
        return TextTaskFeatures(
            positive_review_count=2,
            negative_review_count=1,
            positive_keywords=["cozy noodles"],
            negative_keywords=["noisy"],
            business_scores={
                business_id: TextBusinessScore(
                    business_id=business_id,
                    positive_similarity=0.0,
                    negative_similarity=0.0,
                    text_score=(
                        1.0 if business_id == "business-01" else 0.0
                    ),
                )
                for business_id in task.candidate_business_ids
            },
        )


class FixedQualityStore:
    def score_businesses(
        self,
        business_ids: list[str],
        cutoff_time: object,
    ) -> dict[str, BusinessQuality]:
        return {
            business_id: BusinessQuality(
                business_id=business_id,
                review_count=1,
                mean_rating=3.0,
                bayesian_rating=3.0,
                normalized_bayesian_rating=0.5,
                normalized_popularity=0.5,
                quality_score=(
                    1.0 if business_id == "business-02" else 0.0
                ),
            )
            for business_id in business_ids
        }


class FixedLocationStore:
    def features_for(self, task: RecommendationTask) -> LocationTaskFeatures:
        return LocationTaskFeatures(
            location_center=LocationCenter(
                latitude=39.9526,
                longitude=-75.1652,
            ),
            history_coordinate_count=3,
            business_scores={
                business_id: LocationBusinessScore(
                    business_id=business_id,
                    distance_km=0.0,
                    location_score=(
                        1.0 if business_id == "business-03" else 0.0
                    ),
                )
                for business_id in task.candidate_business_ids
            },
        )


def test_hybrid_ranker_combines_four_scores_and_merges_profile() -> None:
    candidates = [f"business-{index:02d}" for index in range(20)]
    task = RecommendationTask(
        task_id="validation:user-1",
        user_id="user-1",
        cutoff_time="2020-02-01T00:00:00",
        candidate_business_ids=list(reversed(candidates)),
    )
    feature_store = HybridFeatureStore(
        category_store=FixedCategoryStore(),
        text_store=FixedTextStore(),
        quality_store=FixedQualityStore(),
        location_store=FixedLocationStore(),
    )
    weights = HybridWeights(
        category=0.4,
        text=0.3,
        quality=0.2,
        location=0.1,
    )
    ranker = HybridRanker(feature_store, weights)

    scored = ranker.score(task)
    prediction = ranker.rank(task)

    assert isinstance(ranker, Ranker)
    assert scored.score_breakdowns["business-00"].hybrid_score == 0.4
    assert scored.score_breakdowns["business-01"].hybrid_score == 0.3
    assert scored.score_breakdowns["business-02"].hybrid_score == 0.2
    assert scored.score_breakdowns["business-03"].hybrid_score == 0.1
    assert scored.profile.positive_keywords == ["cozy noodles"]
    assert scored.profile.negative_keywords == ["noisy"]
    assert scored.profile.location_center == LocationCenter(
        latitude=39.9526,
        longitude=-75.1652,
    )
    assert prediction.ranking == candidates
    assert prediction.metadata == {
        "method": "hybrid",
        "weights": {
            "category": 0.4,
            "text": 0.3,
            "quality": 0.2,
            "location": 0.1,
        },
        "llm_attempted": False,
    }
