"""Compatibility adapter from rich frozen profiles to the Agent contract."""

from __future__ import annotations

from yelp_agent.models import UserProfile
from yelp_agent.profiles.schema import PreferenceSignal, UserProfileV1


def _scores(signals: list[PreferenceSignal]) -> dict[str, float]:
    return {
        signal.value: abs(signal.score)
        for signal in sorted(signals, key=lambda item: item.value)
    }


def _confidences(*groups: list[PreferenceSignal]) -> dict[str, float]:
    return {
        signal.value: signal.confidence
        for signal in sorted(
            (signal for group in groups for signal in group),
            key=lambda item: item.value,
        )
    }


def to_agent_user_profile(profile: UserProfileV1) -> UserProfile:
    """Expose V1 evidence without making older rankers understand storage."""

    return UserProfile(
        profile_id=profile.profile_id,
        user_id=profile.user_id,
        cutoff_time=profile.cutoff_time,
        history_count=profile.history_length,
        average_rating=profile.average_rating,
        rating_distribution=dict(profile.rating_distribution),
        preferred_categories=_scores(profile.category_preferences),
        disliked_categories=_scores(profile.category_dislikes),
        category_confidences=_confidences(
            profile.category_preferences,
            profile.category_dislikes,
        ),
        aspect_preferences=_scores(profile.aspect_preferences),
        aspect_dislikes=_scores(profile.aspect_dislikes),
        aspect_confidences=_confidences(
            profile.aspect_preferences,
            profile.aspect_dislikes,
        ),
        price_preference=(
            None if profile.price_preference is None else profile.price_preference.value
        ),
        frequent_areas=[signal.value for signal in profile.frequent_areas],
        profile_reliability=profile.reliability,
        profile_version=profile.profile_version,
        location_center=profile.location_center,
    )
