"""读取真实长期画像并转换成新版统一偏好的固定工具。"""

from __future__ import annotations

from typing import Protocol

from yelp_agent.profiles.schema import UserProfileV1
from yelp_agent.recommendation_v2.preference_fusion.profile_adapter import (
    ProfilePreferenceSet,
    adapt_user_profile,
)
from yelp_agent.recommendation_v2.schema import GeoPoint


class LatestUserProfileReader(Protocol):
    """真实画像文件和测试画像都通过同一个最小读取接口接入。"""

    def latest(self, user_id: str) -> UserProfileV1:
        """读取指定用户最新画像。"""


class UserProfileTool:
    """根据真实用户编号读取最新画像，并强制经过统一结构转换。"""

    def __init__(self, store: LatestUserProfileReader) -> None:
        self._store = store

    def load(self, user_id: str) -> tuple[UserProfileV1, ProfilePreferenceSet]:
        """返回原始画像和转换后画像，方便运行时使用与人工核对。"""

        raw_profile = self._store.latest(user_id)
        return raw_profile, adapt_user_profile(raw_profile)

    @staticmethod
    def location(profile: UserProfileV1) -> GeoPoint | None:
        """画像有稳定活动中心时，把它作为当前尚未接地点工具的距离起点。"""

        if profile.location_center is None:
            return None
        return GeoPoint(
            latitude=profile.location_center.latitude,
            longitude=profile.location_center.longitude,
        )
