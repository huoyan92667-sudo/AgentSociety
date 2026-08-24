from datetime import UTC, datetime, timedelta

from yelp_agent.agent.llm import LLMCallResult, LLMMessage
from yelp_agent.models import LocationCenter
from yelp_agent.profiles.schema import (
    PreferenceSignal,
    ProfileEvidenceSummary,
    UserProfileV1,
)
from yelp_agent.recommendation_v2.preference_fusion import (
    CompactHardRequirement,
    CompactSceneSelection,
    CompactSoftRequirement,
    PreferenceFusion,
    PreferenceFusionProposal,
)
from yelp_agent.recommendation_v2.workflow import (
    RecommendationInput,
    RecommendationWorkflow,
)

NOW = datetime(2026, 8, 20, tzinfo=UTC)


class FakeGenerator:
    """返回一次精简需求，并保存大模型实际收到的内容。"""

    def __init__(self, proposal: PreferenceFusionProposal) -> None:
        self._proposal = proposal
        self.messages: list[LLMMessage] = []

    def generate(self, messages: list[LLMMessage]) -> LLMCallResult:
        self.messages = list(messages)
        return LLMCallResult(
            status="success",
            content=self._proposal.model_dump_json(),
            model="fake-model",
            latency_ms=1,
            attempt_count=1,
        )


class FakeProfileStore:
    """模拟真实画像存储，确认入口确实按用户编号读取。"""

    def __init__(self, profile: UserProfileV1) -> None:
        self.profile = profile
        self.requested_users: list[str] = []
        self.closed = False

    def latest(self, user_id: str) -> UserProfileV1:
        self.requested_users.append(user_id)
        return self.profile

    def close(self) -> None:
        self.closed = True


def _signal(kind: str, value: str, score: float) -> PreferenceSignal:
    """建立一条带真实次数和时间的画像信号。"""

    source = {
        "category": "rating_category",
        "aspect": "review_aspect",
    }[kind]
    return PreferenceSignal.model_validate(
        {
            "kind": kind,
            "value": value,
            "score": score,
            "confidence": 0.8,
            "evidence_count": 8,
            "effective_evidence": 6.0,
            "first_seen": NOW - timedelta(days=300),
            "last_confirmed": NOW - timedelta(days=2),
            "source": source,
        }
    )


def _real_shape_profile() -> UserProfileV1:
    """建立和项目真实画像结构完全相同的测试记录。"""

    return UserProfileV1(
        profile_id="a" * 64,
        user_id="real-user",
        cutoff_time=NOW,
        history_length=10,
        average_rating=4,
        rating_distribution={"1": 0, "2": 1, "3": 2, "4": 4, "5": 3},
        category_preferences=[_signal("category", "Japanese", 0.9)],
        category_dislikes=[],
        aspect_preferences=[_signal("aspect", "quiet_environment", 0.8)],
        aspect_dislikes=[],
        frequent_areas=[],
        location_center=LocationCenter(latitude=39.95, longitude=-75.16),
        reliability=0.9,
        evidence_summary=ProfileEvidenceSummary(
            category_evidence_count=8,
            aspect_evidence_count=8,
            price_evidence_count=0,
            area_evidence_count=0,
            first_interaction=NOW - timedelta(days=300),
            last_interaction=NOW - timedelta(days=2),
        ),
        profile_version="1.0.0",
    )


def test_small_entry_loads_and_adapts_profile_then_adds_scene_automatically() -> None:
    """调用方只给三个值，内部完成真实画像转换、菜系硬过滤和场景接入。"""

    proposal = PreferenceFusionProposal(
        scene=CompactSceneSelection(
            kind="friends",
            evidence_text="和朋友聚餐",
            evidence_turn_index=1,
        ),
        hard_constraints=[
            CompactHardRequirement(
                field="category",
                operator="any_of",
                value=["Szechuan"],
                evidence_text="想吃川菜",
                evidence_turn_index=1,
            )
        ],
        soft_preferences=[
            CompactSoftRequirement(
                field="quiet_environment",
                direction="higher",
                priority=1,
                evidence_text="安静最重要",
                evidence_turn_index=1,
            )
        ],
    )
    generator = FakeGenerator(proposal)
    profile_store = FakeProfileStore(_real_shape_profile())
    workflow = RecommendationWorkflow(
        fusion=PreferenceFusion(generator),
        profile_store=profile_store,
    )

    result = workflow.process(
        RecommendationInput(
            user_id="real-user",
            session_id="session-1",
            query_text="和朋友聚餐，想吃川菜，安静最重要",
        )
    )

    assert profile_store.requested_users == ["real-user"]
    assert result.raw_profile is profile_store.profile
    assert result.adapted_profile is not None
    assert any(
        item.target_value == ["Japanese"]
        for item in result.adapted_profile.soft_preferences
    )
    assert result.fusion.status == "success"
    assert result.fusion.state is not None
    state = result.fusion.state
    assert state.scene is not None and state.scene.kind == "friends"
    assert state.default_constraints[0].field == "distance_km"
    assert state.hard_constraints[0].field == "category"
    assert state.hard_constraints[0].value == ["Szechuan"]
    quiet = next(item for item in state.soft_preferences if item.field == "quiet_environment")
    assert quiet.preference_strength == 100
    assert quiet.priority == 1
    japanese = next(
        item
        for item in state.preference_memory
        if item.source == "user_profile" and item.preference.field == "category"
    )
    assert japanese.status == "suppressed"
    assert state.user_location is not None
    assert state.search_center is not None
    assert "profile_preferences" not in generator.messages[1].content
    assert "scene_baseline" not in generator.messages[1].content
    assert "real-user" not in generator.messages[1].content
    assert "session-1" not in generator.messages[1].content

    workflow.close()
    assert profile_store.closed is True
