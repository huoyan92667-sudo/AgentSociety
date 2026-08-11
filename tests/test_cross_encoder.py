"""Step 26 Cross-Encoder Module, cache, fusion, and Agent Adapter tests."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from yelp_agent.agent_tools.adapters.cross_encoder import ComputeCrossEncoderMatchTool
from yelp_agent.agent_tools.schema import ToolExecutionContext
from yelp_agent.agent_tools.tool_schemas import CandidateBusinessIdsInput
from yelp_agent.cross_encoder import (
    CachedCrossEncoderReranker,
    CrossEncoderBusinessMatch,
    CrossEncoderConfig,
    CrossEncoderMatchResult,
    CrossEncoderUsage,
    LocalCrossEncoderEnvironment,
    ScoredPairBatch,
    SqliteCrossEncoderCache,
    fuse_ranking_and_cross_encoder,
    load_cross_encoder_config,
    load_cross_encoder_policy,
)
from yelp_agent.cross_encoder.local_reranker import _sanitize_text
from yelp_agent.data.temporal_view import BusinessRecord


def _config() -> CrossEncoderConfig:
    return CrossEncoderConfig(
        agent_version="test-cross-agent",
        batch_size=2,
        max_sequence_length=512,
        instruction="Judge whether the Yelp business matches the request.",
        business_document_version="test-static-business-v2",
        cache_relative_path="data/features/test-cross",
        policy_relative_path="configs/cross_encoder_policy.json",
    )


def _business(business_id: str, name: str, categories: list[str]) -> BusinessRecord:
    return BusinessRecord(
        business_id=business_id,
        name=name,
        address="1 Test Street",
        city="Philadelphia",
        state="PA",
        postal_code="19103",
        latitude=39.95,
        longitude=-75.16,
        categories=tuple(categories),
        attributes_json=json.dumps({"Ambience": {"romantic": business_id == "steak"}}),
    )


class FixedBusinessReader:
    def __init__(self) -> None:
        self.rows = {
            "steak": _business("steak", "Romantic Steak House", ["Steakhouses"]),
            "dentist": _business("dentist", "Central Dentist", ["Dentists"]),
        }

    def business(self, business_id: str) -> BusinessRecord:
        return self.rows[business_id]


class KeywordPairScorer:
    provider = "local"
    model = "local:fake-qwen-reranker"
    batch_size = 2

    def __init__(self) -> None:
        self.calls = 0

    def score(self, query_text: str, documents: list[str]) -> ScoredPairBatch:
        self.calls += 1
        scores = tuple(0.95 if "Steak" in document else 0.05 for document in documents)
        tokens = tuple(10 for _ in documents)
        return ScoredPairBatch(
            scores=scores,
            input_tokens=sum(tokens),
            per_pair_input_tokens=tokens,
            truncated_pair_count=0,
            latency_ms=2,
        )


def test_production_config_and_policy_freeze_top_five() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_cross_encoder_config(root / "configs" / "cross_encoder.yaml")
    policy = load_cross_encoder_policy(root, config)

    assert config.provider == "local"
    assert config.batch_size == 8
    assert config.max_sequence_length == 512
    assert policy.candidate_limit == 20
    assert policy.display_limit == 5


def test_local_environment_and_legacy_text_sanitization(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    for filename in ("config.json", "model.safetensors", "tokenizer.json"):
        (model / filename).touch()
    environment = LocalCrossEncoderEnvironment(
        model_path=model,
        python_executable=Path(sys.executable),
        device="cpu",
    )
    assert environment.enabled
    assert _sanitize_text("broken\udca2name") == "broken?name"


def test_cross_fusion_reorders_only_scored_prefix_and_keeps_top_five_contract() -> None:
    ranking = ["a", "b", "c", "d", "e", "tail-1", "tail-2"]
    fused = fuse_ranking_and_cross_encoder(
        ranking,
        {"a": 5, "b": 4, "c": 3, "d": 2, "e": 1},
        beta=0.8,
    )
    assert fused[:5] == ["e", "d", "c", "b", "a"]
    assert fused[5:] == ["tail-1", "tail-2"]
    assert fuse_ranking_and_cross_encoder(ranking, {}, beta=0.8) == ranking


def test_cached_reranker_scores_static_documents_once_and_stores_no_raw_query(
    tmp_path: Path,
) -> None:
    scorer = KeywordPairScorer()
    cache = SqliteCrossEncoderCache(tmp_path / "cache")
    reranker = CachedCrossEncoderReranker(
        businesses=FixedBusinessReader(),
        scorer=scorer,
        cache=cache,
        config=_config(),
    )
    first = reranker.rerank(
        query_text="romantic steakhouse for a date",
        business_ids=["dentist", "steak"],
        cutoff_time=datetime(2022, 1, 1),
        usage_scope="request-1",
    )
    second = reranker.rerank(
        query_text="romantic steakhouse for a date",
        business_ids=["dentist", "steak"],
        cutoff_time=datetime(2023, 1, 1),
        usage_scope="request-2",
    )
    assert [item.business_id for item in first.matches] == ["steak", "dentist"]
    assert first.usage.cache_misses == 2
    assert first.usage.api_calls == 0
    assert second.usage.cache_hits == 2
    assert second.usage.input_tokens == 0
    assert scorer.calls == 1
    assert cache.count() == 2
    assert b"romantic steakhouse for a date" not in cache.path.read_bytes()


class FixedCrossService:
    def rerank(
        self,
        *,
        query_text: str,
        business_ids: list[str],
        cutoff_time: datetime,
        usage_scope: str | None = None,
    ) -> CrossEncoderMatchResult:
        assert query_text == "romantic steakhouse"
        assert usage_scope == "f" * 64
        return CrossEncoderMatchResult(
            query_sha256="c" * 64,
            model="local:fake-reranker",
            matches=[
                CrossEncoderBusinessMatch(
                    business_id=business_id,
                    cutoff_time=cutoff_time,
                    document_sha256=("d" if index == 0 else "e") * 64,
                    relevance_score=1.0 - index,
                    cross_encoder_rank=index + 1,
                )
                for index, business_id in enumerate(business_ids)
            ],
            usage=CrossEncoderUsage(
                scorer_calls=1,
                input_tokens=20,
                logical_input_tokens=20,
                cache_hits=0,
                cache_misses=2,
                provider_latency_ms=2,
            ),
        )


def test_agent_tool_requires_exact_step25_prefix_and_returns_evidence() -> None:
    context = ToolExecutionContext(
        request_id="f" * 64,
        user_id="user-1",
        cutoff_time=datetime(2022, 1, 1),
        action="rank_candidates",
        business_scope=("a", "b"),
        business_scope_known=True,
        state_snapshot={
            "request": {"query_text": "romantic steakhouse"},
            "observations": [
                {"payload": {"tool_name": "GET_HYBRID_RANKING", "data": {"ranking": ["a", "b"]}}},
                {"payload": {"tool_name": "COMPUTE_EMBEDDING_MATCH", "data": {"matches": [
                    {"business_id": "b", "semantic_rank": 1},
                    {"business_id": "a", "semantic_rank": 2},
                ]}}},
            ],
        },
    )
    tool = ComputeCrossEncoderMatchTool(FixedCrossService(), embedding_alpha=1.0)
    observation = tool.run(
        CandidateBusinessIdsInput(business_ids=["b", "a"]), context
    )
    invalid = tool.run(
        CandidateBusinessIdsInput(business_ids=["a", "b"]), context
    )

    assert observation.status == "success"
    assert observation.input_tokens == 20
    assert "ranking" not in observation.data
    assert invalid.status == "permanent_error"
    assert invalid.error_code == "STEP25_PREFIX_REQUIRED"
