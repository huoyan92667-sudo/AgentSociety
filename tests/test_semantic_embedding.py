"""Step 25 semantic Module and Agent Adapter tests."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from yelp_agent.agent_tools.adapters.embedding import ComputeEmbeddingMatchTool
from yelp_agent.agent_tools.schema import ToolExecutionContext
from yelp_agent.agent_tools.tool_schemas import CandidateBusinessIdsInput
from yelp_agent.data.temporal_view import BusinessRecord
from yelp_agent.semantic_embedding import (
    CachedEmbeddingGateway,
    DashScopeEmbeddingEncoder,
    EmbeddingUsage,
    EncodedBatch,
    LocalEmbeddingEnvironment,
    SemanticBusinessMatch,
    SemanticEmbeddingConfig,
    SemanticEmbeddingMatcher,
    SemanticMatchResult,
    SqliteEmbeddingCache,
    build_business_document,
    compare_tfidf_and_embedding,
    fuse_hybrid_and_semantic,
    load_dashscope_embedding_environment,
    load_semantic_embedding_config,
)
from yelp_agent.semantic_embedding.local_encoder import _sanitize_text


def _config() -> SemanticEmbeddingConfig:
    return SemanticEmbeddingConfig(
        provider="dashscope",
        agent_version="test-semantic-agent",
        dimension=256,
        batch_size=20,
        timeout_seconds=90,
        max_retries=0,
        candidate_limit=30,
        fusion_alpha=0.3,
        query_instruction="Retrieve matching Yelp businesses.",
        business_document_version="test-business-static-doc-v2",
        cache_relative_path="data/features/test-embedding",
        price_cny_per_1000_input_tokens=0.0005,
    )


def _business(
    business_id: str,
    *,
    name: str,
    categories: list[str],
) -> BusinessRecord:
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
        attributes_json=json.dumps(
            {"Ambience": {"romantic": business_id == "steak"}},
            sort_keys=True,
        ),
    )


class FixedBusinessReader:
    def __init__(self, businesses: dict[str, BusinessRecord]) -> None:
        self.businesses = businesses
        self.requests: list[str] = []

    def business(self, business_id: str) -> BusinessRecord:
        self.requests.append(business_id)
        return self.businesses[business_id]


class KeywordEncoder:
    provider = "dashscope"
    model = "fake-qwen-embedding"
    dimension = 256
    batch_size = 20

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str]] = []

    def encode(self, texts: list[str], *, input_type: str) -> EncodedBatch:
        self.calls.append((list(texts), input_type))
        vectors = []
        for text in texts:
            folded = text.casefold()
            if "dentist" in folded:
                vector = np.zeros(256, dtype=np.float32)
                vector[1] = 1.0
            elif "steak" in folded or "romantic" in folded or "date" in folded:
                vector = np.zeros(256, dtype=np.float32)
                vector[0] = 1.0
            else:
                vector = np.zeros(256, dtype=np.float32)
                vector[2] = 1.0
            vectors.append(vector)
        return EncodedBatch(
            vectors=tuple(vectors),
            model=self.model,
            input_tokens=len(texts) * 5,
            request_id="fake-request",
            latency_ms=1,
        )


def test_environment_keeps_dashscope_key_secret_and_requires_complete_config() -> None:
    environment = load_dashscope_embedding_environment(
        {
            "DASHSCOPE_API_KEY": "secret-key",
            "DASHSCOPE_BASE_URL": "https://example.test/compatible-mode/v1",
            "DASHSCOPE_MODEL": "qwen3.7-text-embedding",
        }
    )

    assert environment.enabled
    assert environment.api_key is not None
    assert environment.api_key.get_secret_value() == "secret-key"
    assert "secret-key" not in repr(environment)


def test_production_config_freezes_static_v2_top_30_and_safe_budget() -> None:
    project_root = Path(__file__).resolve().parents[1]

    config = load_semantic_embedding_config(project_root / "configs" / "embedding.yaml")

    assert config.embedding_version == "2.0.0"
    assert config.provider == "local"
    assert config.batch_size == 16
    assert config.max_sequence_length == 512
    assert config.candidate_limit == 30
    assert config.max_total_tokens_per_turn == 12_000
    assert config.business_document_version == "business-static-semantic-v2.0.0"
    assert config.cache_relative_path == "data/features/semantic_embeddings/v2"
    assert config.price_cny_per_1000_input_tokens == 0


def test_local_environment_validates_model_files_and_sanitizes_legacy_text(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "local-model"
    model_path.mkdir()
    for filename in ("config.json", "model.safetensors", "tokenizer.json"):
        (model_path / filename).touch()

    environment = LocalEmbeddingEnvironment(
        model_path=model_path,
        python_executable=Path(sys.executable),
        device="cpu",
    )

    assert environment.enabled
    assert _sanitize_text("broken\udca2name") == "broken?name"


def test_business_document_is_static_and_contains_no_dynamic_evidence() -> None:
    business = _business(
        "steak",
        name="Romantic Steak House",
        categories=["Restaurants", "Steakhouses"],
    )

    first = build_business_document(business, document_version="v2")
    second = build_business_document(business, document_version="v2")

    assert first == second
    assert first.source_id == "steak"
    assert first.source_kind == "business_static"
    assert "Romantic Steak House" in first.text
    assert "Steakhouses" in first.text
    assert "customer experience" not in first.text.casefold()
    assert "rating" not in first.text.casefold()
    assert "popularity" not in first.text.casefold()
    assert "confidence" not in first.text.casefold()


def test_matcher_reuses_static_business_vectors_across_cutoffs(tmp_path: Path) -> None:
    cutoff = datetime(2022, 1, 1)
    later_cutoff = datetime(2023, 1, 1)
    reader = FixedBusinessReader(
        {
            "steak": _business(
                "steak",
                name="Romantic Steak House",
                categories=["Restaurants", "Steakhouses"],
            ),
            "dentist": _business(
                "dentist",
                name="Central Dentist",
                categories=["Health & Medical", "Dentists"],
            ),
        }
    )
    encoder = KeywordEncoder()
    cache = SqliteEmbeddingCache(tmp_path / "cache")
    matcher = SemanticEmbeddingMatcher(
        businesses=reader,
        gateway=CachedEmbeddingGateway(
            encoder=encoder,
            cache=cache,
            config=_config(),
        ),
        config=_config(),
    )

    first = matcher.match(
        query_text="A romantic steakhouse for a date",
        business_ids=["dentist", "steak"],
        cutoff_time=cutoff,
    )
    second = matcher.match(
        query_text="A romantic steakhouse for a date",
        business_ids=["dentist", "steak"],
        cutoff_time=later_cutoff,
    )

    assert [item.business_id for item in first.matches] == ["steak", "dentist"]
    assert first.usage.api_calls == 2
    assert first.usage.cache_misses == 3
    assert second.usage.api_calls == 0
    assert second.usage.cache_hits == 3
    assert len(encoder.calls) == 2
    assert all(item.cutoff_time == later_cutoff for item in second.matches)
    assert reader.requests == ["dentist", "steak", "dentist", "steak"]
    assert cache.count() == 3
    usage = cache.usage_summary()
    assert usage.event_count == 4
    assert usage.encoded_input_tokens == 15
    assert usage.logical_input_tokens == 30
    assert usage.cache_saved_tokens == 15
    assert usage.encoder_calls == 2
    assert usage.api_calls == 2
    assert b"A romantic steakhouse for a date" not in cache.path.read_bytes()
    exported = cache.export_parquet(tmp_path / "embedding_cache.parquet")
    assert exported.is_file()


def test_fusion_only_reorders_semantically_scored_hybrid_prefix() -> None:
    ranking = ["a", "b", "c", "tail-1", "tail-2"]

    fused = fuse_hybrid_and_semantic(
        ranking,
        {"a": 3, "b": 2, "c": 1},
        alpha=0.6,
    )

    assert fused == ["c", "b", "a", "tail-1", "tail-2"]
    assert fuse_hybrid_and_semantic(ranking, {}, alpha=0.6) == ranking


def test_dashscope_encoder_sends_query_parameters_and_validates_dimension() -> None:
    captured: dict[str, object] = {}

    class FakeEmbeddings:
        def create(self, **kwargs: object) -> object:
            captured.update(kwargs)
            return SimpleNamespace(
                data=[
                    SimpleNamespace(index=0, embedding=[1.0] + [0.0] * 255),
                    SimpleNamespace(index=1, embedding=[0.0, 1.0] + [0.0] * 254),
                ],
                model="fake-qwen-embedding",
                usage=SimpleNamespace(prompt_tokens=7, total_tokens=7),
                id="request-1",
            )

    encoder = DashScopeEmbeddingEncoder(
        client=SimpleNamespace(embeddings=FakeEmbeddings()),
        model="fake-qwen-embedding",
        config=_config(),
    )

    result = encoder.encode(["first", "second"], input_type="query")

    assert result.input_tokens == 7
    assert len(result.vectors) == 2
    assert captured["dimensions"] == 256
    assert captured["extra_body"] == {
        "text_type": "query",
        "output_type": "dense",
        "instruct": "Retrieve matching Yelp businesses.",
    }


class FixedSemanticService:
    def match(
        self,
        *,
        query_text: str,
        business_ids: list[str],
        cutoff_time: datetime,
        usage_scope: str | None = None,
    ) -> SemanticMatchResult:
        assert query_text == "romantic steakhouse"
        assert usage_scope == "f" * 64
        return SemanticMatchResult(
            query_sha256="c" * 64,
            model="fake-qwen-embedding",
            provider="dashscope",
            dimension=256,
            matches=[
                SemanticBusinessMatch(
                    business_id=business_id,
                    cutoff_time=cutoff_time,
                    document_sha256=("d" if index == 0 else "e") * 64,
                    cosine_similarity=1.0 - index,
                    normalized_score=1.0 - index / 2,
                    semantic_rank=index + 1,
                )
                for index, business_id in enumerate(business_ids)
            ],
            usage=EmbeddingUsage(
                api_calls=1,
                input_tokens=9,
                cache_hits=0,
                cache_misses=len(business_ids) + 1,
                estimated_cost_cny=0.00001,
                provider_latency_ms=3,
            ),
        )


def test_agent_tool_returns_semantic_evidence_without_final_ranking() -> None:
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
                {
                    "payload": {
                        "tool_name": "GET_HYBRID_RANKING",
                        "status": "success",
                        "data": {"ranking": ["a", "b"]},
                    }
                }
            ],
        },
    )

    observation = ComputeEmbeddingMatchTool(FixedSemanticService()).run(
        CandidateBusinessIdsInput(business_ids=["a", "b"]),
        context,
    )

    assert observation.status == "success"
    assert observation.input_tokens == 9
    assert "ranking" not in observation.data
    assert [row["business_id"] for row in observation.data["matches"]] == ["a", "b"]


def test_tfidf_embedding_comparison_is_explicitly_label_free() -> None:
    cutoff = datetime(2022, 1, 1)
    result = SemanticMatchResult(
        query_sha256="c" * 64,
        model="fake-qwen-embedding",
        provider="dashscope",
        dimension=256,
        matches=[
            SemanticBusinessMatch(
                business_id="semantic",
                cutoff_time=cutoff,
                document_sha256="d" * 64,
                cosine_similarity=0.9,
                normalized_score=0.95,
                semantic_rank=1,
            ),
            SemanticBusinessMatch(
                business_id="lexical",
                cutoff_time=cutoff,
                document_sha256="e" * 64,
                cosine_similarity=0.5,
                normalized_score=0.75,
                semantic_rank=2,
            ),
        ],
        usage=EmbeddingUsage(
            api_calls=1,
            input_tokens=10,
            cache_hits=0,
            cache_misses=3,
            estimated_cost_cny=0.00001,
            provider_latency_ms=2,
        ),
    )

    comparison = compare_tfidf_and_embedding(
        query_text="a calm place for a proposal",
        business_documents={
            "semantic": "romantic intimate restaurant for engagements",
            "lexical": "a proposal printing and office supply store",
        },
        embedding_result=result,
    )

    assert comparison.has_external_relevance_labels is False
    assert comparison.candidates[0].business_id == "semantic"
