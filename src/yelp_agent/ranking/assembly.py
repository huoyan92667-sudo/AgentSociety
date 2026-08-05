"""Build every Hybrid feature dependency at one shared seam."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from yelp_agent.config import AppConfig
from yelp_agent.features.category import TemporalCategoryStore
from yelp_agent.features.hybrid import HybridFeatureStore, HybridWeights
from yelp_agent.features.location import TemporalLocationStore
from yelp_agent.features.quality import TemporalQualityStore
from yelp_agent.features.text import TemporalTextStore
from yelp_agent.rankers.hybrid_ranker import HybridRanker
from yelp_agent.tuning.hybrid import (
    fingerprint_hybrid_feature_sources,
    load_validated_hybrid_weights,
)


@dataclass(frozen=True)
class HybridSourcePaths:
    """Files that define one point-in-time Hybrid feature runtime."""

    businesses: Path
    reviews: Path
    interactions: Path
    histories: Path
    tfidf_artifact: Path
    tfidf_manifest: Path
    config_dir: Path | None = None


@dataclass(frozen=True)
class HybridAssembly:
    """Shared feature stores and fingerprints before weights are attached."""

    feature_store: HybridFeatureStore
    quality_store: TemporalQualityStore
    location_store: TemporalLocationStore
    feature_sources_sha256: dict[str, str]


@dataclass(frozen=True)
class FrozenHybridRuntime:
    """A fully assembled Hybrid ranker backed by validated frozen weights."""

    assembly: HybridAssembly
    weights: HybridWeights
    ranker: HybridRanker


def _fingerprint_inputs(sources: HybridSourcePaths) -> dict[str, Path]:
    inputs = {
        "businesses": sources.businesses,
        "reviews": sources.reviews,
        "interactions": sources.interactions,
        "histories": sources.histories,
        "tfidf_artifact": sources.tfidf_artifact,
        "tfidf_manifest": sources.tfidf_manifest,
    }
    if sources.config_dir is not None:
        inputs.update(
            {
                "data_config": sources.config_dir / "data.yaml",
                "hybrid_config": sources.config_dir / "hybrid.yaml",
                "tfidf_config": sources.config_dir / "tfidf.yaml",
            }
        )
    return inputs


def _assemble_feature_stores(
    config: AppConfig,
    sources: HybridSourcePaths,
    feature_sources_sha256: dict[str, str],
) -> HybridAssembly:
    quality_store = TemporalQualityStore(
        sources.reviews,
        prior_count=config.hybrid.bayesian_prior_count,
    )
    location_store = TemporalLocationStore(
        sources.businesses,
        sources.interactions,
        sources.histories,
        scale_km=config.hybrid.location_scale_km,
    )
    feature_store = HybridFeatureStore(
        category_store=TemporalCategoryStore(
            sources.businesses,
            sources.interactions,
            sources.histories,
            broad_categories=set(config.data.broad_categories),
        ),
        text_store=TemporalTextStore(
            sources.businesses,
            sources.interactions,
            sources.histories,
            sources.tfidf_artifact,
            sources.tfidf_manifest,
        ),
        quality_store=quality_store,
        location_store=location_store,
    )
    return HybridAssembly(
        feature_store=feature_store,
        quality_store=quality_store,
        location_store=location_store,
        feature_sources_sha256=feature_sources_sha256,
    )


def build_hybrid_assembly(
    config: AppConfig,
    sources: HybridSourcePaths,
) -> HybridAssembly:
    """Build the four feature stores once behind one shared interface."""

    feature_sources_sha256 = fingerprint_hybrid_feature_sources(
        _fingerprint_inputs(sources)
    )
    return _assemble_feature_stores(
        config,
        sources,
        feature_sources_sha256,
    )


def build_frozen_hybrid_runtime(
    config: AppConfig,
    sources: HybridSourcePaths,
    weights_path: str | Path,
) -> FrozenHybridRuntime:
    """Build Hybrid and attach weights only if every source still matches."""

    feature_sources_sha256 = fingerprint_hybrid_feature_sources(
        _fingerprint_inputs(sources)
    )
    weights = load_validated_hybrid_weights(
        weights_path,
        feature_sources_sha256=feature_sources_sha256,
    )
    assembly = _assemble_feature_stores(
        config,
        sources,
        feature_sources_sha256,
    )
    return FrozenHybridRuntime(
        assembly=assembly,
        weights=weights,
        ranker=HybridRanker(assembly.feature_store, weights),
    )
