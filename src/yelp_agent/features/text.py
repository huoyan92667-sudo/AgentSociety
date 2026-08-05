"""Train-safe TF-IDF artifacts and point-in-time text preference features."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

import duckdb
import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from pydantic import Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from yelp_agent.config import TfidfConfig
from yelp_agent.data.temporal_view import TemporalDataView
from yelp_agent.models import RecommendationTask, StrictModel, UnitScore


class TextFeatureError(RuntimeError):
    """Raised when text features cannot be trained or loaded safely."""


class TfidfManifest(StrictModel):
    format_version: int = Field(ge=1)
    sklearn_version: str
    config: TfidfConfig
    source_sha256: dict[str, str]
    artifact_sha256: str
    training_review_count: int = Field(ge=0)
    business_document_count: int = Field(ge=0)
    document_count: int = Field(ge=0)
    vocabulary_size: int = Field(gt=0)


class TfidfFitResult(StrictModel):
    status: Literal["written", "skipped"]
    artifact_path: str
    manifest_path: str
    training_review_count: int = Field(ge=0)
    business_document_count: int = Field(ge=0)
    document_count: int = Field(ge=0)
    vocabulary_size: int = Field(gt=0)


class TextBusinessScore(StrictModel):
    business_id: str = Field(min_length=1)
    positive_similarity: UnitScore
    negative_similarity: UnitScore
    text_score: UnitScore


class TextTaskFeatures(StrictModel):
    positive_review_count: int = Field(ge=0)
    negative_review_count: int = Field(ge=0)
    positive_keywords: list[str] = Field(max_length=10)
    negative_keywords: list[str] = Field(max_length=10)
    business_scores: dict[str, TextBusinessScore]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprints(
    businesses_path: Path,
    interactions_path: Path,
    histories_path: Path,
) -> dict[str, str]:
    paths = {
        "businesses": businesses_path,
        "interactions": interactions_path,
        "histories": histories_path,
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"TF-IDF {label} input does not exist: {path}")
    return {label: _sha256_file(path) for label, path in paths.items()}


def _humanize_attribute_key(value: str) -> str:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    return spaced.replace("_", " ")


def _flatten_attribute(value: object) -> list[str]:
    if isinstance(value, dict):
        flattened: list[str] = []
        for key in sorted(value):
            flattened.append(_humanize_attribute_key(str(key)))
            flattened.extend(_flatten_attribute(value[key]))
        return flattened
    if isinstance(value, (list, tuple, set)):
        flattened = []
        for item in value:
            flattened.extend(_flatten_attribute(item))
        return flattened
    if value is None:
        return []
    return [str(value)]


def _load_business_documents(path: Path) -> tuple[list[str], list[str]]:
    try:
        rows = pq.read_table(
            path,
            columns=["business_id", "name", "categories", "attributes_json"],
        ).to_pylist()
    except (OSError, pa.ArrowException) as exc:
        raise TextFeatureError(
            f"Could not read TF-IDF business documents from {path}: {exc}"
        ) from exc

    business_ids: list[str] = []
    documents: list[str] = []
    seen: set[str] = set()
    for row in rows:
        business_id = row.get("business_id")
        categories = row.get("categories")
        if (
            not isinstance(business_id, str)
            or not business_id
            or business_id in seen
            or not isinstance(categories, list)
        ):
            raise TextFeatureError("TF-IDF business data contains an invalid row")
        try:
            attributes = json.loads(str(row.get("attributes_json") or "{}"))
        except json.JSONDecodeError as exc:
            raise TextFeatureError(
                f"Business {business_id!r} has invalid attributes_json"
            ) from exc
        if not isinstance(attributes, dict):
            raise TextFeatureError(
                f"Business {business_id!r} attributes must be a mapping"
            )
        document_parts = [
            str(row.get("name") or "").strip(),
            *(
                str(category).strip()
                for category in categories
                if str(category).strip()
            ),
            *_flatten_attribute(attributes),
        ]
        document = " ".join(part for part in document_parts if part).strip()
        if not document:
            raise TextFeatureError(
                f"Business {business_id!r} has an empty static document"
            )
        seen.add(business_id)
        business_ids.append(business_id)
        documents.append(document)
    if not documents:
        raise TextFeatureError("TF-IDF business corpus is empty")
    return business_ids, documents


def _business_documents_from_view(
    data_view: TemporalDataView,
) -> tuple[list[str], list[str]]:
    business_ids: list[str] = []
    documents: list[str] = []
    for business in data_view.businesses():
        attributes = business.attributes_dict()
        document_parts = [
            business.name.strip(),
            *business.categories,
            *_flatten_attribute(attributes),
        ]
        document = " ".join(
            part for part in document_parts if part
        ).strip()
        if not document:
            raise TextFeatureError(
                f"Business {business.business_id!r} has an empty static document"
            )
        business_ids.append(business.business_id)
        documents.append(document)
    return business_ids, documents


def _load_training_review_documents(
    interactions_path: Path,
    histories_path: Path,
) -> list[str]:
    try:
        with duckdb.connect() as connection:
            interaction_counts = connection.execute(
                """
                SELECT count(*), count(DISTINCT review_id)
                FROM read_parquet(?)
                """,
                [str(interactions_path)],
            ).fetchone()
            expected_count = connection.execute(
                """
                SELECT count(DISTINCT review_id)
                FROM read_parquet(?)
                WHERE starts_with(task_id, 'validation:')
                """,
                [str(histories_path)],
            ).fetchone()[0]
            rows = connection.execute(
                """
                WITH training_ids AS (
                    SELECT DISTINCT review_id
                    FROM read_parquet(?)
                    WHERE starts_with(task_id, 'validation:')
                )
                SELECT interaction.review_id, interaction.text
                FROM training_ids
                JOIN read_parquet(?) AS interaction USING (review_id)
                ORDER BY interaction.review_id
                """,
                [str(histories_path), str(interactions_path)],
            ).fetchall()
    except duckdb.Error as exc:
        raise TextFeatureError(
            f"Could not load training-safe TF-IDF reviews: {exc}"
        ) from exc

    if interaction_counts[0] != interaction_counts[1]:
        raise TextFeatureError("Interaction review_id values must be unique")
    if len(rows) != expected_count:
        raise TextFeatureError(
            "A validation-history review is missing from interactions"
        )
    documents = [str(text or "").strip() for _, text in rows]
    if not documents:
        raise TextFeatureError("Validation histories contain no training reviews")
    return documents


def load_tfidf_vectorizer(path: str | Path) -> TfidfVectorizer:
    """Load and type-check one persisted TF-IDF vectorizer."""

    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"TF-IDF vectorizer artifact does not exist: {artifact_path}"
        )
    try:
        vectorizer = joblib.load(artifact_path)
    except Exception as exc:
        raise TextFeatureError(
            f"Could not load TF-IDF vectorizer from {artifact_path}: {exc}"
        ) from exc
    if not isinstance(vectorizer, TfidfVectorizer):
        raise TextFeatureError(
            f"TF-IDF artifact has unexpected type: {type(vectorizer).__name__}"
        )
    if not hasattr(vectorizer, "vocabulary_"):
        raise TextFeatureError("TF-IDF vectorizer has not been fitted")
    return vectorizer


def _fit_result(
    status: Literal["written", "skipped"],
    artifact_path: Path,
    manifest_path: Path,
    manifest: TfidfManifest,
) -> TfidfFitResult:
    return TfidfFitResult(
        status=status,
        artifact_path=str(artifact_path),
        manifest_path=str(manifest_path),
        training_review_count=manifest.training_review_count,
        business_document_count=manifest.business_document_count,
        document_count=manifest.document_count,
        vocabulary_size=manifest.vocabulary_size,
    )


def fit_tfidf_model(
    businesses_path: str | Path,
    interactions_path: str | Path,
    histories_path: str | Path,
    artifact_path: str | Path,
    manifest_path: str | Path,
    config: TfidfConfig,
    *,
    force: bool = False,
) -> TfidfFitResult:
    """Fit or safely reuse a train-history-only TF-IDF vectorizer."""

    businesses = Path(businesses_path)
    interactions = Path(interactions_path)
    histories = Path(histories_path)
    artifact = Path(artifact_path)
    manifest_file = Path(manifest_path)
    source_sha256 = _source_fingerprints(
        businesses,
        interactions,
        histories,
    )

    existing = (artifact.is_file(), manifest_file.is_file())
    if any(existing) and not force:
        if not all(existing):
            raise TextFeatureError(
                "TF-IDF artifact outputs are incomplete; use force=True to rebuild"
            )
        try:
            manifest = TfidfManifest.model_validate_json(
                manifest_file.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise TextFeatureError(
                f"TF-IDF manifest is invalid: {manifest_file}"
            ) from exc
        if (
            manifest.format_version != 1
            or manifest.sklearn_version != sklearn.__version__
            or manifest.config != config
            or manifest.source_sha256 != source_sha256
            or manifest.artifact_sha256 != _sha256_file(artifact)
        ):
            raise TextFeatureError(
                "Existing TF-IDF artifact does not match its inputs or config; "
                "use force=True to rebuild"
            )
        vectorizer = load_tfidf_vectorizer(artifact)
        if len(vectorizer.vocabulary_) != manifest.vocabulary_size:
            raise TextFeatureError(
                "Existing TF-IDF vocabulary size does not match its manifest"
            )
        return _fit_result("skipped", artifact, manifest_file, manifest)

    _, business_documents = _load_business_documents(businesses)
    review_documents = _load_training_review_documents(interactions, histories)
    corpus = [*review_documents, *business_documents]
    vectorizer = TfidfVectorizer(
        stop_words=config.stop_words,
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        sublinear_tf=config.sublinear_tf,
        max_features=config.max_features,
        dtype=np.float32,
    )
    try:
        vectorizer.fit(corpus)
    except ValueError as exc:
        raise TextFeatureError(f"Could not fit TF-IDF vocabulary: {exc}") from exc

    artifact.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    artifact_partial = artifact.with_name(artifact.name + ".partial")
    manifest_partial = manifest_file.with_name(manifest_file.name + ".partial")
    artifact_partial.unlink(missing_ok=True)
    manifest_partial.unlink(missing_ok=True)
    try:
        # scikit-learn caches ``id(self.stop_words)`` after fitting. That
        # process-local memory address has no model semantics, but serializing
        # it makes two identical fits produce different Joblib bytes. Reset it
        # before persistence; sklearn safely rebuilds the cache on transform.
        vectorizer._stop_words_id = None
        joblib.dump(vectorizer, artifact_partial)
        manifest = TfidfManifest(
            format_version=1,
            sklearn_version=sklearn.__version__,
            config=config,
            source_sha256=source_sha256,
            artifact_sha256=_sha256_file(artifact_partial),
            training_review_count=len(review_documents),
            business_document_count=len(business_documents),
            document_count=len(corpus),
            vocabulary_size=len(vectorizer.vocabulary_),
        )
        manifest_partial.write_text(
            manifest.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(artifact_partial, artifact)
        os.replace(manifest_partial, manifest_file)
    except Exception:
        artifact_partial.unlink(missing_ok=True)
        manifest_partial.unlink(missing_ok=True)
        raise
    return _fit_result("written", artifact, manifest_file, manifest)


def _load_tfidf_manifest(
    manifest_path: Path,
    artifact_path: Path,
) -> TfidfManifest:
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"TF-IDF manifest does not exist: {manifest_path}"
        )
    try:
        manifest = TfidfManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise TextFeatureError(
            f"Could not parse TF-IDF manifest: {manifest_path}"
        ) from exc
    if (
        manifest.sklearn_version != sklearn.__version__
        or manifest.artifact_sha256 != _sha256_file(artifact_path)
    ):
        raise TextFeatureError(
            "TF-IDF artifact does not match its manifest or sklearn version"
        )
    return manifest


class TemporalTextStore:
    """Transform frozen user histories and candidate static text at one seam."""

    def __init__(
        self,
        data_view: TemporalDataView,
        artifact_path: str | Path,
        manifest_path: str | Path,
    ) -> None:
        artifact = Path(artifact_path)
        self._vectorizer = load_tfidf_vectorizer(artifact)
        self._manifest = _load_tfidf_manifest(Path(manifest_path), artifact)
        self._data_view = data_view
        business_ids, business_documents = _business_documents_from_view(
            data_view
        )
        self._business_row = {
            business_id: index
            for index, business_id in enumerate(business_ids)
        }
        self._business_vectors = self._vectorizer.transform(business_documents)
        self._feature_names = self._vectorizer.get_feature_names_out()
        self._cache: dict[
            tuple[str, str, datetime, tuple[str, ...]],
            TextTaskFeatures,
        ] = {}

    def _top_keywords(self, vector: object) -> list[str]:
        row = vector.getrow(0)
        weighted_terms = [
            (float(weight), str(self._feature_names[index]))
            for index, weight in zip(row.indices, row.data, strict=True)
        ]
        weighted_terms.sort(key=lambda item: (-item[0], item[1]))
        return [
            term
            for _, term in weighted_terms[: self._manifest.config.keyword_count]
        ]

    def features_for(self, task: RecommendationTask) -> TextTaskFeatures:
        """Return positive/negative text affinity for one frozen task."""

        cache_key = (
            task.task_id,
            task.user_id,
            task.cutoff_time,
            tuple(task.candidate_business_ids),
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        history = self._data_view.user_history(
            task.user_id,
            task.cutoff_time,
        )
        if not history:
            raise TextFeatureError(
                f"No frozen TF-IDF history exists for task {task.task_id!r}"
            )
        positive_texts = [
            interaction.text
            for interaction in history
            if interaction.stars >= 4.0 and interaction.text
        ]
        negative_texts = [
            interaction.text
            for interaction in history
            if interaction.stars <= 2.0 and interaction.text
        ]
        positive_vector = self._vectorizer.transform(
            ["\n".join(positive_texts)]
        )
        negative_vector = self._vectorizer.transform(
            ["\n".join(negative_texts)]
        )

        try:
            candidate_rows = [
                self._business_row[business_id]
                for business_id in task.candidate_business_ids
            ]
        except KeyError as exc:
            raise TextFeatureError(
                f"Task references unknown business_id {exc.args[0]!r}"
            ) from exc
        candidate_vectors = self._business_vectors[candidate_rows]
        positive_similarities = cosine_similarity(
            positive_vector,
            candidate_vectors,
        )[0]
        negative_similarities = cosine_similarity(
            negative_vector,
            candidate_vectors,
        )[0]

        business_scores: dict[str, TextBusinessScore] = {}
        for business_id, positive, negative in zip(
            task.candidate_business_ids,
            positive_similarities,
            negative_similarities,
            strict=True,
        ):
            positive_score = float(np.clip(positive, 0.0, 1.0))
            negative_score = float(np.clip(negative, 0.0, 1.0))
            text_score = float(
                np.clip(
                    0.5 + 0.5 * positive_score - 0.5 * negative_score,
                    0.0,
                    1.0,
                )
            )
            business_scores[business_id] = TextBusinessScore(
                business_id=business_id,
                positive_similarity=positive_score,
                negative_similarity=negative_score,
                text_score=text_score,
            )

        features = TextTaskFeatures(
            positive_review_count=len(positive_texts),
            negative_review_count=len(negative_texts),
            positive_keywords=self._top_keywords(positive_vector),
            negative_keywords=self._top_keywords(negative_vector),
            business_scores=business_scores,
        )
        self._cache[cache_key] = features
        return features
