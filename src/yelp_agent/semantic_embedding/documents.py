"""Deterministic query and cutoff-independent static business documents."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from yelp_agent.data.temporal_view import BusinessRecord

from .schema import SemanticDocument


QUERY_DOCUMENT_VERSION = "query-text-v1.0.0"

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_query_document(query_text: str) -> SemanticDocument:
    text = " ".join(query_text.strip().split())
    if not text:
        raise ValueError("query text cannot be blank")
    return SemanticDocument(
        source_id=_sha256(text),
        source_kind="query",
        text=text,
        text_sha256=_sha256(text),
        document_version=QUERY_DOCUMENT_VERSION,
    )


def build_business_document(
    business: BusinessRecord,
    *,
    document_version: str,
) -> SemanticDocument:
    """Render fields whose meaning does not change between task cutoffs.

    Rating, popularity, review evidence, Aspect summaries, coordinates, and
    user-specific signals deliberately cannot enter this function. They stay
    in the structured point-in-time ranker instead of invalidating this vector.
    """

    categories = ", ".join(business.categories)
    attribute_parts = _flatten_attributes(business.attributes_dict())
    sections = [
        f"Business name: {business.name}",
        f"Categories: {categories}",
        f"Location: {business.city}, {business.state}",
    ]
    if attribute_parts:
        sections.append(f"Structured attributes: {'; '.join(attribute_parts)}")
    text = "\n".join(sections)
    return SemanticDocument(
        source_id=business.business_id,
        source_kind="business_static",
        text=text,
        text_sha256=_sha256(text),
        document_version=document_version,
    )


def _humanize(value: str) -> str:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    return spaced.replace("_", " ").strip()


def _flatten_attributes(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        result: list[str] = []
        for key in sorted(value, key=str):
            label = _humanize(str(key))
            path = f"{prefix} {label}".strip()
            result.extend(_flatten_attributes(value[key], path))
        return result
    if isinstance(value, (list, tuple, set)):
        result = []
        for item in sorted(value, key=str):
            result.extend(_flatten_attributes(item, prefix))
        return result
    if value is None or value == "None":
        return []
    rendered = str(value).strip()
    if not rendered:
        return []
    return [f"{prefix}={rendered}" if prefix else rendered]
