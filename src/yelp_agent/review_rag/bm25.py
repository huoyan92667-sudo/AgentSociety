"""Small deterministic BM25 implementation scoped to one business."""

from __future__ import annotations

from collections import Counter
import math
import re
from collections.abc import Sequence


_WORD = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
_CJK = re.compile(r"[\u3400-\u9fff]+")


def tokenize(text: str) -> list[str]:
    lowered = text.casefold()
    tokens = _WORD.findall(lowered)
    for value in _CJK.findall(lowered):
        tokens.extend(value[index : index + 2] for index in range(max(1, len(value) - 1)))
    return tokens


def bm25_scores(
    query: str,
    documents: Sequence[str],
    *,
    expansions: Sequence[str] = (),
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    if not documents:
        return []
    query_terms = list(dict.fromkeys(tokenize(" ".join((query, *expansions)))))
    if not query_terms:
        return [0.0] * len(documents)
    term_frequencies = [Counter(tokenize(document)) for document in documents]
    lengths = [sum(values.values()) for values in term_frequencies]
    average_length = sum(lengths) / len(lengths) or 1.0
    document_frequency = {
        term: sum(term in frequencies for frequencies in term_frequencies)
        for term in query_terms
    }
    scores: list[float] = []
    count = len(documents)
    for frequencies, length in zip(term_frequencies, lengths, strict=True):
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue
            occurrences = document_frequency[term]
            inverse = math.log(1.0 + (count - occurrences + 0.5) / (occurrences + 0.5))
            denominator = frequency + k1 * (1.0 - b + b * length / average_length)
            score += inverse * frequency * (k1 + 1.0) / denominator
        scores.append(score)
    return scores
