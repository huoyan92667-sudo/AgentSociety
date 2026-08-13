"""Deep Step 34 module: one visible turn in, one canonical memory state out."""

from __future__ import annotations

from .config import SessionMemoryConfig
from .extractor import MemoryProposalExtractor, RuleMemoryExtractor
from .reducer import SessionMemoryReducer
from .resolver import SessionReferenceResolver
from .schema import MemoryExtractionTrace, MemoryTurnInput, MemoryTurnResult


class SessionMemoryManager:
    """Hide extraction, fallback, reference resolution, validation, and reduction."""

    def __init__(
        self,
        *,
        config: SessionMemoryConfig,
        primary_extractor: MemoryProposalExtractor | None = None,
        fallback_extractor: MemoryProposalExtractor | None = None,
        resolver: SessionReferenceResolver | None = None,
        reducer: SessionMemoryReducer | None = None,
    ) -> None:
        self._config = config
        self._primary = primary_extractor
        self._fallback = fallback_extractor or RuleMemoryExtractor()
        self._resolver = resolver or SessionReferenceResolver()
        self._reducer = reducer or SessionMemoryReducer(config)

    def update(self, value: MemoryTurnInput) -> MemoryTurnResult:
        """Produce memory without ever exposing an unvalidated model proposal."""

        primary = None
        should_call = self._primary is not None and (
            self._config.call_mode == "always" or value.previous_memory is not None
        )
        if should_call:
            primary = self._primary.extract(value)
        if (
            primary is not None
            and primary.proposal is not None
            and primary.trace.status == "success"
            and primary.proposal.confidence >= self._config.minimum_patch_confidence
        ):
            proposal = primary.proposal
            trace = primary.trace
        else:
            fallback = self._fallback.extract(value)
            if fallback.proposal is None:
                raise RuntimeError("rule memory fallback returned no proposal")
            proposal = fallback.proposal
            trace = (
                fallback.trace.model_copy(update={"status": "deterministic_bootstrap"})
                if (
                    primary is None
                    and self._primary is not None
                    and value.previous_memory is None
                    and self._config.call_mode == "followups_only"
                )
                else _fallback_trace(primary, fallback.trace)
            )
        references = self._resolver.resolve(
            proposal,
            memory=value.previous_memory,
            explicit_business_ids=value.explicit_referenced_business_ids,
        )
        return self._reducer.reduce(
            value,
            proposal=proposal,
            extraction=trace,
            references=references,
        )


def _fallback_trace(primary: object, fallback: MemoryExtractionTrace) -> MemoryExtractionTrace:
    if primary is None:
        return fallback
    trace = getattr(primary, "trace", None)
    if not isinstance(trace, MemoryExtractionTrace):
        return fallback
    return MemoryExtractionTrace(
        status="rule_fallback",
        extractor="rule",
        prompt_version=fallback.prompt_version,
        provider_called=trace.provider_called,
        cache_hit=trace.cache_hit,
        model=trace.model,
        latency_ms=trace.latency_ms,
        attempt_count=trace.attempt_count,
        input_tokens=trace.input_tokens,
        output_tokens=trace.output_tokens,
        total_tokens=trace.total_tokens,
        primary_failure_reason=trace.failure_reason or "proposal_below_confidence",
    )
