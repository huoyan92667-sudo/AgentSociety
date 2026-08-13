"""Scope-safe resolution of ordinal and explicit business references."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import (
    MemoryProposal,
    ResolvedMemoryReference,
    SessionMemory,
)


@dataclass(frozen=True, slots=True)
class ReferenceResolutionResult:
    resolved: tuple[ResolvedMemoryReference, ...]
    rejected: tuple[str, ...]


class SessionReferenceResolver:
    """Map model reference hints only through visible, canonical session state."""

    def resolve(
        self,
        proposal: MemoryProposal,
        *,
        memory: SessionMemory | None,
        explicit_business_ids: list[str],
    ) -> ReferenceResolutionResult:
        displayed = [] if memory is None else memory.last_presented_business_ids
        visible = set(displayed) | set(explicit_business_ids)
        resolved: list[ResolvedMemoryReference] = []
        rejected: list[str] = []
        for mention in proposal.references:
            business_id: str | None = None
            source: str | None = None
            if mention.ordinal is not None:
                index = mention.ordinal - 1
                if 0 <= index < len(displayed):
                    business_id = displayed[index]
                    source = "ordinal"
                else:
                    rejected.append(
                        f"reference:{mention.reference_id}:ordinal_out_of_range"
                    )
            else:
                assert mention.explicit_business_id is not None
                if mention.explicit_business_id in visible:
                    business_id = mention.explicit_business_id
                    source = "explicit_visible_id"
                else:
                    rejected.append(
                        f"reference:{mention.reference_id}:business_out_of_visible_scope"
                    )
            if business_id is not None and source is not None:
                resolved.append(
                    ResolvedMemoryReference(
                        reference_id=mention.reference_id,
                        expression=mention.expression,
                        business_id=business_id,
                        resolution_source=source,  # type: ignore[arg-type]
                    )
                )
        return ReferenceResolutionResult(tuple(resolved), tuple(rejected))
