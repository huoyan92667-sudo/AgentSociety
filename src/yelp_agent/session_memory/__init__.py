"""Authoritative Step 34 session-memory contracts.

Heavy adapters are intentionally not imported here. Keeping package import side
effects small prevents the Harness schema and controlled-LLM adapters from
forming a circular dependency.
"""

from .config import SessionMemoryConfig, load_session_memory_config
from .context import compact_memory
from .effective_request import compile_effective_request, effective_request_from_snapshot
from .schema import (
    ClarificationAnswerProposal,
    EffectiveRelativePreference,
    EffectiveSessionRequest,
    MemoryConditionPatch,
    MemoryExtractionTrace,
    MemoryProposal,
    MemoryReferenceMention,
    MemoryTurnInput,
    MemoryTurnRecord,
    MemoryTurnResult,
    PresentedCandidateSet,
    RelativePreference,
    ResolvedMemoryReference,
    RouterMemoryContext,
    SessionMemory,
)

__all__ = [
    "ClarificationAnswerProposal",
    "EffectiveRelativePreference",
    "EffectiveSessionRequest",
    "MemoryConditionPatch",
    "MemoryExtractionTrace",
    "MemoryProposal",
    "MemoryReferenceMention",
    "MemoryTurnInput",
    "MemoryTurnRecord",
    "MemoryTurnResult",
    "PresentedCandidateSet",
    "RelativePreference",
    "ResolvedMemoryReference",
    "RouterMemoryContext",
    "SessionMemory",
    "SessionMemoryConfig",
    "compact_memory",
    "compile_effective_request",
    "effective_request_from_snapshot",
    "load_session_memory_config",
]
