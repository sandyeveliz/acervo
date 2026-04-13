"""Pydantic schemas for LLM structured output in the extraction pipeline.

These models are used for post-parse validation of LLM responses in
entity_resolution and edge_resolution. They intentionally live separate from
acervo/domain/models.py because those are dataclasses for internal domain state,
while these are pydantic models for external LLM boundary validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

# ── Internal dedup working type ─────────────────────────────────────────────


@dataclass
class DedupNode:
    """Lightweight wrapper used by dedup_helpers.

    dedup_helpers operates on these instead of raw dicts or the full Entity
    dataclass so the Graphiti-derived code stays generic. entity_resolution
    converts between graph-store dicts / domain Entity objects and DedupNode.
    """

    uuid: str
    name: str
    type: str = 'entity'
    # Free-form extra fields we may want to surface back to the caller
    attributes: dict = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.uuid)


# ── LLM output schemas for node dedup escalation ────────────────────────────


class NodeDuplicate(BaseModel):
    """Single dedup decision returned by the LLM for one extracted node."""

    id: int = Field(
        ...,
        description='Position of the extracted entity in the input list (0-indexed).',
    )
    name: str = Field(
        ...,
        description=(
            'Canonical name for the entity. Prefer the most complete form '
            'between the extracted node and its duplicate candidate.'
        ),
    )
    duplicate_candidate_id: int = Field(
        ...,
        description=(
            'candidate_id of the matching existing entity, or -1 if no duplicate exists.'
        ),
    )


class NodeResolutions(BaseModel):
    """Batch of dedup decisions for all unresolved nodes in a turn."""

    entity_resolutions: list[NodeDuplicate] = Field(default_factory=list)


# ── LLM output schema for edge dedup + contradiction detection ──────────────


class EdgeDuplicate(BaseModel):
    """LLM decision for a single extracted edge.

    Indexing is continuous across the two candidate lists sent in the prompt:
    EXISTING_FACTS is indexed first (idx 0..M-1) and INVALIDATION_CANDIDATES
    follows (idx M..M+N-1). duplicate_facts must only contain idx values from
    the EXISTING_FACTS range; contradicted_facts can come from either range.
    """

    duplicate_facts: list[int] = Field(
        default_factory=list,
        description=(
            'idx values of existing facts that are semantically identical to the new fact. '
            'Empty list if none.'
        ),
    )
    contradicted_facts: list[int] = Field(
        default_factory=list,
        description=(
            'idx values of existing facts that the new fact contradicts. Empty list if none.'
        ),
    )


__all__ = [
    'DedupNode',
    'NodeDuplicate',
    'NodeResolutions',
    'EdgeDuplicate',
]
