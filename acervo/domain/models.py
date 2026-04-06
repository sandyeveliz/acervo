"""Domain models — data classes that flow between pipeline stages.

These are the shared contracts. Each stage receives and returns
well-typed objects, not raw dicts.

Backward compat: the original files (extractor.py, s1_unified.py,
s1_5_graph_update.py) re-export these for existing consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Core extraction types ──


@dataclass
class Entity:
    """An entity extracted from conversation or text."""
    name: str
    type: str
    layer: str = ""  # "PERSONAL" or "UNIVERSAL", empty = unset
    attributes: dict = field(default_factory=dict)


@dataclass
class Relation:
    """A directed relation between two entities."""
    source: str
    target: str
    relation: str


@dataclass
class ExtractedFact:
    """A fact attributed to an entity."""
    entity: str
    fact: str
    source: str    # "user", "web", "rag"
    speaker: str = "user"  # "user" | "assistant"


@dataclass
class ExtractionResult:
    """The output of any extraction stage."""
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    facts: list[ExtractedFact] = field(default_factory=list)


# ── S1 result types ──


@dataclass
class TopicResult:
    """Topic classification from S1."""
    action: str  # "same" | "subtopic" | "changed"
    label: str | None  # topic label (only if subtopic/changed)


@dataclass
class S1Result:
    """Output of S1: topic classification + entity extraction."""
    topic: TopicResult
    extraction: ExtractionResult
    intent: str = "specific"  # "overview" | "specific" | "followup" | "chat"
    retrieval: str | None = None  # "summary_only" | "with_chunks" | None
    # Debug data for telemetry/annotation
    prompt_sent: str = ""
    raw_response: str = ""


# ── S2 result types ──


@dataclass
class GatheredNode:
    """A graph node gathered by S2, enriched with relations and hot flag."""
    node: dict[str, Any]  # raw graph node dict
    relations: list[str]  # formatted relation strings (e.g. "uses_technology: Angular")
    hot: bool = True      # True = directly activated, False = neighbor expansion


@dataclass
class RankedChunk:
    """A piece of context with a relevance score, ready for S3 budget selection."""
    text: str
    score: float        # 0.0–1.0 relevance score
    source: str         # "verified_fact", "conversation_fact", "verified_relation", etc.
    label: str          # entity name or file path (for debug)
    tokens: int = 0


@dataclass
class S2Result:
    """Output of S2: activated nodes + ranked chunks for context assembly."""
    activated_nodes: list[GatheredNode] = field(default_factory=list)
    chunks: list[RankedChunk] = field(default_factory=list)
    active_node_ids: set[str] = field(default_factory=set)
    # Debug
    vector_hits: list[dict] = field(default_factory=list)


# ── S3 result types ──


@dataclass
class S3Result:
    """Output of S3: the assembled context stack ready for the LLM."""
    context_stack: list[dict] = field(default_factory=list)
    warm_content: str = ""
    warm_tokens: int = 0
    hot_tokens: int = 0
    total_tokens: int = 0
    has_context: bool = False
    needs_tool: bool = False


# ── S1.5 result types ──


@dataclass
class MergeAction:
    """A node merge action from S1.5 curation."""
    from_id: str
    into_id: str
    reason: str


@dataclass
class TypeCorrection:
    """A type correction from S1.5 curation."""
    node_id: str
    old_type: str
    new_type: str
    reason: str


@dataclass
class DiscardAction:
    """A node discard action from S1.5 curation."""
    node_id: str
    reason: str


@dataclass
class S15Result:
    """Output of S1.5: graph curation + assistant entity extraction."""
    merges: list[MergeAction] = field(default_factory=list)
    new_relations: list[Relation] = field(default_factory=list)
    type_corrections: list[TypeCorrection] = field(default_factory=list)
    discards: list[DiscardAction] = field(default_factory=list)
    assistant_extraction: ExtractionResult = field(default_factory=ExtractionResult)
    # Debug data for telemetry/annotation
    prompt_sent: str = ""
    raw_response: str = ""
