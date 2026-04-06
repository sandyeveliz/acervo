"""TopicNode — modelo de nodo enriquecido para uso futuro.

Stub preparado para cuando se migre el grafo a dataclasses tipadas
en lugar de dicts planos. Por ahora TopicGraph usa dicts internamente.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from acervo.layers import Layer


@dataclass
class TopicNode:
    id: str
    label: str
    type: str
    layer: Layer = Layer.PERSONAL
    source: str = "user_assertion"          # "world" | "user_assertion"
    confidence_for_owner: float = 1.0
    status: str = "complete"               # "complete" | "incomplete" | "pending_verification"
    pending_fields: list[str] = field(default_factory=list)
    facts: list[dict] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)
    session_count: int = 1
    created_at: str = ""
    last_active: str = ""
