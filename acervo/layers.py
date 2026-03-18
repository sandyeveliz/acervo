"""Knowledge layers — separates universal knowledge from personal context.

Layer 1 (UNIVERSAL): verifiable world facts (cities, countries, historical facts).
Layer 2 (PERSONAL): user-specific knowledge (projects, relationships, preferences).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class Layer(Enum):
    UNIVERSAL = 1  # Layer 1: world knowledge, externally verifiable
    PERSONAL = 2   # Layer 2: user knowledge, stated by the user


# Source types for nodes and edges
Source = Literal["world", "user_assertion"]

# Possible node statuses
NodeStatus = Literal["complete", "incomplete", "pending_verification"]


@dataclass
class NodeMeta:
    """Metadata every node must carry. Determines trust, ownership, and completeness."""

    layer: Layer
    owner: str | None = None         # None for Layer.UNIVERSAL
    source: str = "user_assertion"   # "world" or "user_assertion"
    confidence_for_owner: float = 1.0  # 1.0 if source="user_assertion"
    status: str = "complete"         # "complete", "incomplete", "pending_verification"
    pending_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a dict for JSON persistence."""
        return {
            "layer": self.layer.name,
            "owner": self.owner,
            "source": self.source,
            "confidence_for_owner": self.confidence_for_owner,
            "status": self.status,
            "pending_fields": list(self.pending_fields),
        }

    @classmethod
    def from_dict(cls, data: dict) -> NodeMeta:
        """Deserialize from a persisted dict."""
        layer_name = data.get("layer", "PERSONAL")
        try:
            layer = Layer[layer_name]
        except KeyError:
            layer = Layer.PERSONAL
        return cls(
            layer=layer,
            owner=data.get("owner"),
            source=data.get("source", "user_assertion"),
            confidence_for_owner=data.get("confidence_for_owner", 1.0),
            status=data.get("status", "complete"),
            pending_fields=data.get("pending_fields", []),
        )

    @classmethod
    def personal(cls, owner: str | None = None) -> NodeMeta:
        """Create metadata for a user-asserted personal node."""
        return cls(
            layer=Layer.PERSONAL,
            owner=owner,
            source="user_assertion",
            confidence_for_owner=1.0,
            status="complete",
        )

    @classmethod
    def universal(cls) -> NodeMeta:
        """Create metadata for a verified universal knowledge node."""
        return cls(
            layer=Layer.UNIVERSAL,
            owner=None,
            source="world",
            confidence_for_owner=1.0,
            status="complete",
        )

    @classmethod
    def incomplete(cls, owner: str | None = None, pending: list[str] | None = None) -> NodeMeta:
        """Create metadata for a node that needs more information."""
        return cls(
            layer=Layer.PERSONAL,
            owner=owner,
            source="user_assertion",
            confidence_for_owner=1.0,
            status="incomplete",
            pending_fields=pending or ["type"],
        )
