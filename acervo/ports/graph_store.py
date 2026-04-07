"""Graph store port — knowledge graph persistence interface."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GraphStorePort(Protocol):
    """Knowledge graph persistence interface.

    The domain layer depends on this protocol, not on TopicGraph directly.
    This enables testing with mock graphs and alternative storage backends.
    """

    # ── Read ──

    def get_node(self, name_or_id: str) -> dict[str, Any] | None:
        """Get a node by name or ID."""
        ...

    def get_all_nodes(self) -> list[dict[str, Any]]:
        """Return all nodes."""
        ...

    def get_nodes_by_ids(self, node_ids: set[str]) -> list[dict[str, Any]]:
        """Return nodes matching the given IDs."""
        ...

    def get_nodes_by_kind(self, kind: str) -> list[dict[str, Any]]:
        """Return nodes of a specific kind (entity, file, section, etc.)."""
        ...

    def get_edges_for(self, node_id: str) -> list[dict[str, Any]]:
        """Return all edges involving a node (as source or target)."""
        ...

    def get_neighbors(self, node_id: str, max_count: int = 5) -> list[tuple[dict, float]]:
        """Return neighbors of a node with relevance weights."""
        ...

    # ── Write ──

    def upsert_entities(
        self,
        entities: list[tuple[str, str]],
        relations: list[tuple[str, str, str]] | None = None,
        facts: list[tuple[str, str]] | None = None,
        *,
        layer: Any = None,
        source: str = "conversation",
        owner: str | None = None,
    ) -> None:
        """Create or update entities, relations, and facts."""
        ...

    def save(self) -> None:
        """Persist graph state to disk."""
        ...

    # ── Properties ──

    @property
    def node_count(self) -> int: ...

    @property
    def edge_count(self) -> int: ...

    # ── File/chunk operations (for indexed projects) ──

    def get_file_symbols(self, file_path: str) -> list[dict]:
        """Return symbols (functions, classes, etc.) linked to a file."""
        ...

    def get_symbol_content(self, node_id: str, workspace_root: Any) -> str | None:
        """Read symbol content from workspace."""
        ...

    def mark_file_stale(self, file_path: str) -> bool:
        """Mark a file as stale for re-indexing."""
        ...
