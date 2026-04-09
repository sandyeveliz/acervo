"""Graph store port — knowledge graph persistence interface.

The domain layer depends on this protocol, not on TopicGraph directly.
This enables testing with mock graphs and alternative storage backends
(JSON via TopicGraph, LadybugDB via LadybugGraphStore).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GraphStorePort(Protocol):
    """Knowledge graph persistence interface.

    Implemented by:
      - TopicGraph (JSON files)
      - LadybugGraphStore (LadybugDB / KuzuDB)
    """

    # ── Read — nodes ──

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

    # ── Read — edges & neighbors ──

    def get_edges_for(self, node_id: str) -> list[dict[str, Any]]:
        """Return all edges involving a node (as source or target)."""
        ...

    def get_neighbors(self, node_id: str, max_count: int = 5) -> list[tuple[dict, float]]:
        """Return neighbors of a node with relevance weights."""
        ...

    # ── Read — files & symbols ──

    def get_linked_files(self, node_id: str) -> list[str]:
        """Return list of file paths linked to a node."""
        ...

    def get_file_symbols(self, file_path: str) -> list[dict]:
        """Return symbols (functions, classes, etc.) linked to a file."""
        ...

    def get_symbol_content(self, node_id: str, workspace_root: Path) -> str | None:
        """Read symbol content from workspace."""
        ...

    def get_stale_files(self) -> list[dict]:
        """Return all file nodes marked as stale."""
        ...

    # ── Read — chunks ──

    def get_chunks_for_node(self, node_id: str) -> list[str]:
        """Return chunk_ids for a node."""
        ...

    def get_nodes_with_chunks(self) -> list[dict]:
        """Return all nodes that have non-empty chunk_ids."""
        ...

    # ── Write — entities ──

    def upsert_entities(
        self,
        entities: list[tuple[str, str]],
        relations: list[tuple[str, str, str]] | None = None,
        facts: list[tuple[str, str, str]] | None = None,
        *,
        layer: Any = None,
        source: str = "conversation",
        owner: str | None = None,
        confidence: float = 1.0,
    ) -> tuple[int, int]:
        """Create or update entities, relations, and facts.

        Returns (node_count, edge_count) after upsert.
        """
        ...

    # ── Write — node mutations ──

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its edges. Returns True if found."""
        ...

    def update_node(self, node_id: str, **fields: object) -> bool:
        """Update fields on an existing node. Returns True if found."""
        ...

    def merge_nodes(self, keep_id: str, absorb_id: str, alias: str | None = None) -> bool:
        """Merge two nodes: keep one, absorb facts/edges from the other."""
        ...

    def remove_fact(self, entity_name: str, fact_text: str) -> bool:
        """Remove a fact from a node. Returns True if found."""
        ...

    # ── Write — edges ──

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        weight: float = 1.0,
        edge_type: str = "structural",
    ) -> bool:
        """Add an edge between two nodes. Returns True if added."""
        ...

    def remove_edge(self, src_name: str, tgt_name: str, relation: str) -> bool:
        """Remove an edge. Returns True if found and removed."""
        ...

    def remove_edges_by_file(self, file_path: str) -> int:
        """Remove all edges involving a file's nodes. Returns count."""
        ...

    # ── Write — file operations ──

    def link_file(self, node_id: str, file_path: str) -> bool:
        """Link a file to a node. Returns True if added."""
        ...

    def unlink_file(self, node_id: str, file_path: str) -> bool:
        """Remove a file link from a node. Returns True if found."""
        ...

    def mark_file_stale(self, file_path: str) -> bool:
        """Mark a file as stale for re-indexing. Returns True if found."""
        ...

    def upsert_file_structure(
        self,
        file_path: str,
        language: str,
        units: list,
        content_hash: str,
    ) -> int:
        """Create/update graph nodes for a parsed file structure.

        Returns number of symbol/section nodes created.
        """
        ...

    def upsert_folder_node(self, folder_path: str) -> str:
        """Create a folder node if it doesn't exist. Returns node ID."""
        ...

    # ── Write — chunks ──

    def link_chunks(self, node_id: str, chunk_ids: list[str]) -> bool:
        """Set chunk_ids on a node (replaces existing)."""
        ...

    def clear_chunks(self, node_id: str) -> bool:
        """Clear chunk_ids on a node."""
        ...

    # ── Persistence & lifecycle ──

    def save(self) -> None:
        """Persist graph state to disk."""
        ...

    def reload(self) -> None:
        """Reload graph from disk."""
        ...

    def reset(self) -> None:
        """Clear all graph data (nodes, edges, files)."""
        ...

    def repair(self) -> dict:
        """Detect and fix graph corruption. Returns report dict."""
        ...

    # ── Import / Export ──

    def export_json(self) -> dict:
        """Export the full graph as a JSON-serializable dict."""
        ...

    def import_json(self, data: dict, mode: str = "merge") -> tuple[int, int]:
        """Import nodes and edges. Returns (nodes, edges) counts."""
        ...

    # ── Properties ──

    @property
    def node_count(self) -> int: ...

    @property
    def edge_count(self) -> int: ...

    @property
    def session_id(self) -> str: ...

    @property
    def dedup_log(self) -> list[tuple[str, str, str]]: ...

    # ── Traversal (optional override for Cypher backends) ──

    def traverse_bfs(
        self, seed_ids: list[str], max_depth: int = 2,
    ) -> dict[int, list[dict[str, Any]]]:
        """BFS traversal from seed nodes, returns nodes grouped by depth.

        Default implementation uses get_edges_for + get_node (Python BFS).
        LadybugDB overrides with Cypher variable-length path queries.

        Returns: {0: [hot_nodes], 1: [warm_nodes], 2: [cold_nodes]}
        """
        ...
