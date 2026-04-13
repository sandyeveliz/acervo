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
        status: str | None = None,
        updated_by: str | None = None,
    ) -> tuple[int, int]:
        """Create or update entities, relations, and facts.

        ``source`` is the provenance tag applied to both node and fact
        rows — v0.6.1 uses ``"llm"``, ``"user"`` or ``"system"`` at call
        sites, but legacy values like ``"conversation"`` or
        ``"user_assertion"`` are still accepted.

        ``status`` overrides the derived default (``complete`` /
        ``incomplete``) and is used by the confidence pipeline to mark
        newly-extracted low-confidence entities as ``pending_review``.

        ``updated_by`` populates the audit column of the same name on
        both nodes and the associated facts/edges created during this
        upsert.

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

    def merge_nodes(
        self,
        keep_id: str,
        absorb_id: str,
        alias: str | None = None,
        *,
        updated_by: str | None = None,
    ) -> bool:
        """Merge two nodes: keep one, absorb facts/edges from the other.

        ``updated_by`` is stamped on the surviving node and the freshly
        added alias/merge facts so the audit trail reflects who caused
        the merge.
        """
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
        *,
        source: str | None = None,
        updated_by: str | None = None,
    ) -> bool:
        """Add an edge between two nodes. Returns True if added.

        ``source`` and ``updated_by`` are the v0.6.1 audit fields
        (``"llm"`` / ``"user"`` / ``"system"``).
        """
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

    # ── Validation log ──

    def persist_validation_log(self, entries: list) -> int:
        """Persist validation log entries. Returns count written.

        Default no-op for backends that don't support it (e.g. JSON).
        LadybugDB stores entries as ValidationLog nodes.
        """
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

    # ── Phase 2: Semantic search + bi-temporal fact mutations ──

    def entity_similarity_search(
        self,
        query_embedding: list[float],
        *,
        limit: int = 15,
        min_score: float = 0.6,
    ) -> list[tuple[dict[str, Any], float]]:
        """Return entities most similar to ``query_embedding`` by cosine sim.

        Used by the Phase 2 entity resolution pipeline to pre-filter dedup
        candidates before running MinHash LSH. Backends that don't have a
        native vector index should fall back to a Python brute-force scan
        over persisted ``name_embedding`` columns.

        Returns list of ``(node_dict, score)`` sorted by score descending.
        Candidates with score below ``min_score`` are dropped.
        """
        ...

    def fact_fulltext_search(
        self,
        query: str,
        *,
        limit: int = 15,
    ) -> list[dict[str, Any]]:
        """Return fact nodes whose text matches the query.

        Used for BM25-style retrieval in Phase 4 hybrid search. Backends
        without native FTS should fall back to a Python BM25 pass.
        """
        ...

    def invalidate_fact(
        self,
        fact_id: str,
        *,
        expired_at: str,
        invalid_at: str | None = None,
    ) -> bool:
        """Mark a fact as expired without deleting it (append-only model).

        Sets ``expired_at`` (system/ingestion time) and optionally
        ``invalid_at`` (event time — when the fact stopped being true in
        the world). Returns True if the fact was found and updated.
        """
        ...

    def set_entity_embedding(
        self,
        node_id: str,
        embedding: list[float],
    ) -> bool:
        """Persist an entity name embedding to a node.

        Called from the pipeline after S1 computes entity embeddings in
        batch. Kept separate from ``upsert_entities`` so the legacy
        signature stays unchanged and the persistence path is explicit.
        Returns True when the node was found and updated.
        """
        ...

    def set_fact_embedding(
        self,
        fact_id: str,
        embedding: list[float],
    ) -> bool:
        """Persist a fact embedding on an existing Fact row.

        Used by the v0.6.1 fact dedup pass to lazy-populate embeddings on
        historical facts that were written before embedding was tracked
        per-fact. Returns True when the fact was found and updated.
        """
        ...
