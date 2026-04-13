"""Hybrid retrieval over BFS + vector + fulltext, fused with RRF.

This module is the Phase 4 entry point that S2Activator calls in place of
its legacy BFS-only traversal. It runs three complementary retrieval
methods against the same query, collects their top-K identifier lists, and
fuses them with Reciprocal Rank Fusion into a single ordered stream.

The methods — all LLM-free — are:

    1. **BFS** (``graph.traverse_bfs``): graph-native expansion from seed
       nodes. This is what S2 already does.
    2. **Vector / cosine** (``graph.entity_similarity_search``): semantic
       neighbour lookup using the user's query embedding.
    3. **Fulltext / BM25** (``graph.fact_fulltext_search``): token-level
       match against persisted fact text.

Each method returns its own ranked list of node IDs. The caller decides
how many of each to request; we default to 2× the final limit as the
Graphiti recipes do. The fused list preserves Acervo's HOT/WARM/COLD
layering semantics by reprojecting back into depth layers after fusion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from acervo.search.fusion import rrf

log = logging.getLogger(__name__)


DEFAULT_LIMIT = 20
DEFAULT_BFS_DEPTH = 2


@dataclass
class HybridSearchResult:
    """Container for the fused hybrid search output."""

    # Fused list of node dicts in RRF score order.
    fused_nodes: list[dict] = field(default_factory=list)
    # Parallel list of RRF scores (same length as fused_nodes).
    fused_scores: list[float] = field(default_factory=list)
    # Dict of method → list of node IDs it contributed (for diagnostics).
    per_method_ids: dict[str, list[str]] = field(default_factory=dict)


def _collect_bfs(
    graph: Any,
    seed_ids: list[str],
    *,
    max_depth: int,
) -> tuple[list[dict], list[str]]:
    """Run graph.traverse_bfs and flatten to (ordered_nodes, ordered_ids)."""
    if not seed_ids or not hasattr(graph, "traverse_bfs"):
        return [], []
    try:
        depth_map = graph.traverse_bfs(seed_ids, max_depth=max_depth)
    except Exception as exc:
        log.warning("hybrid_search: BFS failed: %s", exc)
        return [], []

    ordered: list[dict] = []
    for depth in sorted(depth_map.keys()):
        ordered.extend(depth_map.get(depth, []))
    ids = [n.get("id") or n.get("uuid") or "" for n in ordered]
    return ordered, ids


def _collect_similarity(
    graph: Any,
    query_embedding: list[float] | None,
    *,
    limit: int,
    min_score: float,
) -> tuple[list[dict], list[str]]:
    """Run graph.entity_similarity_search and return nodes + ids."""
    if not query_embedding or not hasattr(graph, "entity_similarity_search"):
        return [], []
    try:
        hits = graph.entity_similarity_search(
            query_embedding, limit=limit, min_score=min_score,
        )
    except Exception as exc:
        log.warning("hybrid_search: entity_similarity_search failed: %s", exc)
        return [], []
    nodes = [h[0] if isinstance(h, tuple) else h for h in hits]
    ids = [n.get("id") or n.get("uuid") or "" for n in nodes]
    return nodes, ids


def _collect_fulltext(
    graph: Any,
    query: str,
    *,
    limit: int,
) -> tuple[list[dict], list[str]]:
    """Run graph.fact_fulltext_search and resolve fact hits back to their parent nodes.

    ``fact_fulltext_search`` returns fact dicts with (on LadybugGraphStore)
    no back-pointer to the owning node. We ignore the ids there and only
    use the query-text match as a signal — the fused ranking happens on
    ENTITY nodes, not facts. For now we return an empty list when we can't
    map a fact back to its entity. Phase 4.1 can refine this.
    """
    if not query or not hasattr(graph, "fact_fulltext_search"):
        return [], []
    try:
        hits = graph.fact_fulltext_search(query, limit=limit)
    except Exception as exc:
        log.warning("hybrid_search: fact_fulltext_search failed: %s", exc)
        return [], []

    # TopicGraph's fallback attaches ``_node_id`` for each matching fact so
    # we can resolve it back to an entity. LadybugGraphStore's impl doesn't
    # (yet) — those hits are dropped.
    resolved: list[dict] = []
    seen: set[str] = set()
    for hit in hits:
        nid = hit.get("_node_id")
        if not nid or nid in seen:
            continue
        try:
            node = graph.get_node(nid)
        except Exception:
            node = None
        if node is None:
            continue
        seen.add(nid)
        resolved.append(node)
    ids = [n.get("id") or n.get("uuid") or "" for n in resolved]
    return resolved, ids


def hybrid_search(
    *,
    graph: Any,
    query: str,
    seed_ids: list[str] | None = None,
    query_embedding: list[float] | None = None,
    limit: int = DEFAULT_LIMIT,
    bfs_max_depth: int = DEFAULT_BFS_DEPTH,
    methods: set[str] | None = None,
    similarity_min_score: float = 0.6,
) -> HybridSearchResult:
    """Run BFS + vector + fulltext in sequence and fuse the results with RRF.

    Parameters
    ----------
    graph:
        Any GraphStorePort-compatible store.
    query:
        Raw query text. Used for fulltext search; also used implicitly by
        seed-based BFS.
    seed_ids:
        Entity ids to seed BFS from. When None, BFS is skipped.
    query_embedding:
        Query embedding for vector similarity. When None, vector search
        is skipped.
    limit:
        Final desired result count. Each individual method fetches up to
        ``2 * limit`` so the fusion has enough signal to compose.
    bfs_max_depth:
        Max BFS depth when BFS is enabled.
    methods:
        Optional set of method names to enable: ``{"bfs", "vector", "bm25"}``.
        Defaults to all three.
    similarity_min_score:
        Cosine floor for vector search.

    Returns
    -------
    HybridSearchResult
        ``fused_nodes`` / ``fused_scores`` in RRF order. Duplicates across
        methods are merged (higher combined score).
    """
    active = methods if methods is not None else {"bfs", "vector", "bm25"}

    per_method_lists: list[list[str]] = []
    per_method_ids: dict[str, list[str]] = {}
    id_to_node: dict[str, dict] = {}

    if "bfs" in active and seed_ids:
        nodes, ids = _collect_bfs(graph, seed_ids, max_depth=bfs_max_depth)
        for node, nid in zip(nodes, ids, strict=True):
            if nid:
                id_to_node.setdefault(nid, node)
        if ids:
            per_method_lists.append(ids[: 2 * limit])
            per_method_ids["bfs"] = ids[: 2 * limit]

    if "vector" in active and query_embedding:
        nodes, ids = _collect_similarity(
            graph, query_embedding,
            limit=2 * limit, min_score=similarity_min_score,
        )
        for node, nid in zip(nodes, ids, strict=True):
            if nid:
                id_to_node.setdefault(nid, node)
        if ids:
            per_method_lists.append(ids)
            per_method_ids["vector"] = ids

    if "bm25" in active and query:
        nodes, ids = _collect_fulltext(graph, query, limit=2 * limit)
        for node, nid in zip(nodes, ids, strict=True):
            if nid:
                id_to_node.setdefault(nid, node)
        if ids:
            per_method_lists.append(ids)
            per_method_ids["bm25"] = ids

    if not per_method_lists:
        return HybridSearchResult(per_method_ids=per_method_ids)

    fused_ids, fused_scores = rrf(per_method_lists)

    fused_nodes: list[dict] = []
    fused_trim_scores: list[float] = []
    for nid, score in zip(fused_ids, fused_scores, strict=True):
        node = id_to_node.get(nid)
        if node is None:
            continue
        fused_nodes.append(node)
        fused_trim_scores.append(score)
        if len(fused_nodes) >= limit:
            break

    return HybridSearchResult(
        fused_nodes=fused_nodes,
        fused_scores=fused_trim_scores,
        per_method_ids=per_method_ids,
    )


__all__ = ["HybridSearchResult", "hybrid_search", "DEFAULT_LIMIT", "DEFAULT_BFS_DEPTH"]
