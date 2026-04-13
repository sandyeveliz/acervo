"""S2 Activator — BFS-based semantic layer retrieval.

Finds seed nodes from the user message, then does breadth-first
traversal of the knowledge graph. Nodes at depth 0 are HOT (direct
match), depth 1 are WARM (neighbors), depth 2 are COLD (2 hops away).

Phase 4 adds an optional hybrid enrichment: after the BFS layering we
run ``acervo.search.hybrid.hybrid_search`` which fuses BFS + vector +
fulltext via Reciprocal Rank Fusion and exposes the result as
``vector_hits`` on the returned ``S2Result``. This replaces the old
orphan vector-search path while keeping the HOT/WARM/COLD layer
semantics intact for S3.

ONE code path. No conversation/project divergence.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from acervo.domain.models import LayeredContext, S1Result, S2Result

log = logging.getLogger(__name__)


class S2Activator:
    """BFS-based graph traversal for context retrieval."""

    def run(
        self,
        user_text: str,
        s1_result: S1Result,
        graph: Any,  # GraphStorePort
        *,
        intent: str = "specific",
        vector_store: Any | None = None,
        user_embedding: list[float] | None = None,
    ) -> S2Result:
        """Execute S2: find seeds → BFS traverse → return layered context."""

        # ── Step 1: Find seed nodes ──
        seeds = self._find_seeds(user_text, s1_result, graph, intent)

        # ── Step 2: BFS traversal with depth layers ──
        # Use graph.traverse_bfs() if available (Cypher-backed), else Python BFS
        seed_ids = [s.get("id", "") for s in seeds if s.get("id")]
        if hasattr(graph, "traverse_bfs") and seed_ids:
            depth_map = graph.traverse_bfs(seed_ids, max_depth=2)
            layered = LayeredContext(
                hot=depth_map.get(0, []),
                warm=depth_map.get(1, []),
                cold=depth_map.get(2, []),
                seeds_used=[s.get("label", "") for s in seeds],
            )
        else:
            layered = self._traverse(graph, seeds, max_depth=2)

        # ── Step 3: Phase 4 hybrid enrichment — BFS + vector + BM25 fused with RRF.
        # Replaces the legacy orphan _vector_search path. When the graph
        # supports entity_similarity_search / fact_fulltext_search, these
        # contribute extra candidates that are fused with the BFS output.
        vector_hits: list[dict] = []
        if intent != "chat":
            vector_hits = self._hybrid_enrich(
                graph=graph,
                query=user_text,
                seed_ids=seed_ids,
                user_embedding=user_embedding,
                existing_ids={n.get("id") for n in layered.hot + layered.warm + layered.cold},
            )

        # Collect all active node IDs for telemetry
        active_ids: set[str] = set()
        for node in layered.hot + layered.warm + layered.cold:
            active_ids.add(node.get("id", ""))

        log.info(
            "[acervo] S2 — seeds=%d, hot=%d, warm=%d, cold=%d, hybrid_extra=%d",
            len(seeds), len(layered.hot), len(layered.warm),
            len(layered.cold), len(vector_hits),
        )

        return S2Result(
            layered=layered,
            active_node_ids=active_ids,
            vector_hits=vector_hits,
        )

    # ── Phase 4: hybrid enrichment ──

    def _hybrid_enrich(
        self,
        *,
        graph: Any,
        query: str,
        seed_ids: list[str],
        user_embedding: list[float] | None,
        existing_ids: set[str],
    ) -> list[dict]:
        """Run hybrid_search and return nodes not already in the BFS layers.

        Returns an empty list when hybrid_search doesn't add anything or
        when the graph doesn't support any of the required methods. Never
        raises — failures degrade to BFS-only silently.
        """
        try:
            from acervo.search.hybrid import hybrid_search
        except Exception as exc:  # pragma: no cover — import guard
            log.warning("S2 hybrid: import failed: %s", exc)
            return []

        try:
            result = hybrid_search(
                graph=graph,
                query=query,
                seed_ids=seed_ids,
                query_embedding=user_embedding,
                limit=10,
            )
        except Exception as exc:
            log.warning("S2 hybrid: hybrid_search failed: %s", exc)
            return []

        extra: list[dict] = []
        for node in result.fused_nodes:
            nid = node.get("id") or node.get("uuid") or ""
            if nid and nid not in existing_ids:
                extra.append(node)
        return extra

    # ── Seed selection ──

    def _find_seeds(
        self, user_text: str, s1_result: S1Result, graph: Any, intent: str,
    ) -> list[dict]:
        """Find entry points for graph traversal."""
        seeds: list[dict] = []
        seen_ids: set[str] = set()

        def _add(node: dict) -> None:
            nid = node.get("id", "")
            if nid and nid not in seen_ids:
                seen_ids.add(nid)
                seeds.append(node)

        # 1. Entities from S1 extraction → look up in graph
        for entity in s1_result.extraction.entities:
            node = graph.get_node(_make_id(entity.name))
            if node:
                _add(node)

        # 2. Keyword match: words in user_text ≥ 4 chars against node labels
        msg_lower = user_text.lower()
        msg_words = set(msg_lower.split())
        for node in graph.get_all_nodes():
            label = node.get("label", "").lower()
            if not label or len(label) < 3:
                continue
            # Direct substring match
            if label in msg_lower:
                _add(node)
                continue
            # Word prefix match (e.g. "supabase" matches "supa...")
            for word in msg_words:
                if len(word) >= 4 and label.startswith(word):
                    _add(node)
                    break

        # 3. Fallback: if overview intent and no seeds, use all entity nodes
        if not seeds and intent == "overview":
            for node in graph.get_all_nodes():
                if node.get("kind", "entity") == "entity":
                    _add(node)

        # 4. Chat intent: only synthesis nodes (minimal context)
        if intent == "chat":
            synthesis_seeds = []
            for node in graph.get_all_nodes():
                if node.get("kind") == "synthesis":
                    summary = node.get("attributes", {}).get("summary", "").lower()
                    if summary and any(w in summary for w in msg_words if len(w) >= 4):
                        synthesis_seeds.append(node)
            if synthesis_seeds:
                return synthesis_seeds

        return seeds

    # ── BFS traversal ──

    def _traverse(
        self, graph: Any, seeds: list[dict], max_depth: int = 2,
    ) -> LayeredContext:
        """BFS from seed nodes, assigning layers by distance."""
        visited: set[str] = set()
        layers: dict[int, list[dict]] = {0: [], 1: [], 2: []}
        queue: deque[tuple[dict, int]] = deque()

        # Enqueue all seeds at depth 0
        for seed in seeds:
            nid = seed.get("id", "")
            if nid not in visited:
                queue.append((seed, 0))
                visited.add(nid)

        while queue:
            node, depth = queue.popleft()

            if depth > max_depth:
                continue

            if depth in layers:
                layers[depth].append(node)

            # Get ALL neighbors via edges (both directions)
            nid = node.get("id", "")
            for edge in graph.get_edges_for(nid):
                neighbor_id = edge["target"] if edge["source"] == nid else edge["source"]
                if neighbor_id not in visited:
                    neighbor = graph.get_node(neighbor_id)
                    if neighbor:
                        visited.add(neighbor_id)
                        queue.append((neighbor, depth + 1))

        return LayeredContext(
            hot=layers[0],
            warm=layers[1],
            cold=layers[2],
            seeds_used=[s.get("label", "") for s in seeds],
        )

    # (Phase 4 replaced the old _vector_search helper with _hybrid_enrich
    # which fuses BFS + vector + fulltext through RRF. The vector_store
    # parameter on run() is kept for backwards compat but is now a no-op —
    # hybrid retrieval reads directly from the graph store.)


from acervo.graph.ids import _make_id  # noqa: E402 — shared ID generation
