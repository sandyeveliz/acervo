"""S2 Activator — node activation + gathering for context assembly.

ONE code path. No behavioral differences between conversation mode
and project mode. If the graph has nodes, activate them. If a vector
store exists, search it. No guards, no conditional skips.
"""

from __future__ import annotations

import logging
from pathlib import Path, PurePosixPath
from typing import Any

from acervo.domain.models import RankedChunk, S2Result, GatheredNode
from acervo.token_counter import count_tokens

log = logging.getLogger(__name__)


class S2Activator:
    """Activates graph nodes relevant to the user message and gathers context chunks."""

    def run(
        self,
        user_text: str,
        graph: Any,  # GraphStorePort
        *,
        intent: str = "specific",
        current_topic: str = "none",
        vector_store: Any | None = None,
        user_embedding: list[float] | None = None,
        workspace_path: Path | None = None,
    ) -> S2Result:
        """Execute S2: find relevant nodes, traverse edges, gather context.

        Returns S2Result with activated nodes + ranked chunks.
        """
        # ── Step 1: Activate nodes by label matching ──
        active_ids = self._activate_nodes(user_text, graph, intent)

        # ── Step 2: Add topic node if it exists ──
        if current_topic != "none":
            topic_id = _make_id(current_topic)
            if graph.get_node(topic_id):
                active_ids.add(topic_id)

        # ── Step 3: Expand — folder→files, files→sections, entities→neighbors ──
        active_ids = self._expand(active_ids, graph)

        # ── Step 4: Intent-based cap ──
        active_ids = self._apply_cap(active_ids, graph, intent)

        # ── Step 5: Gather nodes with relations ──
        gathered = self._gather(active_ids, graph, workspace_path)

        # ── Step 6: Vector search (graceful no-op if not available) ──
        vector_hits: list[dict] = []
        if vector_store is not None and user_embedding is not None:
            vector_hits = self._vector_search(
                vector_store, user_embedding, user_text,
                active_ids, current_topic, intent,
            )

        # ── Step 7: Build ranked chunks from gathered data ──
        chunks = self._build_chunks(gathered, vector_hits, user_text)

        log.info(
            "[acervo] S2 — %d active nodes, %d gathered, %d chunks, %d vector hits",
            len(active_ids), len(gathered), len(chunks), len(vector_hits),
        )

        return S2Result(
            activated_nodes=gathered,
            chunks=chunks,
            active_node_ids=active_ids,
            vector_hits=vector_hits,
        )

    # ── Node activation (label matching against all graph nodes) ──

    def _activate_nodes(self, user_text: str, graph: Any, intent: str) -> set[str]:
        """Find node IDs relevant to the user message via label matching."""
        active: set[str] = set()
        msg_lower = user_text.lower()
        msg_words = set(msg_lower.split())

        for node in graph.get_all_nodes():
            kind = node.get("kind", "entity")
            label = node.get("label", "").lower()
            nid = node.get("id", "")
            if not label or len(label) < 3:
                continue

            # Overview intent: activate file/folder/synthesis + ALL entities
            if intent == "overview":
                if kind in ("file", "folder", "synthesis", "entity"):
                    active.add(nid)
                continue

            # Chat intent: only synthesis nodes on keyword match
            if intent == "chat":
                if kind == "synthesis":
                    summary = node.get("attributes", {}).get("summary", "").lower()
                    if summary and any(w in summary for w in msg_words if len(w) >= 4):
                        active.add(nid)
                continue

            # Synthesis nodes: keyword match against summary
            if kind == "synthesis":
                summary = node.get("attributes", {}).get("summary", "").lower()
                if summary and any(w in summary for w in msg_words if len(w) >= 4):
                    active.add(nid)
                continue

            # File nodes: match stem or path
            if kind == "file":
                stem = PurePosixPath(label).stem.lower() if "." in label else label
                path_str = node.get("attributes", {}).get("path", "").lower()
                if (len(stem) >= 3 and stem in msg_lower) or (path_str and path_str in msg_lower):
                    active.add(nid)
                continue

            # Folder nodes: match name or path
            if kind == "folder":
                path_str = node.get("attributes", {}).get("path", "").lower()
                if _text_matches(label, msg_lower, msg_words) or (path_str and path_str in msg_lower):
                    active.add(nid)
                continue

            # Entity/section/symbol: text matching on label
            if _text_matches(label, msg_lower, msg_words):
                active.add(nid)
                continue

            # Also match against summary and topics
            summary = node.get("attributes", {}).get("summary", "").lower()
            topics_raw = node.get("attributes", {}).get("topics", [])
            topics_str = " ".join(t.lower() for t in topics_raw) if isinstance(topics_raw, list) else str(topics_raw).lower()
            searchable = f"{summary} {topics_str}"
            if searchable.strip():
                for word in msg_words:
                    if len(word) >= 4 and word in searchable:
                        active.add(nid)
                        break

        return active

    # ── Expansion (folder→file, file→section, entity→neighbor) ──

    def _expand(self, active_ids: set[str], graph: Any) -> set[str]:
        """Expand active set via edge traversal. ONE pass, all node kinds."""
        active = set(active_ids)

        # Folder → contained files
        for nid in list(active):
            node = graph.get_node(nid)
            if not node or node.get("kind") != "folder":
                continue
            for edge in graph.get_edges_for(nid):
                if edge.get("relation") == "contains":
                    child_id = edge["target"] if edge["source"] == nid else edge["source"]
                    active.add(child_id)

        # File → contained sections/symbols
        for nid in list(active):
            node = graph.get_node(nid)
            if not node or node.get("kind") != "file":
                continue
            for edge in graph.get_edges_for(nid):
                if edge.get("relation") == "contains":
                    child_id = edge["target"] if edge["source"] == nid else edge["source"]
                    active.add(child_id)

        # Entity → direct entity neighbors via ANY edge
        for nid in list(active):
            node = graph.get_node(nid)
            if not node or node.get("kind") != "entity":
                continue
            for edge in graph.get_edges_for(nid):
                neighbor_id = edge["target"] if edge["source"] == nid else edge["source"]
                if neighbor_id not in active:
                    neighbor = graph.get_node(neighbor_id)
                    if neighbor and neighbor.get("kind") == "entity":
                        active.add(neighbor_id)

        return active

    # ── Intent-based cap ──

    def _apply_cap(self, active_ids: set[str], graph: Any, intent: str) -> set[str]:
        """Limit active set size based on intent."""
        if intent == "chat":
            return {nid for nid in active_ids
                    if (n := graph.get_node(nid)) and n.get("kind") == "synthesis"}

        if intent in ("specific", "followup") and len(active_ids) > 15:
            structural = {nid for nid in active_ids
                          if (n := graph.get_node(nid))
                          and n.get("kind") in ("entity", "synthesis", "file", "folder")}
            if len(structural) <= 15:
                return structural
            return {nid for nid in active_ids
                    if (n := graph.get_node(nid))
                    and n.get("kind") in ("entity", "synthesis")}

        return active_ids

    # ── Gather nodes with relations ──

    def _gather(
        self, active_ids: set[str], graph: Any, workspace_path: Path | None = None,
    ) -> list[GatheredNode]:
        """Fetch active nodes, attach relations, expand neighbors."""
        gathered: list[GatheredNode] = []
        seen_ids: set[str] = set()

        # Active nodes
        for node in graph.get_nodes_by_ids(active_ids):
            nid = node.get("id", "")
            if nid in seen_ids:
                continue
            seen_ids.add(nid)
            edges = graph.get_edges_for(nid)
            relations = []
            for e in edges[:8]:  # up to 8 relations per node
                other_id = e["target"] if e["source"] == nid else e["source"]
                other = graph.get_node(other_id)
                if other:
                    rel = e.get("relation", "related_to")
                    relations.append(f"{rel}: {other.get('label', other_id)}")
            gathered.append(GatheredNode(node=dict(node), relations=relations, hot=True))

        # Neighbor expansion (1-level, for nodes with facts)
        for gn in list(gathered):
            nid = gn.node.get("id", "")
            neighbors = graph.get_neighbors(nid, max_count=3)
            for nbr, weight in neighbors:
                nbr_id = nbr.get("id", "")
                if nbr_id in seen_ids:
                    continue
                seen_ids.add(nbr_id)
                if nbr.get("facts"):
                    gathered.append(GatheredNode(node=dict(nbr), relations=[], hot=False))

        # Inject summary/content for section/symbol nodes (indexed projects)
        if workspace_path:
            for gn in gathered:
                kind = gn.node.get("kind")
                if kind not in ("section", "symbol"):
                    continue
                if gn.node.get("facts"):
                    continue
                summary = gn.node.get("attributes", {}).get("summary", "")
                if summary:
                    gn.node.setdefault("facts", []).append({"fact": summary, "source": "rag"})
                else:
                    content = graph.get_symbol_content(gn.node["id"], workspace_path)
                    if content:
                        snippet = content[:500].strip()
                        if snippet:
                            gn.node.setdefault("facts", []).append({"fact": snippet, "source": "rag"})

        return gathered

    # ── Vector search ──

    async def _vector_search_async(self, *args, **kwargs) -> list[dict]:
        """Async wrapper — not used directly, kept for future."""
        return self._vector_search(*args, **kwargs)

    def _vector_search(
        self, vector_store: Any, user_embedding: list[float], user_text: str,
        active_ids: set[str], current_topic: str, intent: str,
    ) -> list[dict]:
        """Perform vector search. Returns hits. Skips gracefully on error."""
        # Skip vector for overview or summary-only
        if intent == "overview":
            return []
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context — can't await, skip
                return []
            hits = loop.run_until_complete(
                vector_store.search_with_embedding(user_embedding, n_results=5)
            )
            # Boost on-topic results
            topic_id = _make_id(current_topic) if current_topic != "none" else None
            for hit in hits:
                node_id = hit.get("node_id", "")
                if topic_id and node_id in active_ids:
                    hit["score"] = min(hit.get("score", 0.5) + 0.2, 1.0)
            return hits
        except Exception as e:
            log.warning("S2 vector search failed: %s", e)
            return []

    # ── Build ranked chunks from gathered nodes ──

    def _build_chunks(
        self, gathered: list[GatheredNode], vector_hits: list[dict], user_text: str,
    ) -> list[RankedChunk]:
        """Convert gathered nodes + vector hits into scored chunks for S3."""
        chunks: list[RankedChunk] = []
        user_lower = user_text.lower()

        for gn in gathered:
            node = gn.node
            label = node.get("label", "")
            kind = node.get("kind", "entity")
            has_facts = bool(node.get("facts"))
            has_relations = bool(gn.relations)
            summary = node.get("attributes", {}).get("summary", "")
            description = node.get("attributes", {}).get("description", "")

            # Score: hot nodes rank higher
            base_score = 1.0 if gn.hot else 0.7
            if label.lower() in user_lower:
                base_score = min(base_score + 0.3, 1.0)

            is_verified = (
                node.get("source") in ("world", "web")
                or node.get("attributes", {}).get("verified", False)
                or kind in ("file", "section", "symbol", "folder")
            )

            # Description chunk (NEW — was previously ignored)
            if description and not has_facts:
                text = f"**{label}**: {description}"
                src = "verified_description" if is_verified else "conversation_description"
                chunks.append(RankedChunk(
                    text=text, score=base_score, source=src,
                    label=label, tokens=count_tokens(text),
                ))

            # Summary chunk (for indexed nodes)
            if kind in ("file", "section", "symbol", "folder") and summary and not has_facts:
                text = f"**{label}**: {summary}"
                chunks.append(RankedChunk(
                    text=text, score=base_score, source="verified_summary",
                    label=label, tokens=count_tokens(text),
                ))

            # Fact chunks
            for fact in node.get("facts", []):
                fact_text = fact.get("fact", "") if isinstance(fact, dict) else str(fact)
                text = f"**{label}**: {fact_text}"
                src = "verified_fact" if is_verified else "conversation_fact"
                chunks.append(RankedChunk(
                    text=text, score=base_score, source=src,
                    label=label, tokens=count_tokens(text),
                ))

            # Relation chunks
            for rel in gn.relations:
                text = f"**{label}** → {rel}"
                src = "verified_relation" if is_verified else "conversation_relation"
                chunks.append(RankedChunk(
                    text=text, score=base_score * 0.9, source=src,
                    label=label, tokens=count_tokens(text),
                ))

        # Vector search hits
        for hit in vector_hits:
            text = hit.get("text", "")
            if text:
                chunks.append(RankedChunk(
                    text=text, score=hit.get("score", 0.5), source=hit.get("source", "vector"),
                    label=hit.get("node_id", hit.get("chunk_id", "")),
                    tokens=count_tokens(text),
                ))

        return chunks


# ── Helpers ──

def _make_id(name: str) -> str:
    """Stable node ID from name (must match graph.py:_make_id)."""
    return name.lower().strip().replace(" ", "_")


def _text_matches(label: str, msg_lower: str, msg_words: set[str]) -> bool:
    """Check if a node label is mentioned in the user message."""
    if label in msg_lower:
        return True
    for word in msg_words:
        if len(word) >= 4 and label.startswith(word):
            return True
    label_words = set(label.split())
    if len(label_words) > 1 and label_words.issubset(msg_words):
        return True
    return False
