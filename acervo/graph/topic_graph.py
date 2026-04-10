"""Topic graph — persists entity nodes and edges to JSON.

Each node has:
  layer (Layer): UNIVERSAL or PERSONAL
  source (str): "world" or "user_assertion"
  confidence_for_owner (float): 0.0-1.0
  pending_fields (list[str]): missing fields for incomplete nodes

Runtime activation (hot/warm/cold) is NOT stored in nodes — it belongs to
the conversation context (ephemeral, per-session). The graph only stores
persistent knowledge.

Existing nodes without layer default to PERSONAL with source="user_assertion".
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from acervo.graph.ids import _make_id, _normalize_for_dedup, make_symbol_id  # noqa: F401
from acervo.layers import Layer, NodeMeta
from acervo.ontology import is_known_type

log = logging.getLogger(__name__)

_DEFAULT_PATH = Path("data/graph")


# Relations that are too generic — should be replaced by more specific ones
_GENERIC_RELATIONS = frozenset({
    "related_to", "uses", "uses_technology", "belongs_to", "associated_with",
})


def _default_node_meta(etype: str, owner: str | None = None) -> NodeMeta:
    """Generate layer/ontology metadata for a new node."""
    if is_known_type(etype):
        return NodeMeta.personal(owner=owner)
    return NodeMeta.incomplete(owner=owner, pending=["type"])


# Legacy type → canonical type (extractor model uses lowercase, ontology capitalizes)
_TYPE_MIGRATION: dict[str, str] = {
    "Framework": "Technology",
    "Library": "Technology",
    "Platform": "Technology",
    "Tool": "Technology",
    "Backend_service": "Technology",
    "Design_system": "Technology",
    "Database": "Technology",
    "Language": "Technology",
    "Runtime": "Technology",
    "Api": "Technology",
}


def _migrate_node(node: dict) -> dict:
    """Add layer fields to legacy nodes that don't have them."""
    if "layer" not in node:
        meta = NodeMeta.personal()
        node.update(meta.to_dict())
    if "owner" not in node:
        node["owner"] = None
    if "files" not in node:
        node["files"] = []
    if "kind" not in node:
        node["kind"] = "entity"
    # Staleness fields for file nodes
    if node.get("kind") == "file":
        if "stale" not in node:
            node["stale"] = False
        if "stale_since" not in node:
            node["stale_since"] = None
        if "indexed_at" not in node:
            node["indexed_at"] = node.get("created_at")
    # chunk_ids for document chunk linkage
    if "chunk_ids" not in node:
        node["chunk_ids"] = []
    # Normalize legacy entity types to canonical ontology types
    old_type = node.get("type", "")
    if old_type in _TYPE_MIGRATION:
        node["type"] = _TYPE_MIGRATION[old_type]
    return node


class TopicGraph:
    """In-memory graph with JSON persistence."""

    def __init__(self, path: Path = _DEFAULT_PATH) -> None:
        self._path = path
        self._nodes: dict[str, dict] = {}
        self._edges: list[dict] = []
        self._session_id = f"s_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._load()

    def _load(self) -> None:
        self._path.mkdir(parents=True, exist_ok=True)
        nodes_file = self._path / "nodes.json"
        edges_file = self._path / "edges.json"
        if nodes_file.exists():
            try:
                raw = json.loads(nodes_file.read_text(encoding="utf-8"))
                self._nodes = {n["id"]: _migrate_node(n) for n in raw}
            except (json.JSONDecodeError, KeyError):
                self._nodes = {}
        if edges_file.exists():
            try:
                self._edges = json.loads(edges_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._edges = []

        # Clean up legacy runtime fields that no longer belong in nodes
        for node in self._nodes.values():
            node.pop("_topic_id", None)
            # Remove legacy runtime status if present
            if node.get("status") in ("hot", "warm", "cold"):
                del node["status"]

    def reload(self) -> None:
        """Reload graph from disk (used after external data clear)."""
        self._load()

    def reset(self) -> None:
        """Clear all in-memory data and remove files on disk."""
        self._nodes.clear()
        self._edges.clear()
        nodes_file = self._path / "nodes.json"
        edges_file = self._path / "edges.json"
        if nodes_file.exists():
            nodes_file.unlink()
        if edges_file.exists():
            edges_file.unlink()

    def repair(self) -> dict:
        """Detect and fix graph corruption. Returns a report of fixes applied.

        Checks:
        - Nodes missing required fields (id, label, type, kind)
        - Edges referencing non-existent nodes
        - Duplicate edges
        - Nodes with invalid chunk_ids (non-list)
        """
        report: dict = {"removed_nodes": 0, "removed_edges": 0, "fixed_fields": 0, "deduped_edges": 0}

        # Fix nodes with missing required fields
        bad_node_ids: list[str] = []
        for nid, node in self._nodes.items():
            if not isinstance(node, dict) or not node.get("id") or not node.get("label"):
                bad_node_ids.append(nid)
                continue
            # Ensure required fields exist
            if "type" not in node:
                node["type"] = "unknown"
                report["fixed_fields"] += 1
            if "kind" not in node:
                node["kind"] = "entity"
                report["fixed_fields"] += 1
            if "facts" not in node:
                node["facts"] = []
                report["fixed_fields"] += 1
            if "chunk_ids" not in node or not isinstance(node.get("chunk_ids"), list):
                node["chunk_ids"] = []
                report["fixed_fields"] += 1

        for nid in bad_node_ids:
            del self._nodes[nid]
            report["removed_nodes"] += 1

        # Remove edges referencing non-existent nodes
        valid_ids = set(self._nodes.keys())
        original_edge_count = len(self._edges)
        self._edges = [
            e for e in self._edges
            if isinstance(e, dict) and e.get("source") in valid_ids and e.get("target") in valid_ids
        ]
        report["removed_edges"] = original_edge_count - len(self._edges)

        # Deduplicate edges (same source + target + relation)
        seen: set[tuple[str, str, str]] = set()
        deduped: list[dict] = []
        for e in self._edges:
            key = (e.get("source", ""), e.get("target", ""), e.get("relation", ""))
            if key not in seen:
                seen.add(key)
                deduped.append(e)
            else:
                report["deduped_edges"] += 1
        self._edges = deduped

        if any(v > 0 for v in report.values()):
            self._save()

        return report

    def _save(self) -> None:
        self._path.mkdir(parents=True, exist_ok=True)
        (self._path / "nodes.json").write_text(
            json.dumps(list(self._nodes.values()), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self._path / "edges.json").write_text(
            json.dumps(self._edges, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def persist_validation_log(self, entries: list) -> int:
        """No-op: JSON backend doesn't store validation logs."""
        return 0

    def upsert_entities(
        self,
        entities: list[tuple[str, str]],
        relations: list[tuple[str, str, str]] | None = None,
        facts: list[tuple[str, str, str]] | None = None,
        layer: Layer = Layer.PERSONAL,
        source: str = "user_assertion",
        confidence: float = 1.0,
        owner: str | None = None,
    ) -> tuple[int, int]:
        """Upsert entities, add relations and source-tagged facts.

        Args:
            entities: list of (name, type) pairs
            relations: list of (source_name, target_name, relation) tuples
            facts: list of (entity_name, fact_text, source) tuples
            layer: Layer enum — UNIVERSAL or PERSONAL (default PERSONAL)
            source: "world" or "user_assertion" (default "user_assertion")
            confidence: confidence for the owner (0.0-1.0, default 1.0)
            owner: owner identifier (None for universal knowledge)
        """
        now = datetime.now().isoformat(timespec="seconds")
        node_ids: list[str] = []

        # Upsert nodes
        for name, etype in entities:
            nid = _make_id(name)
            node_ids.append(nid)

            if nid in self._nodes:
                node = self._nodes[nid]
                node["last_active"] = now
                sessions_seen = {f.get("session") for f in node.get("facts", [])}
                if self._session_id not in sessions_seen:
                    node["session_count"] = node.get("session_count", 0) + 1
            else:
                meta = NodeMeta(
                    layer=layer,
                    owner=owner,
                    source=source,
                    confidence_for_owner=confidence,
                    status="complete" if is_known_type(etype) else "incomplete",
                    pending_fields=[] if is_known_type(etype) else ["type"],
                )
                self._nodes[nid] = {
                    "id": nid,
                    "label": name,
                    "type": etype if is_known_type(etype) else "Unknown",
                    "kind": "entity",
                    "created_at": now,
                    "last_active": now,
                    "session_count": 1,
                    "attributes": {},
                    "facts": [],
                    "files": [],
                    "chunk_ids": [],
                    **meta.to_dict(),
                }

        # Add semantic relations (max 1 edge per directed pair)
        if relations:
            for src_name, tgt_name, relation in relations:
                src_id = _make_id(src_name)
                tgt_id = _make_id(tgt_name)
                # Reject self-referential edges
                if src_id == tgt_id:
                    continue
                # Check if any edge between this pair already exists
                existing_edge = self._find_edge_between(src_id, tgt_id)
                if existing_edge:
                    # Replace generic relation with more specific one
                    if existing_edge["relation"] in _GENERIC_RELATIONS and relation not in _GENERIC_RELATIONS:
                        existing_edge["relation"] = relation
                        existing_edge["last_active"] = now
                    # Same relation or both specific — just update timestamp
                    elif existing_edge["relation"] == relation:
                        existing_edge["last_active"] = now
                    # else: keep existing, skip new (don't accumulate edges)
                else:
                    self._edges.append({
                        "source": src_id,
                        "target": tgt_id,
                        "relation": relation,
                        "weight": 1.0,
                        "created_at": now,
                        "last_active": now,
                        "layer": layer.name,
                        "source_type": source,
                    })

        # Add source-tagged facts to nodes (with dedup)
        self._dedup_log: list[tuple[str, str, str]] = []
        if facts:
            for entity_name, fact_text, fact_source in facts:
                nid = _make_id(entity_name)
                node = self._nodes.get(nid)
                if node:
                    dup = self._find_similar_fact(node["facts"], fact_text)
                    if dup:
                        self._dedup_log.append((entity_name, fact_text, f"duplicate of '{dup}'"))
                    else:
                        node["facts"].append({
                            "fact": fact_text,
                            "date": now[:10],
                            "session": self._session_id,
                            "source": fact_source,
                        })

        self._save()
        log.info("graph_update nodes=%d edges=%d", len(self._nodes), len(self._edges))
        return len(self._nodes), len(self._edges)

    @staticmethod
    def _find_similar_fact(existing_facts: list[dict], new_fact: str, threshold: float = 0.65) -> str | None:
        """Check if a similar fact already exists. Returns the existing fact text or None.

        Uses three checks:
        1. Exact normalized match
        2. Substring containment
        3. Word-overlap Jaccard similarity (catches paraphrases like
           "It is a tickets project" vs "Butaco is a ticketing system")
        """
        new_norm = _normalize_for_dedup(new_fact)
        if not new_norm:
            return None
        new_words = set(new_norm.split())
        for f in existing_facts:
            existing_norm = _normalize_for_dedup(f.get("fact", ""))
            if not existing_norm:
                continue
            # Exact normalized match
            if new_norm == existing_norm:
                return f["fact"]
            # Substring containment
            if new_norm in existing_norm or existing_norm in new_norm:
                return f["fact"]
            # Word-overlap Jaccard similarity
            existing_words = set(existing_norm.split())
            intersection = new_words & existing_words
            union = new_words | existing_words
            if union and len(intersection) / len(union) >= threshold:
                return f["fact"]
        return None

    def _find_edge_between(self, src: str, tgt: str) -> dict | None:
        """Find any existing edge between src and tgt (either direction)."""
        pair = {src, tgt}
        for e in self._edges:
            if {e["source"], e["target"]} == pair:
                return e
        return None

    def _edge_exists(self, src: str, tgt: str, relation: str) -> bool:
        pair = {src, tgt}
        return any(
            e.get("relation") == relation and {e["source"], e["target"]} == pair
            for e in self._edges
        )

    def remove_edge(self, src_name: str, tgt_name: str, relation: str) -> bool:
        """Remove an edge between two nodes. Returns True if found and removed."""
        src_id = _make_id(src_name)
        tgt_id = _make_id(tgt_name)
        before = len(self._edges)
        self._edges = [
            e for e in self._edges
            if not (
                e.get("relation") == relation
                and {e["source"], e["target"]} == {src_id, tgt_id}
            )
        ]
        removed = len(self._edges) < before
        if removed:
            log.info("Removed edge: %s -[%s]-> %s", src_name, relation, tgt_name)
        return removed

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its edges. Returns True if found and removed."""
        nid = _make_id(node_id) if node_id not in self._nodes else node_id
        if nid not in self._nodes:
            return False
        del self._nodes[nid]
        self._edges = [
            e for e in self._edges
            if e["source"] != nid and e["target"] != nid
        ]
        log.info("Removed node: %s (and its edges)", nid)
        return True

    def update_node(self, node_id: str, **fields: object) -> bool:
        """Update fields on an existing node. Returns True if node was found."""
        nid = _make_id(node_id) if node_id not in self._nodes else node_id
        node = self._nodes.get(nid)
        if not node:
            return False
        allowed = {"label", "type", "attributes"}
        for key, value in fields.items():
            if key in allowed:
                node[key] = value
        log.info("Updated node %s: %s", nid, list(fields.keys()))
        return True

    def merge_nodes(self, keep_id: str, absorb_id: str, alias: str | None = None) -> bool:
        """Merge two nodes: keep one, absorb the other's facts and edges.

        Args:
            keep_id: Node ID to keep (primary).
            absorb_id: Node ID to absorb (will be deleted).
            alias: Optional alias label (e.g. "Man of Steel") stored as a fact.

        Returns True if both nodes existed and merge succeeded.
        """
        kid = _make_id(keep_id) if keep_id not in self._nodes else keep_id
        aid = _make_id(absorb_id) if absorb_id not in self._nodes else absorb_id

        keep = self._nodes.get(kid)
        absorb = self._nodes.get(aid)
        if not keep or not absorb:
            return False

        # Merge facts (deduplicate by normalized text)
        existing_facts = {_normalize_for_dedup(f.get("fact", "")) for f in keep.get("facts", [])}
        for fact in absorb.get("facts", []):
            norm = _normalize_for_dedup(fact.get("fact", ""))
            if norm and norm not in existing_facts:
                keep["facts"].append(fact)
                existing_facts.add(norm)

        # Add alias as a fact if provided
        if alias:
            alias_fact = f"Also known as: {alias}"
            alias_norm = _normalize_for_dedup(alias_fact)
            if alias_norm not in existing_facts:
                keep["facts"].append({
                    "fact": alias_fact,
                    "date": datetime.now().isoformat(timespec="seconds"),
                    "session": self._session_id,
                    "source": "merge",
                })

        # Merge attributes
        absorb_attrs = absorb.get("attributes", {})
        keep_attrs = keep.setdefault("attributes", {})
        for k, v in absorb_attrs.items():
            if k not in keep_attrs:
                keep_attrs[k] = v

        # Aggregate session count
        keep["session_count"] = keep.get("session_count", 1) + absorb.get("session_count", 1)

        # Re-point edges from absorbed node to keep node
        for edge in self._edges:
            if edge["source"] == aid:
                edge["source"] = kid
            if edge["target"] == aid:
                edge["target"] = kid

        # Remove self-loops created by merge
        self._edges = [
            e for e in self._edges
            if not (e["source"] == kid and e["target"] == kid)
        ]

        # Remove duplicate edges (same source, target, relation)
        seen_edges: set[tuple[str, str, str]] = set()
        deduped: list[dict] = []
        for edge in self._edges:
            key = (edge["source"], edge["target"], edge.get("relation", ""))
            if key not in seen_edges:
                seen_edges.add(key)
                deduped.append(edge)
        self._edges = deduped

        # Delete absorbed node
        del self._nodes[aid]

        log.info("Merged node %s into %s", aid, kid)
        return True

    def remove_fact(self, entity_name: str, fact_text: str) -> bool:
        """Remove a fact from a node. Returns True if found and removed."""
        nid = _make_id(entity_name)
        node = self._nodes.get(nid)
        if not node:
            return False
        before = len(node.get("facts", []))
        node["facts"] = [
            f for f in node.get("facts", [])
            if f.get("fact", "").lower().strip() != fact_text.lower().strip()
        ]
        removed = len(node["facts"]) < before
        if removed:
            log.info("Removed fact from %s: %s", entity_name, fact_text)
        return removed

    # ── Public query API ──

    def get_node(self, name_or_id: str) -> dict | None:
        """Get a node by ID or by name (via _make_id)."""
        node = self._nodes.get(name_or_id)
        if node:
            return node
        return self._nodes.get(_make_id(name_or_id))

    def get_nodes_by_ids(self, node_ids: set[str]) -> list[dict]:
        """Return nodes matching the given IDs."""
        return [self._nodes[nid] for nid in node_ids if nid in self._nodes]

    def get_all_nodes(self) -> list[dict]:
        """Return all nodes as a list of dicts."""
        return list(self._nodes.values())

    def get_neighbors(self, node_id: str, max_count: int = 5) -> list[tuple[dict, float]]:
        """Return neighbor nodes with edge weights, sorted by weight desc."""
        neighbors: dict[str, float] = {}
        for edge in self._edges:
            src, tgt = edge["source"], edge["target"]
            weight = edge.get("weight", 1.0)
            if src == node_id and tgt != node_id:
                neighbors[tgt] = max(neighbors.get(tgt, 0), weight)
            elif tgt == node_id and src != node_id:
                neighbors[src] = max(neighbors.get(src, 0), weight)
        sorted_ids = sorted(neighbors, key=lambda k: neighbors[k], reverse=True)
        result = []
        for nid in sorted_ids[:max_count]:
            node = self._nodes.get(nid)
            if node:
                result.append((node, neighbors[nid]))
        return result

    def get_edges_for(self, node_id: str) -> list[dict]:
        """Return all edges where this node is source or target."""
        return [
            e for e in self._edges
            if e["source"] == node_id or e["target"] == node_id
        ]

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        weight: float = 1.0,
        edge_type: str = "structural",
    ) -> bool:
        """Add an edge between two nodes if it doesn't already exist.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            relation: Relation label (e.g., "imports", "related_to").
            weight: Edge weight (1.0 for structural, 0.x for semantic).
            edge_type: "structural" or "semantic".

        Returns:
            True if the edge was added, False if it already exists.
        """
        if source_id == target_id:
            return False
        if self._edge_exists(source_id, target_id, relation):
            return False

        now = datetime.now().isoformat(timespec="seconds")
        self._edges.append({
            "source": source_id,
            "target": target_id,
            "relation": relation,
            "weight": weight,
            "created_at": now,
            "last_active": now,
            "layer": "UNIVERSAL",
            "source_type": "world",
            "edge_type": edge_type,
        })
        return True

    def remove_edges_by_file(self, file_path: str) -> int:
        """Remove all edges involving a file's nodes. Returns count removed."""
        file_id = _make_id(file_path)
        prefix = file_id + "__"
        before = len(self._edges)
        self._edges = [
            e for e in self._edges
            if not (
                e["source"] == file_id or e["target"] == file_id
                or e["source"].startswith(prefix) or e["target"].startswith(prefix)
            )
        ]
        return before - len(self._edges)

    def link_file(self, node_id: str, file_path: str) -> bool:
        """Link a file to a node. Returns True if added (not a duplicate)."""
        node = self._nodes.get(node_id)
        if not node:
            return False
        files = node.setdefault("files", [])
        normalized = file_path.replace("\\", "/")
        if normalized not in files:
            files.append(normalized)
            return True
        return False

    def unlink_file(self, node_id: str, file_path: str) -> bool:
        """Remove a file link from a node. Returns True if found and removed."""
        node = self._nodes.get(node_id)
        if not node:
            return False
        files = node.get("files", [])
        normalized = file_path.replace("\\", "/")
        if normalized in files:
            files.remove(normalized)
            return True
        return False

    def get_linked_files(self, node_id: str) -> list[str]:
        """Return list of file paths linked to a node."""
        node = self._nodes.get(node_id)
        if not node:
            return []
        return list(node.get("files", []))

    def get_nodes_by_kind(self, kind: str) -> list[dict]:
        """Return all nodes with the given kind (entity/symbol/section/file)."""
        return [n for n in self._nodes.values() if n.get("kind") == kind]

    def upsert_folder_node(self, folder_path: str) -> str:
        """Create a folder node if it doesn't exist. Returns the folder node ID."""
        folder_id = _make_id(folder_path)
        if folder_id not in self._nodes:
            now = datetime.now().isoformat(timespec="seconds")
            folder_name = folder_path.rsplit("/", 1)[-1] if "/" in folder_path else folder_path
            meta = NodeMeta(
                layer=Layer.UNIVERSAL,
                source="world",
                confidence_for_owner=1.0,
                status="complete",
            )
            self._nodes[folder_id] = {
                "id": folder_id,
                "label": folder_name,
                "type": "Folder",
                "kind": "folder",
                "created_at": now,
                "last_active": now,
                "session_count": 1,
                "attributes": {"path": folder_path},
                "facts": [],
                "files": [],
                "chunk_ids": [],
                **meta.to_dict(),
            }
        return folder_id

    def upsert_file_structure(
        self,
        file_path: str,
        language: str,
        units: list,
        content_hash: str,
    ) -> int:
        """Create/update graph nodes for a parsed file structure.

        Creates a file-level node and a symbol/section node for each unit.
        If the file hash is unchanged, skips re-parsing (returns 0).
        On re-parse, removes old symbol nodes for the file first.

        Args:
            file_path: relative file path (forward slashes)
            language: detected language
            units: list of StructuralUnit objects
            content_hash: SHA-256 of file content

        Returns:
            Number of symbol/section nodes created (0 if skipped).
        """
        now = datetime.now().isoformat(timespec="seconds")
        file_id = _make_id(file_path)

        # Create or update file node
        existing_file = self._nodes.get(file_id)
        if existing_file:
            old_hash = existing_file.get("attributes", {}).get("content_hash", "")
            if old_hash == content_hash:
                log.debug("File unchanged (hash match), skipping: %s", file_path)
                return 0
            # Hash changed — remove old symbol children
            self._remove_file_children(file_id)
            existing_file["last_active"] = now
            existing_file["attributes"]["content_hash"] = content_hash
            existing_file["attributes"]["language"] = language
            existing_file["stale"] = False
            existing_file["stale_since"] = None
            existing_file["indexed_at"] = now
        else:
            meta = NodeMeta(
                layer=Layer.UNIVERSAL,
                source="world",
                confidence_for_owner=1.0,
                status="complete",
            )
            self._nodes[file_id] = {
                "id": file_id,
                "label": file_path.rsplit("/", 1)[-1] if "/" in file_path else file_path,
                "type": "File",
                "kind": "file",
                "created_at": now,
                "last_active": now,
                "session_count": 1,
                "attributes": {
                    "path": file_path,
                    "language": language,
                    "content_hash": content_hash,
                },
                "facts": [],
                "files": [file_path],
                "chunk_ids": [],
                "stale": False,
                "stale_since": None,
                "indexed_at": now,
                **meta.to_dict(),
            }

        # Create folder hierarchy: a/b/c/file.md → folder "a" → "a/b" → "a/b/c" → file
        path_parts = file_path.split("/")
        if len(path_parts) > 1:
            for i in range(1, len(path_parts)):
                folder_path = "/".join(path_parts[:i])
                folder_id = self.upsert_folder_node(folder_path)
                if i == len(path_parts) - 1:
                    # Immediate parent folder → file edge
                    if not self._edge_exists(folder_id, file_id, "contains"):
                        self._edges.append({
                            "source": folder_id,
                            "target": file_id,
                            "relation": "contains",
                            "weight": 1.0,
                            "created_at": now,
                            "last_active": now,
                            "layer": "UNIVERSAL",
                            "source_type": "world",
                        })
                else:
                    # Parent folder → child folder edge
                    child_folder = "/".join(path_parts[:i + 1])
                    child_id = _make_id(child_folder)
                    if not self._edge_exists(folder_id, child_id, "contains"):
                        self._edges.append({
                            "source": folder_id,
                            "target": child_id,
                            "relation": "contains",
                            "weight": 1.0,
                            "created_at": now,
                            "last_active": now,
                            "layer": "UNIVERSAL",
                            "source_type": "world",
                        })

        # Create symbol/section nodes
        created = 0
        for unit in units:
            sym_id = make_symbol_id(file_path, unit.name, unit.parent)
            kind = "section" if unit.unit_type == "section" else "symbol"
            node_type = "Section" if kind == "section" else "Symbol"

            meta = NodeMeta(
                layer=Layer.UNIVERSAL,
                source="world",
                confidence_for_owner=1.0,
                status="complete",
            )
            self._nodes[sym_id] = {
                "id": sym_id,
                "label": unit.name,
                "type": node_type,
                "kind": kind,
                "created_at": now,
                "last_active": now,
                "session_count": 1,
                "attributes": {
                    "file_path": file_path,
                    "symbol_type": unit.unit_type,
                    "start_line": unit.start_line,
                    "end_line": unit.end_line,
                    "signature": unit.signature,
                    "language": language,
                },
                "facts": [],
                "files": [file_path],
                "chunk_ids": [],
                **meta.to_dict(),
            }

            # CONTAINS edge: file -> symbol
            if not self._edge_exists(file_id, sym_id, "contains"):
                self._edges.append({
                    "source": file_id,
                    "target": sym_id,
                    "relation": "contains",
                    "weight": 1.0,
                    "created_at": now,
                    "last_active": now,
                    "layer": "UNIVERSAL",
                    "source_type": "world",
                })

            # CHILD_OF edge: nested symbol -> parent symbol
            if unit.parent:
                parent_id = make_symbol_id(file_path, unit.parent)
                if parent_id in self._nodes and not self._edge_exists(sym_id, parent_id, "child_of"):
                    self._edges.append({
                        "source": sym_id,
                        "target": parent_id,
                        "relation": "child_of",
                        "weight": 1.0,
                        "created_at": now,
                        "last_active": now,
                        "layer": "UNIVERSAL",
                        "source_type": "world",
                    })

            created += 1

        log.info("Upserted file structure: %s → %d symbols", file_path, created)
        return created

    def _remove_file_children(self, file_id: str) -> None:
        """Remove all symbol/section nodes that belong to a file node."""
        prefix = file_id + "__"
        child_ids = [nid for nid in self._nodes if nid.startswith(prefix)]
        for cid in child_ids:
            del self._nodes[cid]
        # Remove edges involving those children
        child_set = set(child_ids)
        self._edges = [
            e for e in self._edges
            if e["source"] not in child_set and e["target"] not in child_set
        ]
        # Clear chunk_ids on the parent file node (re-indexing starts clean)
        parent = self._nodes.get(file_id)
        if parent:
            parent["chunk_ids"] = []

    # ── Chunk linkage ──

    def link_chunks(self, node_id: str, chunk_ids: list[str]) -> bool:
        """Set chunk_ids on a node (replaces existing, for re-indexing)."""
        node = self._nodes.get(node_id)
        if not node:
            return False
        node["chunk_ids"] = chunk_ids
        return True

    def get_chunks_for_node(self, node_id: str) -> list[str]:
        """Return chunk_ids for a node, or empty list if not found."""
        node = self._nodes.get(node_id)
        if not node:
            return []
        return node.get("chunk_ids", [])

    def clear_chunks(self, node_id: str) -> bool:
        """Clear chunk_ids on a node."""
        node = self._nodes.get(node_id)
        if not node:
            return False
        node["chunk_ids"] = []
        return True

    def get_nodes_with_chunks(self) -> list[dict]:
        """Return all nodes that have non-empty chunk_ids."""
        return [n for n in self._nodes.values() if n.get("chunk_ids")]

    def get_symbol_content(self, node_id: str, workspace_root: Path) -> str | None:
        """Read and return ONLY the lines for a symbol/section node from disk.

        Returns None if file doesn't exist or node has no line info.
        """
        node = self._nodes.get(node_id)
        if not node or node.get("kind") not in ("symbol", "section"):
            return None

        attrs = node.get("attributes", {})
        file_path = attrs.get("file_path")
        start_line = attrs.get("start_line")
        end_line = attrs.get("end_line")

        if not file_path or not start_line or not end_line:
            return None

        full_path = workspace_root / file_path
        if not full_path.is_file():
            return None

        try:
            lines = full_path.read_text(encoding="utf-8").split("\n")
            # 1-indexed inclusive range
            selected = lines[start_line - 1 : end_line]
            return "\n".join(selected)
        except Exception as e:
            log.warning("Failed to read symbol content from %s: %s", full_path, e)
            return None

    def get_file_symbols(self, file_path: str) -> list[dict]:
        """Return all symbol/section nodes that belong to a file."""
        file_id = _make_id(file_path)
        prefix = file_id + "__"
        return [
            node for nid, node in self._nodes.items()
            if nid.startswith(prefix) and node.get("kind") in ("symbol", "section")
        ]

    def get_stale_files(self) -> list[dict]:
        """Return all file nodes marked as stale."""
        return [
            n for n in self._nodes.values()
            if n.get("kind") == "file" and n.get("stale")
        ]

    def mark_file_stale(self, file_path: str) -> bool:
        """Mark a file node as stale. Returns True if found."""
        file_id = _make_id(file_path)
        node = self._nodes.get(file_id)
        if not node or node.get("kind") != "file":
            return False
        if not node.get("stale"):
            node["stale"] = True
            node["stale_since"] = datetime.now().isoformat(timespec="seconds")
        return True

    def save(self) -> None:
        """Persist current graph state to disk."""
        self._save()

    # ── Import / Export ──

    def export_json(self) -> dict:
        """Export the full graph as a JSON-serializable dict.

        Returns:
            dict with "nodes", "edges", and "metadata" keys.
        """
        return {
            "metadata": {
                "version": "0.2.0",
                "exported_at": datetime.now().isoformat(timespec="seconds"),
                "node_count": len(self._nodes),
                "edge_count": len(self._edges),
                "session_id": self._session_id,
            },
            "nodes": list(self._nodes.values()),
            "edges": list(self._edges),
        }

    def import_json(self, data: dict, mode: str = "merge") -> tuple[int, int]:
        """Import nodes and edges from a previously exported dict.

        Args:
            data: dict with "nodes" and "edges" keys (as from export_json)
            mode: "merge" (upsert, default) or "replace" (clear then load)

        Returns:
            (nodes_imported, edges_imported) counts
        """
        nodes_list = data.get("nodes", [])
        edges_list = data.get("edges", [])

        if mode == "replace":
            self._nodes.clear()
            self._edges.clear()

        nodes_imported = 0
        for node in nodes_list:
            if not isinstance(node, dict) or "id" not in node:
                continue
            nid = node["id"]
            if mode == "merge" and nid in self._nodes:
                # Merge facts from imported node into existing
                existing = self._nodes[nid]
                existing_fact_texts = {
                    _normalize_for_dedup(f.get("fact", ""))
                    for f in existing.get("facts", [])
                }
                for fact in node.get("facts", []):
                    norm = _normalize_for_dedup(fact.get("fact", ""))
                    if norm and norm not in existing_fact_texts:
                        existing.setdefault("facts", []).append(fact)
                        existing_fact_texts.add(norm)
                # Update last_active if imported is newer
                if node.get("last_active", "") > existing.get("last_active", ""):
                    existing["last_active"] = node["last_active"]
            else:
                self._nodes[nid] = _migrate_node(node)
            nodes_imported += 1

        edges_imported = 0
        for edge in edges_list:
            if not isinstance(edge, dict):
                continue
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            rel = edge.get("relation", "")
            if src and tgt and rel:
                if mode == "replace" or not self._edge_exists(src, tgt, rel):
                    self._edges.append(edge)
                    edges_imported += 1

        self._save()
        log.info("graph_import mode=%s nodes=%d edges=%d", mode, nodes_imported, edges_imported)
        return nodes_imported, edges_imported

    @property
    def dedup_log(self) -> list[tuple[str, str, str]]:
        """Return the dedup log from the last upsert_entities call."""
        return getattr(self, "_dedup_log", [])

    @property
    def session_id(self) -> str:
        """Return the current session ID."""
        return self._session_id

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)
