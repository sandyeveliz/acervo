"""Topic graph — persists entity nodes and edges to JSON.

Structure follows docs/CONTEXT_ENGINE_DESIGN.md.

Cada nodo tiene:
  layer (Layer): UNIVERSAL o PERSONAL
  source (str): "world" o "user_assertion"
  confidence_for_owner (float): 0.0-1.0
  status (str): "complete", "incomplete" o "pending_verification"
  pending_fields (list[str]): campos faltantes si status == "incomplete"

Nodos existentes sin layer defaultean a PERSONAL con source="user_assertion".
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from datetime import datetime
from pathlib import Path

from acervo.layers import Layer
from acervo.ontology import is_known_type

log = logging.getLogger(__name__)

_DEFAULT_PATH = Path("data/graph")


def _make_id(name: str) -> str:
    """Convert a name to a stable ASCII node ID. Strips accents."""
    nfkd = unicodedata.normalize("NFKD", name.lower())
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", ascii_str).strip("_")


_DEDUP_STRIP = re.compile(r"[^a-z0-9\s]")
_DEDUP_ARTICLES = re.compile(r"\b(el|la|los|las|un|una|de|del|en|the|a|an|of|in)\b")


def _normalize_for_dedup(text: str) -> str:
    """Normalize text for dedup comparison: lowercase, no punctuation, no articles."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    no_punct = _DEDUP_STRIP.sub("", ascii_str)
    no_articles = _DEDUP_ARTICLES.sub("", no_punct)
    return " ".join(no_articles.split())


def _default_node_meta(etype: str) -> dict:
    """Genera los campos de capa/ontología para un nodo nuevo."""
    known = is_known_type(etype)
    return {
        "layer": Layer.PERSONAL.name,
        "source": "user_assertion",
        "confidence_for_owner": 1.0,
        "status": "complete" if known else "incomplete",
        "pending_fields": [] if known else ["tipo"],
    }


def _migrate_node(node: dict) -> dict:
    """Agrega campos de capa a nodos legacy que no los tienen."""
    if "layer" not in node:
        node["layer"] = Layer.PERSONAL.name
        node["source"] = "user_assertion"
        node["confidence_for_owner"] = 1.0
        node["status"] = "complete"
        node["pending_fields"] = []
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

    def upsert_entities(
        self,
        entities: list[tuple[str, str]],
        relations: list[tuple[str, str, str]] | None = None,
        facts: list[tuple[str, str, str]] | None = None,
        layer: Layer = Layer.PERSONAL,
        source: str = "user_assertion",
        confidence: float = 1.0,
    ) -> tuple[int, int]:
        """Upsert entities, add relations and source-tagged facts.

        Args:
            entities: list of (name, type) pairs
            relations: list of (source_name, target_name, relation) tuples
            facts: list of (entity_name, fact_text, source) tuples
            layer: Layer enum — UNIVERSAL o PERSONAL (default PERSONAL)
            source: "world" o "user_assertion" (default "user_assertion")
            confidence: confianza para el owner (0.0-1.0, default 1.0)
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
                node["status"] = "hot"
                sessions_seen = {f.get("session") for f in node.get("facts", [])}
                if self._session_id not in sessions_seen:
                    node["session_count"] = node.get("session_count", 0) + 1
            else:
                meta = _default_node_meta(etype)
                # Override with caller-provided layer/source if specified
                meta["layer"] = layer.name
                meta["source"] = source
                meta["confidence_for_owner"] = confidence
                self._nodes[nid] = {
                    "id": nid,
                    "label": name,
                    "type": etype if is_known_type(etype) else "Desconocido",
                    "created_at": now,
                    "last_active": now,
                    "session_count": 1,
                    "status": "hot",
                    "attributes": {},
                    "facts": [],
                    **meta,
                }
                # Si el tipo es desconocido, marcar incompleto
                if not is_known_type(etype) and etype not in ("Desconocido",):
                    self._nodes[nid]["status"] = "incomplete"
                    self._nodes[nid]["pending_fields"] = ["tipo"]

        # Add semantic relations
        if relations:
            for src_name, tgt_name, relation in relations:
                src_id = _make_id(src_name)
                tgt_id = _make_id(tgt_name)
                if not self._edge_exists(src_id, tgt_id, relation):
                    self._edges.append({
                        "source": src_id,
                        "target": tgt_id,
                        "relation": relation,
                        "weight": 1.0,
                        "created_at": now,
                        "layer": layer.name,
                        "source_type": source,
                    })

        # Fallback: co_mentioned for entity pairs without explicit relations
        related_pairs = set()
        if relations:
            for src_name, tgt_name, _ in relations:
                related_pairs.add(frozenset([_make_id(src_name), _make_id(tgt_name)]))

        for i, src in enumerate(node_ids):
            for tgt in node_ids[i + 1:]:
                pair = frozenset([src, tgt])
                if pair not in related_pairs:
                    if not self._edge_exists(src, tgt, "co_mentioned"):
                        self._edges.append({
                            "source": src,
                            "target": tgt,
                            "relation": "co_mentioned",
                            "weight": 1.0,
                            "created_at": now,
                            "layer": layer.name,
                            "source_type": source,
                        })
                    else:
                        for edge in self._edges:
                            if (
                                edge.get("relation") == "co_mentioned"
                                and {edge["source"], edge["target"]} == {src, tgt}
                            ):
                                edge["weight"] = edge.get("weight", 1.0) + 0.5
                                break

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
    def _find_similar_fact(existing_facts: list[dict], new_fact: str, threshold: float = 0.9) -> str | None:
        """Check if a similar fact already exists. Returns the existing fact text or None."""
        new_norm = _normalize_for_dedup(new_fact)
        if not new_norm:
            return None
        for f in existing_facts:
            existing_norm = _normalize_for_dedup(f.get("fact", ""))
            if not existing_norm:
                continue
            if new_norm == existing_norm:
                return f["fact"]
            if new_norm in existing_norm or existing_norm in new_norm:
                return f["fact"]
            shorter = min(len(new_norm), len(existing_norm))
            longer = max(len(new_norm), len(existing_norm))
            matches = sum(1 for a, b in zip(new_norm, existing_norm) if a == b)
            if matches / longer >= threshold:
                return f["fact"]
        return None

    def cycle_status(self) -> None:
        """Demote node statuses: hot→warm, warm→cold. Called at turn start."""
        for node in self._nodes.values():
            status = node.get("status", "cold")
            if status == "hot":
                node["status"] = "warm"
            elif status == "warm":
                node["status"] = "cold"

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

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)
