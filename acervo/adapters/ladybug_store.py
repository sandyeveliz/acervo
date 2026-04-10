"""LadybugDB (KuzuDB fork) graph store adapter.

Implements GraphStorePort using an embedded graph database with Cypher queries.
Uses typed node/rel tables with ontology enforcement via OntologyValidator.

The import tries 'ladybug' first, falls back to 'kuzu' for development.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import real_ladybug as _engine  # type: ignore[import-untyped]
except ImportError:
    import kuzu as _engine  # type: ignore[import-untyped]

from acervo.graph.ids import _make_id, _normalize_for_dedup, make_symbol_id

log = logging.getLogger(__name__)

# ── Schema DDL ───────────────────────────────────────────────────────────────

_ENTITY_NODE_DDL = """\
CREATE NODE TABLE IF NOT EXISTS EntityNode (
    id STRING PRIMARY KEY,
    label STRING,
    type STRING,
    kind STRING DEFAULT 'entity',
    created_at STRING,
    last_active STRING,
    session_count INT64 DEFAULT 1,
    attributes STRING DEFAULT '{}',
    files STRING[],
    chunk_ids STRING[],
    layer STRING,
    owner STRING,
    source STRING,
    confidence_for_owner DOUBLE DEFAULT 1.0,
    status STRING DEFAULT 'complete',
    pending_fields STRING[]
)"""

_STRUCTURAL_NODE_DDL = """\
CREATE NODE TABLE IF NOT EXISTS StructuralNode (
    id STRING PRIMARY KEY,
    label STRING,
    type STRING,
    kind STRING,
    created_at STRING,
    last_active STRING,
    session_count INT64 DEFAULT 1,
    attributes STRING DEFAULT '{}',
    files STRING[],
    chunk_ids STRING[],
    layer STRING DEFAULT 'UNIVERSAL',
    owner STRING,
    source STRING DEFAULT 'world',
    confidence_for_owner DOUBLE DEFAULT 1.0,
    status STRING DEFAULT 'complete',
    pending_fields STRING[],
    stale BOOLEAN DEFAULT FALSE,
    stale_since STRING,
    indexed_at STRING
)"""

_FACT_NODE_DDL = """\
CREATE NODE TABLE IF NOT EXISTS Fact (
    id STRING PRIMARY KEY,
    fact_text STRING,
    date STRING,
    session STRING,
    source STRING,
    speaker STRING
)"""

_VALIDATION_LOG_DDL = """\
CREATE NODE TABLE IF NOT EXISTS ValidationLog (
    id STRING PRIMARY KEY,
    timestamp STRING,
    input_type STRING,
    mapped_type STRING,
    input_relation STRING,
    mapped_relation STRING,
    action STRING,
    reason STRING,
    source_stage STRING,
    entity_name STRING,
    session_id STRING
)"""

_REL_DDLS = [
    "CREATE REL TABLE IF NOT EXISTS SemanticRel (FROM EntityNode TO EntityNode, relation STRING, weight DOUBLE DEFAULT 1.0, created_at STRING, last_active STRING, layer STRING, source_type STRING, edge_type STRING DEFAULT 'semantic')",
    "CREATE REL TABLE IF NOT EXISTS StructuralRel (FROM StructuralNode TO StructuralNode, relation STRING, weight DOUBLE DEFAULT 1.0, created_at STRING, last_active STRING, layer STRING DEFAULT 'UNIVERSAL', source_type STRING DEFAULT 'world', edge_type STRING DEFAULT 'structural')",
    "CREATE REL TABLE IF NOT EXISTS EntityToStructural (FROM EntityNode TO StructuralNode, relation STRING, weight DOUBLE DEFAULT 1.0, created_at STRING, last_active STRING, layer STRING, source_type STRING, edge_type STRING DEFAULT 'semantic')",
    "CREATE REL TABLE IF NOT EXISTS StructuralToEntity (FROM StructuralNode TO EntityNode, relation STRING, weight DOUBLE DEFAULT 1.0, created_at STRING, last_active STRING, layer STRING, source_type STRING, edge_type STRING DEFAULT 'semantic')",
    "CREATE REL TABLE IF NOT EXISTS EntityHasFact (FROM EntityNode TO Fact)",
    "CREATE REL TABLE IF NOT EXISTS StructuralHasFact (FROM StructuralNode TO Fact)",
]

# Kinds that go into StructuralNode table
_STRUCTURAL_KINDS = frozenset({"file", "folder", "symbol", "section"})


class LadybugGraphStore:
    """Graph store backed by LadybugDB (KuzuDB fork).

    Implements the full GraphStorePort interface using Cypher queries.
    The database is a single file on disk inside .acervo/data/graphdb/.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = _engine.Database(str(db_path))
        self._conn = _engine.Connection(self._db)
        self._session_id = f"s_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._dedup_log: list[tuple[str, str, str]] = []
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create all tables if they don't exist."""
        for ddl in [_ENTITY_NODE_DDL, _STRUCTURAL_NODE_DDL, _FACT_NODE_DDL, _VALIDATION_LOG_DDL]:
            self._conn.execute(ddl)
        for ddl in _REL_DDLS:
            self._conn.execute(ddl)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _now(self) -> str:
        return datetime.now().isoformat(timespec="seconds")

    def _is_structural(self, kind: str) -> bool:
        return kind in _STRUCTURAL_KINDS

    def _node_table(self, kind: str) -> str:
        return "StructuralNode" if self._is_structural(kind) else "EntityNode"

    def _get_node_table_by_id(self, node_id: str) -> str | None:
        """Determine which table a node is in."""
        r = self._conn.execute(
            "MATCH (n:EntityNode {id: $id}) RETURN 'EntityNode' AS tbl",
            {"id": node_id},
        )
        if r.has_next():
            return "EntityNode"
        r = self._conn.execute(
            "MATCH (n:StructuralNode {id: $id}) RETURN 'StructuralNode' AS tbl",
            {"id": node_id},
        )
        if r.has_next():
            return "StructuralNode"
        return None

    def _rel_table(self, src_table: str, tgt_table: str) -> str:
        """Determine which rel table to use based on source/target node tables."""
        if src_table == "EntityNode" and tgt_table == "EntityNode":
            return "SemanticRel"
        if src_table == "StructuralNode" and tgt_table == "StructuralNode":
            return "StructuralRel"
        if src_table == "EntityNode" and tgt_table == "StructuralNode":
            return "EntityToStructural"
        return "StructuralToEntity"

    def _fact_rel_table(self, node_table: str) -> str:
        return "EntityHasFact" if node_table == "EntityNode" else "StructuralHasFact"

    def _row_to_node(self, row: list, columns: list[str]) -> dict:
        """Convert a Cypher result row to a node dict."""
        d: dict[str, Any] = {}
        for col, val in zip(columns, row):
            # Strip table prefix: EntityNode.id → id
            key = col.split(".")[-1] if "." in col else col
            d[key] = val
        # Deserialize JSON attributes
        if "attributes" in d and isinstance(d["attributes"], str):
            try:
                d["attributes"] = json.loads(d["attributes"])
            except (json.JSONDecodeError, TypeError):
                d["attributes"] = {}
        # Ensure lists
        for field in ("files", "chunk_ids", "pending_fields"):
            if field in d and d[field] is None:
                d[field] = []
        return d

    def _query_node(self, table: str, node_id: str) -> dict | None:
        """Query a single node from a specific table."""
        r = self._conn.execute(f"MATCH (n:{table} {{id: $id}}) RETURN n.*", {"id": node_id})
        if not r.has_next():
            return None
        row = r.get_next()
        cols = r.get_column_names()
        node = self._row_to_node(row, cols)
        # Attach facts
        node["facts"] = self._get_facts_for(node_id, table)
        return node

    def _query_all_from(self, table: str) -> list[dict]:
        """Get all nodes from a table."""
        r = self._conn.execute(f"MATCH (n:{table}) RETURN n.*")
        cols = r.get_column_names()
        nodes = []
        while r.has_next():
            node = self._row_to_node(r.get_next(), cols)
            node["facts"] = self._get_facts_for(node["id"], table)
            nodes.append(node)
        return nodes

    def _get_facts_for(self, node_id: str, node_table: str) -> list[dict]:
        """Get all facts linked to a node."""
        rel = self._fact_rel_table(node_table)
        r = self._conn.execute(
            f"MATCH (n:{node_table} {{id: $id}})-[:{rel}]->(f:Fact) "
            f"RETURN f.fact_text, f.date, f.session, f.source, f.speaker",
            {"id": node_id},
        )
        facts = []
        while r.has_next():
            row = r.get_next()
            facts.append({
                "fact": row[0],
                "date": row[1] or "",
                "session": row[2] or "",
                "source": row[3] or "",
            })
        return facts

    def _add_fact(self, node_id: str, node_table: str, fact_text: str,
                  date: str, session: str, source: str) -> None:
        """Add a fact node and link it to a parent node."""
        # Generate fact ID: count existing facts for this node
        existing = self._get_facts_for(node_id, node_table)
        fact_id = f"{node_id}_f{len(existing)}"
        rel = self._fact_rel_table(node_table)

        self._conn.execute(
            "CREATE (f:Fact {id: $id, fact_text: $text, date: $date, "
            "session: $session, source: $source, speaker: 'user'})",
            {"id": fact_id, "text": fact_text, "date": date,
             "session": session, "source": source},
        )
        self._conn.execute(
            f"MATCH (n:{node_table} {{id: $nid}}), (f:Fact {{id: $fid}}) "
            f"CREATE (n)-[:{rel}]->(f)",
            {"nid": node_id, "fid": fact_id},
        )

    def _find_similar_fact(self, existing_facts: list[dict], new_fact: str,
                           threshold: float = 0.65) -> str | None:
        """Check if a similar fact exists. Same logic as TopicGraph."""
        new_norm = _normalize_for_dedup(new_fact)
        if not new_norm:
            return None
        new_words = set(new_norm.split())
        for f in existing_facts:
            existing_norm = _normalize_for_dedup(f.get("fact", ""))
            if not existing_norm:
                continue
            if new_norm == existing_norm:
                return f.get("fact", "")
            if new_norm in existing_norm or existing_norm in new_norm:
                return f.get("fact", "")
            existing_words = set(existing_norm.split())
            if new_words and existing_words:
                intersection = new_words & existing_words
                union = new_words | existing_words
                if len(intersection) / len(union) >= threshold:
                    return f.get("fact", "")
        return None

    def _edge_exists(self, src_id: str, tgt_id: str, relation: str) -> bool:
        """Check if an edge exists between two nodes."""
        for table in ("SemanticRel", "StructuralRel", "EntityToStructural", "StructuralToEntity"):
            try:
                r = self._conn.execute(
                    f"MATCH (a)-[r:{table}]->(b) "
                    f"WHERE a.id = $src AND b.id = $tgt AND r.relation = $rel "
                    f"RETURN count(r)",
                    {"src": src_id, "tgt": tgt_id, "rel": relation},
                )
                if r.has_next() and r.get_next()[0] > 0:
                    return True
            except Exception:
                continue
        return False

    def _get_all_edges_for(self, node_id: str) -> list[dict]:
        """Get all edges involving a node across all rel tables."""
        edges: list[dict] = []
        for table in ("SemanticRel", "StructuralRel", "EntityToStructural", "StructuralToEntity"):
            for direction in ("out", "in"):
                try:
                    if direction == "out":
                        q = (f"MATCH (a)-[r:{table}]->(b) WHERE a.id = $id "
                             f"RETURN a.id, b.id, r.relation, r.weight, r.created_at, "
                             f"r.last_active, r.layer, r.source_type, r.edge_type")
                    else:
                        q = (f"MATCH (a)-[r:{table}]->(b) WHERE b.id = $id "
                             f"RETURN a.id, b.id, r.relation, r.weight, r.created_at, "
                             f"r.last_active, r.layer, r.source_type, r.edge_type")
                    r = self._conn.execute(q, {"id": node_id})
                    while r.has_next():
                        row = r.get_next()
                        edges.append({
                            "source": row[0],
                            "target": row[1],
                            "relation": row[2] or "",
                            "weight": row[3] or 1.0,
                            "created_at": row[4] or "",
                            "last_active": row[5] or "",
                            "layer": row[6] or "",
                            "source_type": row[7] or "",
                            "edge_type": row[8] or "",
                        })
                except Exception:
                    continue
        # Deduplicate (same edge may appear from both directions)
        seen: set[tuple[str, str, str]] = set()
        deduped: list[dict] = []
        for e in edges:
            key = (e["source"], e["target"], e["relation"])
            if key not in seen:
                seen.add(key)
                deduped.append(e)
        return deduped

    # ── GraphStorePort: Read — nodes ─────────────────────────────────────────

    def get_node(self, name_or_id: str) -> dict | None:
        nid = name_or_id if name_or_id == _make_id(name_or_id) else _make_id(name_or_id)
        node = self._query_node("EntityNode", nid)
        if node:
            return node
        node = self._query_node("StructuralNode", nid)
        if node:
            return node
        # Try original ID as-is
        if nid != name_or_id:
            node = self._query_node("EntityNode", name_or_id)
            if node:
                return node
            node = self._query_node("StructuralNode", name_or_id)
            if node:
                return node
        return None

    def get_all_nodes(self) -> list[dict]:
        return self._query_all_from("EntityNode") + self._query_all_from("StructuralNode")

    def get_nodes_by_ids(self, node_ids: set[str]) -> list[dict]:
        return [n for n in self.get_all_nodes() if n.get("id") in node_ids]

    def get_nodes_by_kind(self, kind: str) -> list[dict]:
        table = self._node_table(kind)
        r = self._conn.execute(
            f"MATCH (n:{table}) WHERE n.kind = $kind RETURN n.*",
            {"kind": kind},
        )
        cols = r.get_column_names()
        nodes = []
        while r.has_next():
            node = self._row_to_node(r.get_next(), cols)
            node["facts"] = self._get_facts_for(node["id"], table)
            nodes.append(node)
        return nodes

    # ── GraphStorePort: Read — edges & neighbors ─────────────────────────────

    def get_edges_for(self, node_id: str) -> list[dict]:
        return self._get_all_edges_for(node_id)

    def get_neighbors(self, node_id: str, max_count: int = 5) -> list[tuple[dict, float]]:
        edges = self.get_edges_for(node_id)
        neighbor_weights: dict[str, float] = {}
        for e in edges:
            other = e["target"] if e["source"] == node_id else e["source"]
            if other != node_id:
                neighbor_weights[other] = max(neighbor_weights.get(other, 0), e.get("weight", 1.0))
        sorted_ids = sorted(neighbor_weights, key=lambda k: neighbor_weights[k], reverse=True)
        result = []
        for nid in sorted_ids[:max_count]:
            node = self.get_node(nid)
            if node:
                result.append((node, neighbor_weights[nid]))
        return result

    # ── GraphStorePort: Read — files & symbols ───────────────────────────────

    def get_linked_files(self, node_id: str) -> list[str]:
        node = self.get_node(node_id)
        return list(node.get("files", [])) if node else []

    def get_file_symbols(self, file_path: str) -> list[dict]:
        file_id = _make_id(file_path)
        prefix = file_id + "__"
        r = self._conn.execute(
            "MATCH (n:StructuralNode) WHERE n.id STARTS WITH $prefix "
            "AND n.kind IN ['symbol', 'section'] RETURN n.*",
            {"prefix": prefix},
        )
        cols = r.get_column_names()
        nodes = []
        while r.has_next():
            node = self._row_to_node(r.get_next(), cols)
            node["facts"] = self._get_facts_for(node["id"], "StructuralNode")
            nodes.append(node)
        return nodes

    def get_symbol_content(self, node_id: str, workspace_root: Path) -> str | None:
        node = self.get_node(node_id)
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
            return "\n".join(lines[start_line - 1 : end_line])
        except Exception as e:
            log.warning("Failed to read symbol content: %s", e)
            return None

    def get_stale_files(self) -> list[dict]:
        r = self._conn.execute(
            "MATCH (n:StructuralNode) WHERE n.kind = 'file' AND n.stale = true RETURN n.*"
        )
        cols = r.get_column_names()
        nodes = []
        while r.has_next():
            node = self._row_to_node(r.get_next(), cols)
            node["facts"] = self._get_facts_for(node["id"], "StructuralNode")
            nodes.append(node)
        return nodes

    # ── GraphStorePort: Read — chunks ────────────────────────────────────────

    def get_chunks_for_node(self, node_id: str) -> list[str]:
        node = self.get_node(node_id)
        return list(node.get("chunk_ids", [])) if node else []

    def get_nodes_with_chunks(self) -> list[dict]:
        nodes = []
        for table in ("EntityNode", "StructuralNode"):
            r = self._conn.execute(
                f"MATCH (n:{table}) WHERE size(n.chunk_ids) > 0 RETURN n.*"
            )
            cols = r.get_column_names()
            while r.has_next():
                node = self._row_to_node(r.get_next(), cols)
                node["facts"] = self._get_facts_for(node["id"], table)
                nodes.append(node)
        return nodes

    # ── GraphStorePort: Write — entities ─────────────────────────────────────

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
        now = self._now()
        layer_name = layer.name if hasattr(layer, "name") else (layer or "PERSONAL")

        for name, etype in entities:
            nid = _make_id(name)
            table = "EntityNode"  # upsert_entities always creates entity nodes
            existing = self._query_node(table, nid)

            if existing:
                # Update last_active + session_count
                sessions = {f.get("session") for f in existing.get("facts", [])}
                new_count = existing.get("session_count", 1)
                if self._session_id not in sessions:
                    new_count += 1
                self._conn.execute(
                    f"MATCH (n:{table} {{id: $id}}) "
                    f"SET n.last_active = $now, n.session_count = $count",
                    {"id": nid, "now": now, "count": new_count},
                )
            else:
                self._conn.execute(
                    f"CREATE (n:{table} {{id: $id, label: $label, type: $type, "
                    f"kind: 'entity', created_at: $now, last_active: $now, "
                    f"session_count: 1, attributes: '{{}}', files: [], chunk_ids: [], "
                    f"layer: $layer, owner: $owner, source: $source, "
                    f"confidence_for_owner: $conf, status: 'complete', pending_fields: []}})",
                    {"id": nid, "label": name, "type": etype, "now": now,
                     "layer": layer_name, "owner": owner or "", "source": source,
                     "conf": confidence},
                )

        # Add relations
        edge_count = 0
        if relations:
            for src_name, tgt_name, relation in relations:
                src_id = _make_id(src_name)
                tgt_id = _make_id(tgt_name)
                if src_id == tgt_id:
                    continue
                if self._edge_exists(src_id, tgt_id, relation):
                    continue
                src_table = self._get_node_table_by_id(src_id)
                tgt_table = self._get_node_table_by_id(tgt_id)
                if not src_table or not tgt_table:
                    continue
                rel_table = self._rel_table(src_table, tgt_table)
                try:
                    self._conn.execute(
                        f"MATCH (a:{src_table} {{id: $src}}), (b:{tgt_table} {{id: $tgt}}) "
                        f"CREATE (a)-[:{rel_table} {{relation: $rel, weight: 1.0, "
                        f"created_at: $now, last_active: $now, layer: $layer, "
                        f"source_type: $source, edge_type: 'semantic'}}]->(b)",
                        {"src": src_id, "tgt": tgt_id, "rel": relation,
                         "now": now, "layer": layer_name, "source": source},
                    )
                    edge_count += 1
                except Exception as e:
                    log.warning("Failed to create edge %s→%s: %s", src_id, tgt_id, e)

        # Add facts
        self._dedup_log = []
        if facts:
            for entity_name, fact_text, fact_source in facts:
                nid = _make_id(entity_name)
                table = self._get_node_table_by_id(nid)
                if not table:
                    continue
                existing_facts = self._get_facts_for(nid, table)
                dup = self._find_similar_fact(existing_facts, fact_text)
                if dup:
                    self._dedup_log.append((entity_name, fact_text, f"duplicate of '{dup}'"))
                else:
                    self._add_fact(nid, table, fact_text, now[:10], self._session_id, fact_source)

        n_count = self.node_count
        e_count = self.edge_count
        log.info("graph_update nodes=%d edges=%d", n_count, e_count)
        return n_count, e_count

    # ── GraphStorePort: Write — node mutations ───────────────────────────────

    def remove_node(self, node_id: str) -> bool:
        nid = _make_id(node_id) if not self._get_node_table_by_id(node_id) else node_id
        table = self._get_node_table_by_id(nid)
        if not table:
            return False
        # Delete linked facts first
        fact_rel = self._fact_rel_table(table)
        self._conn.execute(
            f"MATCH (n:{table} {{id: $id}})-[:{fact_rel}]->(f:Fact) DETACH DELETE f",
            {"id": nid},
        )
        # Delete the node (DETACH removes all remaining edges)
        self._conn.execute(f"MATCH (n:{table} {{id: $id}}) DETACH DELETE n", {"id": nid})
        log.info("Removed node: %s", nid)
        return True

    def update_node(self, node_id: str, **fields: object) -> bool:
        nid = _make_id(node_id) if not self._get_node_table_by_id(node_id) else node_id
        table = self._get_node_table_by_id(nid)
        if not table:
            return False
        allowed = {"label", "type", "attributes"}
        set_clauses = []
        params: dict[str, Any] = {"id": nid}
        for key, value in fields.items():
            if key in allowed:
                if key == "attributes" and isinstance(value, dict):
                    value = json.dumps(value, ensure_ascii=False)
                set_clauses.append(f"n.{key} = ${key}")
                params[key] = value
        if not set_clauses:
            return True
        self._conn.execute(
            f"MATCH (n:{table} {{id: $id}}) SET {', '.join(set_clauses)}",
            params,
        )
        return True

    def merge_nodes(self, keep_id: str, absorb_id: str, alias: str | None = None) -> bool:
        kid = _make_id(keep_id) if not self._get_node_table_by_id(keep_id) else keep_id
        aid = _make_id(absorb_id) if not self._get_node_table_by_id(absorb_id) else absorb_id

        keep_table = self._get_node_table_by_id(kid)
        absorb_table = self._get_node_table_by_id(aid)
        if not keep_table or not absorb_table:
            return False

        keep_node = self._query_node(keep_table, kid)
        absorb_node = self._query_node(absorb_table, aid)
        if not keep_node or not absorb_node:
            return False

        # Merge facts (with dedup)
        keep_facts = keep_node.get("facts", [])
        existing_norms = {_normalize_for_dedup(f.get("fact", "")) for f in keep_facts}
        for fact in absorb_node.get("facts", []):
            norm = _normalize_for_dedup(fact.get("fact", ""))
            if norm and norm not in existing_norms:
                self._add_fact(kid, keep_table, fact["fact"],
                               fact.get("date", ""), fact.get("session", ""),
                               fact.get("source", "merge"))
                existing_norms.add(norm)

        # Alias fact
        if alias:
            alias_fact = f"Also known as: {alias}"
            alias_norm = _normalize_for_dedup(alias_fact)
            if alias_norm not in existing_norms:
                self._add_fact(kid, keep_table, alias_fact,
                               self._now()[:10], self._session_id, "merge")

        # Merge attributes
        keep_attrs = keep_node.get("attributes", {})
        absorb_attrs = absorb_node.get("attributes", {})
        merged_attrs = {**absorb_attrs, **keep_attrs}  # keep takes precedence
        self._conn.execute(
            f"MATCH (n:{keep_table} {{id: $id}}) SET n.attributes = $attrs",
            {"id": kid, "attrs": json.dumps(merged_attrs, ensure_ascii=False)},
        )

        # Update session count
        new_count = keep_node.get("session_count", 1) + absorb_node.get("session_count", 1)
        self._conn.execute(
            f"MATCH (n:{keep_table} {{id: $id}}) SET n.session_count = $count",
            {"id": kid, "count": new_count},
        )

        # Re-point edges: create new edges from/to keep, then delete absorbed
        absorb_edges = self.get_edges_for(aid)
        for e in absorb_edges:
            new_src = kid if e["source"] == aid else e["source"]
            new_tgt = kid if e["target"] == aid else e["target"]
            if new_src == new_tgt:
                continue  # Skip self-loops
            if not self._edge_exists(new_src, new_tgt, e["relation"]):
                src_table = self._get_node_table_by_id(new_src)
                tgt_table = self._get_node_table_by_id(new_tgt)
                if src_table and tgt_table:
                    rel_table = self._rel_table(src_table, tgt_table)
                    try:
                        self._conn.execute(
                            f"MATCH (a:{src_table} {{id: $src}}), (b:{tgt_table} {{id: $tgt}}) "
                            f"CREATE (a)-[:{rel_table} {{relation: $rel, weight: $w, "
                            f"created_at: $ca, last_active: $la, layer: $layer, "
                            f"source_type: $st, edge_type: $et}}]->(b)",
                            {"src": new_src, "tgt": new_tgt, "rel": e["relation"],
                             "w": e.get("weight", 1.0), "ca": e.get("created_at", ""),
                             "la": e.get("last_active", ""), "layer": e.get("layer", ""),
                             "st": e.get("source_type", ""), "et": e.get("edge_type", "")},
                        )
                    except Exception:
                        pass

        # Delete absorbed node
        self.remove_node(aid)
        log.info("Merged node %s into %s", aid, kid)
        return True

    def remove_fact(self, entity_name: str, fact_text: str) -> bool:
        nid = _make_id(entity_name)
        table = self._get_node_table_by_id(nid)
        if not table:
            return False
        fact_rel = self._fact_rel_table(table)
        # Find and delete matching fact
        r = self._conn.execute(
            f"MATCH (n:{table} {{id: $nid}})-[:{fact_rel}]->(f:Fact) "
            f"WHERE toLower(f.fact_text) = toLower($text) RETURN f.id",
            {"nid": nid, "text": fact_text.strip()},
        )
        if r.has_next():
            fid = r.get_next()[0]
            self._conn.execute("MATCH (f:Fact {id: $id}) DETACH DELETE f", {"id": fid})
            return True
        return False

    # ── GraphStorePort: Write — edges ────────────────────────────────────────

    def add_edge(
        self, source_id: str, target_id: str, relation: str,
        weight: float = 1.0, edge_type: str = "structural",
    ) -> bool:
        if source_id == target_id:
            return False
        if self._edge_exists(source_id, target_id, relation):
            return False
        src_table = self._get_node_table_by_id(source_id)
        tgt_table = self._get_node_table_by_id(target_id)
        if not src_table or not tgt_table:
            return False
        rel_table = self._rel_table(src_table, tgt_table)
        now = self._now()
        try:
            self._conn.execute(
                f"MATCH (a:{src_table} {{id: $src}}), (b:{tgt_table} {{id: $tgt}}) "
                f"CREATE (a)-[:{rel_table} {{relation: $rel, weight: $w, "
                f"created_at: $now, last_active: $now, layer: 'UNIVERSAL', "
                f"source_type: 'world', edge_type: $et}}]->(b)",
                {"src": source_id, "tgt": target_id, "rel": relation,
                 "w": weight, "now": now, "et": edge_type},
            )
            return True
        except Exception as e:
            log.warning("Failed to add edge: %s", e)
            return False

    def remove_edge(self, src_name: str, tgt_name: str, relation: str) -> bool:
        src_id = _make_id(src_name)
        tgt_id = _make_id(tgt_name)
        for table in ("SemanticRel", "StructuralRel", "EntityToStructural", "StructuralToEntity"):
            try:
                r = self._conn.execute(
                    f"MATCH (a)-[r:{table}]->(b) "
                    f"WHERE a.id = $src AND b.id = $tgt AND r.relation = $rel "
                    f"RETURN count(r)",
                    {"src": src_id, "tgt": tgt_id, "rel": relation},
                )
                if r.has_next() and r.get_next()[0] > 0:
                    self._conn.execute(
                        f"MATCH (a)-[r:{table}]->(b) "
                        f"WHERE a.id = $src AND b.id = $tgt AND r.relation = $rel "
                        f"DELETE r",
                        {"src": src_id, "tgt": tgt_id, "rel": relation},
                    )
                    return True
            except Exception:
                continue
        return False

    def remove_edges_by_file(self, file_path: str) -> int:
        file_id = _make_id(file_path)
        prefix = file_id + "__"
        count = 0
        for table in ("SemanticRel", "StructuralRel", "EntityToStructural", "StructuralToEntity"):
            try:
                r = self._conn.execute(
                    f"MATCH (a)-[r:{table}]->(b) "
                    f"WHERE a.id = $fid OR b.id = $fid "
                    f"OR a.id STARTS WITH $prefix OR b.id STARTS WITH $prefix "
                    f"RETURN count(r)",
                    {"fid": file_id, "prefix": prefix},
                )
                if r.has_next():
                    c = r.get_next()[0]
                    if c > 0:
                        self._conn.execute(
                            f"MATCH (a)-[r:{table}]->(b) "
                            f"WHERE a.id = $fid OR b.id = $fid "
                            f"OR a.id STARTS WITH $prefix OR b.id STARTS WITH $prefix "
                            f"DELETE r",
                            {"fid": file_id, "prefix": prefix},
                        )
                        count += c
            except Exception:
                continue
        return count

    # ── GraphStorePort: Write — file operations ──────────────────────────────

    def link_file(self, node_id: str, file_path: str) -> bool:
        node = self.get_node(node_id)
        if not node:
            return False
        normalized = file_path.replace("\\", "/")
        files = list(node.get("files", []))
        if normalized in files:
            return False
        files.append(normalized)
        table = self._get_node_table_by_id(node_id)
        self._conn.execute(
            f"MATCH (n:{table} {{id: $id}}) SET n.files = $files",
            {"id": node_id, "files": files},
        )
        return True

    def unlink_file(self, node_id: str, file_path: str) -> bool:
        node = self.get_node(node_id)
        if not node:
            return False
        normalized = file_path.replace("\\", "/")
        files = list(node.get("files", []))
        if normalized not in files:
            return False
        files.remove(normalized)
        table = self._get_node_table_by_id(node_id)
        self._conn.execute(
            f"MATCH (n:{table} {{id: $id}}) SET n.files = $files",
            {"id": node_id, "files": files},
        )
        return True

    def mark_file_stale(self, file_path: str) -> bool:
        file_id = _make_id(file_path)
        r = self._conn.execute(
            "MATCH (n:StructuralNode {id: $id}) WHERE n.kind = 'file' RETURN n.stale",
            {"id": file_id},
        )
        if not r.has_next():
            return False
        self._conn.execute(
            "MATCH (n:StructuralNode {id: $id}) SET n.stale = true, n.stale_since = $now",
            {"id": file_id, "now": self._now()},
        )
        return True

    def upsert_file_structure(
        self, file_path: str, language: str, units: list, content_hash: str,
    ) -> int:
        now = self._now()
        file_id = _make_id(file_path)

        # Check existing
        existing = self._query_node("StructuralNode", file_id)
        if existing:
            old_hash = existing.get("attributes", {}).get("content_hash", "")
            if old_hash == content_hash:
                return 0
            # Remove old children
            self._remove_file_children(file_id)
            attrs = existing.get("attributes", {})
            attrs["content_hash"] = content_hash
            attrs["language"] = language
            self._conn.execute(
                "MATCH (n:StructuralNode {id: $id}) SET n.last_active = $now, "
                "n.attributes = $attrs, n.stale = false, n.stale_since = $none, "
                "n.indexed_at = $now",
                {"id": file_id, "now": now, "none": "",
                 "attrs": json.dumps(attrs, ensure_ascii=False)},
            )
        else:
            label = file_path.rsplit("/", 1)[-1] if "/" in file_path else file_path
            attrs = {"path": file_path, "language": language, "content_hash": content_hash}
            self._conn.execute(
                "CREATE (n:StructuralNode {id: $id, label: $label, type: 'File', "
                "kind: 'file', created_at: $now, last_active: $now, session_count: 1, "
                "attributes: $attrs, files: [$fp], chunk_ids: [], layer: 'UNIVERSAL', "
                "owner: '', source: 'world', confidence_for_owner: 1.0, status: 'complete', "
                "pending_fields: [], stale: false, stale_since: '', indexed_at: $now})",
                {"id": file_id, "label": label, "now": now,
                 "attrs": json.dumps(attrs, ensure_ascii=False), "fp": file_path},
            )

        # Create folder hierarchy
        path_parts = file_path.split("/")
        if len(path_parts) > 1:
            for i in range(1, len(path_parts)):
                folder_path = "/".join(path_parts[:i])
                folder_id = self.upsert_folder_node(folder_path)
                if i == len(path_parts) - 1:
                    if not self._edge_exists(folder_id, file_id, "contains"):
                        self.add_edge(folder_id, file_id, "contains", edge_type="structural")
                else:
                    child_folder = "/".join(path_parts[:i + 1])
                    child_id = _make_id(child_folder)
                    if not self._edge_exists(folder_id, child_id, "contains"):
                        self.add_edge(folder_id, child_id, "contains", edge_type="structural")

        # Create symbol/section nodes
        created = 0
        for unit in units:
            sym_id = make_symbol_id(file_path, unit.name, unit.parent)
            kind = "section" if unit.unit_type == "section" else "symbol"
            node_type = "Section" if kind == "section" else "Symbol"
            attrs = {
                "file_path": file_path,
                "symbol_type": unit.unit_type,
                "start_line": unit.start_line,
                "end_line": unit.end_line,
                "signature": unit.signature,
                "language": language,
            }
            self._conn.execute(
                "CREATE (n:StructuralNode {id: $id, label: $label, type: $type, "
                "kind: $kind, created_at: $now, last_active: $now, session_count: 1, "
                "attributes: $attrs, files: [$fp], chunk_ids: [], layer: 'UNIVERSAL', "
                "owner: '', source: 'world', confidence_for_owner: 1.0, status: 'complete', "
                "pending_fields: [], stale: false, stale_since: '', indexed_at: ''})",
                {"id": sym_id, "label": unit.name, "type": node_type, "kind": kind,
                 "now": now, "attrs": json.dumps(attrs, ensure_ascii=False), "fp": file_path},
            )

            if not self._edge_exists(file_id, sym_id, "contains"):
                self.add_edge(file_id, sym_id, "contains", edge_type="structural")

            if unit.parent:
                parent_id = make_symbol_id(file_path, unit.parent)
                if self._get_node_table_by_id(parent_id):
                    if not self._edge_exists(sym_id, parent_id, "child_of"):
                        self.add_edge(sym_id, parent_id, "child_of", edge_type="structural")

            created += 1

        log.info("Upserted file structure: %s → %d symbols", file_path, created)
        return created

    def _remove_file_children(self, file_id: str) -> None:
        """Remove all children of a file node."""
        prefix = file_id + "__"
        # Delete facts linked to children
        self._conn.execute(
            "MATCH (n:StructuralNode)-[:StructuralHasFact]->(f:Fact) "
            "WHERE n.id STARTS WITH $prefix DETACH DELETE f",
            {"prefix": prefix},
        )
        # Delete children (DETACH removes edges)
        self._conn.execute(
            "MATCH (n:StructuralNode) WHERE n.id STARTS WITH $prefix DETACH DELETE n",
            {"prefix": prefix},
        )
        # Clear chunks on parent
        self._conn.execute(
            "MATCH (n:StructuralNode {id: $id}) SET n.chunk_ids = []",
            {"id": file_id},
        )

    def upsert_folder_node(self, folder_path: str) -> str:
        folder_id = _make_id(folder_path)
        existing = self._query_node("StructuralNode", folder_id)
        if not existing:
            folder_name = folder_path.rsplit("/", 1)[-1] if "/" in folder_path else folder_path
            now = self._now()
            self._conn.execute(
                "CREATE (n:StructuralNode {id: $id, label: $label, type: 'Folder', "
                "kind: 'folder', created_at: $now, last_active: $now, session_count: 1, "
                "attributes: $attrs, files: [], chunk_ids: [], layer: 'UNIVERSAL', "
                "owner: '', source: 'world', confidence_for_owner: 1.0, status: 'complete', "
                "pending_fields: [], stale: false, stale_since: '', indexed_at: ''})",
                {"id": folder_id, "label": folder_name, "now": now,
                 "attrs": json.dumps({"path": folder_path}, ensure_ascii=False)},
            )
        return folder_id

    # ── GraphStorePort: Write — chunks ───────────────────────────────────────

    def link_chunks(self, node_id: str, chunk_ids: list[str]) -> bool:
        table = self._get_node_table_by_id(node_id)
        if not table:
            return False
        self._conn.execute(
            f"MATCH (n:{table} {{id: $id}}) SET n.chunk_ids = $chunks",
            {"id": node_id, "chunks": chunk_ids},
        )
        return True

    def clear_chunks(self, node_id: str) -> bool:
        table = self._get_node_table_by_id(node_id)
        if not table:
            return False
        self._conn.execute(
            f"MATCH (n:{table} {{id: $id}}) SET n.chunk_ids = []",
            {"id": node_id},
        )
        return True

    # ── GraphStorePort: Validation log ────────────────────────────────────────

    def persist_validation_log(self, entries: list) -> int:
        """Persist validation log entries as ValidationLog nodes."""
        import uuid
        count = 0
        for entry in entries:
            try:
                self._conn.execute(
                    "CREATE (v:ValidationLog {"
                    "id: $id, timestamp: $ts, input_type: $it, mapped_type: $mt, "
                    "input_relation: $ir, mapped_relation: $mr, action: $action, "
                    "reason: $reason, source_stage: $ss, entity_name: $en, "
                    "session_id: $sid})",
                    {
                        "id": str(uuid.uuid4()),
                        "ts": entry.timestamp,
                        "it": entry.input_type,
                        "mt": entry.mapped_type,
                        "ir": entry.input_relation,
                        "mr": entry.mapped_relation,
                        "action": entry.action,
                        "reason": entry.reason,
                        "ss": entry.source_stage,
                        "en": entry.entity_name,
                        "sid": entry.session_id,
                    },
                )
                count += 1
            except Exception:
                log.debug("Failed to persist validation log entry: %s", entry)
        return count

    # ── GraphStorePort: Persistence & lifecycle ──────────────────────────────

    def save(self) -> None:
        """No-op: LadybugDB auto-persists."""
        pass

    def reload(self) -> None:
        """Close and re-open connection."""
        self._conn = _engine.Connection(self._db)

    def reset(self) -> None:
        """Drop all data."""
        for table in ("EntityHasFact", "StructuralHasFact",
                       "SemanticRel", "StructuralRel",
                       "EntityToStructural", "StructuralToEntity"):
            try:
                self._conn.execute(f"DROP TABLE {table}")
            except Exception:
                pass
        for table in ("Fact", "ValidationLog", "EntityNode", "StructuralNode"):
            try:
                self._conn.execute(f"DROP TABLE {table}")
            except Exception:
                pass
        self._ensure_schema()

    def repair(self) -> dict:
        """Basic repair: count orphaned edges and nodes without required fields."""
        report: dict[str, Any] = {"orphaned_edges": 0, "fixed_fields": 0}
        # Count nodes without labels
        for table in ("EntityNode", "StructuralNode"):
            r = self._conn.execute(
                f"MATCH (n:{table}) WHERE n.label IS NULL OR n.label = '' RETURN count(n)"
            )
            if r.has_next():
                report["fixed_fields"] += r.get_next()[0]
        return report

    # ── GraphStorePort: Import / Export ──────────────────────────────────────

    def export_json(self) -> dict:
        nodes = self.get_all_nodes()
        # Collect all edges
        all_edges: list[dict] = []
        seen_edges: set[tuple[str, str, str]] = set()
        for node in nodes:
            for e in self.get_edges_for(node["id"]):
                key = (e["source"], e["target"], e["relation"])
                if key not in seen_edges:
                    seen_edges.add(key)
                    all_edges.append(e)

        return {
            "metadata": {
                "version": "0.6.0",
                "exported_at": self._now(),
                "node_count": len(nodes),
                "edge_count": len(all_edges),
                "session_id": self._session_id,
            },
            "nodes": nodes,
            "edges": all_edges,
        }

    def import_json(self, data: dict, mode: str = "merge") -> tuple[int, int]:
        if mode == "replace":
            self.reset()

        nodes_imported = 0
        for node in data.get("nodes", []):
            if not isinstance(node, dict) or "id" not in node:
                continue
            nid = node["id"]
            kind = node.get("kind", "entity")
            table = self._node_table(kind)
            existing = self._query_node(table, nid)

            if existing and mode == "merge":
                # Merge facts
                existing_norms = {_normalize_for_dedup(f.get("fact", "")) for f in existing.get("facts", [])}
                for fact in node.get("facts", []):
                    norm = _normalize_for_dedup(fact.get("fact", ""))
                    if norm and norm not in existing_norms:
                        self._add_fact(nid, table, fact["fact"],
                                       fact.get("date", ""), fact.get("session", ""),
                                       fact.get("source", "import"))
                        existing_norms.add(norm)
            else:
                # Create node
                attrs = node.get("attributes", {})
                facts = node.get("facts", [])
                node_copy = {k: v for k, v in node.items() if k != "facts"}
                if isinstance(attrs, dict):
                    node_copy["attributes"] = json.dumps(attrs, ensure_ascii=False)

                # Build CREATE statement dynamically
                fields = []
                params: dict[str, Any] = {}
                for key, val in node_copy.items():
                    if val is not None:
                        fields.append(f"{key}: ${key}")
                        params[key] = val

                try:
                    self._conn.execute(
                        f"CREATE (n:{table} {{{', '.join(fields)}}})",
                        params,
                    )
                    # Add facts
                    for fact in facts:
                        self._add_fact(nid, table, fact.get("fact", ""),
                                       fact.get("date", ""), fact.get("session", ""),
                                       fact.get("source", "import"))
                except Exception as e:
                    log.warning("Failed to import node %s: %s", nid, e)
                    continue

            nodes_imported += 1

        edges_imported = 0
        for edge in data.get("edges", []):
            if not isinstance(edge, dict):
                continue
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            rel = edge.get("relation", "")
            if src and tgt and rel and not self._edge_exists(src, tgt, rel):
                src_table = self._get_node_table_by_id(src)
                tgt_table = self._get_node_table_by_id(tgt)
                if src_table and tgt_table:
                    rel_table = self._rel_table(src_table, tgt_table)
                    try:
                        self._conn.execute(
                            f"MATCH (a:{src_table} {{id: $src}}), (b:{tgt_table} {{id: $tgt}}) "
                            f"CREATE (a)-[:{rel_table} {{relation: $rel, weight: $w, "
                            f"created_at: $ca, last_active: $la, layer: $layer, "
                            f"source_type: $st, edge_type: $et}}]->(b)",
                            {"src": src, "tgt": tgt, "rel": rel,
                             "w": edge.get("weight", 1.0),
                             "ca": edge.get("created_at", ""),
                             "la": edge.get("last_active", ""),
                             "layer": edge.get("layer", ""),
                             "st": edge.get("source_type", ""),
                             "et": edge.get("edge_type", "")},
                        )
                        edges_imported += 1
                    except Exception as e:
                        log.warning("Failed to import edge: %s", e)

        log.info("graph_import mode=%s nodes=%d edges=%d", mode, nodes_imported, edges_imported)
        return nodes_imported, edges_imported

    # ── GraphStorePort: Properties ───────────────────────────────────────────

    @property
    def node_count(self) -> int:
        total = 0
        for table in ("EntityNode", "StructuralNode"):
            r = self._conn.execute(f"MATCH (n:{table}) RETURN count(n)")
            if r.has_next():
                total += r.get_next()[0]
        return total

    @property
    def edge_count(self) -> int:
        total = 0
        for table in ("SemanticRel", "StructuralRel", "EntityToStructural", "StructuralToEntity"):
            try:
                r = self._conn.execute(f"MATCH ()-[r:{table}]->() RETURN count(r)")
                if r.has_next():
                    total += r.get_next()[0]
            except Exception:
                pass
        return total

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def dedup_log(self) -> list[tuple[str, str, str]]:
        return self._dedup_log

    # ── GraphStorePort: Traversal ────────────────────────────────────────────

    def traverse_bfs(
        self, seed_ids: list[str], max_depth: int = 2,
    ) -> dict[int, list[dict]]:
        """BFS traversal using Cypher variable-length paths."""
        if not seed_ids:
            return {i: [] for i in range(max_depth + 1)}

        # Depth 0: the seeds themselves
        layers: dict[int, list[dict]] = {i: [] for i in range(max_depth + 1)}
        visited: set[str] = set()

        for sid in seed_ids:
            node = self.get_node(sid)
            if node and sid not in visited:
                layers[0].append(node)
                visited.add(sid)

        # For each subsequent depth, use Cypher variable-length paths
        # We query from each table combination
        for depth in range(1, max_depth + 1):
            prev_ids = [n["id"] for n in layers[depth - 1]]
            if not prev_ids:
                break

            neighbor_ids: set[str] = set()
            for table in ("SemanticRel", "StructuralRel", "EntityToStructural", "StructuralToEntity"):
                try:
                    # Outgoing from prev layer
                    r = self._conn.execute(
                        f"MATCH (a)-[:{table}]->(b) WHERE a.id IN $ids RETURN DISTINCT b.id",
                        {"ids": prev_ids},
                    )
                    while r.has_next():
                        nid = r.get_next()[0]
                        if nid not in visited:
                            neighbor_ids.add(nid)

                    # Incoming to prev layer
                    r = self._conn.execute(
                        f"MATCH (a)-[:{table}]->(b) WHERE b.id IN $ids RETURN DISTINCT a.id",
                        {"ids": prev_ids},
                    )
                    while r.has_next():
                        nid = r.get_next()[0]
                        if nid not in visited:
                            neighbor_ids.add(nid)
                except Exception:
                    continue

            for nid in neighbor_ids:
                node = self.get_node(nid)
                if node:
                    layers[depth].append(node)
                    visited.add(nid)

        return layers
