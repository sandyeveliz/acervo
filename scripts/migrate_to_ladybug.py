#!/usr/bin/env python3
"""Migrate Acervo knowledge graph from JSON (nodes.json/edges.json) to LadybugDB.

Usage:
    python scripts/migrate_to_ladybug.py [--data-dir .acervo/data] [--dry-run]

Reads nodes.json + edges.json from {data_dir}/graph/,
creates a LadybugDB database at {data_dir}/graphdb/acervo.db,
validates all types/relations via OntologyValidator,
and logs everything to the ValidationLog table.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from acervo.adapters.ladybug_store import LadybugGraphStore
from acervo.graph.ids import _make_id
from acervo.graph.ontology_validator import OntologyValidator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def load_json_graph(data_dir: Path) -> tuple[list[dict], list[dict]]:
    """Load nodes and edges from JSON files."""
    nodes_file = data_dir / "graph" / "nodes.json"
    edges_file = data_dir / "graph" / "edges.json"

    if not nodes_file.exists():
        log.error("nodes.json not found at %s", nodes_file)
        sys.exit(1)

    nodes = json.loads(nodes_file.read_text(encoding="utf-8"))
    if isinstance(nodes, dict):
        # Old format: dict of id → node
        nodes = list(nodes.values())

    edges = []
    if edges_file.exists():
        edges = json.loads(edges_file.read_text(encoding="utf-8"))

    log.info("Loaded %d nodes and %d edges from JSON", len(nodes), len(edges))
    return nodes, edges


def migrate(data_dir: Path, dry_run: bool = False) -> dict:
    """Run the migration. Returns summary stats."""
    nodes, edges = load_json_graph(data_dir)
    validator = OntologyValidator(source_stage="migration")

    stats = {
        "nodes_total": len(nodes),
        "nodes_migrated": 0,
        "nodes_skipped": 0,
        "edges_total": len(edges),
        "edges_migrated": 0,
        "edges_skipped": 0,
        "types_mapped": 0,
        "relations_mapped": 0,
        "relations_rejected": 0,
        "facts_migrated": 0,
    }

    if dry_run:
        log.info("=== DRY RUN — no database will be created ===")
        for node in nodes:
            vt = validator.validate_entity_type(
                node.get("type", "concept"), entity_name=node.get("label", ""),
            )
            if vt.action == "mapped":
                stats["types_mapped"] += 1
            stats["nodes_migrated"] += 1

        for edge in edges:
            vr = validator.validate_relation(
                edge.get("relation", ""), entity_name=edge.get("source", ""),
            )
            if vr.action == "mapped":
                stats["relations_mapped"] += 1
            elif vr.action == "rejected":
                stats["relations_rejected"] += 1
                stats["edges_skipped"] += 1
                continue
            stats["edges_migrated"] += 1

        log_entries = validator.drain_log()
        stats["validation_log_entries"] = len(log_entries)

        log.info("=== DRY RUN SUMMARY ===")
        for k, v in stats.items():
            log.info("  %s: %d", k, v)
        return stats

    # ── Real migration ──

    db_path = data_dir / "graphdb" / "acervo.db"
    if db_path.exists():
        log.warning("Database already exists at %s — will be overwritten", db_path)
        shutil.rmtree(db_path, ignore_errors=True)

    store = LadybugGraphStore(db_path)
    log.info("Created LadybugDB at %s", db_path)

    # Migrate nodes
    from acervo.graph.layers import Layer

    for node in nodes:
        nid = node.get("id", "")
        kind = node.get("kind", "entity")
        raw_type = node.get("type", "concept")
        label = node.get("label", nid)

        # Validate type
        vt = validator.validate_entity_type(raw_type, entity_name=label)
        if vt.action == "mapped":
            stats["types_mapped"] += 1

        resolved_type = vt.resolved
        layer_str = node.get("layer", "PERSONAL")
        layer = Layer.UNIVERSAL if layer_str == "UNIVERSAL" else Layer.PERSONAL

        if kind in ("file", "folder", "symbol", "section"):
            # Structural node — import via upsert or direct create
            # For simplicity, use import_json for structural nodes
            node_copy = dict(node)
            node_copy["type"] = resolved_type
            store.import_json({"nodes": [node_copy], "edges": []}, mode="merge")
        else:
            # Entity node
            store.upsert_entities(
                [(label, resolved_type)],
                layer=layer,
                source=node.get("source", "user_assertion"),
                owner=node.get("owner") or None,
            )

            # Add facts
            for fact in node.get("facts", []):
                fact_text = fact.get("fact", "")
                if fact_text:
                    store.upsert_entities(
                        [(label, resolved_type)],
                        facts=[(label, fact_text, fact.get("source", "migration"))],
                        layer=layer,
                        source=node.get("source", "user_assertion"),
                    )
                    stats["facts_migrated"] += 1

            # Set attributes if present
            attrs = node.get("attributes", {})
            if attrs:
                store.update_node(nid, attributes=attrs)

            # Link files
            for fp in node.get("files", []):
                store.link_file(nid, fp)

            # Link chunks
            chunk_ids = node.get("chunk_ids", [])
            if chunk_ids:
                store.link_chunks(nid, chunk_ids)

        stats["nodes_migrated"] += 1

    # Migrate edges
    for edge in edges:
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        raw_rel = edge.get("relation", "")

        if not src or not tgt or not raw_rel:
            stats["edges_skipped"] += 1
            continue

        vr = validator.validate_relation(raw_rel, entity_name=src)
        if vr.action == "rejected":
            stats["relations_rejected"] += 1
            stats["edges_skipped"] += 1
            continue
        if vr.action == "mapped":
            stats["relations_mapped"] += 1

        resolved_rel = vr.resolved
        ok = store.add_edge(
            src, tgt, resolved_rel,
            weight=edge.get("weight", 1.0),
            edge_type=edge.get("edge_type", "semantic"),
        )
        if ok:
            stats["edges_migrated"] += 1
        else:
            stats["edges_skipped"] += 1

    # Persist validation log entries to DB
    log_entries = validator.drain_log()
    stats["validation_log_entries"] = len(log_entries)

    import uuid
    for entry in log_entries:
        try:
            store._conn.execute(
                "CREATE (v:ValidationLog {"
                "id: $id, timestamp: $ts, input_type: $it, mapped_type: $mt, "
                "input_relation: $ir, mapped_relation: $mr, action: $action, "
                "reason: $reason, source_stage: $ss, entity_name: $en, session_id: $sid})",
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
        except Exception as e:
            log.warning("Failed to write validation log entry: %s", e)

    # Backup original JSON
    backup_dir = data_dir / "graph_backup"
    if not backup_dir.exists():
        src_dir = data_dir / "graph"
        shutil.copytree(src_dir, backup_dir)
        log.info("Backed up JSON graph to %s", backup_dir)

    # Verification
    log.info("=== MIGRATION SUMMARY ===")
    log.info("  LadybugDB nodes: %d", store.node_count)
    log.info("  LadybugDB edges: %d", store.edge_count)
    for k, v in stats.items():
        log.info("  %s: %d", k, v)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Migrate Acervo graph from JSON to LadybugDB")
    parser.add_argument("--data-dir", default=".acervo/data",
                        help="Path to Acervo data directory (default: .acervo/data)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate without creating database")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        log.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    migrate(data_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
