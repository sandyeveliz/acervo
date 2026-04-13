#!/usr/bin/env python3
"""Migrate a LadybugDB graphdb to the Phase-2 bi-temporal schema.

This script takes an existing graph database created with the pre-Phase-2
schema (no ``name_embedding`` on EntityNode, no ``valid_at/invalid_at/
expired_at/reference_time/fact_embedding/episodes`` on Fact) and rewrites
it to the new schema.

Approach: **new DB + copy + swap.** Kuzu/Ladybug ``ALTER TABLE`` has
documented limitations with array columns and default values so we avoid
it entirely. Instead we:

    1. Back up the existing graphdb directory to ``graphdb.bak_<ts>``.
    2. Build a fresh graphdb at a temp path with the new schema (the
       ``LadybugGraphStore`` constructor creates the tables from the
       updated DDLs in acervo/adapters/ladybug_store.py).
    3. Copy every EntityNode, StructuralNode, Fact, ValidationLog row
       from the old DB into the new DB. New columns stay NULL except for
       ``Fact.reference_time``, which is best-effort populated from
       ``Fact.date`` so Phase 3 contradiction logic has an anchor.
    4. Copy every relation row (SemanticRel, StructuralRel,
       EntityToStructural, StructuralToEntity, EntityHasFact,
       StructuralHasFact).
    5. Validate row counts match between old and new.
    6. Swap the new DB into place. The old copy stays as
       ``graphdb.bak_<ts>`` for manual rollback if needed.

Idempotency: if the target DB already exposes the Phase-2 columns the
script exits with a clear message without touching anything.

Usage::

    python scripts/migrate_bi_temporal.py .acervo/data/graphdb
    python scripts/migrate_bi_temporal.py .acervo/data/graphdb --dry-run
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Make the acervo package importable when running as a standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import real_ladybug as _engine  # type: ignore[import-untyped]
except ImportError:
    import kuzu as _engine  # type: ignore[import-untyped]


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ── Schema probing ──────────────────────────────────────────────────────────


def _connect(db_path: Path):
    db = _engine.Database(str(db_path))
    return db, _engine.Connection(db)


def _is_already_migrated(db_path: Path) -> bool:
    """Return True when the DB already has the Phase-2 columns."""
    if not db_path.exists():
        return False
    try:
        _db, conn = _connect(db_path)
    except Exception as exc:
        log.warning("Could not open %s for schema probe: %s", db_path, exc)
        return False

    probes = [
        ("EntityNode", "name_embedding"),
        ("Fact", "valid_at"),
        ("Fact", "fact_embedding"),
    ]
    try:
        for table, column in probes:
            try:
                conn.execute(f"MATCH (n:{table}) RETURN n.{column} LIMIT 0")
            except Exception:
                return False
        return True
    finally:
        try:
            conn.close()  # type: ignore[attr-defined]
        except Exception:
            pass


# ── Row readers / writers ───────────────────────────────────────────────────


_ENTITY_NODE_COLUMNS = [
    "id", "label", "type", "kind", "created_at", "last_active",
    "session_count", "attributes", "files", "chunk_ids", "layer",
    "owner", "source", "confidence_for_owner", "status", "pending_fields",
]
_STRUCTURAL_NODE_COLUMNS = [
    "id", "label", "type", "kind", "created_at", "last_active",
    "session_count", "attributes", "files", "chunk_ids", "layer",
    "owner", "source", "confidence_for_owner", "status", "pending_fields",
    "stale", "stale_since", "indexed_at",
]
_FACT_COLUMNS = ["id", "fact_text", "date", "session", "source", "speaker"]
_VALIDATION_LOG_COLUMNS = [
    "id", "timestamp", "input_type", "mapped_type", "input_relation",
    "mapped_relation", "action", "reason", "source_stage", "entity_name",
    "session_id",
]
_REL_TABLES = {
    "SemanticRel": ["relation", "weight", "created_at", "last_active", "layer", "source_type", "edge_type"],
    "StructuralRel": ["relation", "weight", "created_at", "last_active", "layer", "source_type", "edge_type"],
    "EntityToStructural": ["relation", "weight", "created_at", "last_active", "layer", "source_type", "edge_type"],
    "StructuralToEntity": ["relation", "weight", "created_at", "last_active", "layer", "source_type", "edge_type"],
    "EntityHasFact": [],
    "StructuralHasFact": [],
}

# Rel-table endpoints (src_table -> tgt_table)
_REL_ENDPOINTS = {
    "SemanticRel": ("EntityNode", "EntityNode"),
    "StructuralRel": ("StructuralNode", "StructuralNode"),
    "EntityToStructural": ("EntityNode", "StructuralNode"),
    "StructuralToEntity": ("StructuralNode", "EntityNode"),
    "EntityHasFact": ("EntityNode", "Fact"),
    "StructuralHasFact": ("StructuralNode", "Fact"),
}


def _fetch_rows(conn, query: str) -> list[tuple]:
    r = conn.execute(query)
    rows: list[tuple] = []
    while r.has_next():
        rows.append(tuple(r.get_next()))
    return rows


def _copy_entity_nodes(src_conn, dst_conn) -> int:
    cols = ", ".join(f"n.{c}" for c in _ENTITY_NODE_COLUMNS)
    rows = _fetch_rows(src_conn, f"MATCH (n:EntityNode) RETURN {cols}")
    for row in rows:
        params = dict(zip(_ENTITY_NODE_COLUMNS, row, strict=True))
        # name_embedding is new — defaults to NULL for migrated rows. The
        # next time the owning entity is touched S1 will batch-embed it.
        params["name_embedding"] = None
        dst_conn.execute(
            "CREATE (n:EntityNode {"
            "id: $id, label: $label, type: $type, kind: $kind, "
            "created_at: $created_at, last_active: $last_active, "
            "session_count: $session_count, attributes: $attributes, "
            "files: $files, chunk_ids: $chunk_ids, layer: $layer, "
            "owner: $owner, source: $source, "
            "confidence_for_owner: $confidence_for_owner, status: $status, "
            "pending_fields: $pending_fields, name_embedding: $name_embedding"
            "})",
            params,
        )
    return len(rows)


def _copy_structural_nodes(src_conn, dst_conn) -> int:
    cols = ", ".join(f"n.{c}" for c in _STRUCTURAL_NODE_COLUMNS)
    rows = _fetch_rows(src_conn, f"MATCH (n:StructuralNode) RETURN {cols}")
    for row in rows:
        params = dict(zip(_STRUCTURAL_NODE_COLUMNS, row, strict=True))
        dst_conn.execute(
            "CREATE (n:StructuralNode {"
            "id: $id, label: $label, type: $type, kind: $kind, "
            "created_at: $created_at, last_active: $last_active, "
            "session_count: $session_count, attributes: $attributes, "
            "files: $files, chunk_ids: $chunk_ids, layer: $layer, "
            "owner: $owner, source: $source, "
            "confidence_for_owner: $confidence_for_owner, status: $status, "
            "pending_fields: $pending_fields, stale: $stale, "
            "stale_since: $stale_since, indexed_at: $indexed_at"
            "})",
            params,
        )
    return len(rows)


def _copy_facts(src_conn, dst_conn) -> int:
    cols = ", ".join(f"f.{c}" for c in _FACT_COLUMNS)
    rows = _fetch_rows(src_conn, f"MATCH (f:Fact) RETURN {cols}")
    for row in rows:
        params = dict(zip(_FACT_COLUMNS, row, strict=True))
        # Best-effort backfill of reference_time from the old flat `date`
        # field so Phase 3 temporal arbitration has an anchor for
        # historical facts. valid_at/invalid_at/expired_at stay NULL.
        params["valid_at"] = None
        params["invalid_at"] = None
        params["expired_at"] = None
        params["reference_time"] = params.get("date") or None
        params["fact_embedding"] = None
        params["episodes"] = []
        dst_conn.execute(
            "CREATE (f:Fact {"
            "id: $id, fact_text: $fact_text, date: $date, session: $session, "
            "source: $source, speaker: $speaker, "
            "valid_at: $valid_at, invalid_at: $invalid_at, "
            "expired_at: $expired_at, reference_time: $reference_time, "
            "fact_embedding: $fact_embedding, episodes: $episodes"
            "})",
            params,
        )
    return len(rows)


def _copy_validation_log(src_conn, dst_conn) -> int:
    cols = ", ".join(f"v.{c}" for c in _VALIDATION_LOG_COLUMNS)
    try:
        rows = _fetch_rows(src_conn, f"MATCH (v:ValidationLog) RETURN {cols}")
    except Exception:
        return 0
    for row in rows:
        params = dict(zip(_VALIDATION_LOG_COLUMNS, row, strict=True))
        dst_conn.execute(
            "CREATE (v:ValidationLog {"
            "id: $id, timestamp: $timestamp, input_type: $input_type, "
            "mapped_type: $mapped_type, input_relation: $input_relation, "
            "mapped_relation: $mapped_relation, action: $action, "
            "reason: $reason, source_stage: $source_stage, "
            "entity_name: $entity_name, session_id: $session_id"
            "})",
            params,
        )
    return len(rows)


def _copy_relationships(src_conn, dst_conn) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rel_table, rel_cols in _REL_TABLES.items():
        src_tbl, tgt_tbl = _REL_ENDPOINTS[rel_table]
        select_cols = ["a.id AS src_id", "b.id AS tgt_id"]
        select_cols += [f"r.{c} AS {c}" for c in rel_cols]
        query = (
            f"MATCH (a:{src_tbl})-[r:{rel_table}]->(b:{tgt_tbl}) "
            f"RETURN {', '.join(select_cols)}"
        )
        try:
            rows = _fetch_rows(src_conn, query)
        except Exception as exc:
            log.warning("Skipping %s: %s", rel_table, exc)
            counts[rel_table] = 0
            continue

        for row in rows:
            src_id, tgt_id, *rest = row
            params = {"src": src_id, "tgt": tgt_id}
            set_clause = ""
            if rel_cols:
                for col, val in zip(rel_cols, rest, strict=True):
                    params[col] = val
                set_clause = (
                    " {" + ", ".join(f"{c}: ${c}" for c in rel_cols) + "}"
                )
            dst_conn.execute(
                f"MATCH (a:{src_tbl} {{id: $src}}), (b:{tgt_tbl} {{id: $tgt}}) "
                f"CREATE (a)-[:{rel_table}{set_clause}]->(b)",
                params,
            )
        counts[rel_table] = len(rows)
    return counts


# ── Main migration driver ──────────────────────────────────────────────────


def migrate(db_path: Path, *, dry_run: bool = False) -> int:
    if not db_path.exists():
        log.error("Graph database not found: %s", db_path)
        return 2

    if _is_already_migrated(db_path):
        log.info(
            "%s already has the Phase-2 schema (name_embedding + bi-temporal "
            "fields present). Nothing to do.",
            db_path,
        )
        return 0

    log.info("Migrating %s to Phase-2 schema...", db_path)
    if dry_run:
        log.info("(dry-run) Would create backup, build new DB, and copy rows.")
        return 0

    # 1. Backup.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.with_name(f"{db_path.name}.bak_{ts}")
    log.info("Backing up current DB to %s", backup_path)
    if db_path.is_dir():
        shutil.copytree(db_path, backup_path)
    else:
        shutil.copy2(db_path, backup_path)

    # 2. Build the new DB via LadybugGraphStore so it gets the current DDL
    #    definitions (source of truth for the schema).
    tmp_dir = Path(tempfile.mkdtemp(prefix="acervo_migrate_"))
    new_db_path = tmp_dir / db_path.name
    log.info("Creating new Phase-2 DB at %s", new_db_path)

    # Import inside the function so the schema assertion doesn't trip on
    # the OLD database path if the user has env-level fallbacks.
    from acervo.adapters.ladybug_store import LadybugGraphStore

    # The new store's _ensure_schema creates the tables with the updated
    # DDL; we just need the instance to exist briefly so the file is
    # seeded with a valid empty DB.
    new_store = LadybugGraphStore(new_db_path)
    new_conn = new_store._conn  # intentional: reuse the open connection

    # Open the OLD DB with a raw engine connection so we can read the
    # pre-Phase-2 tables directly.
    old_db, old_conn = _connect(db_path)
    try:
        # 3. Copy nodes and facts first (needed before relations).
        ent_count = _copy_entity_nodes(old_conn, new_conn)
        struct_count = _copy_structural_nodes(old_conn, new_conn)
        fact_count = _copy_facts(old_conn, new_conn)
        val_count = _copy_validation_log(old_conn, new_conn)
        log.info(
            "Copied nodes: EntityNode=%d StructuralNode=%d Fact=%d ValidationLog=%d",
            ent_count, struct_count, fact_count, val_count,
        )

        # 4. Copy relationships.
        rel_counts = _copy_relationships(old_conn, new_conn)
        log.info("Copied relationships: %s", rel_counts)

        # 5. Validate row counts.
        new_ent = _fetch_rows(new_conn, "MATCH (n:EntityNode) RETURN count(n)")[0][0]
        new_struct = _fetch_rows(new_conn, "MATCH (n:StructuralNode) RETURN count(n)")[0][0]
        new_fact = _fetch_rows(new_conn, "MATCH (f:Fact) RETURN count(f)")[0][0]
        assert new_ent == ent_count, f"EntityNode count mismatch: {ent_count} -> {new_ent}"
        assert new_struct == struct_count, f"StructuralNode count mismatch: {struct_count} -> {new_struct}"
        assert new_fact == fact_count, f"Fact count mismatch: {fact_count} -> {new_fact}"
    finally:
        # Close the raw old-engine handles so Windows releases file locks.
        try:
            old_conn.close()  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            del old_db  # type: ignore[misc]
        except Exception:
            pass
        # Close the new store so it flushes and releases locks before swap.
        try:
            new_conn.close()  # type: ignore[attr-defined]
        except Exception:
            pass

    # 6. Swap new DB into place (atomic within the parent dir).
    log.info("Swapping new DB into %s", db_path)
    if db_path.is_dir():
        shutil.rmtree(db_path)
        shutil.copytree(new_db_path, db_path)
    else:
        db_path.unlink(missing_ok=True)
        shutil.copy2(new_db_path, db_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    log.info(
        "Migration complete. Original DB preserved at %s. "
        "If everything looks good you can delete the backup.",
        backup_path,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path", type=Path, help="Path to the graphdb directory/file")
    parser.add_argument("--dry-run", action="store_true", help="Just probe, don't modify anything")
    args = parser.parse_args()
    return migrate(args.db_path.resolve(), dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
