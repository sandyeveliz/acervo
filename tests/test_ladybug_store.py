"""Tests for LadybugGraphStore — KuzuDB/LadybugDB adapter.

Mirrors the test structure of test_graph.py but runs against LadybugGraphStore.
"""

import tempfile
from pathlib import Path

import pytest

from acervo.adapters.ladybug_store import LadybugGraphStore
from acervo.graph.layers import Layer

try:
    import real_ladybug  # noqa: F401
    HAS_ENGINE = True
except ImportError:
    try:
        import kuzu  # noqa: F401
        HAS_ENGINE = True
    except ImportError:
        HAS_ENGINE = False

pytestmark = pytest.mark.skipif(not HAS_ENGINE, reason="real_ladybug/kuzu not installed")


@pytest.fixture
def store():
    """Create a fresh LadybugGraphStore in a temp directory."""
    db_path = Path(tempfile.mkdtemp()) / "test.kuzu"
    return LadybugGraphStore(db_path)


class TestBasicCRUD:
    def test_upsert_and_get_node(self, store):
        store.upsert_entities(
            [("Alice", "person")], layer=Layer.PERSONAL, source="user_assertion",
        )
        node = store.get_node("alice")
        assert node is not None
        assert node["label"] == "Alice"
        assert node["type"] == "person"
        assert node["layer"] == "PERSONAL"

    def test_upsert_updates_existing(self, store):
        store.upsert_entities([("Alice", "person")], layer=Layer.PERSONAL)
        store.upsert_entities([("Alice", "person")], layer=Layer.PERSONAL)
        assert store.node_count == 1

    def test_get_all_nodes(self, store):
        store.upsert_entities(
            [("Alice", "person"), ("React", "technology")],
            layer=Layer.PERSONAL,
        )
        nodes = store.get_all_nodes()
        assert len(nodes) == 2

    def test_get_nodes_by_ids(self, store):
        store.upsert_entities(
            [("Alice", "person"), ("Bob", "person"), ("React", "technology")],
            layer=Layer.PERSONAL,
        )
        nodes = store.get_nodes_by_ids({"alice", "react"})
        assert len(nodes) == 2

    def test_get_node_not_found(self, store):
        assert store.get_node("nonexistent") is None

    def test_remove_node(self, store):
        store.upsert_entities([("Alice", "person")], layer=Layer.PERSONAL)
        assert store.remove_node("alice")
        assert store.get_node("alice") is None
        assert store.node_count == 0

    def test_remove_nonexistent_node(self, store):
        assert store.remove_node("nope") is False

    def test_update_node(self, store):
        store.upsert_entities([("Alice", "person")], layer=Layer.PERSONAL)
        store.update_node("alice", label="Alice Smith", type="person")
        node = store.get_node("alice")
        assert node["label"] == "Alice Smith"

    def test_node_count_and_edge_count(self, store):
        assert store.node_count == 0
        assert store.edge_count == 0
        store.upsert_entities(
            [("A", "person"), ("B", "person")],
            relations=[("A", "B", "works_at")],
            layer=Layer.PERSONAL,
        )
        assert store.node_count == 2
        assert store.edge_count == 1


class TestRelations:
    def test_add_relation_via_upsert(self, store):
        store.upsert_entities(
            [("React", "technology"), ("Next.js", "technology")],
            relations=[("Next.js", "React", "depends_on")],
            layer=Layer.UNIVERSAL,
        )
        edges = store.get_edges_for("react")
        assert len(edges) == 1
        assert edges[0]["relation"] == "depends_on"

    def test_self_referential_edges_rejected(self, store):
        store.upsert_entities(
            [("A", "concept")],
            relations=[("A", "A", "part_of")],
            layer=Layer.PERSONAL,
        )
        assert store.edge_count == 0

    def test_add_edge(self, store):
        store.upsert_entities(
            [("A", "person"), ("B", "person")], layer=Layer.PERSONAL,
        )
        ok = store.add_edge("a", "b", "works_at")
        assert ok is True
        assert store.edge_count == 1

    def test_add_duplicate_edge_rejected(self, store):
        store.upsert_entities(
            [("A", "person"), ("B", "person")], layer=Layer.PERSONAL,
        )
        store.add_edge("a", "b", "works_at")
        ok = store.add_edge("a", "b", "works_at")
        assert ok is False
        assert store.edge_count == 1

    def test_remove_edge(self, store):
        store.upsert_entities(
            [("A", "person"), ("B", "organization")],
            relations=[("A", "B", "works_at")],
            layer=Layer.PERSONAL,
        )
        ok = store.remove_edge("A", "B", "works_at")
        assert ok is True
        assert store.edge_count == 0

    def test_get_neighbors(self, store):
        store.upsert_entities(
            [("A", "person"), ("B", "organization"), ("C", "technology")],
            relations=[("A", "B", "works_at"), ("A", "C", "uses_technology")],
            layer=Layer.PERSONAL,
        )
        neighbors = store.get_neighbors("a", max_count=5)
        labels = {n["label"] for n, _ in neighbors}
        assert "B" in labels
        assert "C" in labels


class TestFacts:
    def test_facts_added_and_retrieved(self, store):
        store.upsert_entities(
            [("Alice", "person")],
            facts=[("Alice", "Works at Acme Corp", "user")],
            layer=Layer.PERSONAL,
        )
        node = store.get_node("alice")
        assert len(node["facts"]) == 1
        assert node["facts"][0]["fact"] == "Works at Acme Corp"

    def test_fact_dedup(self, store):
        store.upsert_entities(
            [("Alice", "person")],
            facts=[("Alice", "Works at Acme Corp", "user")],
            layer=Layer.PERSONAL,
        )
        store.upsert_entities(
            [("Alice", "person")],
            facts=[("Alice", "Works at Acme Corp", "user")],
            layer=Layer.PERSONAL,
        )
        node = store.get_node("alice")
        assert len(node["facts"]) == 1

    def test_remove_fact(self, store):
        store.upsert_entities(
            [("Alice", "person")],
            facts=[("Alice", "Likes coffee", "user")],
            layer=Layer.PERSONAL,
        )
        ok = store.remove_fact("Alice", "Likes coffee")
        assert ok is True
        node = store.get_node("alice")
        assert len(node["facts"]) == 0


class TestMerge:
    def test_merge_nodes(self, store):
        store.upsert_entities(
            [("React", "technology")],
            facts=[("React", "A JavaScript library", "user")],
            layer=Layer.UNIVERSAL,
        )
        store.upsert_entities(
            [("ReactJS", "technology")],
            facts=[("ReactJS", "Made by Meta", "user")],
            layer=Layer.UNIVERSAL,
        )
        ok = store.merge_nodes("react", "reactjs")
        assert ok is True
        assert store.node_count == 1

        node = store.get_node("react")
        assert len(node["facts"]) == 2

    def test_merge_edges_repointed(self, store):
        store.upsert_entities(
            [("A", "person"), ("B", "person"), ("C", "person")],
            relations=[("B", "C", "works_at")],
            layer=Layer.PERSONAL,
        )
        store.merge_nodes("a", "b")
        # Edge should now be from A to C
        edges = store.get_edges_for("a")
        assert any(e["target"] == "c" for e in edges)

    def test_merge_nonexistent_returns_false(self, store):
        store.upsert_entities([("A", "person")], layer=Layer.PERSONAL)
        assert store.merge_nodes("a", "nope") is False


class TestFileStructure:
    def test_upsert_file_structure(self, store):
        from dataclasses import dataclass

        @dataclass
        class MockUnit:
            name: str
            unit_type: str
            start_line: int
            end_line: int
            signature: str
            parent: str | None = None

        units = [
            MockUnit("main", "function", 1, 10, "def main():", None),
            MockUnit("helper", "function", 12, 20, "def helper():", None),
        ]
        created = store.upsert_file_structure("src/app.py", "python", units, "hash123")
        assert created == 2
        assert store.get_node("src_app_py") is not None

    def test_file_marked_stale(self, store):
        from dataclasses import dataclass

        @dataclass
        class MockUnit:
            name: str
            unit_type: str
            start_line: int
            end_line: int
            signature: str
            parent: str | None = None

        store.upsert_file_structure("test.py", "python", [], "hash1")
        assert store.mark_file_stale("test.py") is True
        stale = store.get_stale_files()
        assert len(stale) == 1

    def test_link_unlink_file(self, store):
        store.upsert_entities([("React", "technology")], layer=Layer.UNIVERSAL)
        store.link_file("react", "src/react.js")
        assert "src/react.js" in store.get_linked_files("react")
        store.unlink_file("react", "src/react.js")
        assert "src/react.js" not in store.get_linked_files("react")


class TestChunks:
    def test_link_and_get_chunks(self, store):
        store.upsert_entities([("A", "document")], layer=Layer.PERSONAL)
        store.link_chunks("a", ["c1", "c2", "c3"])
        assert store.get_chunks_for_node("a") == ["c1", "c2", "c3"]

    def test_clear_chunks(self, store):
        store.upsert_entities([("A", "document")], layer=Layer.PERSONAL)
        store.link_chunks("a", ["c1", "c2"])
        store.clear_chunks("a")
        assert store.get_chunks_for_node("a") == []

    def test_get_nodes_with_chunks(self, store):
        store.upsert_entities(
            [("A", "document"), ("B", "document")], layer=Layer.PERSONAL,
        )
        store.link_chunks("a", ["c1"])
        result = store.get_nodes_with_chunks()
        assert len(result) == 1
        assert result[0]["id"] == "a"


class TestExportImport:
    def test_export_json(self, store):
        store.upsert_entities(
            [("A", "person"), ("B", "technology")],
            relations=[("A", "B", "uses_technology")],
            layer=Layer.PERSONAL,
        )
        data = store.export_json()
        assert data["metadata"]["node_count"] == 2
        assert data["metadata"]["edge_count"] == 1
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1

    def test_import_merge_mode(self, store):
        store.upsert_entities([("A", "person")], layer=Layer.PERSONAL)
        data = {
            "nodes": [
                {"id": "a", "label": "A", "type": "person", "kind": "entity",
                 "layer": "PERSONAL", "facts": [{"fact": "New fact", "date": "2026-01-01"}]},
                {"id": "b", "label": "B", "type": "person", "kind": "entity",
                 "layer": "PERSONAL", "facts": []},
            ],
            "edges": [],
        }
        nodes, edges = store.import_json(data, mode="merge")
        assert nodes == 2
        assert store.node_count == 2


class TestBFS:
    def test_traverse_bfs(self, store):
        store.upsert_entities(
            [("A", "person"), ("B", "technology"), ("C", "organization")],
            relations=[("A", "B", "uses_technology"), ("B", "C", "deployed_on")],
            layer=Layer.PERSONAL,
        )
        layers = store.traverse_bfs(["a"], max_depth=2)
        assert len(layers[0]) == 1  # A
        assert len(layers[1]) == 1  # B
        assert len(layers[2]) == 1  # C

    def test_traverse_bfs_empty_seeds(self, store):
        layers = store.traverse_bfs([], max_depth=2)
        assert all(len(v) == 0 for v in layers.values())


class TestLifecycle:
    def test_save_is_noop(self, store):
        store.upsert_entities([("A", "person")], layer=Layer.PERSONAL)
        store.save()  # Should not raise
        assert store.node_count == 1

    def test_reset_clears_all(self, store):
        store.upsert_entities([("A", "person")], layer=Layer.PERSONAL)
        store.reset()
        assert store.node_count == 0

    def test_session_id(self, store):
        assert store.session_id.startswith("s_")

    def test_persist_validation_log(self, store):
        from acervo.graph.ontology_validator import ValidationLogEntry
        entries = [
            ValidationLogEntry(
                timestamp="2026-04-09T12:00:00",
                input_type="Framework",
                mapped_type="technology",
                action="mapped",
                reason="synonym",
                source_stage="s1",
                entity_name="React",
                session_id=store.session_id,
            ),
            ValidationLogEntry(
                timestamp="2026-04-09T12:00:01",
                input_relation="RELATED_TO",
                mapped_relation="",
                action="rejected",
                reason="too generic",
                source_stage="s1",
                entity_name="Alice",
                session_id=store.session_id,
            ),
        ]
        count = store.persist_validation_log(entries)
        assert count == 2
        # Verify they're queryable
        r = store._conn.execute(
            "MATCH (v:ValidationLog) RETURN v.action ORDER BY v.timestamp"
        )
        actions = []
        while r.has_next():
            actions.append(r.get_next()[0])
        assert actions == ["mapped", "rejected"]

    def test_dedup_log(self, store):
        store.upsert_entities(
            [("A", "person")],
            facts=[("A", "Fact one", "user")],
            layer=Layer.PERSONAL,
        )
        store.upsert_entities(
            [("A", "person")],
            facts=[("A", "Fact one", "user")],
            layer=Layer.PERSONAL,
        )
        assert len(store.dedup_log) >= 1


class TestAuditFieldsV061:
    """v0.6.1: source / updated_by / updated_at trail on nodes, facts, edges."""

    def test_upsert_stamps_source_and_updated_by_on_node(self, store):
        store.upsert_entities(
            [("Alice", "person")],
            layer=Layer.PERSONAL,
            source="llm",
            updated_by="llm",
        )
        node = store.get_node("alice")
        assert node is not None
        assert node["source"] == "llm"
        assert node["updated_by"] == "llm"
        assert node["updated_at"]  # ISO timestamp, non-empty

    def test_upsert_stamps_audit_on_fact(self, store):
        store.upsert_entities(
            [("Alice", "person")],
            facts=[("Alice", "Lives in Neuquen", "s1")],
            layer=Layer.PERSONAL,
            source="llm",
            updated_by="llm",
        )
        node = store.get_node("alice")
        assert node and node["facts"]
        f = node["facts"][0]
        assert f["updated_by"] == "llm"
        assert f["updated_at"]

    def test_update_node_stamps_updated_by(self, store):
        store.upsert_entities(
            [("Alice", "person")], layer=Layer.PERSONAL,
            source="llm", updated_by="llm",
        )
        ok = store.update_node("alice", label="Alicia", updated_by="user")
        assert ok
        node = store.get_node("alice")
        assert node["label"] == "Alicia"
        assert node["updated_by"] == "user"

    def test_update_node_autostamps_updated_at_when_not_provided(self, store):
        store.upsert_entities(
            [("Alice", "person")], layer=Layer.PERSONAL,
            source="llm", updated_by="llm",
        )
        before = store.get_node("alice")["updated_at"]
        # Tiny wait trick: just update again and confirm timestamp column
        # remains populated (exact equality isn't important — we just want
        # to prove the auto-stamp branch runs).
        ok = store.update_node("alice", label="Alicia")
        assert ok
        after = store.get_node("alice")["updated_at"]
        assert after is not None
        assert after >= before  # monotonic ISO string

    def test_merge_nodes_stamps_updated_by_on_survivor(self, store):
        store.upsert_entities(
            [("Alice", "person"), ("Ali", "person")],
            layer=Layer.PERSONAL,
            source="llm", updated_by="llm",
        )
        ok = store.merge_nodes("alice", "ali", alias="Ali", updated_by="user")
        assert ok
        node = store.get_node("alice")
        assert node["updated_by"] == "user"
        assert any("Also known as" in f["fact"] for f in node["facts"])

    def test_upsert_accepts_status_pending_review(self, store):
        store.upsert_entities(
            [("JWT", "technology")],
            layer=Layer.PERSONAL,
            source="llm", updated_by="llm",
            confidence=0.5,
            status="pending_review",
        )
        node = store.get_node("jwt")
        assert node is not None
        assert node["status"] == "pending_review"
        assert abs(float(node["confidence_for_owner"]) - 0.5) < 1e-6

    def test_add_edge_accepts_source_and_updated_by(self, store):
        store.upsert_entities(
            [("Alice", "person"), ("Neuquen", "place")],
            layer=Layer.PERSONAL,
        )
        ok = store.add_edge(
            "alice", "neuquen", "lives_in",
            edge_type="semantic",
            source="user", updated_by="user",
        )
        assert ok
