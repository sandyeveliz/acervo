"""Tests for acervo.graph — TopicGraph with NodeMeta integration."""

import json
import tempfile
from pathlib import Path

from acervo.graph import TopicGraph, _make_id
from acervo.layers import Layer, NodeMeta


class TestGraphWithNodeMeta:
    def _make_graph(self) -> tuple[TopicGraph, Path]:
        """Create a fresh graph in a temp directory."""
        tmp = Path(tempfile.mkdtemp()) / "graph"
        return TopicGraph(tmp), tmp

    def test_upsert_universal_node(self):
        """Cipolletti -> Layer.UNIVERSAL, source='world'."""
        graph, _ = self._make_graph()
        graph.upsert_entities(
            entities=[("Cipolletti", "Place")],
            facts=[("Cipolletti", "City in Rio Negro, Argentina", "world")],
            layer=Layer.UNIVERSAL,
            source="world",
        )
        node = graph.get_node("cipolletti")
        assert node is not None
        assert node["label"] == "Cipolletti"
        assert node["type"] == "Place"
        assert node["layer"] == "UNIVERSAL"
        assert node["source"] == "world"
        assert node["owner"] is None

    def test_upsert_personal_node(self):
        """Altovallestudio -> Layer.PERSONAL, source='user_assertion', owner='Sandy'."""
        graph, _ = self._make_graph()
        graph.upsert_entities(
            entities=[("Altovallestudio", "Organization")],
            facts=[("Altovallestudio", "Sandy's company", "user")],
            layer=Layer.PERSONAL,
            source="user_assertion",
            owner="Sandy",
        )
        node = graph.get_node("altovallestudio")
        assert node is not None
        assert node["label"] == "Altovallestudio"
        assert node["type"] == "Organization"
        assert node["layer"] == "PERSONAL"
        assert node["source"] == "user_assertion"
        assert node["owner"] == "Sandy"
        assert node["confidence_for_owner"] == 1.0

    def test_unknown_type_marked_incomplete(self):
        """Entities with unknown types get status='incomplete'."""
        graph, _ = self._make_graph()
        graph.upsert_entities(
            entities=[("SomeEntity", "weird_type")],
        )
        node = graph.get_node("someentity")
        assert node is not None
        assert node["type"] == "Unknown"
        assert node["pending_fields"] == ["type"]

    def test_persistence_roundtrip(self):
        """Nodes persist to JSON and reload correctly."""
        graph, tmp = self._make_graph()
        graph.upsert_entities(
            entities=[("Sandy", "Person")],
            layer=Layer.PERSONAL,
            source="user_assertion",
            owner="Sandy",
        )

        # Reload from disk
        graph2 = TopicGraph(tmp)
        node = graph2.get_node("sandy")
        assert node is not None
        assert node["layer"] == "PERSONAL"
        assert node["owner"] == "Sandy"

    def test_migrate_legacy_node(self):
        """Legacy nodes without owner get migrated cleanly."""
        graph, tmp = self._make_graph()

        # Write a legacy node without owner field
        legacy = [{
            "id": "old_node",
            "label": "OldNode",
            "type": "Person",
            "layer": "PERSONAL",
            "source": "user_assertion",
            "confidence_for_owner": 1.0,
            "status": "complete",
            "pending_fields": [],
            "created_at": "2026-01-01",
            "last_active": "2026-01-01",
            "session_count": 1,
            "attributes": {},
            "facts": [],
        }]
        tmp.mkdir(parents=True, exist_ok=True)
        (tmp / "nodes.json").write_text(json.dumps(legacy), encoding="utf-8")
        (tmp / "edges.json").write_text("[]", encoding="utf-8")

        graph2 = TopicGraph(tmp)
        node = graph2.get_node("old_node")
        assert node is not None
        assert node["owner"] is None  # migrated
        assert node["layer"] == "PERSONAL"


class TestExportImport:
    def _make_graph(self) -> tuple[TopicGraph, Path]:
        tmp = Path(tempfile.mkdtemp()) / "graph"
        return TopicGraph(tmp), tmp

    def test_export_json_structure(self):
        graph, _ = self._make_graph()
        graph.upsert_entities(
            entities=[("Batman", "Character"), ("DC Universe", "Universe")],
            relations=[("Batman", "DC Universe", "part_of")],
            facts=[("Batman", "Created in 1939", "user")],
        )
        data = graph.export_json()
        assert "metadata" in data
        assert data["metadata"]["version"] == "0.2.0"
        assert data["metadata"]["node_count"] == 2
        assert data["metadata"]["edge_count"] >= 1
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) >= 1

    def test_export_import_roundtrip(self):
        """Export -> new graph -> import -> identical data."""
        graph, _ = self._make_graph()
        graph.upsert_entities(
            entities=[("Batman", "Character"), ("Gotham", "Place")],
            relations=[("Batman", "Gotham", "set_in")],
            facts=[("Batman", "Dark Knight", "user")],
        )
        exported = graph.export_json()

        # Import into fresh graph
        graph2, _ = self._make_graph()
        n, e = graph2.import_json(exported, mode="replace")
        assert n == 2
        assert e >= 1

        node = graph2.get_node("batman")
        assert node is not None
        assert node["label"] == "Batman"
        assert any(f["fact"] == "Dark Knight" for f in node["facts"])

    def test_import_merge_mode(self):
        """Merge mode upserts without losing existing data."""
        graph, _ = self._make_graph()
        graph.upsert_entities(
            entities=[("Batman", "Character")],
            facts=[("Batman", "Created in 1939", "user")],
        )

        # Import additional data via merge
        import_data = {
            "nodes": [{
                "id": "batman",
                "label": "Batman",
                "type": "Character",
                "facts": [{"fact": "Also known as Bruce Wayne", "date": "2026-01-01", "session": "s_import", "source": "user"}],
                "last_active": "2099-01-01",
                "status": "cold",
                "layer": "UNIVERSAL",
                "source": "world",
                "confidence_for_owner": 1.0,
                "pending_fields": [],
                "created_at": "2026-01-01",
                "session_count": 1,
                "attributes": {},
            }],
            "edges": [],
        }
        n, e = graph.import_json(import_data, mode="merge")
        assert n == 1

        node = graph.get_node("batman")
        fact_texts = [f["fact"] for f in node["facts"]]
        assert "Created in 1939" in fact_texts
        assert "Also known as Bruce Wayne" in fact_texts
        assert node["last_active"] == "2099-01-01"


class TestGraphQuality:
    def _make_graph(self) -> TopicGraph:
        tmp = Path(tempfile.mkdtemp()) / "graph"
        return TopicGraph(tmp)

    def test_self_referential_edges_rejected(self):
        """Self-referential edges should be silently dropped."""
        graph = self._make_graph()
        graph.upsert_entities(
            entities=[("Batman", "Character")],
            relations=[("Batman", "Batman", "alias_of")],
        )
        edges = graph.get_edges_for("batman")
        # Should have no self-referential edge
        self_refs = [e for e in edges if e["source"] == e["target"]]
        assert len(self_refs) == 0

    def test_co_mentioned_weight_capped(self):
        """co_mentioned edge weight should not exceed 10.0."""
        graph = self._make_graph()
        # Upsert same pair many times to inflate weight
        for _ in range(25):
            graph.upsert_entities(
                entities=[("Batman", "Character"), ("Robin", "Character")],
            )
        edges = [
            e for e in graph.get_edges_for("batman")
            if e.get("relation") == "co_mentioned"
        ]
        assert len(edges) == 1
        assert edges[0]["weight"] <= 10.0

    def test_edges_have_last_active(self):
        """New edges should have a last_active timestamp."""
        graph = self._make_graph()
        graph.upsert_entities(
            entities=[("Batman", "Character"), ("Gotham", "Place")],
            relations=[("Batman", "Gotham", "set_in")],
        )
        for edge in graph.get_edges_for("batman"):
            assert "last_active" in edge
