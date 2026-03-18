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
            entities=[("Cipolletti", "Lugar")],
            facts=[("Cipolletti", "City in Rio Negro, Argentina", "world")],
            layer=Layer.UNIVERSAL,
            source="world",
        )
        node = graph.get_node("cipolletti")
        assert node is not None
        assert node["label"] == "Cipolletti"
        assert node["type"] == "Lugar"
        assert node["layer"] == "UNIVERSAL"
        assert node["source"] == "world"
        assert node["owner"] is None

    def test_upsert_personal_node(self):
        """Altovallestudio -> Layer.PERSONAL, source='user_assertion', owner='Sandy'."""
        graph, _ = self._make_graph()
        graph.upsert_entities(
            entities=[("Altovallestudio", "Organización")],
            facts=[("Altovallestudio", "Sandy's company", "user")],
            layer=Layer.PERSONAL,
            source="user_assertion",
            owner="Sandy",
        )
        node = graph.get_node("altovallestudio")
        assert node is not None
        assert node["label"] == "Altovallestudio"
        assert node["type"] == "Organización"
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
            entities=[("Sandy", "Persona")],
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
            "type": "Persona",
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
