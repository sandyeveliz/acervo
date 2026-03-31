"""Tests for acervo.synthesizer — context synthesis from graph nodes."""

from __future__ import annotations

import tempfile
from pathlib import Path

from acervo.graph import TopicGraph
from acervo.layers import Layer
from acervo.synthesizer import synthesize, _find_user_identity


def _make_graph() -> TopicGraph:
    """Create a fresh graph in a temp directory."""
    tmp = Path(tempfile.mkdtemp()) / "graph"
    return TopicGraph(tmp)


def test_synthesize_empty_graph():
    """Empty graph returns empty string."""
    graph = _make_graph()
    result = synthesize(graph, "hello")
    assert result == ""


def test_synthesize_active_node_with_facts():
    """Active nodes with facts are included in context."""
    graph = _make_graph()
    graph.upsert_entities(
        entities=[("Sandy", "Person")],
        facts=[("Sandy", "Sandy vive en Cipolletti", "user")],
    )
    # Pass Sandy's ID as active
    result = synthesize(graph, "Sandy", active_node_ids={"sandy"})
    assert "Sandy" in result
    assert "Cipolletti" in result


def test_synthesize_inactive_node_excluded():
    """Nodes not in active set are excluded (unless message-relevant fallback)."""
    graph = _make_graph()
    graph.upsert_entities(
        entities=[("Sandy", "Person")],
        facts=[("Sandy", "Sandy vive en Cipolletti", "user")],
    )
    # Empty active set, message doesn't mention Sandy
    result = synthesize(graph, "something else", active_node_ids=set())
    assert "Sandy" not in result


def test_synthesize_fallback_relevance():
    """Without active_node_ids, falls back to message-based relevance."""
    graph = _make_graph()
    graph.upsert_entities(
        entities=[("River Plate", "Organization")],
        facts=[("River Plate", "Club de futbol argentino", "user")],
    )
    # No active_node_ids → fallback to relevance matching
    # Not mentioned → not included
    result_no_mention = synthesize(graph, "something else")
    assert "River" not in result_no_mention

    # Mentioned → included via fallback
    result_mentioned = synthesize(graph, "que sabes de River Plate")
    assert "River" in result_mentioned


def test_synthesize_neighbor_traversal():
    """Neighbors of active nodes with facts are included."""
    graph = _make_graph()
    graph.upsert_entities(
        entities=[("Sandy", "Person"), ("Cipolletti", "Place")],
        relations=[("Sandy", "Cipolletti", "ubicado_en")],
        facts=[
            ("Sandy", "se llama Sandy", "user"),
            ("Cipolletti", "Ciudad en Rio Negro", "user"),
        ],
    )
    result = synthesize(graph, "Sandy", active_node_ids={"sandy"})
    # Sandy is active -> Cipolletti is a neighbor with facts -> both included
    assert "Sandy" in result
    assert "Cipolletti" in result


def test_find_user_identity():
    """Detects user identity from facts containing name patterns."""
    graph = _make_graph()
    graph.upsert_entities(
        entities=[("Sandy", "Person")],
        facts=[("Sandy", "my name is Sandy", "user")],
    )
    identity = _find_user_identity(graph)
    assert identity == "Sandy"


def test_find_user_identity_none():
    """Returns None when no identity facts exist."""
    graph = _make_graph()
    graph.upsert_entities(
        entities=[("Cipolletti", "Place")],
        facts=[("Cipolletti", "Ciudad en Rio Negro", "user")],
    )
    identity = _find_user_identity(graph)
    assert identity is None
