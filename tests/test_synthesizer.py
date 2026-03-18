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


def test_synthesize_hot_node_with_facts():
    """Hot nodes with facts are included in context."""
    graph = _make_graph()
    graph.upsert_entities(
        entities=[("Sandy", "Persona")],
        facts=[("Sandy", "Sandy vive en Cipolletti", "user")],
    )
    # upsert sets status to "hot"
    result = synthesize(graph, "Sandy")
    assert "Sandy" in result
    assert "Cipolletti" in result


def test_synthesize_cold_node_excluded():
    """Cold nodes are not included in context."""
    graph = _make_graph()
    graph.upsert_entities(
        entities=[("Sandy", "Persona")],
        facts=[("Sandy", "Sandy vive en Cipolletti", "user")],
    )
    # Cycle twice: hot -> warm -> cold
    graph.cycle_status()
    graph.cycle_status()

    result = synthesize(graph, "something else")
    assert "Sandy" not in result


def test_synthesize_warm_node_only_if_mentioned():
    """Warm nodes only included if the message mentions them."""
    graph = _make_graph()
    graph.upsert_entities(
        entities=[("River Plate", "Organización")],
        facts=[("River Plate", "Club de futbol argentino", "user")],
    )
    # Cycle once: hot -> warm
    graph.cycle_status()

    # Not mentioned -> not included
    result_no_mention = synthesize(graph, "something else")
    assert "River" not in result_no_mention

    # Mentioned -> included
    result_mentioned = synthesize(graph, "que sabes de River Plate")
    assert "River" in result_mentioned


def test_synthesize_neighbor_traversal():
    """Neighbors of hot nodes with facts are included."""
    graph = _make_graph()
    graph.upsert_entities(
        entities=[("Sandy", "Persona"), ("Cipolletti", "Lugar")],
        relations=[("Sandy", "Cipolletti", "ubicado_en")],
        facts=[
            ("Sandy", "se llama Sandy", "user"),
            ("Cipolletti", "Ciudad en Rio Negro", "user"),
        ],
    )
    result = synthesize(graph, "Sandy")
    # Sandy is hot -> Cipolletti is a neighbor with facts -> both included
    assert "Sandy" in result
    assert "Cipolletti" in result


def test_find_user_identity():
    """Detects user identity from facts containing name patterns."""
    graph = _make_graph()
    graph.upsert_entities(
        entities=[("Sandy", "Persona")],
        facts=[("Sandy", "el usuario se llama Sandy", "user")],
    )
    identity = _find_user_identity(graph)
    assert identity == "Sandy"


def test_find_user_identity_none():
    """Returns None when no identity facts exist."""
    graph = _make_graph()
    graph.upsert_entities(
        entities=[("Cipolletti", "Lugar")],
        facts=[("Cipolletti", "Ciudad en Rio Negro", "user")],
    )
    identity = _find_user_identity(graph)
    assert identity is None
