"""Tests for acervo.facade — Acervo SDK with mock LLM."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from acervo import Acervo
from acervo.layers import Layer
from tests.conftest import make_mock_llm


def _make_acervo(response: str = "{}") -> Acervo:
    """Create an Acervo instance with a mock LLM and temp graph."""
    tmp = Path(tempfile.mkdtemp()) / "graph"
    llm = make_mock_llm(response)
    return Acervo(llm=llm, owner="Sandy", persist_path=tmp)


@pytest.mark.asyncio
async def test_commit_extracts_and_persists():
    """Mock LLM -> commit -> node appears in graph."""
    response = json.dumps({
        "entities": [{"name": "Sandy", "type": "persona"}],
        "relations": [],
        "facts": [{"entity": "Sandy", "fact": "se llama Sandy", "speaker": "user"}],
    })
    memory = _make_acervo(response)
    result = await memory.commit("Me llamo Sandy", "Hola Sandy!")

    assert len(result.entities) == 1
    node = memory.graph.get_node("sandy")
    assert node is not None
    assert node["type"] == "Person"
    assert node["owner"] == "Sandy"


@pytest.mark.asyncio
async def test_commit_filters_assistant_facts():
    """Only user-spoken facts are stored in the graph."""
    response = json.dumps({
        "entities": [{"name": "Cipolletti", "type": "lugar"}],
        "relations": [],
        "facts": [
            {"entity": "Cipolletti", "fact": "Sandy vive en Cipolletti", "speaker": "user"},
            {"entity": "Cipolletti", "fact": "Cipolletti queda en Rio Negro", "speaker": "assistant"},
        ],
    })
    memory = _make_acervo(response)
    await memory.commit("Vivo en Cipolletti", "Cipolletti queda en Rio Negro")

    node = memory.graph.get_node("cipolletti")
    assert node is not None
    # Only user fact stored
    fact_texts = [f["fact"] for f in node["facts"]]
    assert "Sandy vive en Cipolletti" in fact_texts
    assert "Cipolletti queda en Rio Negro" not in fact_texts


@pytest.mark.asyncio
async def test_materialize_returns_context():
    """After commit, materialize returns relevant context text."""
    response = json.dumps({
        "entities": [{"name": "Sandy", "type": "persona"}],
        "relations": [],
        "facts": [{"entity": "Sandy", "fact": "se llama Sandy", "speaker": "user"}],
    })
    memory = _make_acervo(response)
    await memory.commit("Me llamo Sandy", "")

    context = memory.materialize("Sandy")
    assert "Sandy" in context


@pytest.mark.asyncio
async def test_find_active_node_ids():
    """_find_active_node_ids returns IDs of mentioned nodes."""
    response = json.dumps({
        "entities": [{"name": "Sandy", "type": "persona"}],
        "relations": [],
        "facts": [{"entity": "Sandy", "fact": "se llama Sandy", "speaker": "user"}],
    })
    memory = _make_acervo(response)
    await memory.commit("Me llamo Sandy", "")

    # Node exists but no runtime status on the node
    node = memory.graph.get_node("sandy")
    assert "status" not in node or node.get("status") not in ("hot", "warm", "cold")

    # Activation is ephemeral — returned as a set, not stored on node
    active = memory._find_active_node_ids("Sandy", "none")
    assert "sandy" in active

    # Different message — Sandy not mentioned
    active2 = memory._find_active_node_ids("hello world", "none")
    assert "sandy" not in active2


@pytest.mark.asyncio
async def test_commit_detects_universal_layer():
    """Entities with Lugar type get layer=UNIVERSAL via is_likely_universal."""
    response = json.dumps({
        "entities": [{"name": "Cipolletti", "type": "lugar"}],
        "relations": [],
        "facts": [{"entity": "Cipolletti", "fact": "Ciudad en Rio Negro", "speaker": "user"}],
    })
    memory = _make_acervo(response)
    await memory.commit("Cipolletti", "")

    node = memory.graph.get_node("cipolletti")
    assert node is not None
    assert node["layer"] == "UNIVERSAL"
    assert node["source"] == "world"


@pytest.mark.asyncio
async def test_owner_propagated_to_nodes():
    """Owner from Acervo constructor is propagated to graph nodes."""
    response = json.dumps({
        "entities": [{"name": "Altovallestudio", "type": "entidad"}],
        "relations": [],
        "facts": [],
    })
    memory = _make_acervo(response)
    await memory.commit("Trabajo en Altovallestudio", "")

    node = memory.graph.get_node("altovallestudio")
    assert node is not None
    assert node["owner"] == "Sandy"
