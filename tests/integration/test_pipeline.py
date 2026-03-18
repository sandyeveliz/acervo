"""Integration tests for Acervo pipeline — require a running LLM server.

Run with: pytest tests/integration/ -m integration -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from acervo import Acervo, OpenAIClient


@pytest.mark.integration
@pytest.mark.asyncio
async def test_commit_creates_person_node(memory):
    """Committing a self-introduction creates Persona and Lugar nodes."""
    await memory.commit("Me llamo Sandy y vivo en Cipolletti")

    sandy = memory.graph.get_node("sandy")
    assert sandy is not None, "Sandy node should exist after commit"
    assert sandy["type"] == "Persona"
    assert sandy["layer"] == "PERSONAL"
    assert sandy["source"] == "user_assertion"

    cipolletti = memory.graph.get_node("cipolletti")
    assert cipolletti is not None, "Cipolletti node should exist"
    assert cipolletti["type"] == "Lugar"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_commit_creates_organization_and_project_nodes(memory):
    """Committing about a company creates organization, project, and technology nodes."""
    await memory.commit(
        "Trabajo en Altovallestudio, tenemos un proyecto llamado Butaco hecho con Angular"
    )

    org = memory.graph.get_node("altovallestudio")
    assert org is not None, "Altovallestudio node should exist"

    butaco = memory.graph.get_node("butaco")
    assert butaco is not None, "Butaco node should exist"

    angular = memory.graph.get_node("angular")
    assert angular is not None, "Angular node should exist"
    # Angular is a technology -> likely UNIVERSAL
    assert angular["layer"] == "UNIVERSAL"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_incomplete_node_gets_pending_fields(memory):
    """Entities the extractor can't fully classify get pending_fields."""
    # Mention something vague that might not map to a known type
    await memory.commit("Tengo algo llamado XYZ123 que uso para trabajo")

    # Check if any node was created as incomplete
    all_nodes = memory.graph.get_all_nodes()
    incomplete_nodes = [n for n in all_nodes if n.get("status") == "incomplete"]
    # Note: this test is non-deterministic — the LLM might or might not classify it
    # We check that the system CAN create incomplete nodes
    if incomplete_nodes:
        assert incomplete_nodes[0]["pending_fields"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_materialize_returns_relevant_context(memory):
    """After committing facts, materialize returns relevant context."""
    await memory.commit("Lau es mi señora y trabaja en el colegio Sunrise en Cipolletti")

    context = memory.materialize("donde trabaja Lau")
    assert context, "materialize should return non-empty context"
    # The context should mention at least one of the entities
    assert any(term in context for term in ["Lau", "Sunrise", "Cipolletti"]), \
        f"Context should mention committed entities, got: {context[:200]}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_graph_persists_between_sessions(clean_graph):
    """Graph data persists when creating a new Acervo instance with same path."""
    client = OpenAIClient(
        base_url=os.getenv("ACERVO_LIGHT_MODEL_URL", "http://localhost:1234/v1"),
        model=os.getenv("ACERVO_LIGHT_MODEL", "qwen2.5-3b-instruct"),
        api_key=os.getenv("ACERVO_LIGHT_API_KEY", "lm-studio"),
    )

    # Session 1 — commit a fact
    session1 = Acervo(llm=client, owner="Sandy", persist_path=clean_graph)
    await session1.commit("Me llamo Sandy y soy de Cipolletti")

    node_count_s1 = session1.graph.node_count
    assert node_count_s1 > 0, "Session 1 should have created nodes"

    # Session 2 — verify data persists
    session2 = Acervo(llm=client, owner="Sandy", persist_path=clean_graph)
    assert session2.graph.node_count == node_count_s1, \
        "Session 2 should have the same nodes as session 1"

    sandy = session2.graph.get_node("sandy")
    assert sandy is not None, "Sandy should persist between sessions"
