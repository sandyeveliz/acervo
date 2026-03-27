"""Cross-session persistence test — verifies the graph survives between sessions.

Creates an Acervo instance, runs a few turns to build a graph, destroys the
instance, creates a new one from the same persist_path, and verifies the
graph is intact and context still works.

Requires a running LLM server. Run with:
    pytest tests/integration/test_persistence.py -m integration -v -s
"""

from __future__ import annotations

import logging
from pathlib import Path
import tempfile

import pytest

from acervo import Acervo
from acervo.graph import _make_id

log = logging.getLogger(__name__)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cross_session_persistence(llm_client):
    """Graph and context survive across Acervo instances (simulates app restart)."""
    tmp = Path(tempfile.mkdtemp()) / ".acervo"
    graph_path = tmp / "data" / "graph"
    graph_path.mkdir(parents=True, exist_ok=True)

    # ── Session 1: build knowledge ──
    memory1 = Acervo(llm=llm_client, owner="Sandy", persist_path=graph_path)

    turns_s1 = [
        ("Me llamo Sandy y soy programador, vivo en Cipolletti",
         "Hola Sandy! Cipolletti en la Patagonia, lindo lugar."),
        ("Trabajo en Alto Valle Studio, hacemos software",
         "Genial, que tipo de proyectos hacen?"),
        ("Estamos haciendo un SaaS que se llama Chequear, verificacion con NFC",
         "Interesante, NFC tiene mucho potencial."),
        ("El stack es React con Vite y Supabase como backend",
         "React + Supabase es un buen stack moderno."),
        ("Tambien tengo un proyecto personal con Python que se llama Acervo",
         "Un proyecto de AI, interesante."),
    ]

    history = []
    for user_msg, assistant_msg in turns_s1:
        await memory1.prepare(user_msg, history)
        await memory1.process(user_msg, assistant_msg)
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": assistant_msg})

    # Capture session 1 state
    s1_node_count = memory1.graph.node_count
    s1_edge_count = memory1.graph.edge_count
    s1_nodes = {n.get("label", "").lower() for n in memory1.graph.get_all_nodes()}

    log.info("Session 1: %d nodes, %d edges", s1_node_count, s1_edge_count)
    log.info("Session 1 nodes: %s", sorted(s1_nodes))

    assert s1_node_count >= 3, f"Session 1 should have >= 3 nodes, got {s1_node_count}"

    # ── Destroy instance (simulates app shutdown) ──
    del memory1

    # ── Session 2: new instance, same persist_path ──
    memory2 = Acervo(llm=llm_client, owner="Sandy", persist_path=graph_path)

    # Verify graph survived
    s2_node_count = memory2.graph.node_count
    s2_edge_count = memory2.graph.edge_count
    s2_nodes = {n.get("label", "").lower() for n in memory2.graph.get_all_nodes()}

    log.info("Session 2: %d nodes, %d edges", s2_node_count, s2_edge_count)

    assert s2_node_count == s1_node_count, (
        f"Node count changed: session1={s1_node_count}, session2={s2_node_count}"
    )
    assert s2_edge_count == s1_edge_count, (
        f"Edge count changed: session1={s1_edge_count}, session2={s2_edge_count}"
    )
    assert s2_nodes == s1_nodes, (
        f"Node labels changed: lost={s1_nodes - s2_nodes}, new={s2_nodes - s1_nodes}"
    )

    # Verify context retrieval works in session 2 (no history passed — fresh session)
    # Use a query that references known entities so the topic detector can activate nodes
    prep = await memory2.prepare("Contame sobre Sandy y Chequear", [])
    log.info(
        "Session 2 recall: has_context=%s, warm_tokens=%d, topic=%s",
        prep.has_context, prep.warm_tokens, getattr(prep, "topic", "?"),
    )

    # The graph persists and context is gathered, but has_context may be False
    # because all chunks are "conversation" (unverified). Check warm_content directly.
    warm = (prep.warm_content or "").lower()
    log.info("Session 2 warm_content length: %d", len(warm))

    assert len(warm) > 0, (
        "Session 2 should have warm_content from persisted graph, got empty string"
    )

    # At least one known entity should appear in warm context
    known_entities = ["sandy", "cipolletti", "alto valle", "chequear", "acervo"]
    found = [e for e in known_entities if e in warm]
    log.info("Session 2 warm mentions: %s", found)
    assert len(found) >= 1, (
        f"Session 2 warm context should mention at least 1 known entity, "
        f"found none. Warm content: {warm[:200]}"
    )

    # ── Session 2: add more knowledge ──
    await memory2.prepare("Mi companiero Facu hace el mobile con React Native", [])
    await memory2.process(
        "Mi companiero Facu hace el mobile con React Native",
        "React Native es bueno para mobile.",
    )

    s2_final_nodes = memory2.graph.node_count
    log.info("Session 2 after new turn: %d nodes", s2_final_nodes)
    assert s2_final_nodes >= s2_node_count, "Adding knowledge should not lose nodes"

    # ── Destroy and verify once more ──
    del memory2
    memory3 = Acervo(llm=llm_client, owner="Sandy", persist_path=graph_path)
    s3_node_count = memory3.graph.node_count
    assert s3_node_count == s2_final_nodes, (
        f"Session 3 lost nodes: expected {s2_final_nodes}, got {s3_node_count}"
    )

    print(f"\n{'=' * 60}")
    print(f"  Cross-Session Persistence Test")
    print(f"  Session 1: {s1_node_count} nodes, {s1_edge_count} edges")
    print(f"  Session 2: graph intact, context recall works")
    print(f"  Session 2+: added knowledge, now {s2_final_nodes} nodes")
    print(f"  Session 3: {s3_node_count} nodes (all persisted)")
    print(f"  Warm context mentions: {found}")
    print(f"{'=' * 60}\n")

    # Cleanup
    import shutil
    shutil.rmtree(tmp.parent, ignore_errors=True)
