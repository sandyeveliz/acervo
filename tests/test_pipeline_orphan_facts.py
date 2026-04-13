"""Regression test for the "orphan fact" bug in pipeline._persist_s1_entities.

Root cause discovered in the Phase 3 integration benchmark:

    casa: 7/49 turns passed
      Entity acc:   62% / 76%
      Fact acc:     0% / 42%          ← killer
      Facts:        62/69 (drop=10%)

S1 extracted 62 facts, 90% survived validation, but ``fact_acc=0% exact``
because the pipeline's entity-fact attachment loop only walks entities in
the CURRENT extraction:

    for entity in extraction.entities:
        entity_facts = [f for f in facts if f.entity == entity.name]
        graph.upsert_entities([(entity.name, ...)], None, entity_facts)

When the LLM sees Sandy in EXISTING NODES and correctly chooses not to
re-emit her (using ``existing_id``), but still extracts a fact like
``{entity: "Sandy", fact: "pagó el terreno"}``, that fact is silently
dropped because Sandy is not in ``extraction.entities``.

T01 and T02 of the casa benchmark worked because all entities were new.
From T03 onward the pattern reverses and most facts reference existing
entities, which is why the exact fact accuracy collapsed to 0%.

The fix: after the main loop, walk any unattached facts and upsert them
against existing graph nodes that match by canonical id or fuzzy label.
"""

from __future__ import annotations

import tempfile
import types
from pathlib import Path

from acervo.domain.pipeline import Pipeline
from acervo.extractor import Entity, ExtractedFact, ExtractionResult
from acervo.graph import TopicGraph
from acervo.layers import Layer


def _make_graph() -> TopicGraph:
    tmp = Path(tempfile.mkdtemp()) / "graph"
    return TopicGraph(tmp)


def _make_pipeline(graph: TopicGraph) -> Pipeline:
    """Build just enough of a Pipeline to exercise _persist_s1_entities.

    The real ``Pipeline.__init__`` requires ~10 collaborators we don't
    need here. We sidestep it by instantiating a bare object and patching
    the two attributes the method actually reads — ``self._graph`` and
    ``self._owner``. This keeps the regression test tight and fast (no
    LLM, no async, no real dependencies).
    """
    pipeline = object.__new__(Pipeline)
    pipeline._graph = graph
    pipeline._owner = ""
    return pipeline


def test_orphan_fact_attached_to_existing_node_by_exact_id():
    """Fact references an entity that exists in the graph but is NOT in the
    current S1 extraction. The orphan-fact pass should fuzzy-match by
    canonical id and persist the fact against the existing node.
    """
    graph = _make_graph()
    # Seed the graph with Sandy from a previous "turn".
    graph.upsert_entities(
        [("Sandy", "Person")],
        layer=Layer.PERSONAL,
        source="user_assertion",
    )
    assert graph.get_node("sandy") is not None
    initial_facts = len(graph.get_node("sandy").get("facts", []))

    pipeline = _make_pipeline(graph)

    # Simulate a new turn where the LLM chose NOT to re-emit Sandy
    # (because she's in EXISTING NODES) but DID extract a fact about her.
    extraction = ExtractionResult(
        entities=[],  # ← no new entities; Sandy is not here
        relations=[],
        facts=[
            ExtractedFact(
                entity="Sandy",
                fact="Sandy firmó el contrato con el arquitecto",
                source="user",
                speaker="user",
            )
        ],
    )

    pipeline._persist_s1_entities(extraction, "irrelevant", "topic")

    sandy = graph.get_node("sandy")
    new_facts = len(sandy.get("facts", []))
    assert new_facts == initial_facts + 1, (
        f"Orphan fact should have been attached to Sandy. "
        f"Facts before={initial_facts}, after={new_facts}"
    )
    fact_texts = [f.get("fact", "") for f in sandy.get("facts", [])]
    assert any("contrato con el arquitecto" in f for f in fact_texts)


def test_orphan_fact_attached_via_fuzzy_label_match():
    """Fact references 'Sandy Veliz' but the graph has 'sandy veliz' — fuzzy
    match on the label should still resolve.
    """
    graph = _make_graph()
    graph.upsert_entities(
        [("Sandy Veliz", "Person")],
        layer=Layer.PERSONAL,
        source="user_assertion",
    )
    pipeline = _make_pipeline(graph)

    extraction = ExtractionResult(
        entities=[],
        relations=[],
        facts=[
            ExtractedFact(
                entity="sandy veliz",   # different case
                fact="envió el último diseño el viernes",
                source="user",
                speaker="user",
            )
        ],
    )
    pipeline._persist_s1_entities(extraction, "irrelevant", "topic")

    sandy = graph.get_node("sandy_veliz")
    assert sandy is not None
    fact_texts = [f.get("fact", "") for f in sandy.get("facts", [])]
    assert any("diseño el viernes" in f for f in fact_texts)


def test_orphan_fact_with_no_matching_entity_is_dropped():
    """When the fact references an entity that doesn't exist anywhere,
    the orphan-fact pass drops it silently (with a log message).
    """
    graph = _make_graph()
    graph.upsert_entities(
        [("Alice", "Person")],
        layer=Layer.PERSONAL,
        source="user_assertion",
    )
    pipeline = _make_pipeline(graph)

    extraction = ExtractionResult(
        entities=[],
        relations=[],
        facts=[
            ExtractedFact(
                entity="Bob",  # not in the graph
                fact="Bob fue al mercado",
                source="user",
                speaker="user",
            )
        ],
    )
    pipeline._persist_s1_entities(extraction, "irrelevant", "topic")

    # Alice unchanged, no Bob created
    assert graph.get_node("alice") is not None
    assert graph.get_node("bob") is None
    alice_facts = [
        f.get("fact", "") for f in graph.get_node("alice").get("facts", [])
    ]
    assert not any("Bob" in f for f in alice_facts)


def test_orphan_fact_pass_does_not_double_persist():
    """Facts already attached in pass 1 (because their entity IS in the
    extraction) must not also be persisted in pass 2 under a fuzzy match.
    """
    graph = _make_graph()
    graph.upsert_entities(
        [("Sandy", "Person")],
        layer=Layer.PERSONAL,
        source="user_assertion",
    )
    pipeline = _make_pipeline(graph)

    extraction = ExtractionResult(
        entities=[Entity(name="Sandy", type="person", attributes={})],
        relations=[],
        facts=[
            ExtractedFact(
                entity="Sandy",
                fact="Sandy compró herramientas",
                source="user",
                speaker="user",
            )
        ],
    )
    pipeline._persist_s1_entities(extraction, "irrelevant", "topic")

    sandy = graph.get_node("sandy")
    fact_texts = [f.get("fact", "") for f in sandy.get("facts", [])]
    assert fact_texts.count("Sandy compró herramientas") == 1, (
        f"Fact was persisted twice: {fact_texts}"
    )


def test_orphan_fact_pass_handles_empty_extraction_with_no_facts():
    """Sanity: empty extraction is a no-op."""
    graph = _make_graph()
    pipeline = _make_pipeline(graph)

    extraction = ExtractionResult(entities=[], relations=[], facts=[])
    pipeline._persist_s1_entities(extraction, "irrelevant", "topic")

    assert graph.node_count == 0
