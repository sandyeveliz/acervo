"""Tests for acervo.extraction.entity_resolution — integration glue around dedup_helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

from acervo.extraction.entity_resolution import resolve_extracted_nodes


@dataclass
class _FakeEntity:
    """Minimal Entity-like stand-in (same shape as acervo.extractor.Entity)."""

    name: str
    type: str = "person"
    layer: str = ""
    attributes: dict = field(default_factory=dict)


def test_empty_inputs_return_empty():
    resolved, uuid_map, dupes = resolve_extracted_nodes([], [])
    assert resolved == []
    assert uuid_map == {}
    assert dupes == []


def test_no_existing_graph_keeps_all_as_new():
    extracted = [_FakeEntity(name="Sandy Veliz"), _FakeEntity(name="Cipolletti", type="place")]
    resolved, uuid_map, dupes = resolve_extracted_nodes(extracted, [])
    assert len(resolved) == 2
    assert len(uuid_map) == 2
    # Every UUID maps to itself since nothing to dedup against
    for k, v in uuid_map.items():
        assert k == v
    assert dupes == []


def test_exact_match_dedups_against_existing_node():
    extracted = [_FakeEntity(name="Sandy Veliz", type="person")]
    existing = [{"id": "existing_sandy", "label": "Sandy Veliz", "type": "person"}]
    resolved, uuid_map, dupes = resolve_extracted_nodes(extracted, existing)
    assert len(resolved) == 1
    assert resolved[0].uuid == "existing_sandy"
    # The extracted entity's uuid should map to the existing one
    assert "existing_sandy" in uuid_map.values()
    assert len(dupes) == 1


def test_dict_format_variations_accepted():
    extracted = [_FakeEntity(name="Butaco")]
    # Some stores return `uuid`/`name`, others return `id`/`label` — both work.
    existing = [
        {"uuid": "butaco_id", "name": "Butaco", "type": "project"},
    ]
    resolved, _uuid_map, _dupes = resolve_extracted_nodes(extracted, existing)
    assert resolved[0].uuid == "butaco_id"


def test_unmatched_entity_is_kept_as_new():
    extracted = [_FakeEntity(name="Carlos Pena")]
    existing = [{"id": "alice_id", "label": "Alice", "type": "person"}]
    resolved, uuid_map, dupes = resolve_extracted_nodes(extracted, existing)
    assert len(resolved) == 1
    # Didn't match -> Carlos Pena kept as a fresh node, uuid_map -> itself
    assert resolved[0].name == "Carlos Pena"
    assert len(dupes) == 0


def test_existing_id_attribute_is_used_as_uuid_source():
    # When the LLM hinted that the entity matches an existing graph node,
    # the extractor stamps `_existing_id` in attributes. Our adapter should
    # use that as the DedupNode uuid so the uuid_map stays consistent.
    extracted = [_FakeEntity(name="Sandy", attributes={"_existing_id": "pre_existing_uuid"})]
    resolved, uuid_map, _dupes = resolve_extracted_nodes(extracted, [])
    assert "pre_existing_uuid" in uuid_map
    assert resolved[0].uuid == "pre_existing_uuid"


# ── S1 integration: _resolve_against_graph ──────────────────────────────────


def test_resolve_against_graph_merges_and_rewrites_relations():
    """End-to-end: extracted entity matches existing graph node → relations rewritten."""
    from acervo.extractor import Entity, ExtractedFact, ExtractionResult, Relation
    from acervo.s1_unified import S1Result, TopicResult, _resolve_against_graph

    extracted_entities = [
        Entity(name="sandy veliz", type="person", layer="PERSONAL", attributes={}),
        Entity(name="Cipolletti", type="place", layer="UNIVERSAL", attributes={}),
    ]
    extracted_relations = [
        Relation(source="sandy veliz", target="Cipolletti", relation="located_in"),
    ]
    extracted_facts = [
        ExtractedFact(
            entity="sandy veliz",
            fact="Vive en Cipolletti desde 2020",
            source="user",
            speaker="user",
        ),
    ]

    result = S1Result(
        topic=TopicResult(action="same", label="personal"),
        extraction=ExtractionResult(
            entities=extracted_entities,
            relations=extracted_relations,
            facts=extracted_facts,
        ),
    )

    # Existing graph has a canonical "Sandy Veliz" (different casing)
    existing_nodes = [
        {"id": "sandy_existing", "label": "Sandy Veliz", "type": "person"},
        {"id": "cipolletti_existing", "label": "Cipolletti", "type": "place"},
    ]

    out = _resolve_against_graph(result, existing_nodes)

    # Extracted entity got canonical name + _existing_id stamp
    assert out.extraction.entities[0].name == "Sandy Veliz"
    assert out.extraction.entities[0].attributes["_existing_id"] == "sandy_existing"
    assert out.extraction.entities[1].attributes["_existing_id"] == "cipolletti_existing"

    # Relation source rewritten to the canonical name
    assert out.extraction.relations[0].source == "Sandy Veliz"
    assert out.extraction.relations[0].target == "Cipolletti"

    # Fact entity rewritten too
    assert out.extraction.facts[0].entity == "Sandy Veliz"


def test_resolve_against_graph_no_matches_leaves_result_untouched():
    from acervo.extractor import Entity, ExtractionResult
    from acervo.s1_unified import S1Result, TopicResult, _resolve_against_graph

    entities = [Entity(name="Butaco", type="project", attributes={})]
    result = S1Result(
        topic=TopicResult(action="same", label="personal"),
        extraction=ExtractionResult(entities=entities, relations=[], facts=[]),
    )
    existing = [{"id": "alice_id", "label": "Alice", "type": "person"}]

    out = _resolve_against_graph(result, existing)

    # No match — Butaco entity unchanged, no _existing_id stamped
    assert out.extraction.entities[0].name == "Butaco"
    assert "_existing_id" not in out.extraction.entities[0].attributes


# ── Phase 2: semantic pre-filter ────────────────────────────────────────────


class _FakeSemanticGraph:
    """Graph stub that exposes entity_similarity_search for tests."""

    def __init__(self, hits: list[tuple[dict, float]]):
        self._hits = hits
        self.calls: list[list[float]] = []

    def entity_similarity_search(self, embedding, *, limit=15, min_score=0.6):
        self.calls.append(list(embedding))
        return self._hits


def test_semantic_pre_filter_narrows_candidates_and_merges():
    from acervo.extraction.entity_resolution import resolve_extracted_nodes

    # Extracted entity with a name_embedding already attached (S1 computed it).
    ext = [_FakeEntity(
        name="Sandy Veliz",
        type="person",
        attributes={"name_embedding": [0.9, 0.1, 0.0]},
    )]
    # Fake graph returns a single candidate that matches by exact name.
    graph = _FakeSemanticGraph(hits=[
        ({"id": "sandy_canonical", "label": "Sandy Veliz", "type": "person"}, 0.95),
    ])

    # `existing` is intentionally empty — the semantic search path must feed
    # the candidates entirely via the graph stub.
    resolved, _uuid_map, dupes = resolve_extracted_nodes(ext, [], graph=graph)

    assert len(graph.calls) == 1
    assert resolved[0].uuid == "sandy_canonical"
    assert len(dupes) == 1


def test_semantic_pre_filter_skipped_when_entity_has_no_embedding():
    from acervo.extraction.entity_resolution import resolve_extracted_nodes

    # Extracted entity has NO embedding — the semantic path falls back to
    # the full `existing` list (and must still find the exact match).
    ext = [_FakeEntity(name="Sandy", type="person", attributes={})]
    graph = _FakeSemanticGraph(hits=[])
    existing = [{"id": "existing_sandy", "label": "Sandy", "type": "person"}]

    resolved, _uuid_map, _dupes = resolve_extracted_nodes(
        ext, existing, graph=graph,
    )

    assert graph.calls == []  # semantic search never called — no embedding
    assert resolved[0].uuid == "existing_sandy"


def test_semantic_pre_filter_empty_hits_leaves_entity_as_new():
    from acervo.extraction.entity_resolution import resolve_extracted_nodes

    ext = [_FakeEntity(
        name="Butaco",
        type="project",
        attributes={"name_embedding": [0.1, 0.9, 0.0]},
    )]
    graph = _FakeSemanticGraph(hits=[])  # no neighbours

    resolved, uuid_map, dupes = resolve_extracted_nodes(ext, [], graph=graph)

    assert len(graph.calls) == 1
    assert len(resolved) == 1
    assert resolved[0].name == "Butaco"
    assert dupes == []
    # The entity is its own canonical (uuid_map points to itself)
    assert list(uuid_map.values())[0] == resolved[0].uuid
