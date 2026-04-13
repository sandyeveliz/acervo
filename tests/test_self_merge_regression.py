"""Regression tests for the S1.5 self-merge bug.

Root cause discovered during the first integration run of Phase 3:

    INFO    Merged node cipolletti into cipolletti
    INFO    S1.5 merged: Cipolletti → cipolletti (same place, type corrected)

The LLM emitted ``{"from": "Cipolletti", "into": "cipolletti"}`` (display
name vs. canonical id), the parser compared the raw strings — which were
different — and accepted the merge. Downstream, ``graph.merge_nodes``
resolved both to the same canonical id via ``_make_id`` and then
silently deleted the surviving node at the bottom of the function.

Result: every single turn of every case scenario wiped the graph clean
because S1.5 always proposes a "canonicalization" merge on the new
entities, so the test saw ``graph=0n/0e`` after every turn (49/49 fails).

We close the hole in three places, each tested here:

    1. ``_parse_s1_5_response`` normalizes via ``_make_id`` before the
       equality check and drops self-merges.
    2. ``apply_s1_5_result`` double-checks resolved ids after looking up
       the graph nodes.
    3. ``TopicGraph.merge_nodes`` / ``LadybugGraphStore.merge_nodes``
       refuse to run when ``kid == aid`` — last line of defense before
       ``del self._nodes[aid]``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from acervo.graph import TopicGraph
from acervo.layers import Layer
from acervo.s1_5_graph_update import _parse_s1_5_response, apply_s1_5_result


# ── (1) Parser drops self-merges where display vs id differ only by case ──


def test_parser_drops_case_variant_self_merge():
    raw = json.dumps({
        "merges": [
            {"from": "Cipolletti", "into": "cipolletti", "reason": "canonical"},
        ],
        "new_relations": [],
        "type_corrections": [],
        "discards": [],
        "assistant_entities": [],
        "assistant_facts": [],
        "assistant_relations": [],
    })
    result = _parse_s1_5_response(raw)
    assert result.merges == []  # self-merge rejected


def test_parser_drops_accent_variant_self_merge():
    raw = json.dumps({
        "merges": [
            {"from": "Carlos Peña", "into": "carlos_pena", "reason": "same person"},
        ],
    })
    result = _parse_s1_5_response(raw)
    assert result.merges == []


def test_parser_keeps_real_merge_between_distinct_entities():
    raw = json.dumps({
        "merges": [
            {"from": "carlos_pena", "into": "carlos_arquitecto", "reason": "same person"},
        ],
    })
    result = _parse_s1_5_response(raw)
    assert len(result.merges) == 1
    assert result.merges[0].from_id == "carlos_pena"
    assert result.merges[0].into_id == "carlos_arquitecto"


def test_parser_drops_whitespace_variant_self_merge():
    raw = json.dumps({
        "merges": [
            {"from": "  cipolletti  ", "into": "cipolletti", "reason": "whitespace"},
        ],
    })
    result = _parse_s1_5_response(raw)
    assert result.merges == []


# ── (2) TopicGraph.merge_nodes refuses self-merge ─────────────────────────


def _make_graph() -> TopicGraph:
    tmp = Path(tempfile.mkdtemp()) / "graph"
    return TopicGraph(tmp)


def test_topic_graph_merge_refuses_self_by_canonical_id():
    graph = _make_graph()
    graph.upsert_entities(
        [("Cipolletti", "Place")],
        facts=[("Cipolletti", "City in Rio Negro, Argentina", "world")],
        layer=Layer.UNIVERSAL,
        source="world",
    )
    assert graph.get_node("cipolletti") is not None
    initial_count = graph.node_count

    # Both forms resolve to the same canonical id via _make_id.
    ok = graph.merge_nodes("Cipolletti", "cipolletti")
    assert ok is False, "self-merge must refuse to run"

    # Crucial: the node is still there. Before the fix, del self._nodes[aid]
    # removed the one and only node.
    assert graph.node_count == initial_count
    assert graph.get_node("cipolletti") is not None


def test_topic_graph_merge_still_works_for_real_merges():
    graph = _make_graph()
    graph.upsert_entities(
        [("Alice", "Person"), ("Alice Smith", "Person")],
        layer=Layer.PERSONAL,
    )
    assert graph.node_count == 2

    ok = graph.merge_nodes("alice_smith", "alice")
    assert ok is True
    assert graph.node_count == 1
    assert graph.get_node("alice_smith") is not None
    assert graph.get_node("alice") is None  # absorbed


# ── (3) apply_s1_5_result is defensive against resolved-id collisions ──


def test_apply_s1_5_skips_merge_when_both_sides_resolve_to_same_node():
    """Even if the parser let a self-merge through, apply_s1_5_result
    must not call graph.merge_nodes with equal canonical ids.
    """
    from acervo.s1_5_graph_update import MergeAction, S1_5Result
    from acervo.extractor import ExtractionResult

    graph = _make_graph()
    graph.upsert_entities([("Cipolletti", "Place")], layer=Layer.UNIVERSAL)
    initial_count = graph.node_count

    # Hand-build a S1_5Result with a self-merge that the parser would
    # normally have rejected. This simulates a parser regression.
    result = S1_5Result(
        merges=[MergeAction(from_id="Cipolletti", into_id="cipolletti", reason="bad")],
        assistant_extraction=ExtractionResult(),
    )

    audit = apply_s1_5_result(graph, result, owner="")

    # Merge was not applied, node survives.
    assert audit["merges_applied"] == 0
    assert graph.node_count == initial_count
    assert graph.get_node("cipolletti") is not None
