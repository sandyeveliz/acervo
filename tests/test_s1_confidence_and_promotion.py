"""Unit tests for v0.6.1 Change 2:
- S1 parser reads `confidence` field from entities/relations
- Entities with confidence < 0.7 bypass the garbage filter
- Pipeline persists low-confidence entities with status="pending_review"
- S1.5 _auto_promote_pending_entities graduates them when evidence accrues
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from acervo.graph.topic_graph import TopicGraph
from acervo.layers import Layer
from acervo.s1_unified import _parse_s1_response, _validate_s1
from acervo.s1_5_graph_update import _auto_promote_pending_entities


# ── S1 parser — confidence field ────────────────────────────────────────────


class TestS1ParserConfidence:
    # Use "Butaco" (a non-jargon label) so the parser's tech-jargon
    # heuristic doesn't override the default-confidence branches we want
    # to test in isolation.
    def _make_response(self, entity_extras: dict) -> str:
        payload = {
            "topic": {"action": "same", "label": "proyectos"},
            "intent": {"type": "specific", "retrieval": "with_chunks"},
            "entities": [{
                "id": "butaco",
                "label": "Butaco",
                "type": "project",
                "layer": "PERSONAL",
                "description": "ERP client project",
                **entity_extras,
            }],
            "relations": [],
            "facts": [],
        }
        return json.dumps(payload)

    def test_confidence_is_parsed(self):
        result = _parse_s1_response(self._make_response({"confidence": 0.4}))
        assert len(result.extraction.entities) == 1
        assert result.extraction.entities[0].confidence == 0.4

    def test_missing_confidence_defaults_to_one(self):
        result = _parse_s1_response(self._make_response({}))
        assert result.extraction.entities[0].confidence == 1.0

    def test_malformed_confidence_defaults_to_one(self):
        result = _parse_s1_response(self._make_response({"confidence": "high"}))
        assert result.extraction.entities[0].confidence == 1.0

    def test_out_of_range_confidence_is_clamped(self):
        result = _parse_s1_response(self._make_response({"confidence": 2.5}))
        assert result.extraction.entities[0].confidence == 1.0

    def test_relation_confidence_is_parsed(self):
        payload = {
            "topic": {"action": "same", "label": "x"},
            "intent": {"type": "specific", "retrieval": "with_chunks"},
            "entities": [
                {"id": "a", "label": "Alice", "type": "person", "layer": "PERSONAL"},
                {"id": "b", "label": "Bob", "type": "person", "layer": "PERSONAL"},
            ],
            "relations": [
                {"source": "a", "target": "b", "relation": "member_of",
                 "confidence": 0.6},
            ],
            "facts": [],
        }
        result = _parse_s1_response(json.dumps(payload))
        assert len(result.extraction.relations) == 1
        assert result.extraction.relations[0].confidence == 0.6


# ── _validate_s1 — garbage bypass for low confidence ────────────────────────


class TestValidateS1GarbageBypass:
    def test_low_confidence_bypasses_garbage_filter(self):
        """'config' is on the garbage list, but LLM says confidence=0.5 so
        we should keep the entity under the v0.6.1 rule."""
        from acervo.extraction.extractor import Entity, ExtractionResult
        from acervo.s1_unified import S1Result, TopicResult
        result = S1Result(
            topic=TopicResult(action="same", label="test"),
            extraction=ExtractionResult(
                entities=[Entity(name="config", type="concept", layer="UNIVERSAL",
                                 attributes={}, confidence=0.5)],
            ),
        )
        conv_text = "Necesito ayudar con el config de Prisma"
        validated = _validate_s1(result, conv_text)
        assert len(validated.extraction.entities) == 1
        assert validated.extraction.entities[0].name == "config"

    def test_high_confidence_garbage_entity_still_rejected(self):
        """When the LLM is certain (confidence >= 0.7) we keep the garbage
        filter so we don't regress on the cases that motivated it."""
        from acervo.extraction.extractor import Entity, ExtractionResult
        from acervo.s1_unified import S1Result, TopicResult
        result = S1Result(
            topic=TopicResult(action="same", label="test"),
            extraction=ExtractionResult(
                entities=[Entity(name="config", type="concept", layer="UNIVERSAL",
                                 attributes={}, confidence=1.0)],
            ),
        )
        conv_text = "Necesito ayudar con el config de Prisma"
        validated = _validate_s1(result, conv_text)
        assert validated.extraction.entities == []


# ── TopicGraph — pending_review persistence ─────────────────────────────────


class TestPendingReviewPersistence:
    def _make_graph(self) -> TopicGraph:
        return TopicGraph(Path(tempfile.mkdtemp()) / "graph")

    def test_low_confidence_entity_persists_with_pending_review(self):
        graph = self._make_graph()
        graph.upsert_entities(
            [("JWT", "Technology")],
            layer=Layer.UNIVERSAL,
            source="llm",
            updated_by="llm",
            confidence=0.5,
            status="pending_review",
        )
        node = graph.get_node("jwt")
        assert node["status"] == "pending_review"
        assert node["confidence_for_owner"] == 0.5


# ── _auto_promote_pending_entities ──────────────────────────────────────────


class TestAutoPromotePendingEntities:
    def _make_graph(self) -> TopicGraph:
        return TopicGraph(Path(tempfile.mkdtemp()) / "graph")

    def test_promotes_after_three_sessions(self):
        graph = self._make_graph()
        graph.upsert_entities(
            [("JWT", "Technology")],
            layer=Layer.UNIVERSAL,
            source="llm",
            updated_by="llm",
            confidence=0.5,
            status="pending_review",
        )
        # Bump session count to 3 manually (each upsert only stamps 1
        # session, so we fake it for the test).
        node = graph.get_node("jwt")
        node["session_count"] = 3

        promoted = _auto_promote_pending_entities(graph)
        assert promoted == 1
        node = graph.get_node("jwt")
        assert node["status"] == "confirmed"
        assert node["confidence_for_owner"] >= 0.8

    def test_does_not_promote_below_threshold(self):
        graph = self._make_graph()
        graph.upsert_entities(
            [("JWT", "Technology")],
            layer=Layer.UNIVERSAL,
            source="llm",
            updated_by="llm",
            confidence=0.5,
            status="pending_review",
        )
        node = graph.get_node("jwt")
        node["session_count"] = 2

        promoted = _auto_promote_pending_entities(graph)
        assert promoted == 0
        assert graph.get_node("jwt")["status"] == "pending_review"

    def test_promotes_when_user_edited(self):
        graph = self._make_graph()
        graph.upsert_entities(
            [("JWT", "Technology")],
            layer=Layer.UNIVERSAL,
            source="llm",
            updated_by="llm",
            confidence=0.5,
            status="pending_review",
        )
        # Simulate a user edit via the REST API path.
        graph.update_node("jwt", label="JSON Web Token", updated_by="user")
        promoted = _auto_promote_pending_entities(graph)
        assert promoted == 1
        node = graph.get_node("jwt")
        assert node["status"] == "confirmed"
        assert node["confidence_for_owner"] == 1.0

    def test_noop_when_no_pending_nodes(self):
        graph = self._make_graph()
        graph.upsert_entities(
            [("Sandy", "Person")], layer=Layer.PERSONAL,
            source="llm", updated_by="llm",
        )
        assert _auto_promote_pending_entities(graph) == 0
