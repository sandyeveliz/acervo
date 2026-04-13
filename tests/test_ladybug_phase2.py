"""Phase 2 tests for LadybugGraphStore.

Covers:
    - Schema detection (new fields present, old-schema error path)
    - entity_similarity_search over name_embedding
    - fact_fulltext_search over Fact.fact_text
    - invalidate_fact (expired_at / invalid_at)
    - set_entity_embedding persistence
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest

from acervo.adapters.ladybug_store import LadybugGraphStore
from acervo.layers import Layer


@pytest.fixture()
def store(tmp_path: Path) -> LadybugGraphStore:
    return LadybugGraphStore(tmp_path / "graph.db")


# ── entity_similarity_search ────────────────────────────────────────────────


def test_entity_similarity_search_empty_graph_returns_empty(store):
    assert store.entity_similarity_search([0.1, 0.2, 0.3]) == []


def test_entity_similarity_search_empty_query_returns_empty(store):
    store.upsert_entities([("Sandy", "Person")], layer=Layer.PERSONAL)
    assert store.entity_similarity_search([]) == []


def test_entity_similarity_search_finds_nearest(store):
    # Seed three entities with handcrafted orthogonal embeddings.
    store.upsert_entities(
        [("Alice", "Person"), ("Bob", "Person"), ("Carol", "Person")],
        layer=Layer.PERSONAL,
    )
    store.set_entity_embedding("alice", [1.0, 0.0, 0.0])
    store.set_entity_embedding("bob", [0.0, 1.0, 0.0])
    store.set_entity_embedding("carol", [0.0, 0.0, 1.0])

    hits = store.entity_similarity_search([0.95, 0.05, 0.0], limit=3, min_score=0.5)
    assert len(hits) >= 1
    top_node, top_score = hits[0]
    assert top_node["label"] == "Alice"
    assert top_score > 0.9


def test_entity_similarity_search_respects_min_score(store):
    store.upsert_entities([("Alice", "Person")], layer=Layer.PERSONAL)
    store.set_entity_embedding("alice", [1.0, 0.0, 0.0])

    # Orthogonal query → cosine 0 → dropped by min_score
    hits = store.entity_similarity_search([0.0, 1.0, 0.0], min_score=0.5)
    assert hits == []


def test_entity_similarity_search_respects_limit(store):
    store.upsert_entities(
        [(f"Node{i}", "Person") for i in range(5)],
        layer=Layer.PERSONAL,
    )
    for i in range(5):
        # Everything is similar to the query vector, with slightly
        # decreasing similarity as i grows.
        emb = [1.0 - i * 0.05, 0.0, 0.0]
        store.set_entity_embedding(f"node{i}", emb)

    hits = store.entity_similarity_search([1.0, 0.0, 0.0], limit=3, min_score=0.0)
    assert len(hits) == 3


# ── fact_fulltext_search ────────────────────────────────────────────────────


def test_fact_fulltext_search_empty_query(store):
    assert store.fact_fulltext_search("") == []
    assert store.fact_fulltext_search("   ") == []


def test_fact_fulltext_search_substring_match(store):
    store.upsert_entities(
        [("Sandy", "Person")],
        facts=[("Sandy", "Sandy vive en Cipolletti desde 2020", "user")],
        layer=Layer.PERSONAL,
    )
    hits = store.fact_fulltext_search("Cipolletti")
    assert len(hits) == 1
    assert "Cipolletti" in hits[0]["fact"]


def test_fact_fulltext_search_token_overlap(store):
    store.upsert_entities(
        [("Butaco", "Project")],
        facts=[
            ("Butaco", "Butaco deploy on firebase with angular monorepo", "user"),
            ("Butaco", "Butaco uses capacitor for mobile apps", "user"),
        ],
        layer=Layer.PERSONAL,
    )
    hits = store.fact_fulltext_search("firebase deploy monorepo")
    # First fact has all 3 query tokens, second fact has 0 matches
    assert len(hits) == 1
    assert "firebase" in hits[0]["fact"].lower()


def test_fact_fulltext_search_nothing_matches(store):
    store.upsert_entities(
        [("Sandy", "Person")],
        facts=[("Sandy", "Sandy lives in the city", "user")],
        layer=Layer.PERSONAL,
    )
    assert store.fact_fulltext_search("quantum entanglement") == []


# ── invalidate_fact ─────────────────────────────────────────────────────────


def test_invalidate_fact_unknown_id_returns_false(store):
    assert store.invalidate_fact("nonexistent_id", expired_at="2026-04-12") is False


def test_invalidate_fact_empty_id_returns_false(store):
    assert store.invalidate_fact("", expired_at="2026-04-12") is False


def test_invalidate_fact_sets_expired_at(store):
    store.upsert_entities(
        [("Sandy", "Person")],
        facts=[("Sandy", "Sandy vive en Cipolletti", "user")],
        layer=Layer.PERSONAL,
    )
    # Grab the fact id that _add_fact generated.
    node = store.get_node("sandy")
    assert node is not None
    fact_id = node["facts"][0]["fact_id"]

    ok = store.invalidate_fact(fact_id, expired_at="2026-04-12T00:00:00Z")
    assert ok is True

    # Re-read — expired_at should be populated.
    refreshed = store.get_node("sandy")
    assert refreshed["facts"][0]["expired_at"] == "2026-04-12T00:00:00Z"
    assert refreshed["facts"][0]["invalid_at"] is None


def test_invalidate_fact_sets_both_expired_and_invalid(store):
    store.upsert_entities(
        [("Sandy", "Person")],
        facts=[("Sandy", "Sandy trabaja en Acme", "user")],
        layer=Layer.PERSONAL,
    )
    node = store.get_node("sandy")
    fact_id = node["facts"][0]["fact_id"]

    ok = store.invalidate_fact(
        fact_id,
        expired_at="2026-04-12T00:00:00Z",
        invalid_at="2025-09-01T00:00:00Z",
    )
    assert ok is True

    refreshed = store.get_node("sandy")
    assert refreshed["facts"][0]["expired_at"] == "2026-04-12T00:00:00Z"
    assert refreshed["facts"][0]["invalid_at"] == "2025-09-01T00:00:00Z"


# ── set_entity_embedding ────────────────────────────────────────────────────


def test_set_entity_embedding_unknown_node_returns_false(store):
    assert store.set_entity_embedding("ghost", [0.1, 0.2]) is False


def test_set_entity_embedding_empty_inputs_return_false(store):
    store.upsert_entities([("Sandy", "Person")], layer=Layer.PERSONAL)
    assert store.set_entity_embedding("", [0.1]) is False
    assert store.set_entity_embedding("sandy", []) is False


def test_set_entity_embedding_roundtrip_via_similarity_search(store):
    store.upsert_entities([("Sandy", "Person")], layer=Layer.PERSONAL)
    store.set_entity_embedding("sandy", [0.6, 0.8, 0.0])

    hits = store.entity_similarity_search([0.6, 0.8, 0.0], min_score=0.99)
    assert len(hits) == 1
    assert hits[0][0]["label"] == "Sandy"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-6)


# ── Bi-temporal Fact DDL roundtrip ──────────────────────────────────────────


def test_new_fact_exposes_bi_temporal_fields(store):
    store.upsert_entities(
        [("Sandy", "Person")],
        facts=[("Sandy", "Sandy vive en Cipolletti", "user")],
        layer=Layer.PERSONAL,
    )
    node = store.get_node("sandy")
    assert node is not None
    fact = node["facts"][0]
    # All new columns must be present in the returned dict (even if None)
    for col in ("valid_at", "invalid_at", "expired_at", "reference_time", "episodes"):
        assert col in fact
    # reference_time defaults to ingestion date when not explicitly passed
    assert fact["reference_time"] is not None
    assert fact["expired_at"] is None
