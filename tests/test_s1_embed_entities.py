"""Tests for the S1 batch entity-embedding helper ``_embed_new_entities``.

This function is Phase 2's entry point for attaching name embeddings to
validated entities before they're persisted and before entity_resolution
uses them for semantic pre-filter.
"""

from __future__ import annotations

import pytest

from acervo.extractor import Entity
from acervo.s1_unified import _embed_new_entities


class _FakeBatchEmbedder:
    def __init__(self, vectors: list[list[float]]):
        self._vectors = vectors
        self.calls: list[list[str]] = []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return list(self._vectors)


class _FailingEmbedder:
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedder offline")


@pytest.mark.asyncio
async def test_embeds_only_new_entities():
    entities = [
        Entity(name="Sandy Veliz", type="person", attributes={}),
        Entity(
            name="Cipolletti",
            type="place",
            attributes={"_existing_id": "cipolletti_uuid"},
        ),  # already merged → skipped
        Entity(name="", type="person", attributes={}),  # empty name → skipped
    ]
    embedder = _FakeBatchEmbedder([[0.1, 0.2, 0.3]])

    await _embed_new_entities(entities, embedder)

    # Only Sandy was sent to the embedder
    assert embedder.calls == [["Sandy Veliz"]]
    assert entities[0].attributes["name_embedding"] == [0.1, 0.2, 0.3]
    assert "name_embedding" not in entities[1].attributes
    assert "name_embedding" not in entities[2].attributes


@pytest.mark.asyncio
async def test_already_embedded_entity_not_reembedded():
    entities = [
        Entity(
            name="Sandy",
            type="person",
            attributes={"name_embedding": [0.5, 0.5]},
        ),
    ]
    embedder = _FakeBatchEmbedder([[9.9, 9.9]])

    await _embed_new_entities(entities, embedder)

    # Embedder never called; existing vector preserved.
    assert embedder.calls == []
    assert entities[0].attributes["name_embedding"] == [0.5, 0.5]


@pytest.mark.asyncio
async def test_embedder_failure_does_not_raise():
    entities = [Entity(name="Sandy", type="person", attributes={})]
    embedder = _FailingEmbedder()

    # Must not raise — S1 should degrade gracefully and just skip embedding.
    await _embed_new_entities(entities, embedder)

    assert "name_embedding" not in entities[0].attributes


@pytest.mark.asyncio
async def test_vector_count_mismatch_skips_attach():
    entities = [
        Entity(name="Sandy", type="person", attributes={}),
        Entity(name="Alice", type="person", attributes={}),
    ]
    # Embedder returns one vector for two inputs — the helper must refuse
    # to attach anything rather than associate the wrong vector.
    embedder = _FakeBatchEmbedder([[0.1, 0.2]])

    await _embed_new_entities(entities, embedder)

    assert "name_embedding" not in entities[0].attributes
    assert "name_embedding" not in entities[1].attributes


@pytest.mark.asyncio
async def test_empty_entity_list_is_noop():
    embedder = _FakeBatchEmbedder([])
    await _embed_new_entities([], embedder)
    assert embedder.calls == []
