"""Unit tests for v0.6.1 embedding-based fact deduplication.

Covers:
- cosine similarity helper
- dedupe_facts_by_embedding drop / flag / keep branches
- lazy embedding of existing facts with set_fact_embedding persistence
- graceful no-op when embedder is None or embed_batch fails
- dedupe_s1_5_facts_by_embedding wrapper ordering guarantees
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from acervo.extraction.edge_resolution import (
    _cosine_sim,
    dedupe_facts_by_embedding,
    FactDedupAudit,
)
from acervo.extraction.extractor import ExtractedFact


# ── Fake embedder + fake graph fixtures ─────────────────────────────────────

class FakeEmbedder:
    """Embed texts into handcrafted vectors. Exact matches become the same
    vector so cosine sim hits 1.0; other texts get orthogonal unit vectors."""

    def __init__(self, mapping: dict[str, list[float]]):
        self._mapping = mapping

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._mapping.get(t, [0.0, 0.0, 0.0, 0.0]) for t in texts]


class FakeGraph:
    """Minimal graph stub: holds nodes with facts in-memory."""

    def __init__(self, nodes: dict[str, dict]):
        self._nodes = nodes
        self.set_fact_embedding_calls: list[tuple[str, list[float]]] = []

    def get_node(self, node_id: str) -> dict | None:
        return self._nodes.get(node_id)

    def set_fact_embedding(self, fact_id: str, embedding: list[float]) -> bool:
        self.set_fact_embedding_calls.append((fact_id, list(embedding)))
        for node in self._nodes.values():
            for f in node.get("facts", []):
                if f.get("fact_id") == fact_id:
                    f["fact_embedding"] = list(embedding)
                    return True
        return False


# ── Cosine helper ───────────────────────────────────────────────────────────


class TestCosineSim:
    def test_identical_vectors(self):
        assert _cosine_sim([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_sim([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert _cosine_sim([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_is_safe(self):
        assert _cosine_sim([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_mismatched_length_is_safe(self):
        assert _cosine_sim([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0


# ── dedupe_facts_by_embedding ────────────────────────────────────────────────


class TestDedupeFactsByEmbedding:
    @pytest.mark.asyncio
    async def test_no_op_when_embedder_is_none(self):
        fact = ExtractedFact(entity="Alice", fact="Lives in Neuquen", source="llm")
        kept, audit = await dedupe_facts_by_embedding(
            graph=FakeGraph({}),
            embedder=None,
            node_facts_map={"alice": [fact]},
        )
        assert kept == {"alice": [fact]}
        assert audit.checked == 0

    @pytest.mark.asyncio
    async def test_drops_high_similarity_duplicate(self):
        vec_same = [1.0, 0.0, 0.0, 0.0]
        existing = [{"fact_id": "alice_f0", "fact": "Vive en Neuquén",
                     "fact_embedding": vec_same}]
        graph = FakeGraph({"alice": {"id": "alice", "label": "Alice",
                                     "facts": existing}})
        embedder = FakeEmbedder({"Lives in Neuquen": vec_same})
        new_fact = ExtractedFact(entity="Alice", fact="Lives in Neuquen",
                                 source="llm")
        kept, audit = await dedupe_facts_by_embedding(
            graph=graph, embedder=embedder,
            node_facts_map={"alice": [new_fact]},
        )
        assert kept["alice"] == []
        assert audit.dropped == 1
        assert audit.checked == 1

    @pytest.mark.asyncio
    async def test_keeps_orthogonal_fact(self):
        existing = [{"fact_id": "alice_f0", "fact": "Lives in Neuquen",
                     "fact_embedding": [1.0, 0.0, 0.0, 0.0]}]
        graph = FakeGraph({"alice": {"id": "alice", "label": "Alice",
                                     "facts": existing}})
        embedder = FakeEmbedder({"Has a dog": [0.0, 1.0, 0.0, 0.0]})
        new_fact = ExtractedFact(entity="Alice", fact="Has a dog", source="llm")
        kept, audit = await dedupe_facts_by_embedding(
            graph=graph, embedder=embedder,
            node_facts_map={"alice": [new_fact]},
        )
        assert kept["alice"] == [new_fact]
        assert audit.dropped == 0
        assert new_fact.fact_embedding == [0.0, 1.0, 0.0, 0.0]

    @pytest.mark.asyncio
    async def test_flags_borderline_similarity(self):
        # sim = 0.7 — above flag threshold (0.6) but below drop (0.85)
        existing_emb = [1.0, 0.0]
        new_emb = [0.7, (1.0 - 0.49) ** 0.5]  # cosine ~= 0.7
        existing = [{"fact_id": "alice_f0", "fact": "Original",
                     "fact_embedding": existing_emb}]
        graph = FakeGraph({"alice": {"id": "alice", "label": "Alice",
                                     "facts": existing}})
        embedder = FakeEmbedder({"Paraphrase": new_emb})
        new_fact = ExtractedFact(entity="Alice", fact="Paraphrase", source="llm")
        kept, audit = await dedupe_facts_by_embedding(
            graph=graph, embedder=embedder,
            node_facts_map={"alice": [new_fact]},
        )
        assert kept["alice"] == [new_fact]
        assert audit.dropped == 0
        assert audit.flagged == 1

    @pytest.mark.asyncio
    async def test_lazy_embeds_existing_facts(self):
        # Existing facts lack embeddings — dedup should embed them and call
        # set_fact_embedding to persist.
        existing = [{"fact_id": "alice_f0", "fact": "Lives in Neuquen",
                     "fact_embedding": None}]
        graph = FakeGraph({"alice": {"id": "alice", "label": "Alice",
                                     "facts": existing}})
        embedder = FakeEmbedder({
            "Lives in Neuquen": [1.0, 0.0],
            "Lives in Neuquen!": [1.0, 0.0],
        })
        new_fact = ExtractedFact(entity="Alice", fact="Lives in Neuquen!",
                                 source="llm")
        kept, audit = await dedupe_facts_by_embedding(
            graph=graph, embedder=embedder,
            node_facts_map={"alice": [new_fact]},
        )
        # The existing fact got embedded and persisted
        assert ("alice_f0", [1.0, 0.0]) in graph.set_fact_embedding_calls
        # The new fact matched the now-embedded existing one and was dropped
        assert kept["alice"] == []
        assert audit.dropped == 1

    @pytest.mark.asyncio
    async def test_empty_new_facts_bucket(self):
        graph = FakeGraph({"alice": {"id": "alice", "label": "Alice", "facts": []}})
        embedder = FakeEmbedder({})
        kept, audit = await dedupe_facts_by_embedding(
            graph=graph, embedder=embedder,
            node_facts_map={"alice": []},
        )
        assert kept["alice"] == []
        assert audit.checked == 0


# ── dedupe_s1_5_facts_by_embedding wrapper ──────────────────────────────────


@dataclass
class _FakeAssistantExtraction:
    facts: list = field(default_factory=list)
    entities: list = field(default_factory=list)


@dataclass
class _FakeS15Result:
    assistant_extraction: _FakeAssistantExtraction = field(
        default_factory=_FakeAssistantExtraction
    )


class TestS15WrapperIntegration:
    @pytest.mark.asyncio
    async def test_wrapper_mutates_assistant_facts(self):
        from acervo.s1_5_graph_update import dedupe_s1_5_facts_by_embedding

        f1 = ExtractedFact(entity="Alice", fact="Lives in Neuquen", source="llm")
        f2 = ExtractedFact(entity="Alice", fact="Has a dog", source="llm")
        result = _FakeS15Result(
            assistant_extraction=_FakeAssistantExtraction(facts=[f1, f2]),
        )
        existing = [{"fact_id": "alice_f0", "fact": "Vive en Neuquén",
                     "fact_embedding": [1.0, 0.0]}]
        graph = FakeGraph({"alice": {"id": "alice", "label": "Alice",
                                     "facts": existing}})
        embedder = FakeEmbedder({
            "Lives in Neuquen": [1.0, 0.0],   # will match existing, drop
            "Has a dog": [0.0, 1.0],           # orthogonal, keep
        })

        audit = await dedupe_s1_5_facts_by_embedding(
            result=result,
            last_s1_extraction=None,
            graph=graph,
            embedder=embedder,
        )
        assert result.assistant_extraction.facts == [f2]
        assert audit["dedup_dropped"] == 1
        assert audit["dedup_checked"] == 2

    @pytest.mark.asyncio
    async def test_wrapper_no_op_without_embedder(self):
        from acervo.s1_5_graph_update import dedupe_s1_5_facts_by_embedding

        f1 = ExtractedFact(entity="Alice", fact="Lives in Neuquen", source="llm")
        result = _FakeS15Result(
            assistant_extraction=_FakeAssistantExtraction(facts=[f1]),
        )
        audit = await dedupe_s1_5_facts_by_embedding(
            result=result,
            last_s1_extraction=None,
            graph=FakeGraph({}),
            embedder=None,
        )
        assert audit == {"dedup_checked": 0, "dedup_dropped": 0, "dedup_flagged": 0}
        assert result.assistant_extraction.facts == [f1]
