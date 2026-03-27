"""Tests for M3 — document ingestion with chunks linked to graph nodes."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from acervo.graph import TopicGraph, _make_id, make_symbol_id
from acervo.structural_parser import StructuralParser
from acervo.semantic_enricher import SemanticEnricher, ChunkEmbedding, EnrichmentResult
from acervo.indexer import Indexer


# ── Fixtures ──

@pytest.fixture
def tmp_dir():
    """Temp directory cleaned up after test."""
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_md(tmp_dir):
    """Create a sample .md file with 3 heading sections."""
    md = tmp_dir / "coding-rules.md"
    md.write_text(
        "# Coding Rules\n\n"
        "General rules for our project.\n\n"
        "## Naming Conventions\n\n"
        "Use camelCase for variables and PascalCase for classes.\n"
        "All constants should be UPPER_SNAKE_CASE.\n\n"
        "## Error Handling\n\n"
        "Always use try/catch blocks for async operations.\n"
        "Log errors with structured metadata.\n\n"
        "## Testing\n\n"
        "Write unit tests for all public functions.\n"
        "Integration tests must hit a real database.\n",
        encoding="utf-8",
    )
    return md


@pytest.fixture
def graph(tmp_dir):
    return TopicGraph(tmp_dir / "graph")


def _make_mock_embedder():
    """Mock embedder that returns a deterministic vector per input."""
    mock = AsyncMock()

    async def embed(text: str) -> list[float]:
        # Simple hash-based deterministic embedding (8 dimensions)
        h = hash(text) & 0xFFFFFFFF
        return [(h >> (i * 4) & 0xF) / 15.0 for i in range(8)]

    async def embed_batch(texts: list[str]) -> list[list[float]]:
        return [await embed(t) for t in texts]

    mock.embed = embed
    mock.embed_batch = embed_batch
    return mock


# ── Graph chunk_ids tests ──

class TestGraphChunkIds:
    def test_migrate_node_adds_chunk_ids(self, graph):
        """Legacy nodes get chunk_ids=[] on load."""
        graph.upsert_entities([("Test", "Entity")])
        node = graph.get_node("test")
        assert node is not None
        assert "chunk_ids" in node
        assert node["chunk_ids"] == []

    def test_link_chunks(self, graph):
        """link_chunks sets chunk_ids on a node."""
        graph.upsert_entities([("Test", "Entity")])
        ok = graph.link_chunks("test", ["c1", "c2", "c3"])
        assert ok is True
        assert graph.get_chunks_for_node("test") == ["c1", "c2", "c3"]

    def test_link_chunks_replaces(self, graph):
        """link_chunks replaces existing chunk_ids."""
        graph.upsert_entities([("Test", "Entity")])
        graph.link_chunks("test", ["old1", "old2"])
        graph.link_chunks("test", ["new1"])
        assert graph.get_chunks_for_node("test") == ["new1"]

    def test_link_chunks_nonexistent_node(self, graph):
        """link_chunks returns False for missing node."""
        assert graph.link_chunks("nonexistent", ["c1"]) is False

    def test_clear_chunks(self, graph):
        """clear_chunks sets chunk_ids to []."""
        graph.upsert_entities([("Test", "Entity")])
        graph.link_chunks("test", ["c1", "c2"])
        ok = graph.clear_chunks("test")
        assert ok is True
        assert graph.get_chunks_for_node("test") == []

    def test_get_chunks_for_nonexistent(self, graph):
        """get_chunks_for_node returns [] for missing node."""
        assert graph.get_chunks_for_node("nope") == []

    def test_get_nodes_with_chunks(self, graph):
        """get_nodes_with_chunks returns only nodes with non-empty chunk_ids."""
        graph.upsert_entities([("A", "Entity"), ("B", "Entity"), ("C", "Entity")])
        graph.link_chunks("a", ["c1"])
        graph.link_chunks("c", ["c2", "c3"])
        result = graph.get_nodes_with_chunks()
        ids = {n["id"] for n in result}
        assert ids == {"a", "c"}

    def test_upsert_file_structure_has_chunk_ids(self, graph):
        """File and section nodes created by upsert_file_structure have chunk_ids."""
        from acervo.structural_parser import StructuralUnit
        units = [
            StructuralUnit(
                name="Intro", unit_type="section",
                start_line=1, end_line=5, parent=None,
            ),
        ]
        graph.upsert_file_structure("docs/readme.md", "markdown", units, "hash123")
        file_node = graph.get_node(_make_id("docs/readme.md"))
        assert file_node is not None
        assert file_node["chunk_ids"] == []

        section_id = make_symbol_id("docs/readme.md", "Intro")
        section_node = graph.get_node(section_id)
        assert section_node is not None
        assert section_node["chunk_ids"] == []

    def test_remove_file_children_clears_parent_chunk_ids(self, graph):
        """_remove_file_children clears chunk_ids on the parent file node."""
        from acervo.structural_parser import StructuralUnit
        units = [
            StructuralUnit(
                name="Section", unit_type="section",
                start_line=1, end_line=5, parent=None,
            ),
        ]
        graph.upsert_file_structure("test.md", "markdown", units, "hash1")
        file_id = _make_id("test.md")
        graph.link_chunks(file_id, ["c1", "c2"])
        assert graph.get_chunks_for_node(file_id) == ["c1", "c2"]

        # Re-upsert with different hash triggers _remove_file_children
        graph.upsert_file_structure("test.md", "markdown", units, "hash2")
        assert graph.get_chunks_for_node(file_id) == []

    def test_chunk_ids_persist_across_save_reload(self, tmp_dir):
        """chunk_ids survive save() + reload()."""
        graph_path = tmp_dir / "graph"
        graph = TopicGraph(graph_path)
        graph.upsert_entities([("Test", "Entity")])
        graph.link_chunks("test", ["c1", "c2"])
        graph.save()

        graph2 = TopicGraph(graph_path)
        assert graph2.get_chunks_for_node("test") == ["c1", "c2"]


# ── Vector store tests ──

class TestVectorStoreChunks:
    @pytest.fixture
    def vector_store(self, tmp_dir):
        """Create a ChromaVectorStore with mock embedder."""
        try:
            from acervo.vector_store import ChromaVectorStore
        except ImportError:
            pytest.skip("chromadb not installed")

        embedder = _make_mock_embedder()
        return ChromaVectorStore(
            persist_path=str(tmp_dir / "vectordb"),
            embed_fn=embedder.embed,
            embed_batch_fn=embedder.embed_batch,
        )

    @pytest.mark.asyncio
    async def test_index_with_chunk_ids(self, vector_store):
        """Chunks stored with provided chunk_ids are retrievable by those IDs."""
        stored = await vector_store.index_file_chunks(
            file_path="docs/rules.md",
            chunks=["Rule 1: use camelCase", "Rule 2: always test"],
            chunk_ids=["rules_c0", "rules_c1"],
        )
        assert stored == ["rules_c0", "rules_c1"]

        # Verify retrievable by ID
        results = vector_store._files.get(ids=["rules_c0", "rules_c1"])
        assert len(results["documents"]) == 2

    @pytest.mark.asyncio
    async def test_index_with_precomputed_embeddings(self, vector_store):
        """Pre-computed embeddings skip the embed step."""
        embs = [[0.1] * 8, [0.2] * 8]
        stored = await vector_store.index_file_chunks(
            file_path="test.md",
            chunks=["chunk A", "chunk B"],
            chunk_ids=["a", "b"],
            embeddings=embs,
        )
        assert stored == ["a", "b"]

    @pytest.mark.asyncio
    async def test_search_by_chunk_ids(self, vector_store):
        """search_by_chunk_ids ranks chunks by similarity to query."""
        await vector_store.index_file_chunks(
            file_path="docs.md",
            chunks=[
                "Authentication uses JWT tokens",
                "Database uses PostgreSQL",
                "Deployment on AWS ECS",
            ],
            chunk_ids=["auth_c0", "db_c0", "deploy_c0"],
        )

        # Search for auth-related content
        query_emb = await vector_store._embed("authentication JWT")
        results = await vector_store.search_by_chunk_ids(
            ["auth_c0", "db_c0", "deploy_c0"], query_emb, n_results=2,
        )
        assert len(results) <= 2
        assert all(r["source"] == "node_scoped_chunk" for r in results)
        assert all("chunk_id" in r for r in results)

    @pytest.mark.asyncio
    async def test_remove_by_chunk_ids(self, vector_store):
        """remove_by_chunk_ids deletes specific chunks."""
        await vector_store.index_file_chunks(
            file_path="test.md",
            chunks=["chunk 1", "chunk 2", "chunk 3"],
            chunk_ids=["c1", "c2", "c3"],
        )
        vector_store.remove_by_chunk_ids(["c1", "c3"])

        remaining = vector_store._files.get(ids=["c1", "c2", "c3"])
        # c2 should remain, c1 and c3 removed
        assert "c2" in remaining["ids"]
        assert "c1" not in remaining["ids"]
        assert "c3" not in remaining["ids"]

    @pytest.mark.asyncio
    async def test_backward_compat_no_chunk_ids(self, vector_store):
        """index_file_chunks without chunk_ids still works (auto-generates IDs)."""
        stored = await vector_store.index_file_chunks(
            file_path="legacy.md",
            chunks=["content A", "content B"],
        )
        assert len(stored) == 2
        assert all("legacy_md" in s for s in stored)


# ── Indexer chunk linkage tests ──

class TestIndexerChunkLinkage:
    @pytest.mark.asyncio
    async def test_index_md_links_chunks_to_nodes(self, tmp_dir, sample_md):
        """Indexing a .md file creates nodes with chunk_ids."""
        try:
            from acervo.vector_store import ChromaVectorStore
        except ImportError:
            pytest.skip("chromadb not installed")

        graph = TopicGraph(tmp_dir / "graph")
        embedder = _make_mock_embedder()
        vector_store = ChromaVectorStore(
            persist_path=str(tmp_dir / "vectordb"),
            embed_fn=embedder.embed,
            embed_batch_fn=embedder.embed_batch,
        )

        indexer = Indexer(
            graph=graph,
            llm=None,  # No LLM summaries needed
            embedder=embedder,
            vector_store=vector_store,
        )

        result = await indexer.index(
            tmp_dir,
            extensions={".md"},
        )

        assert result.files_analyzed >= 1
        assert result.total_chunks > 0

        # File node should have chunk_ids
        file_id = _make_id("coding-rules.md")
        file_node = graph.get_node(file_id)
        assert file_node is not None
        chunk_ids = file_node.get("chunk_ids", [])
        assert len(chunk_ids) > 0, "file node should have chunk_ids after indexing"

        # Verify chunk_ids match what's in ChromaDB
        stored = vector_store._files.get(ids=chunk_ids)
        assert len(stored["ids"]) == len(chunk_ids)

    @pytest.mark.asyncio
    async def test_reindex_replaces_chunk_ids(self, tmp_dir, sample_md):
        """Re-indexing a modified file replaces old chunk_ids with new ones."""
        try:
            from acervo.vector_store import ChromaVectorStore
        except ImportError:
            pytest.skip("chromadb not installed")

        graph = TopicGraph(tmp_dir / "graph")
        embedder = _make_mock_embedder()
        vector_store = ChromaVectorStore(
            persist_path=str(tmp_dir / "vectordb"),
            embed_fn=embedder.embed,
            embed_batch_fn=embedder.embed_batch,
        )

        indexer = Indexer(
            graph=graph, embedder=embedder, vector_store=vector_store,
        )

        # First index
        await indexer.index(tmp_dir, extensions={".md"})
        file_id = _make_id("coding-rules.md")
        old_ids = list(graph.get_chunks_for_node(file_id))
        assert len(old_ids) > 0

        # Modify the file (changes content hash)
        sample_md.write_text(
            sample_md.read_text(encoding="utf-8") + "\n## New Section\n\nNew content.\n",
            encoding="utf-8",
        )

        # Re-index
        await indexer.index(tmp_dir, extensions={".md"})
        new_ids = graph.get_chunks_for_node(file_id)

        # IDs should be different (UUIDs regenerated)
        assert set(new_ids) != set(old_ids)
        assert len(new_ids) > 0


# ── Delete document tests ──

class TestDeleteDocument:
    @pytest.mark.asyncio
    async def test_delete_removes_node_and_chunks(self, tmp_dir, sample_md):
        """delete_document removes graph node + ChromaDB chunks."""
        try:
            from acervo.vector_store import ChromaVectorStore
        except ImportError:
            pytest.skip("chromadb not installed")

        graph = TopicGraph(tmp_dir / "graph")
        embedder = _make_mock_embedder()
        vector_store = ChromaVectorStore(
            persist_path=str(tmp_dir / "vectordb"),
            embed_fn=embedder.embed,
            embed_batch_fn=embedder.embed_batch,
        )

        indexer = Indexer(
            graph=graph, embedder=embedder, vector_store=vector_store,
        )
        await indexer.index(tmp_dir, extensions={".md"})

        file_id = _make_id("coding-rules.md")
        chunk_ids = list(graph.get_chunks_for_node(file_id))
        assert len(chunk_ids) > 0

        # Delete
        vector_store.remove_by_chunk_ids(chunk_ids)
        graph._remove_file_children(file_id)
        graph.remove_node(file_id)

        # Verify cleanup
        assert graph.get_node(file_id) is None
        stored = vector_store._files.get(ids=chunk_ids)
        assert len(stored["ids"]) == 0
