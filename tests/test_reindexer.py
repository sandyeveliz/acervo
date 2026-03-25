"""Tests for acervo.reindexer — staleness detection and deferred re-indexing."""

import tempfile
from pathlib import Path

import pytest

from acervo.graph import TopicGraph, _make_id
from acervo.reindexer import Reindexer, hash_file
from acervo.structural_parser import StructuralParser


@pytest.fixture
def workspace(tmp_path):
    """Create a workspace with a sample Python file."""
    f = tmp_path / "example.py"
    f.write_text(
        "def hello():\n    return 'world'\n\ndef goodbye():\n    return 'bye'\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def indexed_graph(tmp_path, workspace):
    """Create a graph with an indexed Python file."""
    graph_dir = tmp_path / "graph"
    graph = TopicGraph(graph_dir)
    parser = StructuralParser()
    structure = parser.parse(workspace / "example.py", workspace)
    graph.upsert_file_structure(
        structure.file_path, structure.language,
        structure.units, structure.content_hash,
    )
    graph.save()
    return graph


class TestFreshnessCheck:
    def test_fresh_file(self, indexed_graph, workspace):
        """Unchanged file → check_freshness returns True."""
        reindexer = Reindexer(indexed_graph, StructuralParser(), workspace)
        assert reindexer.check_freshness("example.py") is True

    def test_stale_file(self, indexed_graph, workspace):
        """Modified file → check_freshness returns False."""
        (workspace / "example.py").write_text(
            "def modified():\n    pass\n", encoding="utf-8",
        )
        reindexer = Reindexer(indexed_graph, StructuralParser(), workspace)
        assert reindexer.check_freshness("example.py") is False

    def test_missing_file(self, indexed_graph, workspace):
        """Deleted file → check_freshness returns False."""
        (workspace / "example.py").unlink()
        reindexer = Reindexer(indexed_graph, StructuralParser(), workspace)
        assert reindexer.check_freshness("example.py") is False

    def test_unindexed_file(self, indexed_graph, workspace):
        """File not in graph → check_freshness returns False."""
        reindexer = Reindexer(indexed_graph, StructuralParser(), workspace)
        assert reindexer.check_freshness("nonexistent.py") is False


class TestMarkStale:
    def test_mark_stale(self, indexed_graph):
        """mark_file_stale sets stale=True and stale_since."""
        assert indexed_graph.mark_file_stale("example.py") is True
        node = indexed_graph.get_node(_make_id("example.py"))
        assert node["stale"] is True
        assert node["stale_since"] is not None

    def test_mark_stale_idempotent(self, indexed_graph):
        """Calling mark_file_stale twice doesn't update stale_since."""
        indexed_graph.mark_file_stale("example.py")
        node = indexed_graph.get_node(_make_id("example.py"))
        first_since = node["stale_since"]

        indexed_graph.mark_file_stale("example.py")
        node = indexed_graph.get_node(_make_id("example.py"))
        assert node["stale_since"] == first_since

    def test_mark_stale_nonexistent(self, indexed_graph):
        """Marking non-existent file returns False."""
        assert indexed_graph.mark_file_stale("nope.py") is False

    def test_get_stale_files(self, indexed_graph):
        """get_stale_files returns only stale file nodes."""
        assert len(indexed_graph.get_stale_files()) == 0
        indexed_graph.mark_file_stale("example.py")
        stale = indexed_graph.get_stale_files()
        assert len(stale) == 1
        assert stale[0]["id"] == _make_id("example.py")


class TestReindexStale:
    async def test_reindex_updates_symbols(self, indexed_graph, workspace):
        """Mark stale → modify file → reindex → new symbols appear."""
        # Original symbols
        orig_symbols = indexed_graph.get_file_symbols("example.py")
        orig_names = {s["label"] for s in orig_symbols}
        assert "hello" in orig_names
        assert "goodbye" in orig_names

        # Modify file
        (workspace / "example.py").write_text(
            "def new_func():\n    pass\n\ndef another():\n    return 1\n",
            encoding="utf-8",
        )
        indexed_graph.mark_file_stale("example.py")

        # Reindex
        reindexer = Reindexer(indexed_graph, StructuralParser(), workspace)
        reindexed = await reindexer.reindex_stale()
        assert "example.py" in reindexed

        # Verify new symbols
        new_symbols = indexed_graph.get_file_symbols("example.py")
        new_names = {s["label"] for s in new_symbols}
        assert "new_func" in new_names
        assert "another" in new_names
        assert "hello" not in new_names

    async def test_reindex_clears_stale_flag(self, indexed_graph, workspace):
        """After reindex, stale=False."""
        indexed_graph.mark_file_stale("example.py")
        reindexer = Reindexer(indexed_graph, StructuralParser(), workspace)
        await reindexer.reindex_stale()

        node = indexed_graph.get_node(_make_id("example.py"))
        assert node["stale"] is False

    async def test_reindex_handles_deleted_file(self, indexed_graph, workspace):
        """Deleted file → node removed from graph."""
        (workspace / "example.py").unlink()
        indexed_graph.mark_file_stale("example.py")

        reindexer = Reindexer(indexed_graph, StructuralParser(), workspace)
        reindexed = await reindexer.reindex_stale()
        assert "example.py" in reindexed

        # File node should be gone
        assert indexed_graph.get_node(_make_id("example.py")) is None

    async def test_reindex_noop_when_nothing_stale(self, indexed_graph, workspace):
        """No stale files → returns empty list."""
        reindexer = Reindexer(indexed_graph, StructuralParser(), workspace)
        result = await reindexer.reindex_stale()
        assert result == []


class TestHashFile:
    def test_hash_deterministic(self, tmp_path):
        """Same content → same hash."""
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        assert hash_file(f) == hash_file(f)

    def test_hash_changes(self, tmp_path):
        """Different content → different hash."""
        f = tmp_path / "test.txt"
        f.write_text("hello", encoding="utf-8")
        h1 = hash_file(f)
        f.write_text("world", encoding="utf-8")
        h2 = hash_file(f)
        assert h1 != h2
