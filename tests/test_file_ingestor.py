"""Tests for acervo.file_ingestor — file ingestion with structural parsing."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from acervo.file_ingestor import FileIngestor
from acervo.graph import TopicGraph, _make_id
from acervo.structural_parser import StructuralParser
from tests.conftest import make_mock_llm


@pytest.fixture
def graph(tmp_path):
    return TopicGraph(tmp_path / "graph")


@pytest.fixture
def parser():
    return StructuralParser()


@pytest.fixture
def mock_extractor_llm():
    """Mock LLM that returns extraction JSON with entities and facts."""
    return make_mock_llm(
        '{"entities": [{"name": "Python", "type": "Technology"}], '
        '"relations": [], '
        '"facts": [{"entity": "Python", "fact": "A programming language", "source": "file", "speaker": "text"}]}'
    )


class TestIngestMarkdown:
    async def test_ingest_md_with_parser(self, tmp_path, graph, parser, mock_extractor_llm):
        """Ingest .md with parser → creates file node + section nodes + entities."""
        md_file = tmp_path / "readme.md"
        md_file.write_text(
            "# Overview\n\nPython is great.\n\n## Installation\n\nRun pip install.\n",
            encoding="utf-8",
        )

        ingestor = FileIngestor(
            llm=mock_extractor_llm, graph=graph,
            structural_parser=parser,
        )
        result = await ingestor.ingest(md_file, tmp_path)

        assert result.file == "readme.md"
        assert result.skipped is False
        assert result.symbols > 0

        # File node exists
        file_node = graph.get_node(_make_id("readme.md"))
        assert file_node is not None
        assert file_node["kind"] == "file"

        # Section nodes exist
        sections = graph.get_file_symbols("readme.md")
        section_names = {s["label"] for s in sections}
        assert "Overview" in section_names
        assert "Installation" in section_names

    async def test_ingest_md_without_parser(self, tmp_path, graph, mock_extractor_llm):
        """Ingest .md without parser → whole-file extraction (legacy flow)."""
        md_file = tmp_path / "notes.md"
        md_file.write_text("# Notes\n\nSome notes about Python.\n", encoding="utf-8")

        ingestor = FileIngestor(llm=mock_extractor_llm, graph=graph)
        result = await ingestor.ingest(md_file, tmp_path)

        assert result.file == "notes.md"
        assert result.symbols == 0  # no structural parser → no symbols
        assert result.entities > 0  # LLM extraction still works

    async def test_skip_unchanged(self, tmp_path, graph, parser, mock_extractor_llm):
        """Ingest same .md twice → second returns skipped=True."""
        md_file = tmp_path / "readme.md"
        md_file.write_text("# Title\n\nContent.\n", encoding="utf-8")

        ingestor = FileIngestor(
            llm=mock_extractor_llm, graph=graph,
            structural_parser=parser,
        )

        result1 = await ingestor.ingest(md_file, tmp_path)
        assert result1.skipped is False

        result2 = await ingestor.ingest(md_file, tmp_path)
        assert result2.skipped is True


class TestIngestCode:
    async def test_ingest_python(self, tmp_path, graph, parser):
        """Ingest .py → creates file + symbol nodes, NO LLM call needed."""
        py_file = tmp_path / "utils.py"
        py_file.write_text(
            "def helper():\n    return 42\n\nclass Config:\n    pass\n",
            encoding="utf-8",
        )

        # Use a mock LLM that should NOT be called for code files
        llm = make_mock_llm("{}")
        ingestor = FileIngestor(
            llm=llm, graph=graph, structural_parser=parser,
        )
        result = await ingestor.ingest(py_file, tmp_path)

        assert result.file == "utils.py"
        assert result.symbols > 0
        assert result.skipped is False

        # Verify symbols
        symbols = graph.get_file_symbols("utils.py")
        names = {s["label"] for s in symbols}
        assert "helper" in names
        assert "Config" in names

        # LLM should NOT have been called (code files = structural only)
        assert llm.chat.call_count == 0

    async def test_ingest_javascript(self, tmp_path, graph, parser):
        """Ingest .js → creates file + symbol nodes."""
        js_file = tmp_path / "app.js"
        js_file.write_text(
            "function main() {\n    console.log('hi');\n}\n",
            encoding="utf-8",
        )

        llm = make_mock_llm("{}")
        ingestor = FileIngestor(
            llm=llm, graph=graph, structural_parser=parser,
        )
        result = await ingestor.ingest(js_file, tmp_path)

        assert result.symbols > 0
        symbols = graph.get_file_symbols("app.js")
        assert any(s["label"] == "main" for s in symbols)
        assert llm.chat.call_count == 0


class TestIngestAll:
    async def test_ingest_all_mixed(self, tmp_path, graph, parser):
        """Workspace with .md + .py → all indexed."""
        (tmp_path / "readme.md").write_text("# Hello\n\nWorld.\n", encoding="utf-8")
        (tmp_path / "utils.py").write_text("def foo():\n    pass\n", encoding="utf-8")

        llm = make_mock_llm(
            '{"entities": [{"name": "Hello", "type": "Topic"}], "relations": [], "facts": []}'
        )
        ingestor = FileIngestor(
            llm=llm, graph=graph, structural_parser=parser,
        )
        results = await ingestor.ingest_all(tmp_path)

        files = {r.file for r in results}
        assert "readme.md" in files
        assert "utils.py" in files
        assert all(not r.skipped for r in results)

    async def test_ingest_all_default_extensions(self, tmp_path, graph):
        """Without parser → only .md files indexed."""
        (tmp_path / "readme.md").write_text("# Hello\n", encoding="utf-8")
        (tmp_path / "utils.py").write_text("def foo():\n    pass\n", encoding="utf-8")

        llm = make_mock_llm('{"entities": [], "relations": [], "facts": []}')
        ingestor = FileIngestor(llm=llm, graph=graph)  # no parser
        results = await ingestor.ingest_all(tmp_path)

        files = {r.file for r in results}
        assert "readme.md" in files
        assert "utils.py" not in files  # no parser → code files skipped
