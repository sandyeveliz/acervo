"""Layer 1: Pipeline Step Validation — diagnose index/curate/synthesize output.

These tests run against the EXISTING graph state (no re-indexing needed).
They inspect what each pipeline step produced and report exactly where
things broke.

Usage:
    pytest tests/integration/test_pipeline_validation.py -v
    pytest tests/integration/test_pipeline_validation.py -k "p2" -v
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from acervo.graph import TopicGraph


# ── Fixtures ──

PROJECT_PATHS = {
    "p1": Path("D:/Development/project-tests/p1-todo-app"),
    "p2": Path("D:/Development/project-tests/p2-books"),
    "p4": Path("D:/Development/project-tests/p4-project-docs"),
}


def _load_graph(project_key: str) -> TopicGraph:
    path = PROJECT_PATHS[project_key] / ".acervo" / "data" / "graph"
    if not path.exists():
        pytest.skip(f"{project_key} not initialized")
    return TopicGraph(path)


def _get_nodes_by_kind(graph: TopicGraph) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for n in graph.get_all_nodes():
        kind = n.get("kind", "entity")
        result.setdefault(kind, []).append(n)
    return result


@pytest.fixture
def p1_graph():
    return _load_graph("p1")


@pytest.fixture
def p2_graph():
    return _load_graph("p2")


@pytest.fixture
def p4_graph():
    return _load_graph("p4")


# ══════════════════════════════════════════════════════════════
# P1 — TODO App (code project)
# ══════════════════════════════════════════════════════════════


class TestIndexP1:
    """Validate indexation output for P1 code project."""

    def test_file_nodes_created(self, p1_graph):
        by_kind = _get_nodes_by_kind(p1_graph)
        files = by_kind.get("file", [])
        assert len(files) >= 20, (
            f"Expected >=20 file nodes for TODO app, got {len(files)}"
        )

    def test_symbol_nodes_created(self, p1_graph):
        by_kind = _get_nodes_by_kind(p1_graph)
        symbols = by_kind.get("symbol", [])
        assert len(symbols) >= 50, (
            f"Expected >=50 symbol nodes (functions/classes), got {len(symbols)}"
        )

    def test_folder_nodes_created(self, p1_graph):
        by_kind = _get_nodes_by_kind(p1_graph)
        folders = by_kind.get("folder", [])
        assert len(folders) >= 5, (
            f"Expected >=5 folder nodes, got {len(folders)}"
        )

    def test_chunk_ids_linked(self, p1_graph):
        nodes_with_chunks = [
            n for n in p1_graph.get_all_nodes() if n.get("chunk_ids")
        ]
        assert len(nodes_with_chunks) >= 100, (
            f"Expected >=100 nodes with chunk_ids, got {len(nodes_with_chunks)}"
        )

    def test_edges_exist(self, p1_graph):
        assert p1_graph.edge_count >= 100, (
            f"Expected >=100 edges (contains + imports), got {p1_graph.edge_count}"
        )


class TestCurateP1:
    """Validate curation output for P1 code project."""

    def test_entities_extracted(self, p1_graph):
        """Curation should extract technology/concept entities from code."""
        by_kind = _get_nodes_by_kind(p1_graph)
        entities = by_kind.get("entity", [])
        # DIAGNOSTIC: this is the key test — curate should produce entities
        entity_info = [
            f"{e.get('label')} (type={e.get('type')}, source={e.get('source')})"
            for e in entities
        ]
        assert len(entities) >= 3, (
            f"Curate produced only {len(entities)} entity nodes for code project. "
            f"Expected >=3 (e.g., React, Express, SQLite, JWT). "
            f"Found: {entity_info}. "
            f"DIAGNOSIS: Curate prompt may not handle code projects well."
        )

    def test_entity_types_valid(self, p1_graph):
        by_kind = _get_nodes_by_kind(p1_graph)
        entities = by_kind.get("entity", [])
        if not entities:
            pytest.skip("No entities to validate (curate didn't produce any)")
        valid_types = {
            "Person", "Organization", "Project", "Technology",
            "Place", "Event", "Document", "Concept", "Symbol",
        }
        for e in entities:
            etype = e.get("type", "")
            assert etype in valid_types, (
                f"Entity '{e.get('label')}' has unexpected type '{etype}'. "
                f"Valid: {valid_types}"
            )


class TestSynthesizeP1:
    """Validate synthesis output for P1 code project."""

    def test_synthesis_node_exists(self, p1_graph):
        by_kind = _get_nodes_by_kind(p1_graph)
        synthesis = by_kind.get("synthesis", [])
        overview = [s for s in synthesis if "overview" in s.get("id", "")]
        assert len(overview) >= 1, (
            f"No synthesis:*overview* node found. "
            f"Synthesis nodes: {[s.get('id') for s in synthesis]}. "
            f"DIAGNOSIS: Synthesize was never run on P1."
        )

    def test_synthesis_mentions_stack(self, p1_graph):
        by_kind = _get_nodes_by_kind(p1_graph)
        synthesis = by_kind.get("synthesis", [])
        overview = next(
            (s for s in synthesis if "overview" in s.get("id", "")), None
        )
        if overview is None:
            pytest.skip("No synthesis overview (synthesize not run)")
        summary = overview.get("attributes", {}).get("summary", "").lower()
        keywords = ["todo", "react", "express", "typescript", "sqlite", "jwt"]
        found = [k for k in keywords if k in summary]
        assert len(found) >= 1, (
            f"Synthesis overview doesn't mention project stack. "
            f"Checked: {keywords}. Summary: {summary[:200]}"
        )


# ══════════════════════════════════════════════════════════════
# P2 — Books (Harry Potter epubs)
# ══════════════════════════════════════════════════════════════


class TestIndexP2:
    """Validate indexation output for P2 books project."""

    def test_file_nodes_for_each_book(self, p2_graph):
        by_kind = _get_nodes_by_kind(p2_graph)
        files = by_kind.get("file", [])
        assert len(files) == 7, (
            f"Expected 7 file nodes (7 HP books), got {len(files)}"
        )

    def test_section_nodes_for_chapters(self, p2_graph):
        by_kind = _get_nodes_by_kind(p2_graph)
        sections = by_kind.get("section", [])
        assert len(sections) >= 200, (
            f"Expected >=200 section nodes (chapter headings), got {len(sections)}"
        )

    def test_summaries_generated(self, p2_graph):
        with_summary = sum(
            1 for n in p2_graph.get_all_nodes()
            if n.get("attributes", {}).get("summary")
        )
        assert with_summary >= 200, (
            f"Expected >=200 nodes with LLM summaries, got {with_summary}"
        )

    def test_chunk_ids_linked(self, p2_graph):
        total = sum(
            len(n.get("chunk_ids", []))
            for n in p2_graph.get_all_nodes()
        )
        assert total >= 1000, (
            f"Expected >=1000 total chunk_ids (epub paragraphs), got {total}"
        )

    def test_cross_file_edges(self, p2_graph):
        """Books should have cross-file semantic edges (shared topics)."""
        assert p2_graph.edge_count >= 500, (
            f"Expected >=500 edges (cross-chapter links), got {p2_graph.edge_count}"
        )


class TestCurateP2:
    """Validate curation output for P2 books project."""

    def test_character_entities(self, p2_graph):
        """Curation should extract HP characters as entities."""
        by_kind = _get_nodes_by_kind(p2_graph)
        entities = by_kind.get("entity", [])
        labels = [e.get("label", "").lower() for e in entities]
        expected = ["harry", "hermione", "ron", "dumbledore", "voldemort"]
        found = [c for c in expected if any(c in l for l in labels)]
        assert len(found) >= 2, (
            f"Expected character entities {expected}. "
            f"Found entities: {labels[:20]}. "
            f"DIAGNOSIS: Curate didn't extract characters from HP books."
        )

    def test_series_relation(self, p2_graph):
        """Curation should discover the books form a series."""
        by_kind = _get_nodes_by_kind(p2_graph)
        entities = by_kind.get("entity", [])
        labels = [e.get("label", "").lower() for e in entities]
        series_terms = ["saga", "series", "collection", "harry potter"]
        found = [t for t in series_terms if any(t in l for l in labels)]
        if not found:
            pytest.xfail(
                f"No series/saga entity found. Entities: {labels[:20]}. "
                f"Curate may need better prompting for book collections."
            )


class TestSynthesizeP2:
    """Validate synthesis output for P2 books project."""

    def test_synthesis_exists(self, p2_graph):
        by_kind = _get_nodes_by_kind(p2_graph)
        synthesis = by_kind.get("synthesis", [])
        assert len(synthesis) >= 1, "No synthesis nodes"

    def test_synthesis_mentions_books(self, p2_graph):
        by_kind = _get_nodes_by_kind(p2_graph)
        synthesis = by_kind.get("synthesis", [])
        overview = next(
            (s for s in synthesis if "overview" in s.get("id", "")), None
        )
        assert overview is not None
        summary = overview.get("attributes", {}).get("summary", "").lower()
        assert "harry potter" in summary, (
            f"Synthesis doesn't mention Harry Potter. Summary: {summary[:200]}"
        )

    def test_synthesis_mentions_count(self, p2_graph):
        by_kind = _get_nodes_by_kind(p2_graph)
        overview = next(
            (s for s in _get_nodes_by_kind(p2_graph).get("synthesis", [])
             if "overview" in s.get("id", "")),
            None,
        )
        if overview is None:
            pytest.skip("No synthesis overview")
        summary = overview.get("attributes", {}).get("summary", "").lower()
        has_count = "7" in summary or "seven" in summary or "siete" in summary
        assert has_count, (
            f"Synthesis doesn't mention book count (7). Summary: {summary[:200]}"
        )


# ══════════════════════════════════════════════════════════════
# P4 — PM Docs (markdown project)
# ══════════════════════════════════════════════════════════════


class TestIndexP4:
    """Validate indexation output for P4 PM docs."""

    def test_file_nodes(self, p4_graph):
        by_kind = _get_nodes_by_kind(p4_graph)
        files = by_kind.get("file", [])
        assert len(files) >= 5, f"Expected >=5 file nodes, got {len(files)}"

    def test_section_nodes(self, p4_graph):
        by_kind = _get_nodes_by_kind(p4_graph)
        sections = by_kind.get("section", [])
        assert len(sections) >= 30, (
            f"Expected >=30 section nodes from markdown headings, got {len(sections)}"
        )


class TestCurateP4:
    """Validate curation output for P4 PM docs."""

    def test_entities_extracted(self, p4_graph):
        by_kind = _get_nodes_by_kind(p4_graph)
        entities = by_kind.get("entity", [])
        entity_info = [
            f"{e.get('label')} (type={e.get('type')})"
            for e in entities
        ]
        assert len(entities) >= 1, (
            f"Curate produced 0 entity nodes from PM docs. "
            f"Expected: project names, technologies, people, deadlines. "
            f"Found: {entity_info}. "
            f"DIAGNOSIS: Curate prompt may not handle PM docs well."
        )


class TestSynthesizeP4:
    """Validate synthesis output for P4 PM docs."""

    def test_synthesis_exists(self, p4_graph):
        by_kind = _get_nodes_by_kind(p4_graph)
        synthesis = by_kind.get("synthesis", [])
        assert len(synthesis) >= 1, "No synthesis nodes"

    def test_module_summaries(self, p4_graph):
        by_kind = _get_nodes_by_kind(p4_graph)
        synthesis = by_kind.get("synthesis", [])
        modules = [s for s in synthesis if "module" in s.get("id", "")]
        assert len(modules) >= 1, (
            f"No module summary nodes. Synthesis IDs: {[s.get('id') for s in synthesis]}"
        )


# ══════════════════════════════════════════════════════════════
# Cross-project diagnostic summary
# ══════════════════════════════════════════════════════════════


class TestPipelineSummary:
    """Print diagnostic summary across all projects."""

    def test_diagnostic_report(self):
        """Generate and print the diagnostic report."""
        lines = []
        lines.append("")
        lines.append("=" * 65)
        lines.append(" PIPELINE DIAGNOSTIC REPORT")
        lines.append("=" * 65)

        for key, path in PROJECT_PATHS.items():
            graph_path = path / ".acervo" / "data" / "graph"
            if not graph_path.exists():
                lines.append(f"\n {key.upper()}: NOT INITIALIZED")
                continue

            g = TopicGraph(graph_path)
            by_kind = _get_nodes_by_kind(g)
            entities = by_kind.get("entity", [])
            synthesis = by_kind.get("synthesis", [])
            files = by_kind.get("file", [])
            sections = by_kind.get("section", [])
            symbols = by_kind.get("symbol", [])
            folders = by_kind.get("folder", [])

            with_summary = sum(
                1 for n in g.get_all_nodes()
                if n.get("attributes", {}).get("summary")
            )
            total_chunks = sum(
                len(n.get("chunk_ids", []))
                for n in g.get_all_nodes()
            )

            idx_ok = len(files) > 0
            cur_ok = len(entities) >= 3
            syn_ok = any("overview" in s.get("id", "") for s in synthesis)

            lines.append(f"\n {'OK' if idx_ok else 'WARN':>4} INDEX     {key.upper()}: "
                         f"{len(files)} files, {len(sections)} sections, "
                         f"{len(symbols)} symbols, {len(folders)} folders, "
                         f"{total_chunks} chunks")
            lines.append(f" {'OK' if cur_ok else 'FAIL':>4} CURATE    "
                         f"{len(entities)} entities"
                         + (f" — {', '.join(e.get('label','?') for e in entities[:5])}"
                            if entities else " — NONE"))
            lines.append(f" {'OK' if syn_ok else 'FAIL':>4} SYNTHESIZE "
                         f"{len(synthesis)} nodes"
                         + (f" — {', '.join(s.get('id','?') for s in synthesis)}"
                            if synthesis else " — NONE"))

        lines.append("")
        lines.append("=" * 65)
        print("\n".join(lines))
