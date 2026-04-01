"""Layer 1: Pipeline Step Validation — diagnose index/curate/synthesize output.

These tests run against the EXISTING graph state (no re-indexing needed).
They inspect what each pipeline step produced and report exactly where
things broke.

Usage:
    pytest tests/integration/test_pipeline_validation.py -v
    pytest tests/integration/test_pipeline_validation.py -k "p2" -v
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from acervo.graph import TopicGraph


# ── Fixtures ──

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FIXTURES = _REPO_ROOT / "tests" / "fixtures"

PROJECT_PATHS = {
    "p1": _FIXTURES / "p1-todo-app",
    "p2": _FIXTURES / "p2-literature",
    "p3": _FIXTURES / "p3-project-docs",
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
def p3_graph():
    return _load_graph("p3")


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
            "person", "organization", "project", "technology",
            "place", "event", "document", "concept", "symbol",
        }
        for e in entities:
            etype = e.get("type", "").lower()
            assert etype in valid_types, (
                f"Entity '{e.get('label')}' has unexpected type '{e.get('type')}'. "
                f"Valid: {valid_types}"
            )


class TestCurateQualityP1:
    """Quality checks for P1 curation output."""

    def test_no_phantom_entities(self, p1_graph):
        """Every entity label should appear in at least one source file."""
        by_kind = _get_nodes_by_kind(p1_graph)
        entities = by_kind.get("entity", [])
        if not entities:
            pytest.skip("No entities")
        # Read all source file content
        all_content = ""
        p1_path = PROJECT_PATHS["p1"]
        for f in p1_path.rglob("*"):
            if f.suffix in (".ts", ".tsx", ".js", ".jsx", ".md", ".json", ".html", ".css"):
                try:
                    all_content += f.read_text(errors="ignore").lower() + "\n"
                except Exception:
                    pass
        phantoms = []
        for e in entities:
            label = e.get("label", "").lower()
            # Check both full label and first word (e.g., "Express.js" → "express")
            first_word = label.split(".")[0].split(" ")[0]
            if label not in all_content and first_word not in all_content:
                phantoms.append(e.get("label"))
        if phantoms:
            pytest.xfail(f"Phantom entities (not in source): {phantoms}")

    def test_entities_have_relations(self, p1_graph):
        """Entities should be connected to file nodes via relations."""
        by_kind = _get_nodes_by_kind(p1_graph)
        entities = by_kind.get("entity", [])
        if not entities:
            pytest.skip("No entities")
        entity_ids = {e.get("id") for e in entities}
        connected = set()
        for node in p1_graph.get_all_nodes():
            for edge in p1_graph.get_edges_for(node.get("id", "")):
                if edge.get("source") in entity_ids:
                    connected.add(edge["source"])
                if edge.get("target") in entity_ids:
                    connected.add(edge["target"])
        orphans = entity_ids - connected
        assert len(orphans) <= len(entities) // 2, (
            f"{len(orphans)}/{len(entities)} entities are orphans (no relations): "
            f"{[e.get('label') for e in entities if e.get('id') in orphans]}"
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
    """Validate indexation output for P2 literature project (Sherlock Holmes)."""

    def test_file_nodes(self, p2_graph):
        by_kind = _get_nodes_by_kind(p2_graph)
        files = by_kind.get("file", [])
        assert len(files) >= 1, (
            f"Expected >=1 file node (epub), got {len(files)}"
        )

    def test_section_nodes_for_chapters(self, p2_graph):
        by_kind = _get_nodes_by_kind(p2_graph)
        sections = by_kind.get("section", [])
        assert len(sections) >= 10, (
            f"Expected >=10 section nodes (story chapters), got {len(sections)}"
        )

    def test_summaries_generated(self, p2_graph):
        with_summary = sum(
            1 for n in p2_graph.get_all_nodes()
            if n.get("attributes", {}).get("summary")
        )
        assert with_summary >= 5, (
            f"Expected >=5 nodes with LLM summaries, got {with_summary}"
        )

    def test_chunk_ids_linked(self, p2_graph):
        total = sum(
            len(n.get("chunk_ids", []))
            for n in p2_graph.get_all_nodes()
        )
        assert total >= 50, (
            f"Expected >=50 total chunk_ids, got {total}"
        )


class TestCurateQualityP2:
    """Quality checks for P2 curation output."""

    def test_no_phantom_entities(self, p2_graph):
        """Entities should be grounded in the epub content."""
        by_kind = _get_nodes_by_kind(p2_graph)
        entities = by_kind.get("entity", [])
        if not entities:
            pytest.skip("No entities")
        # For epub, check against section summaries and facts (can't read binary epub)
        all_text = ""
        for n in p2_graph.get_all_nodes():
            summary = n.get("attributes", {}).get("summary", "")
            all_text += summary.lower() + "\n"
            for f in n.get("facts", []):
                fact_text = f.get("fact", "") if isinstance(f, dict) else str(f)
                all_text += fact_text.lower() + "\n"
            all_text += n.get("label", "").lower() + "\n"
        phantoms = []
        for e in entities:
            label = e.get("label", "").lower()
            first_word = label.split(" ")[0]
            if label not in all_text and first_word not in all_text:
                phantoms.append(e.get("label"))
        if phantoms:
            pytest.xfail(f"Phantom entities: {phantoms}")


class TestCurateP2:
    """Validate curation output for P2 literature (Sherlock Holmes)."""

    def test_character_entities(self, p2_graph):
        """Curation should extract Sherlock Holmes characters as entities."""
        by_kind = _get_nodes_by_kind(p2_graph)
        entities = by_kind.get("entity", [])
        labels = [e.get("label", "").lower() for e in entities]
        expected = ["holmes", "watson", "adler", "irene", "moriarty", "lestrade"]
        found = [c for c in expected if any(c in l for l in labels)]
        assert len(found) >= 2, (
            f"Expected character entities {expected}. "
            f"Found entities: {labels[:20]}. "
            f"DIAGNOSIS: Curate didn't extract characters from Sherlock Holmes."
        )

    def test_location_entities(self, p2_graph):
        """Curation should extract locations from Sherlock Holmes."""
        by_kind = _get_nodes_by_kind(p2_graph)
        entities = by_kind.get("entity", [])
        labels = [e.get("label", "").lower() for e in entities]
        expected = ["baker street", "london"]
        found = [l for l in expected if any(l in el for el in labels)]
        if not found:
            pytest.xfail(
                f"No location entities found. Entities: {labels[:20]}. "
                f"Curate may need better prompting for literature."
            )


class TestSynthesizeP2:
    """Validate synthesis output for P2 literature (Sherlock Holmes)."""

    def test_synthesis_exists(self, p2_graph):
        by_kind = _get_nodes_by_kind(p2_graph)
        synthesis = by_kind.get("synthesis", [])
        assert len(synthesis) >= 1, "No synthesis nodes"

    def test_synthesis_mentions_sherlock(self, p2_graph):
        by_kind = _get_nodes_by_kind(p2_graph)
        synthesis = by_kind.get("synthesis", [])
        overview = next(
            (s for s in synthesis if "overview" in s.get("id", "")), None
        )
        assert overview is not None
        summary = overview.get("attributes", {}).get("summary", "").lower()
        has_sherlock = "sherlock" in summary or "holmes" in summary or "conan doyle" in summary
        assert has_sherlock, (
            f"Synthesis doesn't mention Sherlock Holmes. Summary: {summary[:200]}"
        )


# ══════════════════════════════════════════════════════════════
# P3 — PM Docs (markdown project)
# ══════════════════════════════════════════════════════════════


class TestIndexP3:
    """Validate indexation output for P3 PM docs."""

    def test_file_nodes(self, p3_graph):
        by_kind = _get_nodes_by_kind(p3_graph)
        files = by_kind.get("file", [])
        assert len(files) >= 5, f"Expected >=5 file nodes, got {len(files)}"

    def test_section_nodes(self, p3_graph):
        by_kind = _get_nodes_by_kind(p3_graph)
        sections = by_kind.get("section", [])
        assert len(sections) >= 30, (
            f"Expected >=30 section nodes from markdown headings, got {len(sections)}"
        )


class TestCurateP3:
    """Validate curation output for P3 PM docs."""

    def test_entities_extracted(self, p3_graph):
        by_kind = _get_nodes_by_kind(p3_graph)
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


class TestSynthesizeP3:
    """Validate synthesis output for P3 PM docs."""

    def test_synthesis_exists(self, p3_graph):
        by_kind = _get_nodes_by_kind(p3_graph)
        synthesis = by_kind.get("synthesis", [])
        assert len(synthesis) >= 1, "No synthesis nodes"

    def test_module_summaries(self, p3_graph):
        by_kind = _get_nodes_by_kind(p3_graph)
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
        report_text = "\n".join(lines)
        print(report_text)

        # Export reports
        reports_dir = _REPO_ROOT / "tests" / "integration" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Write JSON report
        report_data = {"generated_at": datetime.now().isoformat(), "projects": {}}
        for key, path in PROJECT_PATHS.items():
            graph_path = path / ".acervo" / "data" / "graph"
            if not graph_path.exists():
                report_data["projects"][key] = {"status": "not_initialized"}
                continue
            g = TopicGraph(graph_path)
            by_kind = _get_nodes_by_kind(g)
            report_data["projects"][key] = {
                "nodes": g.node_count,
                "edges": g.edge_count,
                "files": len(by_kind.get("file", [])),
                "sections": len(by_kind.get("section", [])),
                "symbols": len(by_kind.get("symbol", [])),
                "folders": len(by_kind.get("folder", [])),
                "entities": len(by_kind.get("entity", [])),
                "entity_labels": [e.get("label") for e in by_kind.get("entity", [])],
                "synthesis": len(by_kind.get("synthesis", [])),
                "synthesis_ids": [s.get("id") for s in by_kind.get("synthesis", [])],
                "chunks": sum(len(n.get("chunk_ids", [])) for n in g.get_all_nodes()),
                "summaries": sum(1 for n in g.get_all_nodes() if n.get("attributes", {}).get("summary")),
            }

        json_path = reports_dir / "pipeline_diagnostic.json"
        json_path.write_text(
            json.dumps(report_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Write Markdown report
        md_path = reports_dir / "pipeline_diagnostic.md"
        md_path.write_text(report_text, encoding="utf-8")

        print(f"\n  Reports saved to:")
        print(f"    {json_path}")
        print(f"    {md_path}")
