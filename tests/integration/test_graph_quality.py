"""Layer 1b: Graph Quality Specs — verify indexed graphs contain expected content.

Runs after index+curate+synthesize. Checks that the graph has the right
entities, no phantoms, correct synthesis nodes, and reasonable bounds.

Usage:
    pytest tests/integration/test_graph_quality.py -v -s
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pytest
import yaml

log = logging.getLogger(__name__)

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
_SPECS = Path(__file__).parent / "specs"
_REPORTS = Path(__file__).parent / "reports"

SPEC_FILES = ["p1_graph_quality.yaml", "p2_graph_quality.yaml", "p3_graph_quality.yaml"]


# ── Data structures ──


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class GraphQualityResult:
    name: str
    project: str
    checks: list[Check] = field(default_factory=list)
    node_count: int = 0
    edge_count: int = 0
    entity_count: int = 0
    synthesis_count: int = 0

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    @property
    def total(self) -> int:
        return len(self.checks)


# ── Graph loading ──


def _load_graph(project_dir: Path) -> tuple[list[dict], list[dict]]:
    """Load nodes and edges from a fixture's graph files."""
    graph_dir = project_dir / ".acervo" / "data" / "graph"
    nodes_file = graph_dir / "nodes.json"
    edges_file = graph_dir / "edges.json"

    if not nodes_file.exists():
        return [], []

    nodes = json.loads(nodes_file.read_text(encoding="utf-8"))
    edges = json.loads(edges_file.read_text(encoding="utf-8")) if edges_file.exists() else []
    return nodes, edges


# ── Check functions ──


def _find_entity(nodes: list[dict], spec: dict) -> dict | None:
    """Find an entity node matching the spec."""
    for node in nodes:
        if node.get("kind", "entity") != "entity":
            continue
        label = node.get("label", "").lower()
        label_contains = spec.get("label_contains", "").lower()
        if label_contains and label_contains not in label:
            continue
        if "type" in spec and node.get("type", "").lower() != spec["type"].lower():
            continue
        return node
    return None


def _check_required_entities(nodes: list[dict], spec: dict, result: GraphQualityResult) -> None:
    required = spec.get("expected_entities", {}).get("required", [])
    if isinstance(required, list) and len(required) == 0:
        return
    for expected in required:
        found = _find_entity(nodes, expected)
        label = expected.get("label_contains", "?")
        etype = expected.get("type", "any")
        result.checks.append(Check(
            name=f"Required: {label} ({etype})",
            passed=found is not None,
            detail=f"Found: {found['label']} [{found['type']}]" if found else "NOT FOUND",
        ))


def _check_forbidden_entities(nodes: list[dict], spec: dict, result: GraphQualityResult) -> None:
    forbidden = spec.get("expected_entities", {}).get("forbidden", [])
    for fb in forbidden:
        found = _find_entity(nodes, fb)
        label = fb.get("label_contains", "?")
        result.checks.append(Check(
            name=f"Forbidden: {label}",
            passed=found is None,
            detail=f"PHANTOM: {found['label']} [{found['type']}]" if found else "OK (not found)",
        ))


def _check_optional_entities(nodes: list[dict], spec: dict, result: GraphQualityResult) -> None:
    optional = spec.get("expected_entities", {}).get("optional", [])
    for opt in optional:
        found = _find_entity(nodes, opt)
        label = opt.get("label_contains", "?")
        # Optional checks don't affect pass/fail — they're informational
        result.checks.append(Check(
            name=f"Optional: {label}",
            passed=True,  # always passes (it's optional)
            detail=f"Found: {found['label']}" if found else "Not found (optional)",
        ))


def _check_synthesis(nodes: list[dict], spec: dict, result: GraphQualityResult) -> None:
    required = spec.get("expected_synthesis", {}).get("required", [])
    for synth_spec in required:
        id_contains = synth_spec.get("id_contains", "")
        found = any(
            id_contains.lower() in n.get("id", "").lower()
            for n in nodes if n.get("kind") == "synthesis"
        )
        result.checks.append(Check(
            name=f"Synthesis: {id_contains}",
            passed=found,
            detail="Found" if found else "MISSING",
        ))


def _check_structural(nodes: list[dict], spec: dict, result: GraphQualityResult) -> None:
    structural = spec.get("structural", {})
    if not structural:
        return

    kinds = {}
    for n in nodes:
        k = n.get("kind", "entity")
        kinds[k] = kinds.get(k, 0) + 1

    if "files" in structural:
        actual = kinds.get("file", 0)
        expected = structural["files"]
        result.checks.append(Check(
            name=f"Files: {expected}",
            passed=actual == expected,
            detail=f"Got {actual}",
        ))

    if "files_min" in structural:
        actual = kinds.get("file", 0)
        result.checks.append(Check(
            name=f"Files >= {structural['files_min']}",
            passed=actual >= structural["files_min"],
            detail=f"Got {actual}",
        ))

    if "sections_min" in structural:
        actual = kinds.get("section", 0)
        result.checks.append(Check(
            name=f"Sections >= {structural['sections_min']}",
            passed=actual >= structural["sections_min"],
            detail=f"Got {actual}",
        ))

    if "symbols_min" in structural:
        actual = kinds.get("symbol", 0)
        result.checks.append(Check(
            name=f"Symbols >= {structural['symbols_min']}",
            passed=actual >= structural["symbols_min"],
            detail=f"Got {actual}",
        ))

    if "folders_min" in structural:
        actual = kinds.get("folder", 0)
        result.checks.append(Check(
            name=f"Folders >= {structural['folders_min']}",
            passed=actual >= structural["folders_min"],
            detail=f"Got {actual}",
        ))


def _check_bounds(nodes: list[dict], edges: list[dict], spec: dict, result: GraphQualityResult) -> None:
    bounds = spec.get("graph_bounds", {})
    if not bounds:
        return

    n_count = len(nodes)
    e_count = len(edges)
    entities = [n for n in nodes if n.get("kind") == "entity"]
    synthesis = [n for n in nodes if n.get("kind") == "synthesis"]

    result.node_count = n_count
    result.edge_count = e_count
    result.entity_count = len(entities)
    result.synthesis_count = len(synthesis)

    if "min_nodes" in bounds:
        result.checks.append(Check(
            name=f"Nodes >= {bounds['min_nodes']}",
            passed=n_count >= bounds["min_nodes"],
            detail=f"Got {n_count}",
        ))
    if "max_nodes" in bounds:
        result.checks.append(Check(
            name=f"Nodes <= {bounds['max_nodes']}",
            passed=n_count <= bounds["max_nodes"],
            detail=f"Got {n_count}",
        ))
    if "min_edges" in bounds:
        result.checks.append(Check(
            name=f"Edges >= {bounds['min_edges']}",
            passed=e_count >= bounds["min_edges"],
            detail=f"Got {e_count}",
        ))
    if "min_entities" in bounds:
        result.checks.append(Check(
            name=f"Entities >= {bounds['min_entities']}",
            passed=len(entities) >= bounds["min_entities"],
            detail=f"Got {len(entities)}",
        ))
    if "max_entities" in bounds:
        result.checks.append(Check(
            name=f"Entities <= {bounds['max_entities']}",
            passed=len(entities) <= bounds["max_entities"],
            detail=f"Got {len(entities)}",
        ))
    if "min_synthesis" in bounds:
        result.checks.append(Check(
            name=f"Synthesis >= {bounds['min_synthesis']}",
            passed=len(synthesis) >= bounds["min_synthesis"],
            detail=f"Got {len(synthesis)}",
        ))


# ── Main runner ──


def _run_graph_quality(spec_name: str) -> GraphQualityResult:
    """Run all quality checks for a spec."""
    spec_path = _SPECS / spec_name
    with open(spec_path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    project = spec["project"]
    project_dir = _FIXTURES / project
    result = GraphQualityResult(name=spec["name"], project=project)

    nodes, edges = _load_graph(project_dir)
    if not nodes:
        result.checks.append(Check(name="Graph loaded", passed=False, detail="No graph found"))
        return result

    _check_required_entities(nodes, spec, result)
    _check_forbidden_entities(nodes, spec, result)
    _check_optional_entities(nodes, spec, result)
    _check_synthesis(nodes, spec, result)
    _check_structural(nodes, spec, result)
    _check_bounds(nodes, edges, spec, result)

    return result


def _write_quality_report(results: list[GraphQualityResult]) -> None:
    """Write combined quality report."""
    version_dir = _REPORTS / "v0.5.0"
    version_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    report = {
        "projects": [
            {
                "name": r.name,
                "project": r.project,
                "passed": r.passed,
                "failed": r.failed,
                "total": r.total,
                "node_count": r.node_count,
                "edge_count": r.edge_count,
                "entity_count": r.entity_count,
                "synthesis_count": r.synthesis_count,
                "checks": [asdict(c) for c in r.checks],
            }
            for r in results
        ]
    }
    (version_dir / "graph_quality.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    # Markdown
    lines = ["# Graph Quality Report — v0.5.0", ""]
    for r in results:
        lines.append(f"## {r.project}")
        lines.append(f"**Nodes:** {r.node_count} | **Edges:** {r.edge_count} | "
                      f"**Entities:** {r.entity_count} | **Synthesis:** {r.synthesis_count}")
        lines.append(f"**Passed:** {r.passed}/{r.total}")
        lines.append("")
        lines.append("| Check | Status | Detail |")
        lines.append("|-------|--------|--------|")
        for c in r.checks:
            status = "✓" if c.passed else "✗"
            lines.append(f"| {c.name} | {status} | {c.detail} |")
        lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("| Project | Passed | Failed | Entities | Nodes |")
    lines.append("|---------|--------|--------|----------|-------|")
    for r in results:
        lines.append(f"| {r.project} | {r.passed}/{r.total} | {r.failed} | {r.entity_count} | {r.node_count} |")

    (version_dir / "graph_quality.md").write_text("\n".join(lines), encoding="utf-8")


# ── Test class ──


class TestGraphQuality:

    @pytest.mark.parametrize("spec_file", SPEC_FILES)
    def test_graph_quality(self, spec_file: str):
        result = _run_graph_quality(spec_file)

        # Print results
        print(f"\n  {result.project}: {result.passed}/{result.total} checks passed")
        for c in result.checks:
            if not c.passed:
                print(f"    X {c.name}: {c.detail}")

        # Fail on required/forbidden/structural/bounds failures (not optionals)
        real_failures = [c for c in result.checks if not c.passed and "Optional" not in c.name]
        assert len(real_failures) == 0, (
            f"{result.project}: {len(real_failures)} failures: "
            + "; ".join(f"{c.name}: {c.detail}" for c in real_failures)
        )

    def test_all_quality(self):
        """Run all specs and generate combined report."""
        results = [_run_graph_quality(f) for f in SPEC_FILES]
        _write_quality_report(results)

        total_passed = sum(r.passed for r in results)
        total_checks = sum(r.total for r in results)
        print(f"\n  Graph Quality: {total_passed}/{total_checks} total checks passed")
        for r in results:
            print(f"    {r.project}: {r.passed}/{r.total}")
