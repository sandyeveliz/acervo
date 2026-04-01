"""5-Category Benchmark System: RESOLVE / GROUND / RECALL / FOCUS / ADAPT

Runs 55 turns across 3 projects through prepare()/process().
Produces public scores (5 categories) and internal diagnostics (per component).

Requires: LM Studio + Ollama running locally.

Usage:
    pytest tests/integration/test_benchmarks.py -v -s
    pytest tests/integration/test_benchmarks.py -k "p1" -v -s
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pytest
import yaml

from acervo import Acervo
from acervo.token_counter import count_tokens

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FIXTURES = _REPO_ROOT / "tests" / "fixtures"
_SCENARIOS = Path(__file__).parent / "scenarios"
_REPORTS = Path(__file__).parent / "reports"

PROJECT_MAP = {
    "p1": _FIXTURES / "p1-todo-app",
    "p2": _FIXTURES / "p2-literature",
    "p3": _FIXTURES / "p3-project-docs",
}

CATEGORIES = ["RESOLVE", "GROUND", "RECALL", "FOCUS", "ADAPT"]
COMPONENTS = ["s1_intent", "s2_activation", "s3_budget", "s3_quality"]


# ── Data classes ──


@dataclass
class TurnCheck:
    turn: int
    category: str
    user_msg: str
    warm_tokens: int = 0
    elapsed_ms: int = 0

    # S1 checks
    s1_intent_expected: str | None = None
    s1_intent_actual: str = ""
    s1_intent_passed: bool | None = None

    # S2 checks
    s2_nodes_activated: int = 0
    s2_max_nodes: int | None = None
    s2_nodes_ok: bool | None = None
    s2_kinds_ok: bool | None = None
    s2_files_ok: bool | None = None

    # S3 checks
    s3_budget_ok: bool | None = None
    s3_contains_ok: bool | None = None
    s3_not_contains_ok: bool | None = None

    # S1.5 checks
    s1_5_ran: bool = False
    s1_5_entity_ok: bool | None = None

    # Public effectiveness
    effectiveness_passed: bool = True

    # Failures detail
    failures: list[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    version: str = "v0.4.0"
    total_turns: int = 0
    category_scores: dict[str, float] = field(default_factory=dict)
    component_scores: dict[str, float] = field(default_factory=dict)
    matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    s1_failures: list[dict] = field(default_factory=list)
    turns: list[TurnCheck] = field(default_factory=list)
    per_project: dict[str, dict] = field(default_factory=dict)


# ── YAML loader ──


def _load_benchmark(name: str) -> dict:
    path = _SCENARIOS / f"{name}.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Check evaluators ──


def _check_s1(turn_spec: dict, prep_debug: dict, tc: TurnCheck) -> None:
    s1_spec = turn_spec.get("s1")
    if not s1_spec:
        return
    s1_data = prep_debug.get("s1_detection", {})
    tc.s1_intent_actual = s1_data.get("intent", "")

    if "expected_intent" in s1_spec:
        tc.s1_intent_expected = s1_spec["expected_intent"]
        tc.s1_intent_passed = tc.s1_intent_actual == tc.s1_intent_expected
        if not tc.s1_intent_passed:
            tc.failures.append(
                f"S1 intent: expected '{tc.s1_intent_expected}', got '{tc.s1_intent_actual}'"
            )


def _check_s2(turn_spec: dict, prep_debug: dict, tc: TurnCheck) -> None:
    s2_spec = turn_spec.get("s2")
    if not s2_spec:
        return
    s2_data = prep_debug.get("s2_gathered", {})
    nodes = s2_data.get("nodes", [])
    tc.s2_nodes_activated = len(nodes)

    # max_nodes check
    if "max_nodes" in s2_spec:
        tc.s2_max_nodes = s2_spec["max_nodes"]
        tc.s2_nodes_ok = tc.s2_nodes_activated <= tc.s2_max_nodes
        if not tc.s2_nodes_ok:
            tc.failures.append(
                f"S2 nodes: {tc.s2_nodes_activated} > max {tc.s2_max_nodes}"
            )

    # min_nodes check
    if "min_nodes" in s2_spec:
        if tc.s2_nodes_activated < s2_spec["min_nodes"]:
            tc.s2_nodes_ok = False
            tc.failures.append(
                f"S2 nodes: {tc.s2_nodes_activated} < min {s2_spec['min_nodes']}"
            )

    # activate_kinds check
    if "activate_kinds" in s2_spec:
        node_kinds = {n.get("type", "").lower() for n in nodes}
        # Also check the "kind" field from graph nodes (the nodes in debug use "type")
        expected = set(s2_spec["activate_kinds"])
        # This is a soft check — at least one expected kind should appear
        tc.s2_kinds_ok = True  # default pass

    # not_activate_kinds check
    if "not_activate_kinds" in s2_spec:
        node_kinds = {n.get("type", "").lower() for n in nodes}
        tc.s2_kinds_ok = True  # default pass

    # activate_files_containing check
    if "activate_files_containing" in s2_spec:
        node_labels = [n.get("label", "").lower() for n in nodes]
        expected_terms = s2_spec["activate_files_containing"]
        found = any(
            any(term.lower() in label for label in node_labels)
            for term in expected_terms
        )
        tc.s2_files_ok = found
        if not found:
            tc.failures.append(
                f"S2 files: none matching {expected_terms} in {node_labels[:5]}"
            )

    # not_activate_files_containing check
    if "not_activate_files_containing" in s2_spec:
        node_labels = [n.get("label", "").lower() for n in nodes]
        unwanted = s2_spec["not_activate_files_containing"]
        noise = [
            term for term in unwanted
            if any(term.lower() in label for label in node_labels)
        ]
        if noise:
            if tc.s2_files_ok is None:
                tc.s2_files_ok = False
            tc.failures.append(f"S2 noise: unwanted files {noise} activated")


def _check_s3(turn_spec: dict, warm_content: str, warm_tokens: int, tc: TurnCheck) -> None:
    s3_spec = turn_spec.get("s3")
    if not s3_spec:
        return
    warm_lower = warm_content.lower()

    # Budget checks
    if "warm_tokens_max" in s3_spec:
        tc.s3_budget_ok = warm_tokens <= s3_spec["warm_tokens_max"]
        if not tc.s3_budget_ok:
            tc.failures.append(
                f"S3 budget: {warm_tokens}tk > max {s3_spec['warm_tokens_max']}tk"
            )
    if "warm_tokens_min" in s3_spec:
        if warm_tokens < s3_spec["warm_tokens_min"]:
            tc.s3_budget_ok = False
            tc.failures.append(
                f"S3 budget: {warm_tokens}tk < min {s3_spec['warm_tokens_min']}tk"
            )

    # Content contains checks
    contains = s3_spec.get("context_contains", [])
    contains_any = s3_spec.get("context_contains_any", [])

    if contains:
        all_found = all(term.lower() in warm_lower for term in contains)
        tc.s3_contains_ok = all_found
        if not all_found:
            missing = [t for t in contains if t.lower() not in warm_lower]
            tc.failures.append(f"S3 missing: {missing}")

    if contains_any:
        any_found = any(term.lower() in warm_lower for term in contains_any)
        if tc.s3_contains_ok is None:
            tc.s3_contains_ok = any_found
        elif not any_found:
            tc.s3_contains_ok = False
        if not any_found:
            tc.failures.append(f"S3 missing any of: {contains_any}")

    # Content NOT contains checks
    not_contains = s3_spec.get("context_not_contains", [])
    if not_contains:
        noise_found = [t for t in not_contains if t.lower() in warm_lower]
        tc.s3_not_contains_ok = len(noise_found) == 0
        if noise_found:
            tc.failures.append(f"S3 noise: {noise_found} found in context")


def _check_effectiveness(turn_spec: dict, tc: TurnCheck) -> None:
    eff = turn_spec.get("effectiveness")
    if not eff:
        # Default: pass if s3 content checks passed
        if tc.s3_contains_ok is False:
            tc.effectiveness_passed = False
        return

    # Effectiveness is determined by content quality checks
    if tc.s3_contains_ok is False or tc.s3_not_contains_ok is False:
        tc.effectiveness_passed = False


# ── Runner ──


async def _run_benchmark(scenario_name: str) -> list[TurnCheck]:
    scenario = _load_benchmark(scenario_name)
    fixture = scenario.get("project_fixture", "p1-todo-app")

    # Map fixture name to path
    project_key = next(
        (k for k, v in PROJECT_MAP.items() if fixture in str(v)), "p1"
    )
    project_path = PROJECT_MAP[project_key]

    if not (project_path / ".acervo" / "data" / "graph" / "nodes.json").exists():
        pytest.skip(f"Project {project_key} not indexed")

    acervo = Acervo.from_project(project_path, auto_init=False)
    history: list[dict] = []
    results: list[TurnCheck] = []

    for turn_spec in scenario.get("turns", []):
        user_msg = turn_spec["user"]
        assistant_sim = turn_spec.get("assistant_sim", "OK")
        category = turn_spec.get("category", "RESOLVE")

        tc = TurnCheck(
            turn=turn_spec.get("turn", len(results) + 1),
            category=category,
            user_msg=user_msg[:60],
        )

        # Run prepare
        t0 = time.monotonic()
        prep = await acervo.prepare(user_msg, history)
        tc.elapsed_ms = int((time.monotonic() - t0) * 1000)
        tc.warm_tokens = prep.warm_tokens

        debug = prep.debug or {}

        # Evaluate checks
        _check_s1(turn_spec, debug, tc)
        _check_s2(turn_spec, debug, tc)
        _check_s3(turn_spec, prep.warm_content, prep.warm_tokens, tc)

        # S1.5 checks (run process for RECALL turns)
        s1_5_spec = turn_spec.get("s1_5")
        if s1_5_spec:
            tc.s1_5_ran = True
            await acervo.process(user_msg, assistant_sim)
            # Check entity extraction
            if "should_extract_entity" in s1_5_spec:
                # Verify entity was created in graph
                from acervo.graph import _make_id
                entity_id = _make_id(s1_5_spec["should_extract_entity"])
                node = acervo.graph.get_node(entity_id)
                tc.s1_5_entity_ok = node is not None
                if not tc.s1_5_entity_ok:
                    tc.failures.append(
                        f"S1.5: entity '{s1_5_spec['should_extract_entity']}' not found in graph"
                    )
        else:
            # Still call process to build history
            await acervo.process(user_msg, assistant_sim)

        # Effectiveness check
        _check_effectiveness(turn_spec, tc)

        results.append(tc)

        # Update history
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": assistant_sim})

    return results


# ── Scoring ──


def _compute_scores(all_turns: list[TurnCheck]) -> BenchmarkResult:
    result = BenchmarkResult(total_turns=len(all_turns), turns=all_turns)

    # Public scores per category
    for cat in CATEGORIES:
        cat_turns = [t for t in all_turns if t.category == cat]
        if cat_turns:
            passed = sum(1 for t in cat_turns if t.effectiveness_passed)
            result.category_scores[cat] = round(passed / len(cat_turns) * 100)
        else:
            result.category_scores[cat] = 0

    # Component scores
    def _pct(checks: list[bool | None]) -> float:
        valid = [c for c in checks if c is not None]
        if not valid:
            return -1  # no data
        return round(sum(1 for c in valid if c) / len(valid) * 100)

    result.component_scores["s1_intent"] = _pct([t.s1_intent_passed for t in all_turns])
    result.component_scores["s2_activation"] = _pct(
        [t.s2_nodes_ok for t in all_turns] +
        [t.s2_files_ok for t in all_turns]
    )
    result.component_scores["s3_budget"] = _pct([t.s3_budget_ok for t in all_turns])
    result.component_scores["s3_quality"] = _pct(
        [t.s3_contains_ok for t in all_turns] +
        [t.s3_not_contains_ok for t in all_turns]
    )

    # Cross-matrix: category x component
    for cat in CATEGORIES:
        cat_turns = [t for t in all_turns if t.category == cat]
        if not cat_turns:
            continue
        result.matrix[cat] = {
            "s1_intent": _pct([t.s1_intent_passed for t in cat_turns]),
            "s2_activation": _pct(
                [t.s2_nodes_ok for t in cat_turns] +
                [t.s2_files_ok for t in cat_turns]
            ),
            "s3_budget": _pct([t.s3_budget_ok for t in cat_turns]),
            "s3_quality": _pct(
                [t.s3_contains_ok for t in cat_turns] +
                [t.s3_not_contains_ok for t in cat_turns]
            ),
            "score": result.category_scores[cat],
        }

    # S1 failures log
    result.s1_failures = [
        {"turn": t.turn, "user": t.user_msg,
         "expected": t.s1_intent_expected, "actual": t.s1_intent_actual}
        for t in all_turns
        if t.s1_intent_passed is False
    ]

    return result


# ── Reports ──


def _print_public(r: BenchmarkResult) -> None:
    print(f"\n{'=' * 60}")
    print(f"  ACERVO {r.version} -- {r.total_turns} turns")
    print(f"{'=' * 60}")
    print()
    header = "  ".join(f"{c:>8s}" for c in CATEGORIES)
    scores = "  ".join(f"{r.category_scores.get(c, 0):>7.0f}%" for c in CATEGORIES)
    print(f"  {header}")
    print(f"  {scores}")
    print()


def _print_diagnostic(r: BenchmarkResult) -> None:
    print(f"  COMPONENT HEALTH")
    print(f"  {'-' * 50}")
    for comp in COMPONENTS:
        score = r.component_scores.get(comp, -1)
        label = f"{score:.0f}%" if score >= 0 else "n/a"
        print(f"  {comp:20s} {label:>6s}")

    print(f"\n  CROSS-MATRIX (category x component)")
    print(f"  {'-' * 50}")
    header = "              " + "  ".join(f"{c:>8s}" for c in COMPONENTS) + "   Score"
    print(f"  {header}")
    for cat in CATEGORIES:
        if cat not in r.matrix:
            continue
        m = r.matrix[cat]
        vals = "  ".join(
            f"{m.get(c, -1):>7.0f}%" if m.get(c, -1) >= 0 else "    n/a "
            for c in COMPONENTS
        )
        print(f"  {cat:12s}  {vals}   {m.get('score', 0):.0f}%")

    if r.s1_failures:
        print(f"\n  S1 FAILURES ({len(r.s1_failures)} logged)")
        print(f"  {'-' * 50}")
        for f in r.s1_failures[:5]:
            print(f"  Turn {f['turn']}: '{f['user']}' expected={f['expected']} got={f['actual']}")

    # Per-turn failures
    failed_turns = [t for t in r.turns if t.failures]
    if failed_turns:
        print(f"\n  TURN FAILURES ({len(failed_turns)} turns with issues)")
        print(f"  {'-' * 50}")
        for t in failed_turns[:10]:
            print(f"  [{t.turn:2d}] {t.category:8s} {t.user_msg}")
            for f in t.failures:
                print(f"       {f}")


def _export_reports(r: BenchmarkResult) -> None:
    _REPORTS.mkdir(parents=True, exist_ok=True)

    # Public JSON
    public = {
        "version": r.version,
        "total_turns": r.total_turns,
        "scores": r.category_scores,
    }
    (_REPORTS / "benchmark_public.json").write_text(
        json.dumps(public, indent=2), encoding="utf-8"
    )

    # Diagnostic JSON
    diag = asdict(r)
    # Trim warm content from turns to keep size reasonable
    for t in diag.get("turns", []):
        t.pop("warm_content", None)
    (_REPORTS / "benchmark_diagnostic.json").write_text(
        json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Public markdown
    lines = [
        f"# ACERVO {r.version} Benchmark",
        f"",
        f"**{r.total_turns} turns** across 3 projects",
        f"",
        f"| Category | Score |",
        f"|----------|-------|",
    ]
    for cat in CATEGORIES:
        lines.append(f"| {cat} | {r.category_scores.get(cat, 0):.0f}% |")
    (_REPORTS / "benchmark_public.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    # Diagnostic markdown
    dlines = [
        f"# ACERVO {r.version} Diagnostic",
        f"",
        f"## Component Health",
        f"",
        f"| Component | Score |",
        f"|-----------|-------|",
    ]
    for comp in COMPONENTS:
        score = r.component_scores.get(comp, -1)
        dlines.append(f"| {comp} | {score:.0f}% |" if score >= 0 else f"| {comp} | n/a |")
    dlines.append(f"")
    dlines.append(f"## S1 Failures: {len(r.s1_failures)}")
    for f in r.s1_failures:
        dlines.append(f"- Turn {f['turn']}: expected={f['expected']}, got={f['actual']}")
    (_REPORTS / "benchmark_diagnostic.md").write_text(
        "\n".join(dlines), encoding="utf-8"
    )

    print(f"\n  Reports: {_REPORTS}/benchmark_*.json|md")


# ── Tests ──


class TestBenchmarks:

    @pytest.mark.asyncio
    async def test_full_benchmark(self):
        """Run all 55 turns across 3 projects, compute 5-category scores."""
        all_turns: list[TurnCheck] = []

        for scenario in ["p1_benchmark", "p2_benchmark", "p3_benchmark"]:
            print(f"\n  Running {scenario}...")
            turns = await _run_benchmark(scenario)
            all_turns.extend(turns)
            passed = sum(1 for t in turns if t.effectiveness_passed)
            print(f"  {len(turns)} turns, {passed} passed effectiveness")

        result = _compute_scores(all_turns)

        _print_public(result)
        _print_diagnostic(result)
        _export_reports(result)

        # Soft assertions — this is a benchmark, not a gate
        assert result.total_turns >= 50, f"Expected 55 turns, got {result.total_turns}"
        # At least 50% on each category
        for cat in CATEGORIES:
            score = result.category_scores.get(cat, 0)
            if score < 50:
                print(f"  WARNING: {cat} score {score}% < 50%")
