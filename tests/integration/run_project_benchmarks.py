"""Project benchmark runner — tests context retrieval on indexed projects.

Unlike run_benchmarks.py (which tests conversation memory), this tests
the full S1->S2->S3 pipeline against pre-indexed project directories.

Usage:
    python -m tests.integration.run_project_benchmarks
    python -m tests.integration.run_project_benchmarks --scenario p1_code_project
    python -m tests.integration.run_project_benchmarks --format json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from acervo import Acervo
from acervo.token_counter import count_tokens

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
REPORTS_DIR = Path(__file__).parent / "reports"

log = logging.getLogger(__name__)


@dataclass
class QuestionResult:
    text: str
    intent: str
    warm_tokens: int = 0
    warm_content: str = ""
    stages: list[str] = field(default_factory=list)
    context_checks_passed: int = 0
    context_checks_total: int = 0
    context_failures: list[str] = field(default_factory=list)
    elapsed_ms: int = 0
    passed: bool = False


@dataclass
class ProjectBenchmarkResult:
    name: str
    project_path: str
    node_count: int = 0
    edge_count: int = 0
    questions: list[QuestionResult] = field(default_factory=list)
    total_passed: int = 0
    total_questions: int = 0
    elapsed_seconds: float = 0.0


def load_project_scenario(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


async def run_project_benchmark(scenario_path: Path) -> ProjectBenchmarkResult:
    scenario = load_project_scenario(scenario_path)
    name = scenario["name"]
    project_path = Path(scenario["project_path"])

    result = ProjectBenchmarkResult(
        name=name,
        project_path=str(project_path),
    )

    if not (project_path / ".acervo").exists():
        print(f"  SKIP: {name} — no .acervo/ at {project_path}")
        return result

    t0 = time.monotonic()

    # Load Acervo from the project (uses existing graph, no re-indexing)
    try:
        acervo = Acervo.from_project(project_path, auto_init=False)
    except Exception as e:
        print(f"  ERROR: {name} — failed to load: {e}")
        return result

    stats = acervo.get_graph_stats()
    result.node_count = stats.get("node_count", 0)
    result.edge_count = stats.get("edge_count", 0)

    print(f"  Loaded: {result.node_count} nodes, {result.edge_count} edges")

    for q in scenario.get("questions", []):
        qr = await _run_question(acervo, q)
        result.questions.append(qr)
        if qr.passed:
            result.total_passed += 1
        result.total_questions += 1

        status = "PASS" if qr.passed else "FAIL"
        print(f"    [{status}] {q['text'][:60]} — {qr.warm_tokens}tk, {qr.elapsed_ms}ms")
        for f in qr.context_failures:
            print(f"           {f}")

    result.elapsed_seconds = time.monotonic() - t0
    return result


async def _run_question(acervo: Acervo, q: dict) -> QuestionResult:
    text = q["text"]
    expected_intent = q.get("intent", "specific")

    t0 = time.monotonic()

    try:
        prep = await acervo.prepare(text, [])
    except Exception as e:
        return QuestionResult(
            text=text,
            intent=expected_intent,
            context_failures=[f"prepare() crashed: {e}"],
        )

    elapsed = int((time.monotonic() - t0) * 1000)

    qr = QuestionResult(
        text=text,
        intent=expected_intent,
        warm_tokens=prep.warm_tokens,
        warm_content=prep.warm_content,
        stages=list(prep.stages),
        elapsed_ms=elapsed,
    )

    # Check context contains expected strings
    warm_lower = prep.warm_content.lower()
    checks_total = 0
    checks_passed = 0

    for expected in q.get("expect_in_context", []):
        checks_total += 1
        if expected.lower() in warm_lower:
            checks_passed += 1
        else:
            qr.context_failures.append(f"expected in context: '{expected}'")

    for not_expected in q.get("expect_not_in_context", []):
        checks_total += 1
        if not_expected.lower() not in warm_lower:
            checks_passed += 1
        else:
            qr.context_failures.append(f"should NOT be in context: '{not_expected}'")

    # Check warm_tokens > 0 for non-chat intents
    if expected_intent != "chat":
        checks_total += 1
        if prep.warm_tokens > 0:
            checks_passed += 1
        else:
            qr.context_failures.append("warm_tokens is 0 (no context injected)")

    qr.context_checks_passed = checks_passed
    qr.context_checks_total = checks_total
    qr.passed = checks_passed == checks_total

    return qr


def console_report(results: list[ProjectBenchmarkResult]) -> None:
    print("\n" + "=" * 60)
    print("PROJECT BENCHMARK RESULTS")
    print("=" * 60)

    total_pass = 0
    total_q = 0

    for r in results:
        pct = (r.total_passed / r.total_questions * 100) if r.total_questions else 0
        icon = "PASS" if r.total_passed == r.total_questions else "FAIL"
        print(f"\n[{icon}] {r.name}: {r.total_passed}/{r.total_questions} ({pct:.0f}%) — {r.node_count}n, {r.elapsed_seconds:.1f}s")
        total_pass += r.total_passed
        total_q += r.total_questions

    print(f"\n{'=' * 60}")
    pct = (total_pass / total_q * 100) if total_q else 0
    print(f"TOTAL: {total_pass}/{total_q} ({pct:.0f}%)")
    print("=" * 60)


def json_report(results: list[ProjectBenchmarkResult], path: Path) -> None:
    data = [asdict(r) for r in results]
    # Strip warm_content to keep report small
    for r in data:
        for q in r["questions"]:
            q["warm_content_preview"] = q["warm_content"][:200]
            del q["warm_content"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"\nJSON report: {path}")


async def main():
    parser = argparse.ArgumentParser(description="Run project benchmarks")
    parser.add_argument("--scenario", help="Run specific scenario (e.g., p1_code_project)")
    parser.add_argument("--format", choices=["console", "json", "both"], default="both")
    parser.add_argument("--list", action="store_true", help="List available scenarios")
    args = parser.parse_args()

    # Discover project scenarios (p1_, p2_, p3_, p4_ prefix)
    scenarios = sorted(
        s for s in SCENARIOS_DIR.glob("p[0-9]*.yaml")
    )

    if args.list:
        for s in scenarios:
            data = yaml.safe_load(s.read_text())
            print(f"  {s.stem}: {data.get('description', '')}")
        return

    if args.scenario:
        scenarios = [s for s in scenarios if args.scenario in s.stem]
        if not scenarios:
            print(f"No scenario matching '{args.scenario}'")
            return

    print(f"Running {len(scenarios)} project benchmark(s)...\n")

    results = []
    for scenario_path in scenarios:
        print(f"--- {scenario_path.stem} ---")
        r = await run_project_benchmark(scenario_path)
        results.append(r)

    if args.format in ("console", "both"):
        console_report(results)

    if args.format in ("json", "both"):
        json_report(results, REPORTS_DIR / "project_benchmarks.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())
