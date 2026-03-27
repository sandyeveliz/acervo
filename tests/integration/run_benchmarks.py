"""Standalone benchmark runner — no pytest required.

Usage:
    python -m tests.integration.run_benchmarks
    python -m tests.integration.run_benchmarks --scenario 01_programming
    python -m tests.integration.run_benchmarks --format json
    python -m tests.integration.run_benchmarks --list
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Ensure project root is on path
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from acervo import Acervo, OpenAIClient
from tests.integration.framework import (
    ScenarioRunner,
    discover_scenarios,
    load_scenario,
    run_final_assertions,
)
from tests.integration.reporter import (
    console_report,
    html_report,
    json_report,
    markdown_report,
)

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
REPORTS_DIR = Path(__file__).parent / "reports"


def create_memory() -> tuple[Acervo, Path]:
    """Create an Acervo instance with temp storage."""
    client = OpenAIClient(
        base_url=os.getenv("ACERVO_LLM_BASE_URL",
                           os.getenv("ACERVO_LIGHT_MODEL_URL", "http://localhost:1234/v1")),
        model=os.getenv("ACERVO_LLM_MODEL",
                        os.getenv("ACERVO_LIGHT_MODEL", "qwen2.5-3b-instruct")),
        api_key=os.getenv("ACERVO_LIGHT_API_KEY", "lm-studio"),
    )
    tmp = Path(tempfile.mkdtemp()) / ".acervo"
    graph_path = tmp / "data" / "graph"
    graph_path.mkdir(parents=True, exist_ok=True)
    memory = Acervo(llm=client, owner="Sandy", persist_path=graph_path)
    return memory, tmp.parent


async def run_scenario(scenario_path: Path, fmt: str) -> dict:
    """Run a single scenario and return results."""
    scenario = load_scenario(scenario_path)
    memory, tmp_root = create_memory()

    try:
        runner = ScenarioRunner(memory)
        result = await runner.run_with_acervo(scenario)

        # Run assertions (don't abort — report results)
        try:
            run_final_assertions(result)
            passed = True
        except AssertionError as e:
            passed = False
            logging.warning("Hard assertion failed: %s", e)

        # Generate reports
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        scenario_id = scenario_path.stem

        md = markdown_report(result)
        (REPORTS_DIR / f"{scenario_id}.md").write_text(md, encoding="utf-8")

        jr = json_report(result)
        (REPORTS_DIR / f"{scenario_id}.json").write_text(
            json.dumps(jr, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        if fmt == "json":
            print(json.dumps(jr, indent=2, ensure_ascii=False))
        elif fmt == "markdown":
            print(md)
        # Console format is printed at the end for all scenarios

        return {
            "result": result,
            "passed": passed,
            "scenario_id": scenario_id,
        }
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


async def main():
    parser = argparse.ArgumentParser(description="Acervo integration benchmarks")
    parser.add_argument(
        "--scenario", type=str, default=None,
        help="Run a specific scenario (e.g., 01_programming)",
    )
    parser.add_argument(
        "--format", type=str, default="console",
        choices=["console", "markdown", "json", "html"],
        help="Output format (default: console)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available scenarios",
    )
    args = parser.parse_args()

    if args.list:
        for p in discover_scenarios(SCENARIOS_DIR):
            scenario = load_scenario(p)
            print(f"  {p.stem:<30} {scenario.category:<16} {len(scenario.turns)} turns")
        return

    # Find scenarios to run
    if args.scenario:
        matches = [
            p for p in discover_scenarios(SCENARIOS_DIR)
            if args.scenario in p.stem
        ]
        if not matches:
            print(f"No scenario matching '{args.scenario}' found.")
            sys.exit(1)
    else:
        matches = discover_scenarios(SCENARIOS_DIR)

    if not matches:
        print("No scenarios found in", SCENARIOS_DIR)
        sys.exit(1)

    # Run scenarios
    all_results = []
    for scenario_path in matches:
        print(f"\nRunning: {scenario_path.stem} ...")
        run = await run_scenario(scenario_path, args.format)
        all_results.append(run)

    # Print summary
    results = [r["result"] for r in all_results]

    if args.format == "html" and results:
        # Load prompt comparison data if available
        prompt_json_path = REPORTS_DIR / "prompt_comparison.json"
        prompt_data = None
        if prompt_json_path.exists():
            prompt_data = json.loads(prompt_json_path.read_text(encoding="utf-8"))

        html = html_report(results, prompt_data)
        html_path = REPORTS_DIR / "report.html"
        html_path.write_text(html, encoding="utf-8")
        print(f"  HTML report: {html_path}")

    if args.format == "console" and results:
        print(console_report(results))

    # Status
    if results:
        passed = sum(1 for r in all_results if r["passed"])
        total = len(all_results)
        print(f"  Results: {passed}/{total} passed")
        if passed < total:
            failed = [r["scenario_id"] for r in all_results if not r["passed"]]
            print(f"  Failed: {', '.join(failed)}")
            sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(name)s:%(message)s",
    )
    asyncio.run(main())
