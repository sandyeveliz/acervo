#!/usr/bin/env python3
"""Generate README demo tables from existing benchmark data.

Reads JSON reports from tests/integration/reports/ and produces formatted
output suitable for the README "Acervo in Action" section.

Usage:
    python scripts/generate_demo.py                    # all scenarios, console
    python scripts/generate_demo.py --scenario 01      # specific scenario
    python scripts/generate_demo.py --format markdown   # markdown tables
    python scripts/generate_demo.py --turns 1,10,25,50  # custom turn selection
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPORTS_DIR = Path(__file__).parent.parent / "tests" / "integration" / "reports"


def load_scenario(name: str) -> dict:
    """Load a scenario JSON report."""
    path = REPORTS_DIR / f"{name}.json"
    if not path.exists():
        # Try partial match
        matches = [f for f in REPORTS_DIR.glob("*.json") if name in f.stem]
        if matches:
            path = matches[0]
        else:
            print(f"Error: no report found for '{name}'", file=sys.stderr)
            print(f"Available: {', '.join(f.stem for f in REPORTS_DIR.glob('*.json'))}", file=sys.stderr)
            sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def discover_scenarios() -> list[str]:
    """List available scenario names (only full scenario reports, not prompt tests)."""
    results = []
    for f in sorted(REPORTS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if "summary" in data and "turns" in data:
                results.append(f.stem)
        except Exception:
            pass
    return results


def print_turn_detail(turn: dict, show_user_msg: bool = True) -> None:
    """Print a single turn in the side-by-side format."""
    n = turn["turn"]
    acervo = turn["acervo_tokens"]
    baseline = turn["baseline_tokens"]
    savings = turn["savings_pct"]
    nodes = turn["node_count"]
    hit = turn["context_hit"]
    user_msg = turn.get("user_msg", "")

    print(f'TURN {n} -- "{user_msg[:70]}{"..." if len(user_msg) > 70 else ""}"')
    print(f"                                           Without Acervo   With Acervo")
    print(f"                                           --------------   -----------")
    print(f"  Tokens sent to LLM:                      {baseline:>8,} tk    {acervo:>6,} tk")
    if savings > 0:
        print(f"  Savings:                                      --         {savings:.0f}%")
    print(f"  Graph nodes:                                  --         {nodes}")
    print(f"  Context hit:                                  --         {'Yes' if hit else 'No'}")
    print()


def print_progression_table(turns: list[dict], milestones: list[int]) -> None:
    """Print a token progression table for selected turns."""
    print("Turn     Without Acervo   With Acervo   Savings   Graph")
    print("-----    --------------   -----------   -------   -----")
    for t in turns:
        if t["turn"] in milestones:
            print(
                f"{t['turn']:>5d}    {t['baseline_tokens']:>8,} tk   {t['acervo_tokens']:>7,} tk"
                f"    {t['savings_pct']:>4.0f}%   {t['node_count']:>2d} nodes"
            )
    print()


def print_summary_table(scenarios: list[dict]) -> None:
    """Print the cross-scenario summary table."""
    print("| Scenario | Turns | Avg Acervo | Avg Baseline | Savings | Context Hit | Nodes |")
    print("|----------|------:|----------:|------------:|--------:|:----------:|------:|")
    total_turns = 0
    total_savings = 0
    total_hit = 0
    for data in scenarios:
        s = data["summary"]
        name = data["name"].replace("_", " ").replace("real", "").strip().title()
        print(
            f"| {name:<25s} | {s['total_turns']:>4d} "
            f"| {s['avg_acervo_tokens']:>5.0f} tk "
            f"| {s['avg_baseline_tokens']:>7,.0f} tk "
            f"| {s['avg_savings_pct']:>4.0f}% "
            f"| {s['context_hit_rate']*100:>3.0f}% "
            f"| {s['final_nodes']:>3d} |"
        )
        total_turns += s["total_turns"]
        total_savings += s["avg_savings_pct"] * s["total_turns"]
        total_hit += s["context_hit_rate"] * s["total_turns"]

    avg_savings = total_savings / total_turns if total_turns else 0
    avg_hit = total_hit / total_turns if total_turns else 0
    print()
    print(f"Average across {total_turns} turns: {avg_savings:.0f}% token savings, {avg_hit*100:.0f}% context hit rate.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate README demo from benchmark data")
    parser.add_argument("--scenario", help="Scenario name or partial match (default: all)")
    parser.add_argument("--format", choices=["console", "markdown"], default="console")
    parser.add_argument("--turns", help="Comma-separated turn numbers for detail view (default: 1,10,25,50)")
    parser.add_argument("--list", action="store_true", help="List available scenarios")
    parser.add_argument("--summary-only", action="store_true", help="Only print summary table")
    args = parser.parse_args()

    if args.list:
        for name in discover_scenarios():
            data = load_scenario(name)
            s = data["summary"]
            print(f"  {name:<35s}  {s['total_turns']} turns, {s['avg_savings_pct']:.0f}% savings")
        return

    milestones = [int(x) for x in args.turns.split(",")] if args.turns else [1, 10, 25, 50, 75, 100]

    if args.scenario:
        scenarios = [load_scenario(args.scenario)]
    else:
        scenarios = [load_scenario(name) for name in discover_scenarios()]

    if args.summary_only:
        print_summary_table(scenarios)
        return

    # Per-scenario detail
    for data in scenarios:
        s = data["summary"]
        name = data["name"]
        turns = data["turns"]

        display_name = name.replace("_", " ").title()
        print("=" * 70)
        print(f"  {display_name}")
        print(f"  {s['total_turns']} turns, {s['final_nodes']} nodes, {s['final_edges']} edges")
        print(f"  {s['avg_savings_pct']:.0f}% avg savings, {s['context_hit_rate']*100:.0f}% context hit rate")
        print("=" * 70)
        print()

        # Progression table
        available = [t["turn"] for t in turns]
        relevant_milestones = [m for m in milestones if m in available]
        if relevant_milestones:
            print_progression_table(turns, relevant_milestones)

        # Detail for selected turns
        for t in turns:
            if t["turn"] in relevant_milestones[:4]:  # limit detail to 4 turns
                print_turn_detail(t)

        print()

    # Summary
    if len(scenarios) > 1:
        print("=" * 70)
        print("  SUMMARY — All Scenarios")
        print("=" * 70)
        print()
        print_summary_table(scenarios)


if __name__ == "__main__":
    main()
