"""Export HTML report from existing JSON results — no LLM required.

Usage:
    python -m tests.integration.export_report
    python -m tests.integration.export_report --open
    python -m tests.integration.export_report --tier quick
    python -m tests.integration.export_report --tier full --version v0.2.2-1
    python -m tests.integration.export_report --tier full --open
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from tests.integration.metrics import ScenarioResult, TurnResult
from tests.integration.narrative import build_narrative
from tests.integration.reporter import html_report, html_report_narrative, html_report_quick

REPORTS_DIR = Path(__file__).parent / "reports"
ARCHIVE_DIR = REPORTS_DIR / "archive"
QUICK_PREFIXES = {"01_", "02_", "03_", "04_"}


def _load_result(path: Path) -> ScenarioResult:
    """Reconstruct a ScenarioResult from a JSON report file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    turns = [
        TurnResult(
            turn_number=t["turn"],
            phase=t["phase"],
            description=t.get("description", ""),
            acervo_tokens=t["acervo_tokens"],
            baseline_tokens=t["baseline_tokens"],
            savings_pct=t["savings_pct"],
            warm_tokens=t.get("warm_tokens", 0),
            hot_tokens=t.get("hot_tokens", 0),
            system_tokens=t.get("system_tokens", 0),
            user_tokens=t.get("user_tokens", 0),
            overhead_tokens=t.get("overhead_tokens", 0),
            context_hit=t["context_hit"],
            node_count=t["node_count"],
            edge_count=t["edge_count"],
            entities_expected=t.get("entities_expected", 0),
            entities_found=t.get("entities_found", 0),
            entities_missing=t.get("entities_missing", []),
            prepare_ms=t.get("prepare_ms", 0),
            process_ms=t.get("process_ms", 0),
            user_msg=t.get("user_msg", ""),
            assistant_msg=t.get("assistant_msg", ""),
            warm_context=t.get("warm_context", ""),
            entities_extracted=t.get("entities_extracted", []),
            topic=t.get("topic", ""),
        )
        for t in data.get("turns", [])
    ]

    summary = data.get("summary", {})
    return ScenarioResult(
        name=data.get("name", path.stem),
        category=data.get("category", ""),
        turns=turns,
        total_turns=summary.get("total_turns", len(turns)),
        final_node_count=summary.get("final_nodes", 0),
        final_edge_count=summary.get("final_edges", 0),
        personal_nodes=summary.get("personal_nodes", 0),
        universal_nodes=summary.get("universal_nodes", 0),
        phantom_entities=summary.get("phantom_entities", []),
        graph_nodes=data.get("graph_nodes", []),
        soft_failures=data.get("soft_failures", []),
        soft_total=max(
            len(data.get("soft_failures", [])),
            int(summary.get("soft_pass_rate", 1) > 0 and
                len(data.get("soft_failures", [])) / (1 - summary.get("soft_pass_rate", 1))
                if summary.get("soft_pass_rate", 1) < 1 else
                len(data.get("soft_failures", []))),
        ),
    )


def _read_project_version() -> str:
    """Read version from pyproject.toml."""
    pyproject = _root / "pyproject.toml"
    if pyproject.exists():
        text = pyproject.read_text(encoding="utf-8")
        match = re.search(r'version\s*=\s*"([^"]+)"', text)
        if match:
            return match.group(1)
    return "0.0.0"


def _next_version() -> str:
    """Auto-generate version tag from pyproject.toml + sequence number."""
    base = _read_project_version()
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(ARCHIVE_DIR.glob(f"v{base}-*"))
    seq = len(existing) + 1
    return f"v{base}-{seq}"


def _git_sha() -> str:
    """Get current git short SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(_root),
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _load_scenario_meta() -> dict[str, dict]:
    """Load persona/hook metadata from YAML scenario files."""
    import yaml

    meta = {}
    scenarios_dir = Path(__file__).parent / "scenarios"
    for path in sorted(scenarios_dir.glob("*.yaml")):
        if path.stem.startswith("prompt_"):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            meta[data.get("name", path.stem)] = {
                "persona": data.get("persona", ""),
                "persona_role": data.get("persona_role", ""),
                "narrative_hook": data.get("narrative_hook", ""),
            }
        except Exception:
            pass
    return meta


def _open_in_browser(path: Path) -> None:
    """Open a file in the default browser."""
    if sys.platform == "win32":
        subprocess.run(["start", "", str(path)], shell=True)
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)])
    else:
        subprocess.run(["xdg-open", str(path)])


def main():
    parser = argparse.ArgumentParser(description="Export Acervo HTML report from JSON results")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output HTML file path (default: reports/report.html)",
    )
    parser.add_argument(
        "--open", action="store_true",
        help="Open the report in the default browser",
    )
    parser.add_argument(
        "--tier", choices=["quick", "full"], default=None,
        help="Report tier: 'quick' for regression dashboard, 'full' for narrative report (default: auto)",
    )
    parser.add_argument(
        "--version", type=str, default=None,
        help="Version tag for archiving (default: auto from pyproject.toml)",
    )
    parser.add_argument(
        "--no-archive", action="store_true",
        help="Skip archiving, just produce the report",
    )
    args = parser.parse_args()

    # Find JSON reports (exclude prompt_* files)
    json_files = sorted(
        p for p in REPORTS_DIR.glob("*.json")
        if not p.stem.startswith("prompt_") and p.stem != "prompt_comparison"
    )
    if not json_files:
        print(f"No JSON reports found in {REPORTS_DIR}")
        print("Run the benchmarks first: python -m tests.integration.run_benchmarks")
        sys.exit(1)

    # Load results
    results = []
    for p in json_files:
        try:
            results.append(_load_result(p))
            print(f"  Loaded: {p.stem}")
        except Exception as e:
            print(f"  Skipped {p.stem}: {e}")

    if not results:
        print("No valid results to export.")
        sys.exit(1)

    # Auto-detect tier if not specified
    has_full = any(not any(r.name.startswith(p.rstrip("_")) for p in QUICK_PREFIXES)
                   for r in results)
    tier = args.tier or ("full" if has_full else "quick")

    # Load prompt comparison if available
    prompt_path = REPORTS_DIR / "prompt_comparison.json"
    prompt_data = None
    if prompt_path.exists():
        prompt_data = json.loads(prompt_path.read_text(encoding="utf-8"))
        print(f"  Loaded: prompt comparison ({len(prompt_data)} variants)")

    # Generate HTML based on tier
    if tier == "quick":
        print(f"\n  Generating quick regression dashboard...")
        html = html_report_quick(results)
    else:
        print(f"\n  Generating narrative report...")
        scenario_meta = _load_scenario_meta()
        narratives = []
        for r in results:
            meta = scenario_meta.get(r.name, {})
            narratives.append(build_narrative(
                r,
                persona=meta.get("persona", ""),
                persona_role=meta.get("persona_role", ""),
                hook=meta.get("narrative_hook", ""),
            ))
        version = args.version or _next_version()
        html = html_report_narrative(results, narratives, prompt_data, version=version)

    # Write latest report
    output_path = Path(args.output) if args.output else REPORTS_DIR / "report.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"  HTML report: {output_path}")
    print(f"  Size: {len(html) // 1024}KB")
    print(f"  Tier: {tier}")

    # Archive (full tier only, unless --no-archive)
    if tier == "full" and not args.no_archive and not args.output:
        version = args.version or _next_version()
        archive_path = ARCHIVE_DIR / version
        archive_path.mkdir(parents=True, exist_ok=True)

        # Copy report
        shutil.copy2(output_path, archive_path / "report.html")

        # Copy source JSONs
        for p in json_files:
            shutil.copy2(p, archive_path / p.name)

        # Write meta
        meta = {
            "version": version,
            "tier": tier,
            "timestamp": datetime.now().isoformat(),
            "git_sha": _git_sha(),
            "scenarios": [r.name for r in results],
            "avg_savings_pct": round(sum(r.avg_savings_pct for r in results) / len(results), 1),
            "total_turns": sum(r.total_turns for r in results),
        }
        (archive_path / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8",
        )
        print(f"  Archived: {archive_path}")

    if args.open:
        _open_in_browser(output_path)


if __name__ == "__main__":
    main()
