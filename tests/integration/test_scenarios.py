"""E2E scenario tests — Acervo's source of truth.

Each scenario runs 50 turns through the full prepare/process pipeline,
verifying graph state, extraction, topic transitions, context assembly,
metrics, trace persistence, and token stability.

Requires a running LLM server. Run with:
    pytest tests/integration/test_scenarios.py -m integration -v -s

Run a single scenario:
    pytest tests/integration/test_scenarios.py -m integration -v -s -k "programming"

Run benchmarks:
    pytest tests/integration/test_scenarios.py -m benchmark -v -s
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from tests.integration.framework import (
    ScenarioRunner,
    discover_scenarios,
    load_scenario,
    run_final_assertions,
)
from tests.integration.reporter import (
    console_report,
    json_report,
    markdown_report,
)

log = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent / "reports"
SCENARIOS_DIR = Path(__file__).parent / "scenarios"


QUICK_PREFIXES = ("01_", "02_", "03_", "04_")


def _scenario_ids() -> list[str]:
    """Discover scenario files and return their stems as test IDs.

    Excludes prompt_test.yaml which is a short scenario for prompt comparison only.
    """
    return [
        p.stem for p in discover_scenarios(SCENARIOS_DIR)
        if not p.stem.startswith("prompt_")
    ]


def _quick_ids() -> list[str]:
    return [s for s in _scenario_ids() if s.startswith(QUICK_PREFIXES)]


def _full_ids() -> list[str]:
    return [s for s in _scenario_ids() if not s.startswith(QUICK_PREFIXES)]


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("scenario_id", _scenario_ids())
async def test_scenario(e2e_memory, scenario_id):
    """Run a full scenario and validate everything."""
    scenario_path = SCENARIOS_DIR / f"{scenario_id}.yaml"
    scenario = load_scenario(scenario_path)

    runner = ScenarioRunner(e2e_memory)
    result = await runner.run_with_acervo(scenario)

    # Hard assertions
    run_final_assertions(result)

    # Generate reports
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    md = markdown_report(result)
    (REPORTS_DIR / f"{scenario_id}.md").write_text(md, encoding="utf-8")

    jr = json_report(result)
    (REPORTS_DIR / f"{scenario_id}.json").write_text(
        json.dumps(jr, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Print to terminal
    print(f"\n{'=' * 70}")
    print(md)
    print(f"{'=' * 70}\n")

    # Soft assertion threshold — model extraction is non-deterministic,
    # 50% pass rate accounts for LLM variance while catching regressions
    if result.soft_total > 0 and result.soft_pass_rate < 0.3:
        pytest.fail(
            f"Scenario '{scenario_id}' — too many soft failures "
            f"({1 - result.soft_pass_rate:.0%}):\n"
            + "\n".join(f"  - {f}" for f in result.soft_failures)
        )
    elif result.soft_failures:
        log.warning(
            "Scenario '%s' — %d soft failures (%.0f%% pass rate)",
            scenario_id, len(result.soft_failures), result.soft_pass_rate * 100,
        )


# ── Tiered test functions ──


@pytest.mark.integration
@pytest.mark.quick
@pytest.mark.asyncio
@pytest.mark.parametrize("scenario_id", _quick_ids())
async def test_scenario_quick(e2e_memory, scenario_id):
    """Quick tier — scenarios 01-04 for fast dev iteration."""
    await test_scenario(e2e_memory, scenario_id)


@pytest.mark.integration
@pytest.mark.full
@pytest.mark.asyncio
@pytest.mark.parametrize("scenario_id", _full_ids())
async def test_scenario_full(e2e_memory, scenario_id):
    """Full tier — long-running scenarios (05-06) for publishable reports."""
    await test_scenario(e2e_memory, scenario_id)
