"""Layer 3: Conversation benchmarks over indexed projects.

Runs multi-turn conversations (15-20 turns) through prepare()/process()
against each indexed fixture project. Measures token efficiency, context
quality, and compression ratio.

Requires: LM Studio running (for S1 utility model calls)

Usage:
    pytest tests/integration/test_indexed_conversations.py -v -s
    pytest tests/integration/test_indexed_conversations.py -k "p1" -v -s
"""

from __future__ import annotations

import json
import sys
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


@dataclass
class TurnMetric:
    turn: int
    phase: str
    user_msg: str
    warm_tokens: int = 0
    hot_tokens: int = 0
    total_tokens: int = 0
    baseline_tokens: int = 0
    has_context: bool = False
    intent: str = ""
    nodes_activated: int = 0
    chunks_selected: int = 0
    context_checks_passed: int = 0
    context_checks_total: int = 0
    elapsed_ms: int = 0


@dataclass
class ConversationResult:
    name: str
    project: str
    total_turns: int = 0
    avg_warm_tokens: float = 0
    avg_baseline_tokens: float = 0
    savings_pct: float = 0
    context_hit_rate: float = 0
    turns: list[TurnMetric] = field(default_factory=list)


def _load_scenario(name: str) -> dict:
    path = _SCENARIOS / f"{name}.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


async def _run_conversation(scenario_name: str) -> ConversationResult:
    scenario = _load_scenario(scenario_name)
    project_key = scenario.get("project_path_key", "p1")
    project_path = PROJECT_MAP[project_key]

    if not (project_path / ".acervo" / "data" / "graph" / "nodes.json").exists():
        pytest.skip(f"Project {project_key} not indexed")

    acervo = Acervo.from_project(project_path, auto_init=False)
    result = ConversationResult(name=scenario_name, project=project_key)

    history: list[dict] = []
    baseline_running = 0

    for i, turn in enumerate(scenario.get("turns", [])):
        user_msg = turn["user_msg"]
        assistant_msg = turn.get("assistant_msg", "OK")
        phase = turn.get("phase", "")

        # Baseline: cumulative full history tokens
        baseline_running += count_tokens(user_msg)
        if assistant_msg:
            baseline_running += count_tokens(assistant_msg)

        import time
        t0 = time.monotonic()

        prep = await acervo.prepare(user_msg, history)
        elapsed = int((time.monotonic() - t0) * 1000)

        # Process (simulated assistant response)
        await acervo.process(user_msg, assistant_msg)

        # Update history
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": assistant_msg})

        # Extract metrics
        debug = prep.debug or {}
        s1 = debug.get("s1_detection", {})
        s2 = debug.get("s2_gathered", {})

        metric = TurnMetric(
            turn=i + 1,
            phase=phase,
            user_msg=user_msg[:60],
            warm_tokens=prep.warm_tokens,
            hot_tokens=prep.hot_tokens,
            total_tokens=prep.total_tokens,
            baseline_tokens=baseline_running,
            has_context=prep.has_context,
            intent=s1.get("intent", ""),
            nodes_activated=len(s2.get("nodes", [])),
            chunks_selected=s2.get("chunks_selected", 0) if isinstance(s2.get("chunks_selected"), int) else 0,
            elapsed_ms=elapsed,
        )

        # Context content checks
        expect_in = turn.get("expect_in_context", [])
        if expect_in:
            warm_lower = prep.warm_content.lower()
            metric.context_checks_total = len(expect_in)
            metric.context_checks_passed = sum(
                1 for e in expect_in if e.lower() in warm_lower
            )

        result.turns.append(metric)

    result.total_turns = len(result.turns)
    if result.turns:
        result.avg_warm_tokens = sum(t.warm_tokens for t in result.turns) / len(result.turns)
        result.avg_baseline_tokens = sum(t.baseline_tokens for t in result.turns) / len(result.turns)
        if result.avg_baseline_tokens > 0:
            result.savings_pct = (1 - result.avg_warm_tokens / result.avg_baseline_tokens) * 100
        hits = sum(1 for t in result.turns if t.has_context)
        result.context_hit_rate = hits / len(result.turns) * 100

    return result


def _print_result(r: ConversationResult) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {r.name} ({r.project}) - {r.total_turns} turns")
    print(f"{'-' * 60}")
    print(f"  Avg Acervo:   {r.avg_warm_tokens:.0f} tk")
    print(f"  Avg Baseline: {r.avg_baseline_tokens:.0f} tk")
    print(f"  Savings:      {r.savings_pct:.0f}%")
    print(f"  Hit Rate:     {r.context_hit_rate:.0f}%")
    print()
    for t in r.turns:
        status = "HIT" if t.has_context else "---"
        print(f"  [{t.turn:2d}] {status} {t.warm_tokens:4d}tk "
              f"({t.intent:8s}) {t.phase:12s} {t.user_msg}")


def _export_result(r: ConversationResult) -> None:
    _REPORTS.mkdir(parents=True, exist_ok=True)
    path = _REPORTS / f"conversation_{r.name}.json"
    data = asdict(r)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Tests ──


class TestConversationP1:
    @pytest.mark.asyncio
    async def test_p1_conversation(self):
        result = await _run_conversation("p1_indexed_code")
        _print_result(result)
        _export_result(result)

        assert result.total_turns >= 15
        assert result.context_hit_rate >= 60, (
            f"Context hit rate {result.context_hit_rate:.0f}% < 60%"
        )


class TestConversationP2:
    @pytest.mark.asyncio
    async def test_p2_conversation(self):
        result = await _run_conversation("p2_indexed_literature")
        _print_result(result)
        _export_result(result)

        assert result.total_turns >= 15
        assert result.context_hit_rate >= 60


class TestConversationP3:
    @pytest.mark.asyncio
    async def test_p3_conversation(self):
        result = await _run_conversation("p3_indexed_pm")
        _print_result(result)
        _export_result(result)

        assert result.total_turns >= 15
        assert result.context_hit_rate >= 60
