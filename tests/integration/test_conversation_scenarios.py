"""Layer 3: Conversation Test Scenarios with Graph Evolution Tracking.

Tests the FULL cycle: user message → S1 extracts → graph grows →
S2 retrieves (BFS) → S3 injects → LLM responds.

Unlike Layer 2 (indexed project benchmarks), these start with an empty
graph and build knowledge through conversation.

Usage:
    pytest tests/integration/test_conversation_scenarios.py -v -s
    pytest tests/integration/test_conversation_scenarios.py -k "c1" -v -s
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pytest
import yaml

from acervo import Acervo
from acervo.openai_client import OpenAIClient

log = logging.getLogger(__name__)

_SCENARIOS = Path(__file__).parent / "scenarios"
_REPORTS = Path(__file__).parent / "reports"

CONVERSATION_SCENARIOS = ["c1_multi_project", "c2_personal_knowledge", "c3_progressive_building"]


# ── Data structures ──


@dataclass
class TurnSnapshot:
    """Per-turn result with graph evolution data."""
    turn: int
    phase: str
    category: str
    user_msg: str
    assistant_sim: str = ""
    elapsed_ms: int = 0

    # Graph state
    graph_node_count: int = 0
    graph_edge_count: int = 0
    graph_node_delta: int = 0
    graph_edge_delta: int = 0

    # S1 extraction
    s1_intent_expected: str | None = None
    s1_intent_actual: str = ""
    s1_intent_passed: bool | None = None
    expected_entities: list[dict] = field(default_factory=list)
    actual_entities: list[dict] = field(default_factory=list)
    entity_accuracy: float = -1.0  # -1 = not checked
    expected_relations: list[dict] = field(default_factory=list)
    actual_relations: list[dict] = field(default_factory=list)
    relation_accuracy: float = -1.0

    # S2 layers (BFS)
    s2_seeds: list[str] = field(default_factory=list)
    s2_hot: list[str] = field(default_factory=list)
    s2_warm: list[str] = field(default_factory=list)
    s2_cold: list[str] = field(default_factory=list)

    # S3 context
    warm_tokens: int = 0
    s3_contains_ok: bool | None = None
    s3_not_contains_ok: bool | None = None

    # Graph assertions
    graph_assertions_ok: bool | None = None

    # Overall
    passed: bool = True
    failures: list[str] = field(default_factory=list)


@dataclass
class ConversationResult:
    """Result of running a full conversation scenario."""
    name: str
    turns: list[TurnSnapshot] = field(default_factory=list)
    total_turns: int = 0
    passed_turns: int = 0

    # Graph evolution
    final_node_count: int = 0
    final_edge_count: int = 0
    avg_entity_accuracy: float = 0.0
    avg_relation_accuracy: float = 0.0


# ── Scenario runner ──


def _load_scenario(name: str) -> dict:
    path = _SCENARIOS / f"{name}.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


async def _run_conversation(scenario_name: str) -> ConversationResult:
    """Run a conversation scenario on an empty graph."""
    scenario = _load_scenario(scenario_name)
    print(f"\n  Running {scenario_name}...")

    # Create Acervo with empty graph in temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        llm = OpenAIClient(
            base_url="http://localhost:11434/v1",
            model="acervo-extractor-v3-Q4_K_M",
            api_key="ollama",
        )
        acervo = Acervo(
            llm=llm,
            persist_path=str(Path(tmpdir) / "graph"),
        )

        history: list[dict] = []
        results: list[TurnSnapshot] = []
        prev_nodes, prev_edges = 0, 0

        for turn_spec in scenario.get("turns", []):
            user_msg = turn_spec["user"].strip()
            assistant_sim = turn_spec.get("assistant_sim", "OK").strip()
            phase = turn_spec.get("phase", "unknown")
            category = turn_spec.get("category", "RESOLVE")

            ts = TurnSnapshot(
                turn=turn_spec.get("turn", len(results) + 1),
                phase=phase,
                category=category,
                user_msg=user_msg[:80],
                assistant_sim=assistant_sim[:80],
            )

            # ── Run prepare ──
            t0 = time.monotonic()
            prep = await acervo.prepare(user_msg, history)
            ts.elapsed_ms = int((time.monotonic() - t0) * 1000)
            ts.warm_tokens = prep.warm_tokens
            debug = prep.debug or {}

            # ── Check S1 intent ──
            _check_s1(turn_spec, debug, ts)

            # ── Check S2 layers ──
            _check_s2_layers(turn_spec, debug, ts)

            # ── Check S3 context ──
            _check_s3(turn_spec, prep.warm_content, prep.warm_tokens, ts)

            # ── Run process (extract + persist to graph) ──
            await acervo.process(user_msg, assistant_sim)

            # ── Graph snapshot ──
            ts.graph_node_count = acervo.graph.node_count
            ts.graph_edge_count = acervo.graph.edge_count
            ts.graph_node_delta = ts.graph_node_count - prev_nodes
            ts.graph_edge_delta = ts.graph_edge_count - prev_edges
            prev_nodes = ts.graph_node_count
            prev_edges = ts.graph_edge_count

            # ── Check extraction accuracy ──
            _check_extraction(turn_spec, debug, ts)

            # ── Check graph assertions ──
            _check_graph_assertions(turn_spec, ts)

            # ── Determine pass/fail ──
            ts.passed = len(ts.failures) == 0

            results.append(ts)

            # Update history
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": assistant_sim})

            # Log progress
            _log_turn(ts)

        # Build result
        entity_accs = [t.entity_accuracy for t in results if t.entity_accuracy >= 0]
        relation_accs = [t.relation_accuracy for t in results if t.relation_accuracy >= 0]

        return ConversationResult(
            name=scenario_name,
            turns=results,
            total_turns=len(results),
            passed_turns=sum(1 for t in results if t.passed),
            final_node_count=prev_nodes,
            final_edge_count=prev_edges,
            avg_entity_accuracy=sum(entity_accs) / len(entity_accs) if entity_accs else 0,
            avg_relation_accuracy=sum(relation_accs) / len(relation_accs) if relation_accs else 0,
        )


# ── Check functions ──


def _check_s1(turn_spec: dict, debug: dict, ts: TurnSnapshot) -> None:
    s1_spec = turn_spec.get("s1", {})
    if not s1_spec:
        return
    s1_det = debug.get("s1_detection", {})
    ts.s1_intent_actual = s1_det.get("intent", "")

    expected = s1_spec.get("expected_intent")
    if expected:
        ts.s1_intent_expected = expected
        ts.s1_intent_passed = ts.s1_intent_actual == expected
        if not ts.s1_intent_passed:
            ts.failures.append(f"S1 intent: expected={expected}, got={ts.s1_intent_actual}")


def _check_s2_layers(turn_spec: dict, debug: dict, ts: TurnSnapshot) -> None:
    s2 = debug.get("s2_gathered", {})
    ts.s2_seeds = s2.get("seeds", [])
    nodes = s2.get("nodes", [])
    ts.s2_hot = [n["id"] for n in nodes if n.get("layer") == "hot"]
    ts.s2_warm = [n["id"] for n in nodes if n.get("layer") == "warm"]
    ts.s2_cold = [n["id"] for n in nodes if n.get("layer") == "cold"]

    expected_s2 = turn_spec.get("expected_s2", {})
    if not expected_s2:
        return

    if "seeds" in expected_s2:
        expected_seeds = {s.lower() for s in expected_s2["seeds"]}
        actual_seeds = {s.lower() for s in ts.s2_seeds}
        if not expected_seeds.issubset(actual_seeds):
            ts.failures.append(f"S2 seeds: expected {expected_seeds}, got {actual_seeds}")

    if "warm" in expected_s2:
        expected_warm = {s.lower() for s in expected_s2["warm"]}
        actual_warm = {s.lower() for s in ts.s2_warm}
        if not expected_warm.issubset(actual_warm):
            ts.failures.append(f"S2 warm: expected {expected_warm}, got {actual_warm}")


def _check_s3(turn_spec: dict, warm_content: str, warm_tokens: int, ts: TurnSnapshot) -> None:
    s3_spec = turn_spec.get("s3", {})
    if not s3_spec:
        return

    warm_lower = warm_content.lower() if warm_content else ""

    if "warm_tokens_min" in s3_spec:
        if warm_tokens < s3_spec["warm_tokens_min"]:
            ts.failures.append(f"S3 warm_tokens={warm_tokens} < min={s3_spec['warm_tokens_min']}")

    if "warm_tokens_max" in s3_spec:
        if warm_tokens > s3_spec["warm_tokens_max"]:
            ts.failures.append(f"S3 warm_tokens={warm_tokens} > max={s3_spec['warm_tokens_max']}")

    if "context_contains" in s3_spec:
        missing = [w for w in s3_spec["context_contains"] if w.lower() not in warm_lower]
        ts.s3_contains_ok = len(missing) == 0
        if missing:
            ts.failures.append(f"S3 missing: {missing}")

    if "context_contains_any" in s3_spec:
        found = any(w.lower() in warm_lower for w in s3_spec["context_contains_any"])
        if not found:
            ts.failures.append(f"S3 missing_any: {s3_spec['context_contains_any']}")

    if "context_not_contains" in s3_spec:
        found_bad = [w for w in s3_spec["context_not_contains"] if w.lower() in warm_lower]
        ts.s3_not_contains_ok = len(found_bad) == 0
        if found_bad:
            ts.failures.append(f"S3 unwanted: {found_bad}")


def _check_extraction(turn_spec: dict, debug: dict, ts: TurnSnapshot) -> None:
    expected = turn_spec.get("expected_entities", [])
    if not expected:
        return

    s1_det = debug.get("s1_detection", {})
    actual = s1_det.get("entities", [])

    ts.expected_entities = expected
    ts.actual_entities = actual

    expected_labels = {e["label"].lower() for e in expected}
    actual_labels = {e.get("name", e.get("label", "")).lower() for e in actual}

    matched = expected_labels & actual_labels
    ts.entity_accuracy = len(matched) / len(expected_labels) if expected_labels else 1.0

    missing = expected_labels - actual_labels
    if missing:
        ts.failures.append(f"Extraction missing entities: {missing}")

    # Relations
    expected_rels = turn_spec.get("expected_relations", [])
    if expected_rels:
        actual_rels = s1_det.get("relations", [])
        ts.expected_relations = expected_rels
        ts.actual_relations = actual_rels

        expected_pairs = {(r["source"].lower(), r["target"].lower()) for r in expected_rels}
        actual_pairs = {(r.get("source", "").lower(), r.get("target", "").lower()) for r in actual_rels}
        rel_matched = expected_pairs & actual_pairs
        ts.relation_accuracy = len(rel_matched) / len(expected_pairs) if expected_pairs else 1.0


def _check_graph_assertions(turn_spec: dict, ts: TurnSnapshot) -> None:
    ga = turn_spec.get("graph_assertions", {})
    if not ga:
        return

    if "min_nodes" in ga and ts.graph_node_count < ga["min_nodes"]:
        ts.failures.append(f"Graph nodes={ts.graph_node_count} < min={ga['min_nodes']}")
    if "min_edges" in ga and ts.graph_edge_count < ga["min_edges"]:
        ts.failures.append(f"Graph edges={ts.graph_edge_count} < min={ga['min_edges']}")
    if "node_delta_min" in ga and ts.graph_node_delta < ga["node_delta_min"]:
        ts.failures.append(f"Graph Δnodes={ts.graph_node_delta} < min={ga['node_delta_min']}")
    if "node_delta_max" in ga and ts.graph_node_delta > ga["node_delta_max"]:
        ts.failures.append(f"Graph Δnodes={ts.graph_node_delta} > max={ga['node_delta_max']}")
    if "edge_delta_min" in ga and ts.graph_edge_delta < ga["edge_delta_min"]:
        ts.failures.append(f"Graph Δedges={ts.graph_edge_delta} < min={ga['edge_delta_min']}")

    ts.graph_assertions_ok = not any("Graph" in f for f in ts.failures)


# ── Logging ──


def _log_turn(ts: TurnSnapshot) -> None:
    status = "✓" if ts.passed else "✗"
    extras = []
    if ts.entity_accuracy >= 0:
        extras.append(f"entities={ts.entity_accuracy:.0%}")
    if ts.warm_tokens > 0:
        extras.append(f"warm={ts.warm_tokens}tk")
    extras.append(f"graph={ts.graph_node_count}n/{ts.graph_edge_count}e")
    if ts.graph_node_delta:
        extras.append(f"Δ+{ts.graph_node_delta}n")
    extra_str = " ".join(extras)
    print(f"    {status} Turn {ts.turn} [{ts.phase}] {extra_str} ({ts.elapsed_ms}ms)")
    for f in ts.failures:
        print(f"      ✗ {f}")


# ── Report generation ──


def _write_conversation_report(result: ConversationResult) -> None:
    version_dir = _REPORTS / "v0.5.0"
    version_dir.mkdir(parents=True, exist_ok=True)

    # JSON report
    report = {
        "name": result.name,
        "total_turns": result.total_turns,
        "passed_turns": result.passed_turns,
        "pass_rate": round(result.passed_turns / result.total_turns * 100) if result.total_turns else 0,
        "final_graph": {
            "nodes": result.final_node_count,
            "edges": result.final_edge_count,
        },
        "avg_entity_accuracy": round(result.avg_entity_accuracy * 100),
        "avg_relation_accuracy": round(result.avg_relation_accuracy * 100),
        "turns": [asdict(t) for t in result.turns],
    }

    (version_dir / f"conversation_{result.name}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # Markdown report
    lines = [
        f"# Graph Evolution: {result.name}",
        "",
        f"**Turns:** {result.total_turns} | **Passed:** {result.passed_turns}/{result.total_turns} "
        f"| **Graph:** {result.final_node_count} nodes, {result.final_edge_count} edges",
        f"**Avg entity accuracy:** {result.avg_entity_accuracy:.0%} "
        f"| **Avg relation accuracy:** {result.avg_relation_accuracy:.0%}",
        "",
        "## Turn-by-turn",
        "",
        "| Turn | Phase | Nodes | Edges | Δn | Δe | Warm | Entity% | Status |",
        "|------|-------|-------|-------|----|----|------|---------|--------|",
    ]
    for t in result.turns:
        ea = f"{t.entity_accuracy:.0%}" if t.entity_accuracy >= 0 else "—"
        status = "✓" if t.passed else "✗"
        lines.append(
            f"| {t.turn} | {t.phase} | {t.graph_node_count} | {t.graph_edge_count} "
            f"| {t.graph_node_delta:+d} | {t.graph_edge_delta:+d} | {t.warm_tokens} | {ea} | {status} |"
        )

    # Failures
    failures = [(t.turn, f) for t in result.turns for f in t.failures]
    if failures:
        lines.extend(["", "## Failures", ""])
        for turn, fail in failures:
            lines.append(f"- **Turn {turn}:** {fail}")

    lines.append("")
    (version_dir / f"conversation_{result.name}.md").write_text(
        "\n".join(lines), encoding="utf-8",
    )


# ── Test class ──


class TestConversationScenarios:

    @pytest.mark.asyncio
    async def test_c1_multi_project(self):
        result = await _run_conversation("c1_multi_project")
        _write_conversation_report(result)
        print(f"\n  C1: {result.passed_turns}/{result.total_turns} passed, "
              f"graph={result.final_node_count}n/{result.final_edge_count}e, "
              f"entity_acc={result.avg_entity_accuracy:.0%}")
        assert result.passed_turns >= result.total_turns * 0.5, (
            f"C1: {result.passed_turns}/{result.total_turns} passed"
        )

    @pytest.mark.asyncio
    async def test_c2_personal_knowledge(self):
        result = await _run_conversation("c2_personal_knowledge")
        _write_conversation_report(result)
        print(f"\n  C2: {result.passed_turns}/{result.total_turns} passed, "
              f"graph={result.final_node_count}n/{result.final_edge_count}e")
        assert result.passed_turns >= result.total_turns * 0.5

    @pytest.mark.asyncio
    async def test_c3_progressive_building(self):
        result = await _run_conversation("c3_progressive_building")
        _write_conversation_report(result)
        print(f"\n  C3: {result.passed_turns}/{result.total_turns} passed, "
              f"graph={result.final_node_count}n/{result.final_edge_count}e")
        assert result.passed_turns >= result.total_turns * 0.5

    @pytest.mark.asyncio
    async def test_all_conversations(self):
        """Run all conversation scenarios and produce combined report."""
        all_results: list[ConversationResult] = []
        for name in CONVERSATION_SCENARIOS:
            result = await _run_conversation(name)
            _write_conversation_report(result)
            all_results.append(result)

        # Combined summary
        total = sum(r.total_turns for r in all_results)
        passed = sum(r.passed_turns for r in all_results)
        print(f"\n  ALL CONVERSATIONS: {passed}/{total} turns passed")
        for r in all_results:
            print(f"    {r.name}: {r.passed_turns}/{r.total_turns} "
                  f"({r.final_node_count}n/{r.final_edge_count}e) "
                  f"entity={r.avg_entity_accuracy:.0%}")
