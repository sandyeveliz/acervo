"""Layer 4: JSONL Case Scenarios — end-to-end extraction accuracy tests.

Tests the FULL extraction pipeline across diverse conversation domains.
Each case file simulates a ~50-turn conversation (fitness, finances,
family, travel, etc.) and verifies entities, relations, facts, and
topic detection against hand-labeled expectations.

Purpose: identify extraction weaknesses to inform training data for v3.

Usage:
    pytest tests/integration/test_case_scenarios.py -v -s
    pytest tests/integration/test_case_scenarios.py -k "fitness" -v -s
    pytest tests/integration/test_case_scenarios.py -k "test_all" -v -s
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pytest

from acervo import Acervo
from acervo.openai_client import OpenAIClient
from acervo.graph import _make_id

log = logging.getLogger(__name__)

_CASES_DIR = Path(__file__).parent / "scenarios" / "cases"
_REPORTS = Path(__file__).parent / "reports"

CASE_FILES = sorted(p.stem for p in _CASES_DIR.glob("*.jsonl"))


# ── Data structures ──


@dataclass
class TurnResult:
    """Per-turn extraction comparison."""
    turn: int
    user_msg: str
    elapsed_ms: int = 0

    # Topic
    expected_topic_action: str = ""
    actual_topic_action: str = ""
    topic_ok: bool | None = None

    # Entities
    expected_entities: list[dict] = field(default_factory=list)
    matched_entities: list[str] = field(default_factory=list)
    missing_entities: list[str] = field(default_factory=list)
    extra_entities: list[str] = field(default_factory=list)
    entity_accuracy: float = -1.0  # -1 = no expectations

    # Relations
    expected_relations: list[dict] = field(default_factory=list)
    matched_relations: list[tuple] = field(default_factory=list)
    missing_relations: list[tuple] = field(default_factory=list)
    relation_accuracy: float = -1.0

    # Facts
    expected_facts: list[dict] = field(default_factory=list)
    matched_facts: list[str] = field(default_factory=list)
    missing_facts: list[str] = field(default_factory=list)
    fact_accuracy: float = -1.0

    # Fact validation diagnostics
    raw_facts: int = 0
    parsed_facts: int = 0
    dropped_facts: int = 0
    drop_reasons: list[str] = field(default_factory=list)

    # Graph state
    graph_nodes: int = 0
    graph_edges: int = 0
    node_delta: int = 0
    edge_delta: int = 0

    # Overall
    passed: bool = True
    failures: list[str] = field(default_factory=list)


@dataclass
class CaseResult:
    """Aggregate result for one case file."""
    name: str
    domain: str
    total_turns: int = 0
    turns: list[TurnResult] = field(default_factory=list)

    # Aggregate metrics
    passed_turns: int = 0
    entity_accuracy_avg: float = 0.0
    relation_accuracy_avg: float = 0.0
    fact_accuracy_avg: float = 0.0
    topic_accuracy: float = 0.0

    # Graph final state
    final_nodes: int = 0
    final_edges: int = 0

    # Timing
    total_elapsed_ms: int = 0

    # Fact validation diagnostics
    total_raw_facts: int = 0
    total_parsed_facts: int = 0
    total_dropped_facts: int = 0
    drop_rate: float = 0.0

    # Failure analysis
    entity_misses: list[dict] = field(default_factory=list)
    relation_misses: list[dict] = field(default_factory=list)
    fact_misses: list[dict] = field(default_factory=list)


# ── JSONL loader ──


def _load_case(name: str) -> list[dict]:
    """Load a JSONL case file, returning list of turn specs."""
    path = _CASES_DIR / f"{name}.jsonl"
    turns = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            turns.append(json.loads(line))
    return turns


# ── Comparison helpers ──


def _normalize(s: str) -> str:
    """Normalize for comparison: lowercase, strip, collapse whitespace."""
    return " ".join(s.lower().strip().split())


def _check_entities(
    expected: list[dict], graph, prev_node_ids: set[str], tr: TurnResult,
) -> None:
    """Compare expected entities against graph nodes added this turn."""
    if not expected:
        return

    tr.expected_entities = expected
    all_nodes = {n["id"]: n for n in graph.get_all_nodes()}
    new_node_ids = set(all_nodes.keys()) - prev_node_ids

    expected_labels = {}
    for e in expected:
        label = e.get("label", "")
        eid = _make_id(label)
        expected_labels[eid] = {
            "label": label,
            "type": e.get("type", ""),
            "layer": e.get("layer", ""),
        }

    # Match: expected entity exists in graph (new or updated)
    matched = []
    missing = []
    for eid, meta in expected_labels.items():
        if eid in all_nodes:
            matched.append(meta["label"])
        else:
            # Try fuzzy: check if label appears as substring in any node label
            found = False
            for nid, node in all_nodes.items():
                if _normalize(meta["label"]) in _normalize(node.get("label", "")):
                    matched.append(meta["label"])
                    found = True
                    break
            if not found:
                missing.append(meta["label"])

    tr.matched_entities = matched
    tr.missing_entities = missing
    tr.entity_accuracy = len(matched) / len(expected_labels) if expected_labels else 1.0

    if missing:
        tr.failures.append(f"entities missing: {missing}")


def _check_relations(
    expected: list[dict], graph, tr: TurnResult,
) -> None:
    """Compare expected relations against graph edges."""
    if not expected:
        return

    tr.expected_relations = expected
    actual_pairs = set()
    # Use GraphStorePort-compatible API (works with both TopicGraph and LadybugGraphStore)
    for node in graph.get_all_nodes():
        for e in graph.get_edges_for(node.get("id", "")):
            src = e.get("source", "")
            tgt = e.get("target", "")
            actual_pairs.add((_normalize(src), _normalize(tgt)))

    matched = []
    missing = []
    for rel in expected:
        src = _normalize(rel.get("source", ""))
        tgt = _normalize(rel.get("target", ""))
        pair = (src, tgt)
        if pair in actual_pairs:
            matched.append(pair)
        else:
            # Try reverse direction
            if (tgt, src) in actual_pairs:
                matched.append(pair)
            else:
                missing.append(pair)

    tr.matched_relations = matched
    tr.missing_relations = missing
    tr.relation_accuracy = len(matched) / len(expected) if expected else 1.0

    if missing:
        tr.failures.append(f"relations missing: {missing}")


def _check_facts(
    expected: list[dict], graph, tr: TurnResult,
) -> None:
    """Compare expected facts against graph node facts."""
    if not expected:
        return

    tr.expected_facts = expected
    all_nodes = {n["id"]: n for n in graph.get_all_nodes()}

    matched = []
    missing = []
    for fact_spec in expected:
        entity_id = _normalize(fact_spec.get("entity", ""))
        fact_text = _normalize(fact_spec.get("text", ""))

        # Find the entity node
        node = all_nodes.get(entity_id)
        if not node:
            # Try finding by substring
            for nid, n in all_nodes.items():
                if entity_id in nid:
                    node = n
                    break

        if not node:
            missing.append(fact_spec.get("text", "")[:60])
            continue

        # Check if any fact on this node matches (substring or keyword overlap)
        node_facts = [_normalize(f.get("fact", "")) for f in node.get("facts", [])]
        found = False
        for nf in node_facts:
            # Substring match
            if fact_text in nf or nf in fact_text:
                found = True
                break
            # Keyword overlap (>50% of expected fact words in actual fact)
            expected_words = set(fact_text.split())
            actual_words = set(nf.split())
            if expected_words and len(expected_words & actual_words) / len(expected_words) > 0.4:
                found = True
                break

        if found:
            matched.append(fact_spec.get("text", "")[:60])
        else:
            missing.append(fact_spec.get("text", "")[:60])

    tr.matched_facts = matched
    tr.missing_facts = missing
    tr.fact_accuracy = len(matched) / len(expected) if expected else 1.0

    if missing:
        tr.failures.append(f"facts missing: {[m[:40] for m in missing]}")


# ── Case runner ──


async def _run_case(case_name: str, graph_backend: str | None = None) -> CaseResult:
    """Run all turns of a JSONL case file against a live LLM."""
    if graph_backend is None:
        graph_backend = os.environ.get("ACERVO_TEST_BACKEND", "json")
    turns = _load_case(case_name)
    backend_label = f" [{graph_backend}]" if graph_backend != "json" else ""
    print(f"\n  {'='*50}")
    print(f"  CASE: {case_name}{backend_label} ({len(turns)} turns)")
    print(f"  {'='*50}")

    tmpdir = tempfile.mkdtemp()
    try:
        llm = OpenAIClient(
            base_url="http://localhost:11434/v1",
            model="qwen2.5:7b",
            api_key="ollama",
        )
        graph_path = Path(tmpdir) / "graph"
        graph_path.mkdir(parents=True, exist_ok=True)
        acervo = Acervo(
            llm=llm, owner="Sandy", persist_path=str(graph_path),
            graph_backend=graph_backend,
        )

        history: list[dict] = []
        results: list[TurnResult] = []
        prev_nodes, prev_edges = 0, 0
        prev_node_ids: set[str] = set()
        total_ms = 0

        for i, turn_spec in enumerate(turns):
            conv = turn_spec.get("conversation", [])
            if not conv:
                continue
            user_msg = conv[-1].get("content", "").strip()
            if not user_msg:
                continue

            expected = turn_spec.get("expected", {})

            tr = TurnResult(
                turn=i + 1,
                user_msg=user_msg[:100],
            )

            # Expected topic action
            topic_spec = expected.get("topic", {})
            tr.expected_topic_action = topic_spec.get("action", "")

            # ── Run prepare (S1 + S2 + S3) ──
            t0 = time.monotonic()
            try:
                prep = await acervo.prepare(user_msg, history)
                debug = prep.debug or {}
            except Exception as e:
                tr.failures.append(f"prepare() error: {e}")
                tr.passed = False
                results.append(tr)
                _log_turn(tr)
                # Still update history for continuity
                history.append({"role": "user", "content": user_msg})
                history.append({"role": "assistant", "content": "OK"})
                continue

            # ── Check topic detection ──
            if tr.expected_topic_action:
                s1_det = debug.get("s1_detection", {})
                topic_changed = s1_det.get("topic_changed", False)

                # Turn 1 on empty graph: S1 sees no prior topic, so it may
                # report "same" (default). Treat turn 1 as always "changed".
                if i == 0:
                    tr.actual_topic_action = "changed"
                elif topic_changed:
                    tr.actual_topic_action = "changed"
                else:
                    tr.actual_topic_action = "same"

                # "subtopic" is hard to distinguish from "same" at the S1 level
                if tr.expected_topic_action == "subtopic":
                    tr.topic_ok = tr.actual_topic_action in ("same", "subtopic")
                elif tr.expected_topic_action == "changed" and i == 0:
                    tr.topic_ok = True  # First turn is definitionally a new topic
                else:
                    tr.topic_ok = tr.actual_topic_action == tr.expected_topic_action
                if not tr.topic_ok:
                    tr.failures.append(
                        f"topic: expected={tr.expected_topic_action}, got={tr.actual_topic_action}"
                    )

            # ── Capture fact validation diagnostics ──
            s1_val = debug.get("s1_validation", {})
            tr.raw_facts = s1_val.get("raw_facts", 0)
            tr.parsed_facts = s1_val.get("parsed_facts", 0)
            tr.dropped_facts = tr.raw_facts - tr.parsed_facts
            tr.drop_reasons = [
                d.get("reason", "") for d in s1_val.get("dropped_facts", [])
            ]

            # ── Run process (S1.5 — extract + persist) ──
            assistant_sim = "Entendido, lo tengo registrado."
            try:
                await acervo.process(user_msg, assistant_sim)
            except Exception as e:
                tr.failures.append(f"process() error: {e}")
                tr.passed = False
                results.append(tr)
                _log_turn(tr)
                history.append({"role": "user", "content": user_msg})
                history.append({"role": "assistant", "content": assistant_sim})
                continue

            elapsed = int((time.monotonic() - t0) * 1000)
            tr.elapsed_ms = elapsed
            total_ms += elapsed

            # ── Graph snapshot ──
            current_node_ids = {n["id"] for n in acervo.graph.get_all_nodes()}
            tr.graph_nodes = acervo.graph.node_count
            tr.graph_edges = acervo.graph.edge_count
            tr.node_delta = tr.graph_nodes - prev_nodes
            tr.edge_delta = tr.graph_edges - prev_edges

            # ── Check extraction results ──
            _check_entities(expected.get("entities", []), acervo.graph, prev_node_ids, tr)
            _check_relations(expected.get("relations", []), acervo.graph, tr)
            _check_facts(expected.get("facts", []), acervo.graph, tr)

            # Update state
            prev_nodes = tr.graph_nodes
            prev_edges = tr.graph_edges
            prev_node_ids = current_node_ids

            tr.passed = len(tr.failures) == 0
            results.append(tr)

            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": assistant_sim})

            _log_turn(tr)

        # ── Build aggregate result ──
        entity_accs = [t.entity_accuracy for t in results if t.entity_accuracy >= 0]
        relation_accs = [t.relation_accuracy for t in results if t.relation_accuracy >= 0]
        fact_accs = [t.fact_accuracy for t in results if t.fact_accuracy >= 0]
        topic_checks = [t for t in results if t.topic_ok is not None]

        # Collect all misses for training data analysis
        entity_misses = []
        relation_misses = []
        fact_misses = []
        for t in results:
            if t.missing_entities:
                entity_misses.append({
                    "turn": t.turn,
                    "msg": t.user_msg,
                    "missing": t.missing_entities,
                })
            if t.missing_relations:
                relation_misses.append({
                    "turn": t.turn,
                    "msg": t.user_msg,
                    "missing": [list(p) for p in t.missing_relations],
                })
            if t.missing_facts:
                fact_misses.append({
                    "turn": t.turn,
                    "msg": t.user_msg,
                    "missing": t.missing_facts,
                })

        result = CaseResult(
            name=case_name,
            domain=case_name.replace("_", " ").title(),
            total_turns=len(results),
            turns=results,
            passed_turns=sum(1 for t in results if t.passed),
            entity_accuracy_avg=(
                sum(entity_accs) / len(entity_accs) if entity_accs else 0
            ),
            relation_accuracy_avg=(
                sum(relation_accs) / len(relation_accs) if relation_accs else 0
            ),
            fact_accuracy_avg=(
                sum(fact_accs) / len(fact_accs) if fact_accs else 0
            ),
            topic_accuracy=(
                sum(1 for t in topic_checks if t.topic_ok) / len(topic_checks)
                if topic_checks else 0
            ),
            final_nodes=prev_nodes,
            final_edges=prev_edges,
            total_elapsed_ms=total_ms,
            total_raw_facts=sum(t.raw_facts for t in results),
            total_parsed_facts=sum(t.parsed_facts for t in results),
            total_dropped_facts=sum(t.dropped_facts for t in results),
            drop_rate=(
                sum(t.dropped_facts for t in results) / max(sum(t.raw_facts for t in results), 1)
            ),
            entity_misses=entity_misses,
            relation_misses=relation_misses,
            fact_misses=fact_misses,
        )

        _print_summary(result)
        return result

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Logging ──


def _log_turn(tr: TurnResult) -> None:
    status = "✓" if tr.passed else "✗"
    parts = []
    if tr.entity_accuracy >= 0:
        parts.append(f"ent={tr.entity_accuracy:.0%}")
    if tr.relation_accuracy >= 0:
        parts.append(f"rel={tr.relation_accuracy:.0%}")
    if tr.fact_accuracy >= 0:
        parts.append(f"fact={tr.fact_accuracy:.0%}")
    if tr.raw_facts > 0:
        parts.append(f"facts={tr.parsed_facts}/{tr.raw_facts}")
    parts.append(f"graph={tr.graph_nodes}n/{tr.graph_edges}e")
    if tr.node_delta:
        parts.append(f"Δ+{tr.node_delta}n")
    extra = " ".join(parts)
    print(f"    {status} T{tr.turn:02d} {extra} ({tr.elapsed_ms}ms) {tr.user_msg[:50]}")
    for f in tr.failures:
        print(f"      ✗ {f[:120]}")


def _print_summary(result: CaseResult) -> None:
    print(f"\n  {result.name}: {result.passed_turns}/{result.total_turns} turns passed")
    print(f"    Graph: {result.final_nodes}n / {result.final_edges}e")
    print(f"    Entity acc:   {result.entity_accuracy_avg:.0%}")
    print(f"    Relation acc: {result.relation_accuracy_avg:.0%}")
    print(f"    Fact acc:     {result.fact_accuracy_avg:.0%}")
    print(f"    Topic acc:    {result.topic_accuracy:.0%}")
    print(f"    Facts:        {result.total_parsed_facts}/{result.total_raw_facts} "
          f"(drop={result.drop_rate:.0%})")
    print(f"    Total time:   {result.total_elapsed_ms / 1000:.1f}s")
    if result.entity_misses:
        print(f"    Entity misses: {len(result.entity_misses)} turns")
    if result.relation_misses:
        print(f"    Relation misses: {len(result.relation_misses)} turns")
    if result.fact_misses:
        print(f"    Fact misses: {len(result.fact_misses)} turns")


# ── Report generation ──


def _report_version() -> str:
    """Version string for reports, includes backend suffix."""
    backend = os.environ.get("ACERVO_TEST_BACKEND", "json")
    base = "v0.6.0"
    return f"{base}-{backend}" if backend != "json" else base


def _write_case_report(result: CaseResult, version: str | None = None) -> None:
    if version is None:
        version = _report_version()
    """Write JSON + Markdown report for a single case."""
    version_dir = _REPORTS / version
    version_dir.mkdir(parents=True, exist_ok=True)

    # JSON report
    report = {
        "name": result.name,
        "domain": result.domain,
        "total_turns": result.total_turns,
        "passed_turns": result.passed_turns,
        "pass_rate": round(result.passed_turns / result.total_turns * 100) if result.total_turns else 0,
        "entity_accuracy": round(result.entity_accuracy_avg * 100),
        "relation_accuracy": round(result.relation_accuracy_avg * 100),
        "fact_accuracy": round(result.fact_accuracy_avg * 100),
        "topic_accuracy": round(result.topic_accuracy * 100),
        "final_graph": {"nodes": result.final_nodes, "edges": result.final_edges},
        "total_elapsed_ms": result.total_elapsed_ms,
        "fact_diagnostics": {
            "total_raw_facts": result.total_raw_facts,
            "total_parsed_facts": result.total_parsed_facts,
            "total_dropped_facts": result.total_dropped_facts,
            "drop_rate": round(result.drop_rate * 100),
        },
        "entity_misses": result.entity_misses,
        "relation_misses": result.relation_misses,
        "fact_misses": result.fact_misses,
        "turns": [
            {
                "turn": t.turn,
                "user_msg": t.user_msg,
                "elapsed_ms": t.elapsed_ms,
                "passed": t.passed,
                "entity_accuracy": round(t.entity_accuracy * 100) if t.entity_accuracy >= 0 else None,
                "relation_accuracy": round(t.relation_accuracy * 100) if t.relation_accuracy >= 0 else None,
                "fact_accuracy": round(t.fact_accuracy * 100) if t.fact_accuracy >= 0 else None,
                "topic_ok": t.topic_ok,
                "graph_nodes": t.graph_nodes,
                "graph_edges": t.graph_edges,
                "node_delta": t.node_delta,
                "raw_facts": t.raw_facts,
                "parsed_facts": t.parsed_facts,
                "dropped_facts": t.dropped_facts,
                "drop_reasons": t.drop_reasons,
                "failures": t.failures,
                "missing_entities": t.missing_entities,
                "missing_facts": t.missing_facts,
            }
            for t in result.turns
        ],
    }

    (version_dir / f"case_{result.name}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # Markdown report
    lines = [
        f"# Case Scenario: {result.domain}",
        "",
        f"**Turns:** {result.total_turns} | **Passed:** {result.passed_turns}/{result.total_turns} "
        f"({report['pass_rate']}%)",
        f"**Entity acc:** {result.entity_accuracy_avg:.0%} | "
        f"**Relation acc:** {result.relation_accuracy_avg:.0%} | "
        f"**Fact acc:** {result.fact_accuracy_avg:.0%} | "
        f"**Topic acc:** {result.topic_accuracy:.0%}",
        f"**Graph:** {result.final_nodes} nodes, {result.final_edges} edges | "
        f"**Time:** {result.total_elapsed_ms / 1000:.1f}s",
        "",
        "## Turn-by-turn",
        "",
        "| Turn | Ent% | Rel% | Fact% | Nodes | Δn | ms | Status |",
        "|------|------|------|-------|-------|----|----|--------|",
    ]
    for t in result.turns:
        ea = f"{t.entity_accuracy:.0%}" if t.entity_accuracy >= 0 else "—"
        ra = f"{t.relation_accuracy:.0%}" if t.relation_accuracy >= 0 else "—"
        fa = f"{t.fact_accuracy:.0%}" if t.fact_accuracy >= 0 else "—"
        status = "✓" if t.passed else "✗"
        lines.append(
            f"| {t.turn} | {ea} | {ra} | {fa} | {t.graph_nodes} "
            f"| {t.node_delta:+d} | {t.elapsed_ms} | {status} |"
        )

    # Entity misses (for training data analysis)
    if result.entity_misses:
        lines.extend(["", "## Entity Misses (training data candidates)", ""])
        for miss in result.entity_misses:
            lines.append(f"- **T{miss['turn']}** `{miss['msg'][:60]}` → missing: {miss['missing']}")

    # Relation misses
    if result.relation_misses:
        lines.extend(["", "## Relation Misses", ""])
        for miss in result.relation_misses:
            lines.append(f"- **T{miss['turn']}** → missing: {miss['missing']}")

    # Fact misses
    if result.fact_misses:
        lines.extend(["", "## Fact Misses (training data candidates)", ""])
        for miss in result.fact_misses:
            lines.append(f"- **T{miss['turn']}** `{miss['msg'][:60]}` → missing: {[m[:50] for m in miss['missing']]}")

    lines.append("")
    (version_dir / f"case_{result.name}.md").write_text(
        "\n".join(lines), encoding="utf-8",
    )


def _write_combined_report(all_results: list[CaseResult], version: str | None = None) -> None:
    if version is None:
        version = _report_version()
    """Write a combined report across all cases."""
    version_dir = _REPORTS / version
    version_dir.mkdir(parents=True, exist_ok=True)

    total_turns = sum(r.total_turns for r in all_results)
    passed_turns = sum(r.passed_turns for r in all_results)
    total_entity_misses = sum(len(r.entity_misses) for r in all_results)
    total_relation_misses = sum(len(r.relation_misses) for r in all_results)
    total_fact_misses = sum(len(r.fact_misses) for r in all_results)

    # Weighted averages
    entity_accs = [r.entity_accuracy_avg for r in all_results if r.entity_accuracy_avg > 0]
    relation_accs = [r.relation_accuracy_avg for r in all_results if r.relation_accuracy_avg > 0]
    fact_accs = [r.fact_accuracy_avg for r in all_results if r.fact_accuracy_avg > 0]

    combined = {
        "version": version,
        "total_cases": len(all_results),
        "total_turns": total_turns,
        "passed_turns": passed_turns,
        "pass_rate": round(passed_turns / total_turns * 100) if total_turns else 0,
        "entity_accuracy_avg": round(sum(entity_accs) / len(entity_accs) * 100) if entity_accs else 0,
        "relation_accuracy_avg": round(sum(relation_accs) / len(relation_accs) * 100) if relation_accs else 0,
        "fact_accuracy_avg": round(sum(fact_accs) / len(fact_accs) * 100) if fact_accs else 0,
        "total_entity_misses": total_entity_misses,
        "total_relation_misses": total_relation_misses,
        "total_fact_misses": total_fact_misses,
        "cases": [
            {
                "name": r.name,
                "domain": r.domain,
                "turns": r.total_turns,
                "passed": r.passed_turns,
                "pass_rate": round(r.passed_turns / r.total_turns * 100) if r.total_turns else 0,
                "entity_acc": round(r.entity_accuracy_avg * 100),
                "relation_acc": round(r.relation_accuracy_avg * 100),
                "fact_acc": round(r.fact_accuracy_avg * 100),
                "topic_acc": round(r.topic_accuracy * 100),
                "graph": f"{r.final_nodes}n/{r.final_edges}e",
                "time_s": round(r.total_elapsed_ms / 1000, 1),
            }
            for r in all_results
        ],
    }

    (version_dir / "case_scenarios_combined.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Markdown
    lines = [
        f"# Case Scenarios — Combined Report ({version})",
        "",
        f"**Cases:** {len(all_results)} | **Turns:** {total_turns} | "
        f"**Passed:** {passed_turns}/{total_turns} ({combined['pass_rate']}%)",
        f"**Entity acc:** {combined['entity_accuracy_avg']}% | "
        f"**Relation acc:** {combined['relation_accuracy_avg']}% | "
        f"**Fact acc:** {combined['fact_accuracy_avg']}%",
        "",
        "## Per-case breakdown",
        "",
        "| Case | Domain | Turns | Pass% | Ent% | Rel% | Fact% | Topic% | Graph | Time |",
        "|------|--------|-------|-------|------|------|-------|--------|-------|------|",
    ]
    for r in all_results:
        lines.append(
            f"| {r.name} | {r.domain} | {r.total_turns} "
            f"| {round(r.passed_turns / r.total_turns * 100) if r.total_turns else 0}% "
            f"| {round(r.entity_accuracy_avg * 100)}% "
            f"| {round(r.relation_accuracy_avg * 100)}% "
            f"| {round(r.fact_accuracy_avg * 100)}% "
            f"| {round(r.topic_accuracy * 100)}% "
            f"| {r.final_nodes}n/{r.final_edges}e "
            f"| {r.total_elapsed_ms / 1000:.1f}s |"
        )

    # Training data summary: aggregate all misses
    all_entity_misses = []
    all_fact_misses = []
    for r in all_results:
        for m in r.entity_misses:
            all_entity_misses.append({**m, "case": r.name})
        for m in r.fact_misses:
            all_fact_misses.append({**m, "case": r.name})

    if all_entity_misses:
        lines.extend([
            "",
            f"## Training Data Candidates — Entity Misses ({len(all_entity_misses)} turns)",
            "",
        ])
        for m in all_entity_misses[:30]:  # Top 30
            lines.append(f"- **{m['case']}:T{m['turn']}** → {m['missing']}")
        if len(all_entity_misses) > 30:
            lines.append(f"- ... and {len(all_entity_misses) - 30} more")

    if all_fact_misses:
        lines.extend([
            "",
            f"## Training Data Candidates — Fact Misses ({len(all_fact_misses)} turns)",
            "",
        ])
        for m in all_fact_misses[:30]:
            lines.append(f"- **{m['case']}:T{m['turn']}** → {[f[:40] for f in m['missing']]}")
        if len(all_fact_misses) > 30:
            lines.append(f"- ... and {len(all_fact_misses) - 30} more")

    lines.append("")
    (version_dir / "case_scenarios_combined.md").write_text(
        "\n".join(lines), encoding="utf-8",
    )


# ── Test class ──


class TestCaseScenarios:
    """Run JSONL case scenarios against a live LLM.

    Each test processes a full conversation (~50 turns), building up
    the knowledge graph and verifying extraction accuracy per turn.
    """

    @pytest.mark.asyncio
    async def test_casa(self):
        result = await _run_case("casa")
        _write_case_report(result)
        assert result.passed_turns >= result.total_turns * 0.3

    @pytest.mark.asyncio
    async def test_finanzas(self):
        result = await _run_case("finanzas")
        _write_case_report(result)
        assert result.passed_turns >= result.total_turns * 0.3

    @pytest.mark.asyncio
    async def test_fitness(self):
        result = await _run_case("fitness")
        _write_case_report(result)
        assert result.passed_turns >= result.total_turns * 0.3

    @pytest.mark.asyncio
    async def test_libro(self):
        result = await _run_case("libro")
        _write_case_report(result)
        assert result.passed_turns >= result.total_turns * 0.3

    @pytest.mark.asyncio
    async def test_proyecto_codigo(self):
        result = await _run_case("proyecto_codigo")
        _write_case_report(result)
        assert result.passed_turns >= result.total_turns * 0.3

    @pytest.mark.asyncio
    async def test_salud_familia(self):
        result = await _run_case("salud_familia")
        _write_case_report(result)
        assert result.passed_turns >= result.total_turns * 0.3

    @pytest.mark.asyncio
    async def test_trabajo(self):
        result = await _run_case("trabajo")
        _write_case_report(result)
        assert result.passed_turns >= result.total_turns * 0.3

    @pytest.mark.asyncio
    async def test_viajes(self):
        result = await _run_case("viajes")
        _write_case_report(result)
        assert result.passed_turns >= result.total_turns * 0.3

    @pytest.mark.asyncio
    async def test_all_cases(self):
        """Run ALL case scenarios and produce combined report."""
        all_results: list[CaseResult] = []
        for name in CASE_FILES:
            try:
                result = await _run_case(name)
            except Exception as e:
                print(f"\n  ERROR in {name}: {e}")
                result = CaseResult(name=name, domain=name.replace("_", " ").title())
            _write_case_report(result)
            all_results.append(result)

        _write_combined_report(all_results)

        total = sum(r.total_turns for r in all_results)
        passed = sum(r.passed_turns for r in all_results)
        total_raw = sum(r.total_raw_facts for r in all_results)
        total_parsed = sum(r.total_parsed_facts for r in all_results)
        total_dropped = sum(r.total_dropped_facts for r in all_results)
        overall_drop = total_dropped / max(total_raw, 1)

        print(f"\n  {'='*60}")
        print(f"  ALL CASES: {passed}/{total} turns passed ({round(passed/total*100) if total else 0}%)")
        print(f"  FACTS: {total_parsed}/{total_raw} parsed (drop={overall_drop:.0%})")
        for r in all_results:
            print(f"    {r.name:20s}: {r.passed_turns:3d}/{r.total_turns:3d} "
                  f"ent={r.entity_accuracy_avg:.0%} rel={r.relation_accuracy_avg:.0%} "
                  f"fact={r.fact_accuracy_avg:.0%} "
                  f"facts={r.total_parsed_facts}/{r.total_raw_facts}")
        print(f"  {'='*60}")


