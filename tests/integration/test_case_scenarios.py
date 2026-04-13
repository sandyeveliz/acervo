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

    # Entities (exact = strict ID match, fuzzy = difflib + substring)
    expected_entities: list[dict] = field(default_factory=list)
    matched_entities: list[str] = field(default_factory=list)
    missing_entities: list[str] = field(default_factory=list)
    extra_entities: list[str] = field(default_factory=list)
    entity_accuracy: float = -1.0  # exact
    entity_accuracy_fuzzy: float = -1.0

    # Relations
    expected_relations: list[dict] = field(default_factory=list)
    matched_relations: list[tuple] = field(default_factory=list)
    missing_relations: list[tuple] = field(default_factory=list)
    relation_accuracy: float = -1.0  # exact
    relation_accuracy_fuzzy: float = -1.0
    matched_relations_fuzzy: list[tuple] = field(default_factory=list)

    # Facts
    expected_facts: list[dict] = field(default_factory=list)
    matched_facts: list[str] = field(default_factory=list)
    missing_facts: list[str] = field(default_factory=list)
    fact_accuracy: float = -1.0  # exact
    fact_accuracy_fuzzy: float = -1.0
    matched_facts_fuzzy: list[str] = field(default_factory=list)

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

    # Aggregate metrics (exact / fuzzy)
    passed_turns: int = 0
    entity_accuracy_avg: float = 0.0
    entity_accuracy_fuzzy_avg: float = 0.0
    relation_accuracy_avg: float = 0.0
    relation_accuracy_fuzzy_avg: float = 0.0
    fact_accuracy_avg: float = 0.0
    fact_accuracy_fuzzy_avg: float = 0.0
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

from difflib import get_close_matches, SequenceMatcher


def _normalize(s: str) -> str:
    """Normalize for comparison: lowercase, strip, collapse whitespace."""
    return " ".join(s.lower().strip().split())


def _fuzzy_find_node(target_id: str, target_label: str, all_nodes: dict, all_labels: dict) -> bool:
    """Check if a target entity exists in the graph via exact or fuzzy matching."""
    # 1. Exact ID
    if target_id in all_nodes:
        return True
    # 2. Fuzzy ID (fondo_de_emergencia ↔ fondo_emergencia)
    if get_close_matches(target_id, all_nodes.keys(), n=1, cutoff=0.7):
        return True
    # 3. Label substring (bidirectional, min 4 chars)
    norm_label = _normalize(target_label)
    if len(norm_label) >= 4:
        for lbl in all_labels.values():
            if len(lbl) >= 4 and (norm_label in lbl or lbl in norm_label):
                return True
    # 4. Fuzzy label
    if norm_label and get_close_matches(norm_label, list(all_labels.values()), n=1, cutoff=0.7):
        return True
    return False


def _check_entities(
    expected: list[dict], graph, prev_node_ids: set[str], tr: TurnResult,
) -> None:
    """Compare expected entities against graph nodes."""
    if not expected:
        return

    tr.expected_entities = expected
    all_nodes = {n["id"]: n for n in graph.get_all_nodes()}
    all_labels = {n["id"]: _normalize(n.get("label", "")) for n in graph.get_all_nodes()}

    expected_labels = {}
    for e in expected:
        label = e.get("label", "")
        eid = _make_id(label)
        expected_labels[eid] = {
            "label": label,
            "type": e.get("type", ""),
            "layer": e.get("layer", ""),
        }

    # Exact matching (strict ID or label substring — original behavior)
    matched_exact = []
    missing_exact = []
    for eid, meta in expected_labels.items():
        if eid in all_nodes:
            matched_exact.append(meta["label"])
        else:
            found = False
            norm = _normalize(meta["label"])
            for nid, node in all_nodes.items():
                if norm in _normalize(node.get("label", "")):
                    matched_exact.append(meta["label"])
                    found = True
                    break
            if not found:
                missing_exact.append(meta["label"])

    # Fuzzy matching (difflib + substring + fuzzy label)
    matched_fuzzy = []
    missing_fuzzy = []
    for eid, meta in expected_labels.items():
        if _fuzzy_find_node(eid, meta["label"], all_nodes, all_labels):
            matched_fuzzy.append(meta["label"])
        else:
            missing_fuzzy.append(meta["label"])

    tr.matched_entities = matched_exact
    tr.missing_entities = missing_exact
    tr.entity_accuracy = len(matched_exact) / len(expected_labels) if expected_labels else 1.0
    tr.entity_accuracy_fuzzy = len(matched_fuzzy) / len(expected_labels) if expected_labels else 1.0

    # Pass/fail uses the fuzzy miss set (entities that failed both exact
    # label match AND difflib/substring fuzzy resolution). Exact match is
    # too strict for a local 9B model that may extract "Punto Sur" while
    # the spec calls it "Punto Sur Inmobiliaria"; fuzzy already handles
    # those legitimate variants so we trust its verdict.
    if missing_fuzzy:
        tr.failures.append(f"entities missing: {missing_fuzzy}")


def _check_relations(
    expected: list[dict], graph, tr: TurnResult,
) -> None:
    """Compare expected relations against graph edges."""
    if not expected:
        return

    tr.expected_relations = expected
    actual_pairs = set()
    actual_endpoints = set()
    for node in graph.get_all_nodes():
        for e in graph.get_edges_for(node.get("id", "")):
            src = _normalize(e.get("source", ""))
            tgt = _normalize(e.get("target", ""))
            actual_pairs.add((src, tgt))
            actual_endpoints.add(src)
            actual_endpoints.add(tgt)

    # Exact matching
    matched_exact = []
    missing_exact = []
    for rel in expected:
        src = _normalize(rel.get("source", ""))
        tgt = _normalize(rel.get("target", ""))
        if (src, tgt) in actual_pairs or (tgt, src) in actual_pairs:
            matched_exact.append((src, tgt))
        else:
            missing_exact.append((src, tgt))

    # Fuzzy matching — resolve endpoints via difflib
    matched_fuzzy = []
    missing_fuzzy = []
    ep_list = list(actual_endpoints)
    for rel in expected:
        src = _normalize(rel.get("source", ""))
        tgt = _normalize(rel.get("target", ""))
        if (src, tgt) in actual_pairs or (tgt, src) in actual_pairs:
            matched_fuzzy.append((src, tgt))
            continue
        # Fuzzy resolve src and tgt
        src_matches = get_close_matches(src, ep_list, n=1, cutoff=0.7)
        tgt_matches = get_close_matches(tgt, ep_list, n=1, cutoff=0.7)
        fsrc = src_matches[0] if src_matches else src
        ftgt = tgt_matches[0] if tgt_matches else tgt
        if (fsrc, ftgt) in actual_pairs or (ftgt, fsrc) in actual_pairs:
            matched_fuzzy.append((src, tgt))
        else:
            missing_fuzzy.append((src, tgt))

    tr.matched_relations = matched_exact
    tr.missing_relations = missing_exact
    tr.relation_accuracy = len(matched_exact) / len(expected) if expected else 1.0
    tr.matched_relations_fuzzy = matched_fuzzy
    tr.relation_accuracy_fuzzy = len(matched_fuzzy) / len(expected) if expected else 1.0

    # Same rationale as ``_check_entities`` and ``_check_facts``: pass/fail
    # uses the fuzzy miss set. This covers relations where the LLM picked
    # ``located_in`` while the spec expected ``part_of`` etc.; fuzzy
    # endpoint resolution via difflib matches them as long as the node
    # ids are close enough.
    if missing_fuzzy:
        tr.failures.append(f"relations missing: {missing_fuzzy}")


def _fact_matches(fact_text: str, candidate: str) -> bool:
    """Check if a fact text matches a candidate via substring, keyword overlap, or sequence similarity."""
    if not fact_text or not candidate:
        return False
    # Substring
    if fact_text in candidate or candidate in fact_text:
        return True
    # Keyword overlap (>40% of expected words in actual)
    expected_words = set(fact_text.split())
    actual_words = set(candidate.split())
    if expected_words and len(expected_words & actual_words) / len(expected_words) > 0.4:
        return True
    # Sequence similarity (catches "4.000.000 ARS" vs "4M ARS")
    if SequenceMatcher(None, fact_text, candidate).ratio() >= 0.5:
        return True
    return False


def _check_facts(
    expected: list[dict], graph, tr: TurnResult,
) -> None:
    """Compare expected facts against graph node facts."""
    if not expected:
        return

    tr.expected_facts = expected
    all_nodes = {n["id"]: n for n in graph.get_all_nodes()}
    node_ids = list(all_nodes.keys())

    # Collect ALL facts in the graph for fuzzy fallback
    all_facts_flat: list[str] = []
    for n in all_nodes.values():
        for f in n.get("facts", []):
            all_facts_flat.append(_normalize(f.get("fact", "")))

    matched_exact = []
    missing_exact = []
    matched_fuzzy = []
    missing_fuzzy = []

    for fact_spec in expected:
        entity_id = _normalize(fact_spec.get("entity", ""))
        fact_text = _normalize(fact_spec.get("text", ""))
        fact_label = fact_spec.get("text", "")[:60]

        # ── Exact: find entity by exact ID or substring ──
        node = all_nodes.get(entity_id)
        if not node:
            for nid, n in all_nodes.items():
                if entity_id and entity_id in nid:
                    node = n
                    break

        if node:
            node_facts = [_normalize(f.get("fact", "")) for f in node.get("facts", [])]
            if any(_fact_matches(fact_text, nf) for nf in node_facts):
                matched_exact.append(fact_label)
            else:
                missing_exact.append(fact_label)
        else:
            missing_exact.append(fact_label)

        # ── Fuzzy: resolve entity via difflib, then search facts ──
        fuzzy_node = node
        if not fuzzy_node and entity_id:
            # Fuzzy ID match
            matches = get_close_matches(entity_id, node_ids, n=1, cutoff=0.6)
            if matches:
                fuzzy_node = all_nodes.get(matches[0])

        if fuzzy_node:
            node_facts = [_normalize(f.get("fact", "")) for f in fuzzy_node.get("facts", [])]
            if any(_fact_matches(fact_text, nf) for nf in node_facts):
                matched_fuzzy.append(fact_label)
            else:
                # Fact not on this entity — search ALL facts in graph
                if any(_fact_matches(fact_text, af) for af in all_facts_flat):
                    matched_fuzzy.append(fact_label)
                else:
                    missing_fuzzy.append(fact_label)
        else:
            # Entity not found even with fuzzy — search ALL facts
            if any(_fact_matches(fact_text, af) for af in all_facts_flat):
                matched_fuzzy.append(fact_label)
            else:
                missing_fuzzy.append(fact_label)
    tr.matched_facts = matched_exact
    tr.missing_facts = missing_exact
    tr.fact_accuracy = len(matched_exact) / len(expected) if expected else 1.0
    tr.matched_facts_fuzzy = matched_fuzzy
    tr.fact_accuracy_fuzzy = len(matched_fuzzy) / len(expected) if expected else 1.0

    # Pass/fail uses the FUZZY miss set (facts that failed BOTH exact and
    # fuzzy strategies), not the strict exact set. With a local 9B model
    # the LLM phrases facts with natural variance ("seña 5.000 USD" vs
    # "Comprado: seña 5.000 USD"); requiring literal matches is not the
    # right bar. ``_fact_matches`` already does substring + token-overlap
    # + SequenceMatcher-based fuzzy so a real match should always
    # survive; only genuinely missing facts end up in ``missing_fuzzy``.
    if missing_fuzzy:
        tr.failures.append(f"facts missing: {[m[:40] for m in missing_fuzzy]}")


# ── Case runner ──


async def _run_case(case_name: str, graph_backend: str | None = None) -> CaseResult:
    """Run all turns of a JSONL case file against a live LLM."""
    if graph_backend is None:
        # Default backend is ladybug as of v0.6.0 — matches production.
        # Set ACERVO_TEST_BACKEND=json to exercise the TopicGraph fallback.
        graph_backend = os.environ.get("ACERVO_TEST_BACKEND", "ladybug")
    turns = _load_case(case_name)
    # Only tag the report with a suffix when using the non-default backend,
    # so ladybug runs land in ``v0.6.0/`` (the canonical directory) and
    # json runs land in ``v0.6.0-json/``.
    backend_label = f" [{graph_backend}]" if graph_backend != "ladybug" else ""
    print(f"\n  {'='*50}")
    print(f"  CASE: {case_name}{backend_label} ({len(turns)} turns)")
    print(f"  {'='*50}")

    tmpdir = tempfile.mkdtemp()
    try:
        # Apply the facade's Ollama auto-detection so qwen3+/qwq/deepseek-r1
        # models use the native /api/chat dialect with think=false.
        # Before this fix, the benchmark hit /v1/chat/completions which puts
        # thinking into message.reasoning and leaves content empty — every
        # run was corrupted by partial/empty LLM responses.
        from acervo.facade import _ollama_dialect_kwargs

        base_url = os.getenv("ACERVO_LIGHT_MODEL_URL", "http://localhost:11434/v1")
        model = os.getenv("ACERVO_LIGHT_MODEL", "qwen3.5:9b")
        llm = OpenAIClient(
            base_url=base_url,
            model=model,
            api_key=os.getenv("ACERVO_LIGHT_API_KEY", "ollama"),
            **_ollama_dialect_kwargs(base_url, model),
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
        entity_accs_fuzzy = [t.entity_accuracy_fuzzy for t in results if t.entity_accuracy_fuzzy >= 0]
        relation_accs = [t.relation_accuracy for t in results if t.relation_accuracy >= 0]
        relation_accs_fuzzy = [t.relation_accuracy_fuzzy for t in results if t.relation_accuracy_fuzzy >= 0]
        fact_accs = [t.fact_accuracy for t in results if t.fact_accuracy >= 0]
        fact_accs_fuzzy = [t.fact_accuracy_fuzzy for t in results if t.fact_accuracy_fuzzy >= 0]
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
            entity_accuracy_fuzzy_avg=(
                sum(entity_accs_fuzzy) / len(entity_accs_fuzzy) if entity_accs_fuzzy else 0
            ),
            relation_accuracy_avg=(
                sum(relation_accs) / len(relation_accs) if relation_accs else 0
            ),
            relation_accuracy_fuzzy_avg=(
                sum(relation_accs_fuzzy) / len(relation_accs_fuzzy) if relation_accs_fuzzy else 0
            ),
            fact_accuracy_avg=(
                sum(fact_accs) / len(fact_accs) if fact_accs else 0
            ),
            fact_accuracy_fuzzy_avg=(
                sum(fact_accs_fuzzy) / len(fact_accs_fuzzy) if fact_accs_fuzzy else 0
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
        fuzzy = f"/{tr.entity_accuracy_fuzzy:.0%}" if tr.entity_accuracy_fuzzy >= 0 else ""
        parts.append(f"ent={tr.entity_accuracy:.0%}{fuzzy}")
    if tr.relation_accuracy >= 0:
        fuzzy = f"/{tr.relation_accuracy_fuzzy:.0%}" if tr.relation_accuracy_fuzzy >= 0 else ""
        parts.append(f"rel={tr.relation_accuracy:.0%}{fuzzy}")
    if tr.fact_accuracy >= 0:
        fuzzy = f"/{tr.fact_accuracy_fuzzy:.0%}" if tr.fact_accuracy_fuzzy >= 0 else ""
        parts.append(f"fact={tr.fact_accuracy:.0%}{fuzzy}")
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
    print(f"    Entity acc:   {result.entity_accuracy_avg:.0%} / {result.entity_accuracy_fuzzy_avg:.0%} (exact/fuzzy)")
    print(f"    Relation acc: {result.relation_accuracy_avg:.0%} / {result.relation_accuracy_fuzzy_avg:.0%}")
    print(f"    Fact acc:     {result.fact_accuracy_avg:.0%} / {result.fact_accuracy_fuzzy_avg:.0%}")
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
    """Version string for reports, includes backend suffix.

    As of v0.6.0 the canonical backend is ``ladybug`` and its reports
    live in the un-suffixed directory (``v0.6.0/``). Non-default
    backends (``json`` for TopicGraph fallback runs) get a suffix so
    they don't overwrite the canonical snapshots.
    """
    backend = os.environ.get("ACERVO_TEST_BACKEND", "ladybug")
    base = "v0.6.0"
    return f"{base}-{backend}" if backend != "ladybug" else base


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
        "entity_accuracy_fuzzy": round(result.entity_accuracy_fuzzy_avg * 100),
        "relation_accuracy": round(result.relation_accuracy_avg * 100),
        "relation_accuracy_fuzzy": round(result.relation_accuracy_fuzzy_avg * 100),
        "fact_accuracy": round(result.fact_accuracy_avg * 100),
        "fact_accuracy_fuzzy": round(result.fact_accuracy_fuzzy_avg * 100),
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

        print(f"\n  {'='*70}")
        print(f"  ALL CASES: {passed}/{total} turns passed ({round(passed/total*100) if total else 0}%)")
        print(f"  FACTS: {total_parsed}/{total_raw} parsed (drop={overall_drop:.0%})")
        print(f"  {'':20s}  {'ent exact/fuzzy':>16s}  {'rel exact/fuzzy':>16s}  {'fact exact/fuzzy':>16s}")
        for r in all_results:
            print(f"    {r.name:20s}: "
                  f"ent={r.entity_accuracy_avg:.0%}/{r.entity_accuracy_fuzzy_avg:.0%} "
                  f"rel={r.relation_accuracy_avg:.0%}/{r.relation_accuracy_fuzzy_avg:.0%} "
                  f"fact={r.fact_accuracy_avg:.0%}/{r.fact_accuracy_fuzzy_avg:.0%} "
                  f"facts={r.total_parsed_facts}/{r.total_raw_facts}")
        print(f"  {'='*70}")


