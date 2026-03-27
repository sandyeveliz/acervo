"""Scenario framework — loads YAML scenarios and runs them through Acervo.

Supports two modes:
  - With Acervo: prepare() → process() per turn (graph compression)
  - Baseline: full history token counting (no Acervo, no LLM calls)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from acervo.graph import _make_id
from acervo.token_counter import count_tokens
from tests.integration.metrics import (
    Checkpoint,
    EntityExpectation,
    FactExpectation,
    RelationExpectation,
    Scenario,
    ScenarioResult,
    ScenarioTurn,
    TurnResult,
)

if TYPE_CHECKING:
    from acervo import Acervo

log = logging.getLogger(__name__)

SCENARIOS_DIR = Path(__file__).parent / "scenarios"

GARBAGE_LABELS = frozenset({
    "bien", "calor", "verano", "hoy", "che", "bueno", "dale", "ok",
    "si", "no", "hola", "chau", "nada", "todo", "uf",
})


# ── YAML Loader ──


def load_scenario(path: str | Path) -> Scenario:
    """Load a scenario from a YAML file."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    turns = []
    for t in data.get("turns", []):
        checkpoint = None
        if "checkpoint" in t:
            cp = t["checkpoint"]
            checkpoint = Checkpoint(
                should_have_context=cp.get("should_have_context", True),
                context_should_mention=cp.get("context_should_mention", []),
                context_should_not_mention=cp.get("context_should_not_mention", []),
            )

        turns.append(ScenarioTurn(
            user_msg=t["user_msg"],
            assistant_msg=t.get("assistant_msg", ""),
            phase=t.get("phase", ""),
            description=t.get("description", ""),
            expected_entities=[
                EntityExpectation(**e) for e in t.get("expected_entities", [])
            ],
            expected_relations=[
                RelationExpectation(**r) for r in t.get("expected_relations", [])
            ],
            expected_facts=[
                FactExpectation(**f) for f in t.get("expected_facts", [])
            ],
            checkpoint=checkpoint,
            max_new_nodes=t.get("max_new_nodes"),
        ))

    return Scenario(
        name=data.get("name", path.stem),
        category=data.get("category", ""),
        description=data.get("description", ""),
        turns=turns,
        system_prompt=data.get("system_prompt", ""),
        persona=data.get("persona", ""),
        persona_role=data.get("persona_role", ""),
        narrative_hook=data.get("narrative_hook", ""),
    )


def discover_scenarios(directory: Path | None = None) -> list[Path]:
    """Find all YAML scenario files in the scenarios directory."""
    d = directory or SCENARIOS_DIR
    return sorted(d.glob("*.yaml"))


# ── Scenario Runner ──


class ScenarioRunner:
    """Runs a scenario through the full Acervo pipeline."""

    def __init__(self, memory: "Acervo"):
        self._memory = memory
        self._soft_failures: list[str] = []
        self._soft_total: int = 0

    def _soft_check(self, condition: bool, msg: str) -> None:
        self._soft_total += 1
        if not condition:
            self._soft_failures.append(msg)
            log.warning("SOFT FAIL: %s", msg)

    async def run_with_acervo(self, scenario: Scenario) -> ScenarioResult:
        """Run all turns through prepare/process and collect metrics."""
        turn_results: list[TurnResult] = []
        baseline_tokens_running = 0

        # Add system prompt to history — build_context_stack() expects history[0]
        # to be a system message (same as the proxy path). Without this, the first
        # user message gets misinterpreted as the system prompt.
        system_prompt = scenario.system_prompt or "You are a helpful assistant."
        system_prompt_tokens = count_tokens(system_prompt)
        history: list[dict] = [{"role": "system", "content": system_prompt}]

        for i, turn in enumerate(scenario.turns):
            turn_num = i + 1
            log.info(
                "── Turn %d [%s]: %s ──",
                turn_num, turn.phase, turn.description or turn.user_msg[:50],
            )

            # Calculate baseline tokens (what full history would cost without Acervo).
            # Includes system prompt (constant) + cumulative messages.
            baseline_tokens_running += count_tokens(turn.user_msg)
            if turn.assistant_msg:
                baseline_tokens_running += count_tokens(turn.assistant_msg)
            baseline_total = system_prompt_tokens + baseline_tokens_running

            # Run Acervo pipeline
            prep = await self._memory.prepare(turn.user_msg, history)
            result = await self._memory.process(turn.user_msg, turn.assistant_msg)

            # Get timing from metrics
            m_turn = self._memory.metrics.turns[-1] if self._memory.metrics.turns else None
            prepare_ms = m_turn.prepare_ms if m_turn else 0
            process_ms = m_turn.process_ms if m_turn else 0

            # Capture extracted entity labels from this turn
            extracted_labels = []
            if result and hasattr(result, "entities"):
                extracted_labels = [e.label for e in result.entities]

            # Token breakdown: decompose prep.total_tokens into components
            user_tokens = count_tokens(turn.user_msg)
            # system_tokens comes from context_stack[0] if it's a system message
            sys_tk = 0
            if prep.context_stack and prep.context_stack[0].get("role") == "system":
                sys_tk = count_tokens(prep.context_stack[0].get("content", ""))
            # overhead = total - known components
            overhead_tk = max(0, prep.total_tokens - prep.warm_tokens - prep.hot_tokens - user_tokens - sys_tk)

            # Build turn result
            tr = TurnResult(
                turn_number=turn_num,
                phase=turn.phase,
                description=turn.description,
                acervo_tokens=prep.total_tokens,
                warm_tokens=prep.warm_tokens,
                hot_tokens=prep.hot_tokens,
                system_tokens=sys_tk,
                user_tokens=user_tokens,
                overhead_tokens=overhead_tk,
                baseline_tokens=baseline_total,
                savings_pct=(
                    (1 - prep.total_tokens / baseline_total) * 100
                    if baseline_total > 0 else 0
                ),
                context_hit=prep.has_context,
                node_count=self._memory.graph.node_count,
                edge_count=self._memory.graph.edge_count,
                prepare_ms=prepare_ms,
                process_ms=process_ms,
                user_msg=turn.user_msg,
                assistant_msg=turn.assistant_msg,
                warm_context=prep.warm_content or "",
                entities_extracted=extracted_labels,
                topic=prep.topic if hasattr(prep, "topic") else "",
            )

            # Validate expected entities
            if turn.expected_entities:
                tr.entities_expected = len(turn.expected_entities)
                for ee in turn.expected_entities:
                    node_id = _make_id(ee.label)
                    node = self._memory.graph.get_node(node_id)
                    if node:
                        tr.entities_found += 1
                        if ee.type:
                            self._soft_check(
                                node.get("type", "").lower() == ee.type.lower(),
                                f"Turn {turn_num}: '{ee.label}' type={node.get('type')}, expected={ee.type}",
                            )
                        if ee.layer:
                            self._soft_check(
                                node.get("layer", "") == ee.layer,
                                f"Turn {turn_num}: '{ee.label}' layer={node.get('layer')}, expected={ee.layer}",
                            )
                    else:
                        tr.entities_missing.append(ee.label)
                        self._soft_check(
                            False,
                            f"Turn {turn_num}: expected node '{ee.label}' (id={node_id}) not found",
                        )

            # Validate expected facts
            if turn.expected_facts:
                tr.facts_expected = len(turn.expected_facts)
                for ef in turn.expected_facts:
                    node_id = _make_id(ef.entity)
                    node = self._memory.graph.get_node(node_id)
                    if node:
                        facts = node.get("facts", [])
                        found = any(
                            ef.substring.lower() in f.get("fact", "").lower()
                            for f in facts
                        )
                        if found:
                            tr.facts_found += 1
                        else:
                            self._soft_check(
                                False,
                                f"Turn {turn_num}: fact '{ef.substring}' not found on '{ef.entity}'",
                            )

            # Validate checkpoint
            if turn.checkpoint:
                cp = turn.checkpoint
                warm = (prep.warm_content or "").lower()

                if cp.should_have_context:
                    self._soft_check(
                        prep.has_context,
                        f"Turn {turn_num}: expected context but has_context=False",
                    )

                for mention in cp.context_should_mention:
                    if mention.lower() in warm:
                        tr.context_mentions_ok.append(mention)
                    else:
                        tr.context_mentions_missing.append(mention)
                        self._soft_check(
                            False,
                            f"Turn {turn_num}: warm_content should mention '{mention}'",
                        )

                for mention in cp.context_should_not_mention:
                    if mention.lower() in warm:
                        tr.context_mentions_unwanted.append(mention)
                        self._soft_check(
                            False,
                            f"Turn {turn_num}: warm_content should NOT mention '{mention}'",
                        )

            # Validate max_new_nodes (small talk guard)
            if turn.max_new_nodes is not None and turn_results:
                prev_count = turn_results[-1].node_count
                delta = self._memory.graph.node_count - prev_count
                self._soft_check(
                    delta <= turn.max_new_nodes,
                    f"Turn {turn_num}: created {delta} nodes, max was {turn.max_new_nodes}",
                )

            turn_results.append(tr)

            # Update history
            history.append({"role": "user", "content": turn.user_msg})
            if turn.assistant_msg:
                history.append({"role": "assistant", "content": turn.assistant_msg})

            log.info(
                "  tokens=%d (baseline=%d, savings=%.0f%%) nodes=%d edges=%d prep=%dms proc=%dms",
                tr.acervo_tokens, tr.baseline_tokens, tr.savings_pct,
                tr.node_count, tr.edge_count, tr.prepare_ms, tr.process_ms,
            )

        # Build final result
        g = self._memory.graph
        all_nodes = g.get_all_nodes()
        entity_nodes = [n for n in all_nodes if n.get("kind", "entity") == "entity"]
        entity_labels = {n.get("label", "").lower() for n in entity_nodes}
        phantoms = list(entity_labels & GARBAGE_LABELS)

        m = self._memory.metrics
        trace_path = str(m.trace_path) if m.trace_path else ""
        trace_lines = 0
        if m.trace_path and m.trace_path.exists():
            trace_lines = len(m.trace_path.read_text(encoding="utf-8").strip().split("\n"))

        return ScenarioResult(
            name=scenario.name,
            category=scenario.category,
            turns=turn_results,
            total_turns=len(turn_results),
            final_node_count=len(entity_nodes),
            final_edge_count=g.edge_count,
            personal_nodes=sum(1 for n in entity_nodes if n.get("layer") == "PERSONAL"),
            universal_nodes=sum(1 for n in entity_nodes if n.get("layer") == "UNIVERSAL"),
            phantom_entities=phantoms,
            graph_nodes=[
                {
                    "label": n.get("label", ""),
                    "type": n.get("type", ""),
                    "layer": n.get("layer", ""),
                    "facts": len(n.get("facts", [])),
                    "status": n.get("status", ""),
                }
                for n in sorted(entity_nodes, key=lambda n: n.get("label", ""))
            ],
            trace_path=trace_path,
            trace_lines=trace_lines,
            soft_failures=list(self._soft_failures),
            soft_total=self._soft_total,
        )


# ── Hard Assertions ──


def run_final_assertions(result: ScenarioResult) -> None:
    """Hard assertions that must pass for every scenario."""
    # Graph integrity
    assert result.final_node_count >= 5, (
        f"Expected >=5 entity nodes, got {result.final_node_count}"
    )
    assert result.final_edge_count >= 3, (
        f"Expected >=3 edges, got {result.final_edge_count}"
    )
    assert not result.phantom_entities, (
        f"Phantom nodes from small talk: {result.phantom_entities}"
    )

    # Metrics completeness
    assert result.total_turns == len(result.turns)
    for t in result.turns:
        assert t.prepare_ms > 0, f"Turn {t.turn_number} has prepare_ms=0"

    # Token count present on most turns
    turns_with_tokens = sum(1 for t in result.turns if t.acervo_tokens > 0)
    assert turns_with_tokens >= result.total_turns * 0.8, (
        f"Only {turns_with_tokens}/{result.total_turns} turns have acervo_tokens > 0"
    )

    # Token stability (sub-linear growth)
    third = result.total_turns // 3
    if third > 0:
        first_avg = sum(t.acervo_tokens for t in result.turns[:third]) / third
        last_avg = sum(t.acervo_tokens for t in result.turns[-third:]) / third
        # Use a floor of 150tk for first_avg to avoid false positives when
        # early turns have near-zero tokens (empty graph produces minimal context)
        effective_first = max(first_avg, 150)
        assert last_avg < effective_first * 4, (
            f"Token growth unbounded: first_third={first_avg:.0f}, last_third={last_avg:.0f}"
        )

    # Context hits
    hits = sum(1 for t in result.turns if t.context_hit)
    assert hits >= 3, f"Only {hits} context hits in {result.total_turns} turns"

    # Trace persistence
    assert result.trace_lines == result.total_turns, (
        f"Trace has {result.trace_lines} lines, expected {result.total_turns}"
    )
