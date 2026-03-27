"""Prompt variation tests — model-dependent behavior validation.

Tests that different extraction prompts produce different results, and
compares their entity recall, fact extraction, and noise rejection.

This validates that Acervo's extraction quality is prompt-dependent
(not just model-dependent), and helps find optimal prompt configurations.

Requires a running LLM server. Run with:
    pytest tests/integration/test_prompts.py -m integration -v -s

Run a single variant:
    pytest tests/integration/test_prompts.py -m integration -v -s -k "strict"
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path

import pytest

from acervo import Acervo
from acervo.graph import _make_id
from tests.integration.framework import (
    GARBAGE_LABELS,
    ScenarioRunner,
    load_scenario,
)

log = logging.getLogger(__name__)

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
REPORTS_DIR = Path(__file__).parent / "reports"

# ── Prompt Variants ──
# Each variant overrides the S1 Unified system prompt.
# The default is the one baked into s1_unified.py.

PROMPT_VARIANTS: dict[str, str | None] = {
    "default": None,  # use built-in prompt (no override)
    "strict": (
        "You are a strict knowledge extractor for a personal knowledge graph. "
        "Only extract entities that are EXPLICITLY named in the conversation. "
        "Do not infer entities, relations, or facts that are not directly stated. "
        "If an entity is referenced by nickname or shortened name, use ONLY that form. "
        "Output valid JSON only, no markdown, no explanation."
    ),
    "verbose": (
        "You are a thorough knowledge extractor for a personal knowledge graph. "
        "Extract ALL entities mentioned or implied in the conversation, including "
        "abbreviated references, nicknames, and implicit entities. "
        "For each entity, also extract any facts that can be inferred from context. "
        "Be comprehensive — it's better to extract too much than too little. "
        "Output valid JSON only, no markdown, no explanation."
    ),
    "structured": (
        "You are a knowledge extractor for a personal knowledge graph. "
        "Analyze the conversation and return a single JSON object with topic "
        "classification, entities, relations, and facts. "
        "Rules: "
        "1. Use the person's FULL name as the label (e.g., 'Alan Moore' not 'Moore'). "
        "2. Include work titles exactly as mentioned (e.g., 'The Dark Knight Returns'). "
        "3. Classify entities by type: Person, Organization, Place, Work, Technology, Event, Concept. "
        "4. Only extract facts that are explicitly stated, never infer. "
        "Output valid JSON only, no markdown, no explanation."
    ),
}


# ── Fixtures ──


def _create_memory(llm_client, prompt_override: str | None = None):
    """Create an Acervo instance with optional prompt override."""
    tmp = Path(tempfile.mkdtemp()) / ".acervo"
    graph_path = tmp / "data" / "graph"
    graph_path.mkdir(parents=True, exist_ok=True)

    prompts = {}
    if prompt_override is not None:
        prompts["s1_unified"] = prompt_override

    memory = Acervo(
        llm=llm_client,
        owner="Sandy",
        persist_path=graph_path,
        prompts=prompts if prompts else None,
    )
    return memory, tmp.parent


# ── Small Scenario for Prompt Testing ──
# 10 turns is enough to compare prompt behavior without 5-min waits.

PROMPT_TEST_SCENARIO = SCENARIOS_DIR / "prompt_test.yaml"


def _variant_ids() -> list[str]:
    return list(PROMPT_VARIANTS.keys())


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("variant_id", _variant_ids())
async def test_prompt_variant(llm_client, variant_id):
    """Run a short scenario with each prompt variant and compare extraction quality."""
    if not PROMPT_TEST_SCENARIO.exists():
        pytest.skip(f"Prompt test scenario not found: {PROMPT_TEST_SCENARIO}")

    scenario = load_scenario(PROMPT_TEST_SCENARIO)
    prompt_override = PROMPT_VARIANTS[variant_id]

    memory, tmp_root = _create_memory(llm_client, prompt_override)
    try:
        runner = ScenarioRunner(memory)
        result = await runner.run_with_acervo(scenario)

        # Collect per-variant metrics
        g = memory.graph
        all_nodes = g.get_all_nodes()
        entity_nodes = [n for n in all_nodes if n.get("kind", "entity") == "entity"]
        entity_labels = {n.get("label", "").lower() for n in entity_nodes}
        phantoms = list(entity_labels & GARBAGE_LABELS)

        variant_report = {
            "variant": variant_id,
            "prompt": prompt_override or "(built-in default)",
            "node_count": len(entity_nodes),
            "edge_count": g.edge_count,
            "entity_recall": result.entity_recall,
            "phantom_count": len(phantoms),
            "phantoms": phantoms,
            "avg_savings_pct": result.avg_savings_pct,
            "context_hit_rate": result.context_hit_rate,
            "soft_pass_rate": result.soft_pass_rate,
            "soft_total": result.soft_total,
            "soft_failures": result.soft_failures,
            "entities": sorted(entity_labels),
        }

        # Save per-variant report
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORTS_DIR / f"prompt_{variant_id}.json"
        report_path.write_text(
            json.dumps(variant_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Print summary
        print(f"\n{'=' * 60}")
        print(f"  Prompt variant: {variant_id}")
        print(f"  Nodes: {len(entity_nodes)}  Edges: {g.edge_count}  Phantoms: {len(phantoms)}")
        print(f"  Entity recall: {result.entity_recall:.0%}")
        print(f"  Token savings: {result.avg_savings_pct:.1f}%")
        print(f"  Context hits: {result.context_hit_rate:.0%}")
        print(f"  Soft pass rate: {result.soft_pass_rate:.0%} ({result.soft_total - len(result.soft_failures)}/{result.soft_total})")
        print(f"  Entities: {sorted(entity_labels)}")
        print(f"{'=' * 60}\n")

        # Hard assertions — prompt variants should still produce valid graphs
        assert len(entity_nodes) >= 3, (
            f"Variant '{variant_id}': too few entities ({len(entity_nodes)})"
        )
        assert not phantoms, (
            f"Variant '{variant_id}': phantom nodes: {phantoms}"
        )

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_prompt_comparison_report(llm_client):
    """Run all prompt variants and generate a comparison report.

    This is a single test that runs all variants sequentially and
    produces a side-by-side comparison. Slower but gives a unified view.
    """
    if not PROMPT_TEST_SCENARIO.exists():
        pytest.skip(f"Prompt test scenario not found: {PROMPT_TEST_SCENARIO}")

    scenario = load_scenario(PROMPT_TEST_SCENARIO)
    all_variants = {}

    for variant_id, prompt_override in PROMPT_VARIANTS.items():
        log.info("Running prompt variant: %s", variant_id)
        memory, tmp_root = _create_memory(llm_client, prompt_override)
        try:
            runner = ScenarioRunner(memory)
            result = await runner.run_with_acervo(scenario)

            g = memory.graph
            all_nodes = g.get_all_nodes()
            entity_nodes = [n for n in all_nodes if n.get("kind", "entity") == "entity"]
            entity_labels = {n.get("label", "").lower() for n in entity_nodes}
            phantoms = list(entity_labels & GARBAGE_LABELS)

            all_variants[variant_id] = {
                "node_count": len(entity_nodes),
                "edge_count": g.edge_count,
                "entity_recall": result.entity_recall,
                "phantom_count": len(phantoms),
                "avg_savings_pct": result.avg_savings_pct,
                "context_hit_rate": result.context_hit_rate,
                "soft_pass_rate": result.soft_pass_rate,
                "entities": sorted(entity_labels),
            }
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    # Generate comparison report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    lines = ["# Prompt Variant Comparison Report", ""]
    lines.append("| Metric | " + " | ".join(all_variants.keys()) + " |")
    lines.append("|--------|" + "|".join("-------" for _ in all_variants) + "|")

    metrics = [
        ("Nodes", "node_count", "d"),
        ("Edges", "edge_count", "d"),
        ("Entity Recall", "entity_recall", ".0%"),
        ("Phantoms", "phantom_count", "d"),
        ("Token Savings", "avg_savings_pct", ".1f"),
        ("Context Hits", "context_hit_rate", ".0%"),
        ("Soft Pass Rate", "soft_pass_rate", ".0%"),
    ]
    for label, key, fmt in metrics:
        values = []
        for v in all_variants.values():
            val = v[key]
            if fmt == ".0%":
                values.append(f"{val:.0%}")
            elif fmt == ".1f":
                values.append(f"{val:.1f}%")
            else:
                values.append(str(val))
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    lines.append("")
    lines.append("## Entities per Variant")
    lines.append("")
    for variant_id, data in all_variants.items():
        lines.append(f"### {variant_id}")
        lines.append(f"  {', '.join(data['entities'])}")
        lines.append("")

    # Unique entities analysis
    all_entity_sets = {k: set(v["entities"]) for k, v in all_variants.items()}
    all_entities = set()
    for s in all_entity_sets.values():
        all_entities |= s

    lines.append("## Entity Coverage Matrix")
    lines.append("")
    lines.append("| Entity | " + " | ".join(all_variants.keys()) + " |")
    lines.append("|--------|" + "|".join("-------" for _ in all_variants) + "|")
    for entity in sorted(all_entities):
        marks = []
        for variant_id in all_variants:
            marks.append("Y" if entity in all_entity_sets[variant_id] else "-")
        lines.append(f"| {entity} | " + " | ".join(marks) + " |")
    lines.append("")

    report = "\n".join(lines)
    (REPORTS_DIR / "prompt_comparison.md").write_text(report, encoding="utf-8")
    (REPORTS_DIR / "prompt_comparison.json").write_text(
        json.dumps(all_variants, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    print(f"\n{report}")

    # Save per-variant JSON too
    for variant_id, data in all_variants.items():
        (REPORTS_DIR / f"prompt_{variant_id}.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8",
        )
