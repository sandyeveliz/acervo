#!/usr/bin/env python3
"""Run a few turns of a case scenario and dump raw S1 responses to JSONL.

Usage:
    python scripts/debug_extraction.py [--case casa] [--turns 5]

Output: tests/integration/reports/v0.6.0-ladybug/debug_raw_extraction.jsonl
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from acervo import Acervo
from acervo.openai_client import OpenAIClient
from acervo.s1_unified import S1Unified, build_graph_summary, generate_topic_hint

_CASES_DIR = Path(__file__).parent.parent / "tests" / "integration" / "scenarios" / "cases"
_REPORTS = Path(__file__).parent.parent / "tests" / "integration" / "reports" / "v0.6.0-ladybug"


def load_case(name: str) -> list[dict]:
    path = _CASES_DIR / f"{name}.jsonl"
    turns = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                turns.append(json.loads(line))
    return turns


async def run_debug(case_name: str, max_turns: int = 5):
    turns = load_case(case_name)
    print(f"Loaded {len(turns)} turns from {case_name}, running {max_turns}")

    llm = OpenAIClient(
        base_url="http://localhost:11434/v1",
        model="acervo-extractor-v3",
        api_key="ollama",
    )

    tmpdir = tempfile.mkdtemp()
    graph_path = Path(tmpdir) / "graph"
    graph_path.mkdir(parents=True, exist_ok=True)
    acervo = Acervo(llm=llm, owner="Sandy", persist_path=str(graph_path), graph_backend="ladybug")

    s1 = S1Unified(llm)

    history: list[dict] = []
    records: list[dict] = []

    for i, turn_spec in enumerate(turns[:max_turns]):
        conv = turn_spec.get("conversation", [])
        if not conv:
            continue
        user_msg = conv[-1].get("content", "").strip()
        if not user_msg:
            continue

        # Build context like the pipeline does
        all_nodes = acervo.graph.get_all_nodes()
        existing_summary = build_graph_summary([dict(n) for n in all_nodes], user_msg)
        existing_names = {n.get("label", "") for n in all_nodes if n.get("label")}

        prev_assistant = ""
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                prev_assistant = msg.get("content", "")
                break

        # Call S1 directly to get raw response
        result = await s1.run(
            user_msg=user_msg,
            prev_assistant_msg=prev_assistant,
            current_topic="none",
            topic_hint="",
            existing_nodes_summary=existing_summary,
            existing_node_names=existing_names,
        )

        # Also run through acervo for graph updates
        try:
            prep = await acervo.prepare(user_msg, history)
            await acervo.process(user_msg, "Entendido.")
        except Exception as e:
            print(f"  T{i+1} pipeline error: {e}")

        expected = turn_spec.get("expected", {})

        record = {
            "turn": i + 1,
            "user_msg": user_msg[:200],
            "raw_response": result.raw_response[:3000] if result.raw_response else "",
            "parsed": {
                "topic": {"action": result.topic.action, "label": result.topic.label},
                "intent": result.intent,
                "entities": [{"name": e.name, "type": e.type} for e in result.extraction.entities],
                "relations": [{"src": r.source, "tgt": r.target, "rel": r.relation} for r in result.extraction.relations],
                "facts": [{"entity": f.entity, "text": f.fact} for f in result.extraction.facts],
            },
            "expected_entities": expected.get("entities", []),
            "expected_facts": expected.get("facts", []),
            "graph_nodes": acervo.graph.node_count,
            "graph_edges": acervo.graph.edge_count,
        }
        records.append(record)

        e_count = len(result.extraction.entities)
        r_count = len(result.extraction.relations)
        f_count = len(result.extraction.facts)
        print(f"  T{i+1:02d} {e_count}E {r_count}R {f_count}F | graph={acervo.graph.node_count}n/{acervo.graph.edge_count}e | {user_msg[:60]}")

        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": "Entendido."})

    # Write output
    _REPORTS.mkdir(parents=True, exist_ok=True)
    out_path = _REPORTS / "debug_raw_extraction.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="casa")
    parser.add_argument("--turns", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(run_debug(args.case, args.turns))
