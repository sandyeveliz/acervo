#!/usr/bin/env python3
"""Quick prompt iteration test — samples worst turns from the last benchmark.

Picks turns that had fact drops, low entity accuracy, or missing relations
across ALL 8 cases. Re-runs only those through the pipeline.

Usage:
    python scripts/test_prompt_quick.py                          # 20 worst, qwen2.5:7b
    python scripts/test_prompt_quick.py --count 30               # 30 worst
    python scripts/test_prompt_quick.py --model qwen3.5:9b       # test different model
    python scripts/test_prompt_quick.py --model qwen2.5:3b       # test small model
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from acervo import Acervo
from acervo.openai_client import OpenAIClient

_CASES_DIR = Path(__file__).parent.parent / "tests" / "integration" / "scenarios" / "cases"
_REPORTS = Path(__file__).parent.parent / "tests" / "integration" / "reports" / "v0.6.0-ladybug"

CASES = ["casa", "finanzas", "fitness", "libro", "proyecto_codigo",
         "salud_familia", "trabajo", "viajes"]


def find_worst_turns(count: int = 20) -> list[dict]:
    """Find worst turns across all cases — mix of fact drops, entity misses, relation misses."""
    candidates = []

    for case in CASES:
        report = _REPORTS / f"case_{case}.json"
        if not report.exists():
            continue
        data = json.loads(report.read_text(encoding="utf-8"))
        for t in data.get("turns", []):
            # Score: higher = worse. Weight fact drops, entity misses, relation misses
            dropped = t.get("dropped_facts", 0)
            ent_acc = t.get("entity_accuracy")
            ent_miss = len(t.get("missing_entities", []))
            rel_miss = len(t.get("missing_relations", []))  # from failures text
            fact_miss = len(t.get("missing_facts", []))

            score = dropped * 3 + ent_miss * 2 + fact_miss * 2 + rel_miss
            if ent_acc is not None and ent_acc == 0:
                score += 5  # bonus for total entity failure

            if score > 0:
                candidates.append({
                    "case": case,
                    "turn_idx": t["turn"] - 1,
                    "turn_num": t["turn"],
                    "msg": t["user_msg"],
                    "score": score,
                    "prev_dropped": dropped,
                    "prev_ent_acc": ent_acc,
                    "prev_fact_acc": t.get("fact_accuracy"),
                })

    # Sort by score descending, then take top N spread across cases
    candidates.sort(key=lambda x: -x["score"])

    # Ensure diversity: max 4 per case
    selected = []
    per_case: dict[str, int] = {}
    for c in candidates:
        case_count = per_case.get(c["case"], 0)
        if case_count >= 4:
            continue
        selected.append(c)
        per_case[c["case"]] = case_count + 1
        if len(selected) >= count:
            break

    return selected


def load_turn(case: str, turn_idx: int) -> dict | None:
    path = _CASES_DIR / f"{case}.jsonl"
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == turn_idx:
                return json.loads(line.strip())
    return None


async def run_quick_test(count: int = 20, model: str = "qwen2.5:7b"):
    worst = find_worst_turns(count)
    print(f"Testing {len(worst)} worst turns across {len(set(w['case'] for w in worst))} cases")
    print(f"Model: {model}\n")

    llm = OpenAIClient(
        base_url="http://localhost:11434/v1",
        model=model,
        api_key="ollama",
    )

    # Stats
    better = same = worse = errors = 0
    total_raw = total_parsed = 0
    total_prev_dropped = 0

    for i, w in enumerate(worst):
        turn = load_turn(w["case"], w["turn_idx"])
        if not turn:
            continue

        conv = turn.get("conversation", [])
        if not conv:
            continue
        user_msg = conv[-1].get("content", "").strip()

        # Fresh DB per turn
        tmpdir = tempfile.mkdtemp()
        graph_path = Path(tmpdir) / "graph"
        graph_path.mkdir(parents=True, exist_ok=True)
        acervo = Acervo(llm=llm, owner="Sandy", persist_path=str(graph_path), graph_backend="ladybug")

        history = [msg for msg in conv[:-1]]

        try:
            prep = await acervo.prepare(user_msg, history)
            debug = prep.debug or {}
            s1_val = debug.get("s1_validation", {})
            s1_det = debug.get("s1_detection", {})

            raw_f = s1_val.get("raw_facts", 0)
            parsed_f = s1_val.get("parsed_facts", 0)
            dropped_f = raw_f - parsed_f
            ent_count = s1_det.get("entities_extracted", 0)
            rel_count = s1_det.get("relations_extracted", 0)
            reasons = [d.get("reason", "")[:50] for d in s1_val.get("dropped_facts", [])]

            total_raw += raw_f
            total_parsed += parsed_f
            total_prev_dropped += w["prev_dropped"]

            # Compare
            if dropped_f < w["prev_dropped"]:
                status, emoji = "BETTER", "+"
                better += 1
            elif dropped_f == w["prev_dropped"]:
                status, emoji = "SAME", "="
                same += 1
            else:
                status, emoji = "WORSE", "-"
                worse += 1

            print(f"  {emoji} {w['case']:15s}:T{w['turn_num']:02d} "
                  f"{ent_count}E {rel_count}R facts={parsed_f}/{raw_f} "
                  f"(was {w['prev_dropped']}drop) {status}")
            if reasons:
                print(f"      drop: {reasons[0]}")
            print(f"      {user_msg[:70]}")

        except Exception as e:
            errors += 1
            print(f"  ! {w['case']:15s}:T{w['turn_num']:02d} ERROR: {e}")

        print()

    # Summary
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Turns tested: {len(worst)}")
    print(f"Results: {better} BETTER, {same} SAME, {worse} WORSE, {errors} ERROR")
    print(f"Facts: {total_parsed}/{total_raw} parsed "
          f"(drop={total_raw - total_parsed})")
    if total_raw > 0:
        print(f"Drop rate: {(total_raw - total_parsed) * 100 // total_raw}%")
    print(f"Previous total drops on these turns: {total_prev_dropped}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--model", default=os.environ.get("ACERVO_LIGHT_MODEL", "qwen2.5:7b"))
    args = parser.parse_args()
    asyncio.run(run_quick_test(args.count, args.model))
