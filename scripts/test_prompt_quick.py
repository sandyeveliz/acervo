#!/usr/bin/env python3
"""Quick prompt iteration test — samples worst turns from the last benchmark.

Picks turns that had fact drops, low entity accuracy, or missing relations
across ALL 8 cases. Re-runs only those through the pipeline.
Shows dual metrics (exact/fuzzy) for real comparison.

Usage:
    python scripts/test_prompt_quick.py                          # 20 worst, default model
    python scripts/test_prompt_quick.py --count 30               # 30 worst
    python scripts/test_prompt_quick.py --model qwen3.5:9b       # test different model
    python scripts/test_prompt_quick.py --model acervo-extractor-v4
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from difflib import get_close_matches, SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from acervo import Acervo
from acervo.openai_client import OpenAIClient
from acervo.graph.ids import _make_id

_CASES_DIR = Path(__file__).parent.parent / "tests" / "integration" / "scenarios" / "cases"
_REPORTS = Path(__file__).parent.parent / "tests" / "integration" / "reports" / "v0.6.0-ladybug"

CASES = ["casa", "finanzas", "fitness", "libro", "proyecto_codigo",
         "salud_familia", "trabajo", "viajes"]


# ── Fuzzy matching (same logic as test_case_scenarios.py) ──

def _normalize(s: str) -> str:
    return " ".join(s.lower().strip().split())


def _fuzzy_find_node(target_id, target_label, all_nodes, all_labels):
    if target_id in all_nodes:
        return True
    if get_close_matches(target_id, all_nodes.keys(), n=1, cutoff=0.7):
        return True
    norm = _normalize(target_label)
    if len(norm) >= 4:
        for lbl in all_labels.values():
            if len(lbl) >= 4 and (norm in lbl or lbl in norm):
                return True
    if norm and get_close_matches(norm, list(all_labels.values()), n=1, cutoff=0.7):
        return True
    return False


def _fact_matches(fact_text, candidate):
    if not fact_text or not candidate:
        return False
    if fact_text in candidate or candidate in fact_text:
        return True
    ew = set(fact_text.split())
    aw = set(candidate.split())
    if ew and len(ew & aw) / len(ew) > 0.4:
        return True
    if SequenceMatcher(None, fact_text, candidate).ratio() >= 0.5:
        return True
    return False


def compute_accuracy(expected_spec: list[dict], graph) -> dict:
    """Compute exact + fuzzy accuracy for entities, relations, facts."""
    all_nodes = {n["id"]: n for n in graph.get_all_nodes()}
    all_labels = {n["id"]: _normalize(n.get("label", "")) for n in graph.get_all_nodes()}

    # Entities
    exp_entities = [e for e in expected_spec.get("entities", [])]
    ent_exact = ent_fuzzy = 0
    ent_total = len(exp_entities)
    for e in exp_entities:
        eid = _make_id(e.get("label", ""))
        label = e.get("label", "")
        if eid in all_nodes or any(_normalize(label) in _normalize(n.get("label", "")) for n in all_nodes.values()):
            ent_exact += 1
        if _fuzzy_find_node(eid, label, all_nodes, all_labels):
            ent_fuzzy += 1

    # Facts
    all_facts_flat = []
    for n in all_nodes.values():
        for f in n.get("facts", []):
            all_facts_flat.append(_normalize(f.get("fact", "")))

    exp_facts = expected_spec.get("facts", [])
    fact_exact = fact_fuzzy = 0
    fact_total = len(exp_facts)
    for fs in exp_facts:
        entity_id = _normalize(fs.get("entity", ""))
        fact_text = _normalize(fs.get("text", ""))

        # Exact: find node, search its facts
        node = all_nodes.get(entity_id)
        if not node:
            for nid in all_nodes:
                if entity_id and entity_id in nid:
                    node = all_nodes[nid]
                    break
        if node:
            nf = [_normalize(f.get("fact", "")) for f in node.get("facts", [])]
            if any(_fact_matches(fact_text, x) for x in nf):
                fact_exact += 1

        # Fuzzy: try fuzzy node resolve, fallback to all facts
        fuzzy_node = node
        if not fuzzy_node and entity_id:
            m = get_close_matches(entity_id, list(all_nodes.keys()), n=1, cutoff=0.6)
            if m:
                fuzzy_node = all_nodes.get(m[0])
        if fuzzy_node:
            nf = [_normalize(f.get("fact", "")) for f in fuzzy_node.get("facts", [])]
            if any(_fact_matches(fact_text, x) for x in nf):
                fact_fuzzy += 1
            elif any(_fact_matches(fact_text, x) for x in all_facts_flat):
                fact_fuzzy += 1
        elif any(_fact_matches(fact_text, x) for x in all_facts_flat):
            fact_fuzzy += 1

    # Relations
    actual_pairs = set()
    actual_eps = set()
    for n in all_nodes.values():
        for e in graph.get_edges_for(n.get("id", "")):
            s, t = _normalize(e.get("source", "")), _normalize(e.get("target", ""))
            actual_pairs.add((s, t))
            actual_eps.add(s)
            actual_eps.add(t)

    exp_rels = expected_spec.get("relations", [])
    rel_exact = rel_fuzzy = 0
    rel_total = len(exp_rels)
    ep_list = list(actual_eps)
    for r in exp_rels:
        s, t = _normalize(r.get("source", "")), _normalize(r.get("target", ""))
        if (s, t) in actual_pairs or (t, s) in actual_pairs:
            rel_exact += 1
            rel_fuzzy += 1
        else:
            sm = get_close_matches(s, ep_list, n=1, cutoff=0.7)
            tm = get_close_matches(t, ep_list, n=1, cutoff=0.7)
            fs = sm[0] if sm else s
            ft = tm[0] if tm else t
            if (fs, ft) in actual_pairs or (ft, fs) in actual_pairs:
                rel_fuzzy += 1

    return {
        "ent_exact": ent_exact, "ent_fuzzy": ent_fuzzy, "ent_total": ent_total,
        "fact_exact": fact_exact, "fact_fuzzy": fact_fuzzy, "fact_total": fact_total,
        "rel_exact": rel_exact, "rel_fuzzy": rel_fuzzy, "rel_total": rel_total,
    }


# ── Turn selection ──

def find_worst_turns(count: int = 20) -> list[dict]:
    candidates = []
    for case in CASES:
        report = _REPORTS / f"case_{case}.json"
        if not report.exists():
            # Try v0.6.0 fallback
            report = _REPORTS.parent / "v0.6.0" / f"case_{case}.json"
        if not report.exists():
            continue
        data = json.loads(report.read_text(encoding="utf-8"))
        for t in data.get("turns", []):
            dropped = t.get("dropped_facts", 0)
            ent_miss = len(t.get("missing_entities", []))
            fact_miss = len(t.get("missing_facts", []))
            score = dropped * 3 + ent_miss * 2 + fact_miss * 2
            if t.get("entity_accuracy") is not None and t["entity_accuracy"] == 0:
                score += 5
            if score > 0:
                candidates.append({
                    "case": case, "turn_idx": t["turn"] - 1, "turn_num": t["turn"],
                    "msg": t["user_msg"], "score": score, "prev_dropped": dropped,
                })
    candidates.sort(key=lambda x: -x["score"])
    selected = []
    per_case: dict[str, int] = {}
    for c in candidates:
        cc = per_case.get(c["case"], 0)
        if cc >= 4:
            continue
        selected.append(c)
        per_case[c["case"]] = cc + 1
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

    llm = OpenAIClient(base_url="http://localhost:11434/v1", model=model, api_key="ollama")

    # Aggregates
    better = same = worse = errors = 0
    total_raw = total_parsed = 0
    agg = {"ent_exact": 0, "ent_fuzzy": 0, "ent_total": 0,
           "fact_exact": 0, "fact_fuzzy": 0, "fact_total": 0,
           "rel_exact": 0, "rel_fuzzy": 0, "rel_total": 0}

    for w in worst:
        turn = load_turn(w["case"], w["turn_idx"])
        if not turn:
            continue
        conv = turn.get("conversation", [])
        if not conv:
            continue
        user_msg = conv[-1].get("content", "").strip()
        expected = turn.get("expected", {})

        tmpdir = tempfile.mkdtemp()
        graph_path = Path(tmpdir) / "graph"
        graph_path.mkdir(parents=True, exist_ok=True)
        acervo = Acervo(llm=llm, owner="Sandy", persist_path=str(graph_path), graph_backend="ladybug")

        history = [msg for msg in conv[:-1]]

        try:
            prep = await acervo.prepare(user_msg, history)
            await acervo.process(user_msg, "Entendido.")

            debug = prep.debug or {}
            s1_val = debug.get("s1_validation", {})
            s1_det = debug.get("s1_detection", {})

            raw_f = s1_val.get("raw_facts", 0)
            parsed_f = s1_val.get("parsed_facts", 0)
            dropped_f = raw_f - parsed_f
            ent_count = s1_det.get("entities_extracted", 0)
            rel_count = s1_det.get("relations_extracted", 0)

            total_raw += raw_f
            total_parsed += parsed_f

            # Compute accuracy against expected
            acc = compute_accuracy(expected, acervo.graph)
            for k in agg:
                agg[k] += acc[k]

            # Drop comparison
            if dropped_f < w["prev_dropped"]:
                status, emoji = "BETTER", "+"
                better += 1
            elif dropped_f == w["prev_dropped"]:
                status, emoji = "SAME", "="
                same += 1
            else:
                status, emoji = "WORSE", "-"
                worse += 1

            # Format accuracy as exact/fuzzy
            ent_str = ""
            if acc["ent_total"] > 0:
                ee = acc["ent_exact"] * 100 // acc["ent_total"]
                ef = acc["ent_fuzzy"] * 100 // acc["ent_total"]
                ent_str = f"ent={ee}%/{ef}% "
            fact_str = ""
            if acc["fact_total"] > 0:
                fe = acc["fact_exact"] * 100 // acc["fact_total"]
                ff = acc["fact_fuzzy"] * 100 // acc["fact_total"]
                fact_str = f"fact={fe}%/{ff}% "

            print(f"  {emoji} {w['case']:15s}:T{w['turn_num']:02d} "
                  f"{ent_count}E {rel_count}R facts={parsed_f}/{raw_f} "
                  f"{ent_str}{fact_str}{status}")
            print(f"      {user_msg[:70]}")

        except Exception as e:
            errors += 1
            print(f"  ! {w['case']:15s}:T{w['turn_num']:02d} ERROR: {e}")

        print()

    # Summary
    def pct(n, t): return f"{n*100//t}%" if t > 0 else "—"

    print(f"{'='*70}")
    print(f"Model: {model}")
    print(f"Turns tested: {len(worst)}")
    print(f"Results: {better} BETTER, {same} SAME, {worse} WORSE, {errors} ERROR")
    print(f"Facts: {total_parsed}/{total_raw} parsed "
          f"(drop={total_raw - total_parsed})")
    print(f"")
    print(f"{'':20s} {'exact':>8s} {'fuzzy':>8s} {'total':>8s}")
    print(f"{'Entity accuracy':20s} {pct(agg['ent_exact'], agg['ent_total']):>8s} "
          f"{pct(agg['ent_fuzzy'], agg['ent_total']):>8s} {agg['ent_total']:>8d}")
    print(f"{'Relation accuracy':20s} {pct(agg['rel_exact'], agg['rel_total']):>8s} "
          f"{pct(agg['rel_fuzzy'], agg['rel_total']):>8s} {agg['rel_total']:>8d}")
    print(f"{'Fact accuracy':20s} {pct(agg['fact_exact'], agg['fact_total']):>8s} "
          f"{pct(agg['fact_fuzzy'], agg['fact_total']):>8s} {agg['fact_total']:>8d}")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--model", default=os.environ.get("ACERVO_LIGHT_MODEL", "qwen2.5:7b"))
    args = parser.parse_args()
    asyncio.run(run_quick_test(args.count, args.model))
