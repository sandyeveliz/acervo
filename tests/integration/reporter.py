"""Multi-format report generator for E2E scenario results.

Produces console, markdown, and JSON reports from ScenarioResult data.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tests.integration.metrics import ScenarioResult
    from tests.integration.narrative import ScenarioNarrative


# ── Console Report ──


def console_report(results: list["ScenarioResult"]) -> str:
    """Compact console report for CI output."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("  ACERVO INTEGRATION BENCHMARK REPORT")
    lines.append("=" * 72)
    lines.append("")
    lines.append("  COMPARATIVE RESULTS (with vs without Acervo)")
    lines.append("-" * 72)
    lines.append(
        f"  {'Scenario':<28} {'Category':<16} {'Savings':>8} "
        f"{'Recall':>8} {'Checks':>8}"
    )
    lines.append("-" * 72)

    for r in results:
        checks = f"{r.checkpoint_pass_count}/{r.checkpoint_total}" if r.checkpoint_total else "N/A"
        lines.append(
            f"  {r.name:<28} {r.category:<16} {r.avg_savings_pct:>7.1f}% "
            f"{r.entity_recall:>7.0%} {checks:>8}"
        )

    lines.append("-" * 72)
    if results:
        avg_savings = sum(r.avg_savings_pct for r in results) / len(results)
        avg_recall = sum(r.entity_recall for r in results) / len(results)
        total_cp = sum(r.checkpoint_pass_count for r in results)
        total_ct = sum(r.checkpoint_total for r in results)
        checks_str = f"{total_cp}/{total_ct}" if total_ct else "N/A"
        lines.append(
            f"  {'AVERAGE':<28} {'':<16} {avg_savings:>7.1f}% "
            f"{avg_recall:>7.0%} {checks_str:>8}"
        )
    lines.append("")
    return "\n".join(lines)


# ── Markdown Report ──


def markdown_report(result: "ScenarioResult") -> str:
    """Full markdown report for a single scenario."""
    lines: list[str] = []
    lines.append(f"# Acervo E2E Report: {result.name.replace('_', ' ').title()}")
    lines.append(
        f"> {result.total_turns} turns | {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
        f"Category: {result.category}"
    )
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total turns | {result.total_turns} |")
    lines.append(f"| Graph nodes (entity) | {result.final_node_count} |")
    lines.append(f"| Graph edges | {result.final_edge_count} |")
    lines.append(f"| Personal / Universal | {result.personal_nodes} / {result.universal_nodes} |")
    lines.append(f"| Avg tokens/turn (Acervo) | {result.avg_acervo_tokens:.0f} |")
    lines.append(f"| Avg tokens/turn (baseline) | {result.avg_baseline_tokens:.0f} |")
    lines.append(f"| **Avg token savings** | **{result.avg_savings_pct:.1f}%** |")
    lines.append(f"| Context hit rate | {result.context_hit_rate:.0%} |")
    lines.append(f"| Entity recall | {result.entity_recall:.0%} |")
    lines.append(f"| Avg prepare time | {result.avg_prepare_ms:.0f}ms |")
    lines.append(f"| Avg process time | {result.avg_process_ms:.0f}ms |")
    lines.append(f"| Phantom entities | {len(result.phantom_entities)} |")
    lines.append(f"| Soft pass rate | {result.soft_pass_rate:.0%} ({result.soft_total - len(result.soft_failures)}/{result.soft_total}) |")
    lines.append("")

    # Token comparison by phase
    phase_data = _phase_stats(result)
    if phase_data:
        lines.append("## Token Comparison by Phase")
        lines.append("")
        lines.append("| Phase | Turns | Acervo Avg | Baseline Avg | Savings | Hits |")
        lines.append("|-------|-------|-----------|-------------|---------|------|")
        for phase, stats in phase_data.items():
            lines.append(
                f"| {phase} | {stats['range']} | "
                f"{stats['avg_acervo']:.0f} | {stats['avg_baseline']:.0f} | "
                f"{stats['savings']:.0f}% | {stats['hits']}/{stats['count']} |"
            )
        lines.append("")

    # Token trend
    if result.total_turns >= 6:
        third = result.total_turns // 3
        first_avg = sum(t.acervo_tokens for t in result.turns[:third]) / third
        last_avg = sum(t.acervo_tokens for t in result.turns[-third:]) / third
        growth = ((last_avg / first_avg) - 1) * 100 if first_avg > 0 else 0
        stability = "STABLE" if abs(growth) < 100 else "GROWING"
        lines.append(
            f"**Token trend:** first third avg={first_avg:.0f}, "
            f"last third avg={last_avg:.0f} ({growth:+.0f}%) — **{stability}**"
        )
        lines.append("")

    # Graph nodes
    if result.graph_nodes:
        lines.append("## Graph Nodes")
        lines.append("")
        lines.append("| Node | Type | Layer | Facts | Status |")
        lines.append("|------|------|-------|-------|--------|")
        for n in result.graph_nodes:
            lines.append(
                f"| {n['label']} | {n['type']} | {n['layer']} | "
                f"{n['facts']} | {n['status']} |"
            )
        lines.append("")

    # Context restoration
    return_turns = [t for t in result.turns if t.phase == "return"]
    if return_turns:
        lines.append("## Context Restoration")
        lines.append("")
        for t in return_turns[:5]:
            status = "HIT" if t.context_hit else "MISS"
            lines.append(
                f"- Turn {t.turn_number} ({t.description}): "
                f"context={status}, warm={t.warm_tokens}tk"
            )
        lines.append("")

    # Timing
    if result.turns:
        prep = [t.prepare_ms for t in result.turns]
        proc = [t.process_ms for t in result.turns]
        lines.append("## Timing")
        lines.append("")
        lines.append("| Stage | Avg | Min | Max | P95 |")
        lines.append("|-------|-----|-----|-----|-----|")
        lines.append(
            f"| prepare | {_avg(prep):.0f}ms | {min(prep)}ms | "
            f"{max(prep)}ms | {_p95(prep):.0f}ms |"
        )
        lines.append(
            f"| process | {_avg(proc):.0f}ms | {min(proc)}ms | "
            f"{max(proc)}ms | {_p95(proc):.0f}ms |"
        )
        lines.append("")

    # Soft assertion failures
    if result.soft_failures:
        lines.append(f"## Soft Assertion Failures ({len(result.soft_failures)}/{result.soft_total})")
        lines.append("")
        for f in result.soft_failures:
            lines.append(f"- {f}")
        lines.append("")
    elif result.soft_total:
        lines.append(f"## Soft Assertions: All {result.soft_total} passed")
        lines.append("")

    # Trace info
    if result.trace_path:
        lines.append("## Trace")
        lines.append("")
        lines.append(f"- Path: `{result.trace_path}`")
        lines.append(f"- Lines: {result.trace_lines}")
        lines.append("")

    return "\n".join(lines)


# ── JSON Report ──


def json_report(result: "ScenarioResult") -> dict:
    """Machine-readable JSON report."""
    return {
        "name": result.name,
        "category": result.category,
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_turns": result.total_turns,
            "final_nodes": result.final_node_count,
            "final_edges": result.final_edge_count,
            "personal_nodes": result.personal_nodes,
            "universal_nodes": result.universal_nodes,
            "avg_acervo_tokens": round(result.avg_acervo_tokens, 1),
            "avg_baseline_tokens": round(result.avg_baseline_tokens, 1),
            "avg_savings_pct": round(result.avg_savings_pct, 1),
            "context_hit_rate": round(result.context_hit_rate, 3),
            "entity_recall": round(result.entity_recall, 3),
            "avg_prepare_ms": round(result.avg_prepare_ms, 1),
            "avg_process_ms": round(result.avg_process_ms, 1),
            "phantom_entities": result.phantom_entities,
            "soft_pass_rate": round(result.soft_pass_rate, 3),
        },
        "turns": [
            {
                "turn": t.turn_number,
                "phase": t.phase,
                "description": t.description,
                "acervo_tokens": t.acervo_tokens,
                "baseline_tokens": t.baseline_tokens,
                "savings_pct": round(t.savings_pct, 1),
                "warm_tokens": t.warm_tokens,
                "hot_tokens": t.hot_tokens,
                "system_tokens": t.system_tokens,
                "user_tokens": t.user_tokens,
                "overhead_tokens": t.overhead_tokens,
                "context_hit": t.context_hit,
                "node_count": t.node_count,
                "edge_count": t.edge_count,
                "entities_expected": t.entities_expected,
                "entities_found": t.entities_found,
                "entities_missing": t.entities_missing,
                "prepare_ms": t.prepare_ms,
                "process_ms": t.process_ms,
                "user_msg": t.user_msg,
                "assistant_msg": t.assistant_msg,
                "warm_context": t.warm_context,
                "entities_extracted": t.entities_extracted,
                "topic": t.topic,
            }
            for t in result.turns
        ],
        "graph_nodes": result.graph_nodes,
        "soft_failures": result.soft_failures,
    }


# ── Helpers ──


def _phase_stats(result: "ScenarioResult") -> dict[str, dict]:
    """Group turns by phase and compute stats."""
    phases: dict[str, list] = defaultdict(list)
    for t in result.turns:
        phases[t.phase].append(t)

    out = {}
    for phase, turns in phases.items():
        if not turns:
            continue
        first = turns[0].turn_number
        last = turns[-1].turn_number
        avg_a = sum(t.acervo_tokens for t in turns) / len(turns)
        avg_b = sum(t.baseline_tokens for t in turns) / len(turns)
        savings = (1 - avg_a / avg_b) * 100 if avg_b > 0 else 0
        hits = sum(1 for t in turns if t.context_hit)
        out[phase] = {
            "range": f"{first}-{last}",
            "count": len(turns),
            "avg_acervo": avg_a,
            "avg_baseline": avg_b,
            "savings": savings,
            "hits": hits,
        }
    return out


def _avg(values: list[int | float]) -> float:
    return sum(values) / len(values) if values else 0


def _p95(values: list[int | float]) -> float:
    if not values:
        return 0
    s = sorted(values)
    return s[min(int(len(s) * 0.95), len(s) - 1)]


# ── HTML Report ──


def html_report(results: list["ScenarioResult"], prompt_data: dict | None = None) -> str:
    """Self-contained HTML report with embedded Chart.js charts.

    Designed for blog posts and sharing — single file, no external dependencies.
    Uses Chart.js via CDN for interactive charts.
    Includes: summary, token charts, cost estimation, conversation evidence,
    prompt comparison, and image export buttons.
    """
    from datetime import datetime

    scenarios_json = json.dumps([
        {
            "name": r.name.replace("_", " ").title(),
            "category": r.category,
            "turns": r.total_turns,
            "nodes": r.final_node_count,
            "edges": r.final_edge_count,
            "savings": round(r.avg_savings_pct, 1),
            "recall": round(r.entity_recall * 100, 1),
            "context_hit_rate": round(r.context_hit_rate * 100, 1),
            "phantoms": len(r.phantom_entities),
            "soft_pass_rate": round(r.soft_pass_rate * 100, 1),
            "avg_prepare_ms": round(r.avg_prepare_ms),
            "avg_process_ms": round(r.avg_process_ms),
            "personal": r.personal_nodes,
            "universal": r.universal_nodes,
        }
        for r in results
    ])

    # Per-turn token data for the line chart (all scenarios)
    turn_data = {}
    for r in results:
        turn_data[r.name] = {
            "acervo": [t.acervo_tokens for t in r.turns],
            "baseline": [t.baseline_tokens for t in r.turns],
            "system": [t.system_tokens for t in r.turns],
            "warm": [t.warm_tokens for t in r.turns],
            "hot": [t.hot_tokens for t in r.turns],
            "user": [t.user_tokens for t in r.turns],
            "overhead": [t.overhead_tokens for t in r.turns],
        }
    turns_json = json.dumps(turn_data)

    # Detailed turn data (messages, context, entities) for evidence section
    detail_data = {}
    for r in results:
        detail_data[r.name] = [
            {
                "turn": t.turn_number,
                "phase": t.phase,
                "desc": t.description,
                "user": t.user_msg,
                "assistant": t.assistant_msg,
                "warm": t.warm_context,
                "entities": t.entities_extracted,
                "topic": t.topic,
                "acervo_tk": t.acervo_tokens,
                "baseline_tk": t.baseline_tokens,
                "system_tk": t.system_tokens,
                "warm_tk": t.warm_tokens,
                "hot_tk": t.hot_tokens,
                "user_tk": t.user_tokens,
                "overhead_tk": t.overhead_tokens,
                "hit": t.context_hit,
                "nodes": t.node_count,
            }
            for t in r.turns
            if t.user_msg  # only include turns with captured messages
        ]
    detail_json = json.dumps(detail_data, ensure_ascii=False)

    # Cost estimation (based on real API pricing per 1M tokens)
    # GPT-4o: $2.50/1M input, Claude Sonnet: $3/1M input, GPT-4o-mini: $0.15/1M input
    cost_models = {
        "GPT-4o": {"input": 2.50, "output": 10.00},
        "Claude Sonnet": {"input": 3.00, "output": 15.00},
        "GPT-4o-mini": {"input": 0.15, "output": 0.60},
    }
    cost_data = {}
    for model_name, prices in cost_models.items():
        for r in results:
            total_acervo = sum(t.acervo_tokens for t in r.turns)
            total_baseline = sum(t.baseline_tokens for t in r.turns)
            # Assume output tokens ≈ 30% of input for cost estimation
            acervo_cost = (total_acervo * prices["input"] + total_acervo * 0.3 * prices["output"]) / 1_000_000
            baseline_cost = (total_baseline * prices["input"] + total_baseline * 0.3 * prices["output"]) / 1_000_000
            key = r.name.replace("_", " ").title()
            if key not in cost_data:
                cost_data[key] = {}
            cost_data[key][model_name] = {
                "acervo": round(acervo_cost, 4),
                "baseline": round(baseline_cost, 4),
                "saved": round(baseline_cost - acervo_cost, 4),
            }
    cost_json = json.dumps(cost_data)

    prompt_json = json.dumps(prompt_data) if prompt_data else "null"
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Acervo Integration Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0d1117; color: #c9d1d9; line-height: 1.6; padding: 2rem; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{ color: #58a6ff; font-size: 2rem; margin-bottom: 0.5rem; }}
  h2 {{ color: #58a6ff; font-size: 1.4rem; margin: 2rem 0 1rem; border-bottom: 1px solid #30363d; padding-bottom: 0.5rem; }}
  h3 {{ color: #8b949e; font-size: 1.1rem; margin: 1.5rem 0 0.5rem; }}
  .subtitle {{ color: #8b949e; margin-bottom: 2rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin: 1rem 0; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1.5rem; }}
  .card-value {{ font-size: 2rem; font-weight: bold; color: #58a6ff; }}
  .card-label {{ color: #8b949e; font-size: 0.9rem; }}
  .card.green .card-value {{ color: #3fb950; }}
  .card.yellow .card-value {{ color: #d29922; }}
  .card.red .card-value {{ color: #f85149; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
  th, td {{ padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid #30363d; }}
  th {{ background: #161b22; color: #8b949e; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; }}
  td {{ color: #c9d1d9; }}
  tr:hover td {{ background: #161b22; }}
  .chart-container {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1.5rem; margin: 1rem 0; position: relative; }}
  .badge {{ display: inline-block; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }}
  .badge-green {{ background: #238636; color: #fff; }}
  .badge-yellow {{ background: #9e6a03; color: #fff; }}
  .badge-blue {{ background: #1f6feb; color: #fff; }}
  .footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #30363d; color: #484f58; font-size: 0.85rem; }}
  .entity-grid {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.5rem 0; }}
  .entity-tag {{ background: #1f6feb22; border: 1px solid #1f6feb55; color: #58a6ff;
                 padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.8rem; }}
  .btn {{ background: #21262d; border: 1px solid #30363d; color: #8b949e; padding: 0.4rem 0.8rem;
          border-radius: 6px; cursor: pointer; font-size: 0.8rem; }}
  .btn:hover {{ background: #30363d; color: #c9d1d9; }}
  .btn-export {{ position: absolute; top: 0.75rem; right: 0.75rem; }}
  .turn-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; }}
  .turn-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }}
  .turn-phase {{ font-size: 0.75rem; text-transform: uppercase; }}
  .msg {{ padding: 0.5rem 0.75rem; border-radius: 6px; margin: 0.25rem 0; font-size: 0.9rem; }}
  .msg-user {{ background: #1f6feb22; border-left: 3px solid #1f6feb; }}
  .msg-assistant {{ background: #23863622; border-left: 3px solid #238636; }}
  .msg-context {{ background: #d2992222; border-left: 3px solid #d29922; font-size: 0.8rem; }}
  .msg-label {{ font-size: 0.7rem; color: #8b949e; text-transform: uppercase; margin-bottom: 0.2rem; }}
  .cost-saved {{ color: #3fb950; font-weight: bold; }}
  .scenario-select {{ background: #21262d; border: 1px solid #30363d; color: #c9d1d9; padding: 0.4rem;
                      border-radius: 6px; font-size: 0.9rem; }}
  .turn-meta {{ display: flex; gap: 1rem; font-size: 0.8rem; color: #8b949e; flex-wrap: wrap; }}
  .turn-meta span {{ background: #21262d; padding: 0.1rem 0.5rem; border-radius: 4px; }}
  .collapse-btn {{ background: none; border: none; color: #58a6ff; cursor: pointer; font-size: 0.85rem; padding: 0; }}
</style>
</head>
<body>
<div class="container">
  <h1>Acervo Integration Report</h1>
  <p class="subtitle">Generated {generated} &mdash; {sum(r.total_turns for r in results)} total turns across {len(results)} scenarios</p>

  <div class="grid" id="summary-cards"></div>

  <h2>Token Savings: Acervo vs Full History <button class="btn btn-export" onclick="exportChart('savingsChart')">Export PNG</button></h2>
  <div class="chart-container"><canvas id="savingsChart" height="80"></canvas></div>

  <h2>Token Usage per Turn</h2>
  <div id="turn-charts"></div>

  <h2>Token Anatomy (Component Breakdown)</h2>
  <p style="color:#8b949e;margin-bottom:1rem;">Shows where Acervo's tokens go: system prompt (constant), warm context (graph knowledge), hot layer (recent history), user message, and framing overhead.</p>
  <div id="anatomy-charts"></div>

  <h2>Estimated API Cost Impact</h2>
  <p style="color:#8b949e;margin-bottom:1rem;">Projected costs based on real API pricing. Shows what each scenario would cost with and without Acervo compression.</p>
  <div id="cost-section"></div>

  <h2>Scenario Breakdown</h2>
  <table id="scenario-table">
    <thead>
      <tr><th>Scenario</th><th>Category</th><th>Turns</th><th>Nodes</th><th>Edges</th>
          <th>Savings</th><th>Recall</th><th>Context Hits</th><th>Phantoms</th></tr>
    </thead>
    <tbody></tbody>
  </table>

  <h2>Conversation Evidence</h2>
  <p style="color:#8b949e;margin-bottom:1rem;">Actual messages, extracted entities, and injected context for each turn. This is what was really tested.</p>
  <div style="margin-bottom:1rem;">
    <label style="color:#8b949e;">Scenario: </label>
    <select id="evidence-select" class="scenario-select" onchange="renderEvidence()"></select>
    <button class="btn" onclick="toggleAllTurns()" style="margin-left:0.5rem;">Expand/Collapse All</button>
  </div>
  <div id="evidence-section"></div>

  <div id="prompt-section"></div>

  <div class="footer">
    <p>Acervo &mdash; Context proxy for AI agents. Graph-based memory compression.</p>
    <p>Hard assertions verify infrastructure (token stability, graph integrity, trace persistence).
       Soft assertions measure model+prompt extraction quality (model-dependent).</p>
  </div>
</div>

<script>
const scenarios = {scenarios_json};
const turnData = {turns_json};
const detailData = {detail_json};
const costData = {cost_json};
const promptData = {prompt_json};
const chartInstances = {{}};

// ── Export chart as PNG ──
function exportChart(canvasId) {{
  const canvas = document.getElementById(canvasId);
  const link = document.createElement('a');
  link.download = canvasId + '.png';
  link.href = canvas.toDataURL('image/png', 1.0);
  link.click();
}}

// ── Summary cards ──
const totalTurns = scenarios.reduce((s, r) => s + r.turns, 0);
const avgSavings = scenarios.reduce((s, r) => s + r.savings, 0) / scenarios.length;
const totalNodes = scenarios.reduce((s, r) => s + r.nodes, 0);
const avgHits = scenarios.reduce((s, r) => s + r.context_hit_rate, 0) / scenarios.length;
const totalPhantoms = scenarios.reduce((s, r) => s + r.phantoms, 0);

const cards = [
  {{ value: avgSavings.toFixed(1) + '%', label: 'Avg Token Savings', cls: 'green' }},
  {{ value: totalTurns, label: 'Total Turns Tested', cls: '' }},
  {{ value: totalNodes, label: 'Graph Nodes Extracted', cls: '' }},
  {{ value: avgHits.toFixed(0) + '%', label: 'Context Hit Rate', cls: 'green' }},
  {{ value: totalPhantoms, label: 'Phantom Entities', cls: totalPhantoms === 0 ? 'green' : 'red' }},
  {{ value: scenarios.length, label: 'Scenarios', cls: '' }},
];
document.getElementById('summary-cards').innerHTML = cards.map(c =>
  `<div class="card ${{c.cls}}"><div class="card-value">${{c.value}}</div><div class="card-label">${{c.label}}</div></div>`
).join('');

// ── Savings bar chart ──
new Chart(document.getElementById('savingsChart'), {{
  type: 'bar',
  data: {{
    labels: scenarios.map(s => s.name),
    datasets: [
      {{ label: 'Acervo (avg tk/turn)', data: scenarios.map(s => {{
           const td = turnData[Object.keys(turnData)[scenarios.indexOf(s)]];
           return td ? Math.round(td.acervo.reduce((a,b)=>a+b,0)/td.acervo.length) : 0;
         }}), backgroundColor: '#238636' }},
      {{ label: 'Baseline (avg tk/turn)', data: scenarios.map(s => {{
           const td = turnData[Object.keys(turnData)[scenarios.indexOf(s)]];
           return td ? Math.round(td.baseline.reduce((a,b)=>a+b,0)/td.baseline.length) : 0;
         }}), backgroundColor: '#f8514966' }},
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#656d76' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#656d76' }}, grid: {{ color: '#30363d' }} }},
      y: {{ ticks: {{ color: '#656d76' }}, grid: {{ color: '#30363d' }}, title: {{ display: true, text: 'Tokens', color: '#656d76' }} }},
    }}
  }}
}});

// ── Per-scenario turn charts ──
const turnChartsEl = document.getElementById('turn-charts');
Object.entries(turnData).forEach(([name, data]) => {{
  const safeId = name.replace(/[^a-zA-Z0-9]/g, '_');
  const div = document.createElement('div');
  div.className = 'chart-container';
  div.innerHTML = `<button class="btn btn-export" onclick="exportChart('turn_${{safeId}}')">Export PNG</button><canvas id="turn_${{safeId}}" height="60"></canvas>`;
  turnChartsEl.appendChild(div);
  const labels = data.acervo.map((_, i) => i + 1);
  chartInstances[name] = new Chart(document.getElementById(`turn_${{safeId}}`), {{
    type: 'line',
    data: {{
      labels,
      datasets: [
        {{ label: `${{name}} — Acervo`, data: data.acervo, borderColor: '#238636', backgroundColor: '#23863622', fill: true, tension: 0.3, pointRadius: 1 }},
        {{ label: `${{name}} — Baseline`, data: data.baseline, borderColor: '#f85149', backgroundColor: '#f8514922', fill: true, tension: 0.3, pointRadius: 1 }},
      ]
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ labels: {{ color: '#656d76' }} }} }},
      scales: {{
        x: {{ ticks: {{ color: '#656d76' }}, grid: {{ color: '#30363d' }}, title: {{ display: true, text: 'Turn', color: '#656d76' }} }},
        y: {{ ticks: {{ color: '#656d76' }}, grid: {{ color: '#30363d' }}, title: {{ display: true, text: 'Tokens', color: '#656d76' }} }},
      }}
    }}
  }});
}});

// ── Token Anatomy (stacked area) ──
const anatomyEl = document.getElementById('anatomy-charts');
Object.entries(turnData).forEach(([name, data]) => {{
  if (!data.system) return;
  const safeId = name.replace(/[^a-zA-Z0-9]/g, '_');
  const div = document.createElement('div');
  div.className = 'chart-container';
  div.innerHTML = `<button class="btn btn-export" onclick="exportChart('anatomy_${{safeId}}')">Export PNG</button><canvas id="anatomy_${{safeId}}" height="60"></canvas>`;
  anatomyEl.appendChild(div);
  const labels = data.acervo.map((_, i) => i + 1);
  new Chart(document.getElementById(`anatomy_${{safeId}}`), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [
        {{ label: 'System prompt', data: data.system, backgroundColor: '#1f6feb88', stack: 'stack' }},
        {{ label: 'Warm (graph)', data: data.warm, backgroundColor: '#3fb95088', stack: 'stack' }},
        {{ label: 'Hot (history)', data: data.hot, backgroundColor: '#d2992288', stack: 'stack' }},
        {{ label: 'User message', data: data.user, backgroundColor: '#58a6ff88', stack: 'stack' }},
        {{ label: 'Overhead', data: data.overhead, backgroundColor: '#8b949e88', stack: 'stack' }},
        {{ label: 'Baseline', data: data.baseline, type: 'line', borderColor: '#f85149', backgroundColor: 'transparent', borderWidth: 2, pointRadius: 0, tension: 0.3, order: -1 }},
      ]
    }},
    options: {{
      responsive: true,
      plugins: {{
        legend: {{ labels: {{ color: '#656d76' }} }},
        title: {{ display: true, text: name.replace(/_/g, ' '), color: '#656d76' }},
      }},
      scales: {{
        x: {{ stacked: true, ticks: {{ color: '#656d76' }}, grid: {{ color: '#30363d' }}, title: {{ display: true, text: 'Turn', color: '#656d76' }} }},
        y: {{ stacked: true, ticks: {{ color: '#656d76' }}, grid: {{ color: '#30363d' }}, title: {{ display: true, text: 'Tokens', color: '#656d76' }} }},
      }}
    }}
  }});
}});

// Component averages summary
const avgSummaryEl = document.createElement('div');
avgSummaryEl.className = 'card';
avgSummaryEl.style.marginTop = '1rem';
let avgHtml = '<h3 style="margin-bottom:0.5rem;">Average Token Composition</h3><table><thead><tr><th>Scenario</th><th>System</th><th>Warm</th><th>Hot</th><th>User</th><th>Overhead</th><th>Total</th></tr></thead><tbody>';
Object.entries(turnData).forEach(([name, data]) => {{
  if (!data.system) return;
  const n = data.acervo.length || 1;
  const avgS = Math.round(data.system.reduce((a,b)=>a+b,0)/n);
  const avgW = Math.round(data.warm.reduce((a,b)=>a+b,0)/n);
  const avgH = Math.round(data.hot.reduce((a,b)=>a+b,0)/n);
  const avgU = Math.round(data.user.reduce((a,b)=>a+b,0)/n);
  const avgO = Math.round(data.overhead.reduce((a,b)=>a+b,0)/n);
  const avgT = Math.round(data.acervo.reduce((a,b)=>a+b,0)/n);
  avgHtml += `<tr><td>${{name.replace(/_/g,' ')}}</td><td>${{avgS}} (${{avgT>0?((avgS/avgT)*100).toFixed(0):0}}%)</td><td>${{avgW}} (${{avgT>0?((avgW/avgT)*100).toFixed(0):0}}%)</td><td>${{avgH}} (${{avgT>0?((avgH/avgT)*100).toFixed(0):0}}%)</td><td>${{avgU}} (${{avgT>0?((avgU/avgT)*100).toFixed(0):0}}%)</td><td>${{avgO}} (${{avgT>0?((avgO/avgT)*100).toFixed(0):0}}%)</td><td>${{avgT}}</td></tr>`;
}});
avgHtml += '</tbody></table>';
avgSummaryEl.innerHTML = avgHtml;
anatomyEl.appendChild(avgSummaryEl);

// ── Cost estimation ──
if (costData && Object.keys(costData).length > 0) {{
  const costEl = document.getElementById('cost-section');
  const models = Object.keys(Object.values(costData)[0]);
  let html = '<table><thead><tr><th>Scenario</th>';
  models.forEach(m => {{ html += `<th colspan="3">${{m}}</th>`; }});
  html += '</tr><tr><th></th>';
  models.forEach(() => {{ html += '<th>Baseline</th><th>Acervo</th><th>Saved</th>'; }});
  html += '</tr></thead><tbody>';
  Object.entries(costData).forEach(([scenario, mdata]) => {{
    html += `<tr><td>${{scenario}}</td>`;
    models.forEach(m => {{
      const d = mdata[m];
      html += `<td>$${{d.baseline.toFixed(4)}}</td><td>$${{d.acervo.toFixed(4)}}</td><td class="cost-saved">$${{d.saved.toFixed(4)}}</td>`;
    }});
    html += '</tr>';
  }});
  // Totals row
  html += '<tr style="border-top:2px solid #30363d;font-weight:bold;"><td>TOTAL (all scenarios)</td>';
  models.forEach(m => {{
    let tb = 0, ta = 0;
    Object.values(costData).forEach(mdata => {{ tb += mdata[m].baseline; ta += mdata[m].acervo; }});
    html += `<td>$${{tb.toFixed(4)}}</td><td>$${{ta.toFixed(4)}}</td><td class="cost-saved">$${{(tb-ta).toFixed(4)}}</td>`;
  }});
  html += '</tr>';
  // Extrapolated row (1000 conversations)
  html += '<tr style="color:#d29922;"><td>Projected: 1,000 conversations</td>';
  models.forEach(m => {{
    let tb = 0, ta = 0;
    Object.values(costData).forEach(mdata => {{ tb += mdata[m].baseline; ta += mdata[m].acervo; }});
    const scale = 1000 / Object.keys(costData).length;
    html += `<td>$${{(tb*scale).toFixed(2)}}</td><td>$${{(ta*scale).toFixed(2)}}</td><td class="cost-saved">$${{((tb-ta)*scale).toFixed(2)}}</td>`;
  }});
  html += '</tr>';
  html += '</tbody></table>';
  html += '<p style="color:#484f58;font-size:0.8rem;margin-top:0.5rem;">Costs based on published API pricing (per 1M tokens). Output estimated at 30% of input tokens. Actual costs depend on response length and model.</p>';
  costEl.innerHTML = html;
}}

// ── Scenario table ──
const tbody = document.querySelector('#scenario-table tbody');
scenarios.forEach(s => {{
  tbody.innerHTML += `<tr>
    <td>${{s.name}}</td><td><span class="badge badge-blue">${{s.category}}</span></td>
    <td>${{s.turns}}</td><td>${{s.nodes}}</td><td>${{s.edges}}</td>
    <td><span class="badge badge-green">${{s.savings}}%</span></td>
    <td>${{s.recall}}%</td><td>${{s.context_hit_rate}}%</td>
    <td>${{s.phantoms === 0 ? '<span class="badge badge-green">0</span>' : '<span class="badge badge-red">' + s.phantoms + '</span>'}}</td>
  </tr>`;
}});

// ── Conversation evidence ──
const evidenceSelect = document.getElementById('evidence-select');
const evidenceNames = Object.keys(detailData).filter(k => detailData[k].length > 0);
evidenceNames.forEach(name => {{
  const opt = document.createElement('option');
  opt.value = name;
  opt.textContent = name.replace(/_/g, ' ');
  evidenceSelect.appendChild(opt);
}});

let allExpanded = false;
function toggleAllTurns() {{
  allExpanded = !allExpanded;
  document.querySelectorAll('.turn-detail').forEach(el => {{
    el.style.display = allExpanded ? 'block' : 'none';
  }});
}}

function renderEvidence() {{
  const name = evidenceSelect.value;
  const turns = detailData[name] || [];
  const el = document.getElementById('evidence-section');
  if (!turns.length) {{ el.innerHTML = '<p style="color:#8b949e;">No detailed data captured for this scenario. Re-run to capture messages.</p>'; return; }}

  el.innerHTML = turns.map(t => `
    <div class="turn-card">
      <div class="turn-header">
        <span><strong>Turn ${{t.turn}}</strong> &mdash; ${{t.desc || 'No description'}}
          <button class="collapse-btn" onclick="this.closest('.turn-card').querySelector('.turn-detail').style.display=this.closest('.turn-card').querySelector('.turn-detail').style.display==='none'?'block':'none'">
            [toggle]
          </button>
        </span>
        <span class="turn-phase"><span class="badge badge-blue">${{t.phase}}</span></span>
      </div>
      <div class="turn-meta">
        <span>Acervo: ${{t.acervo_tk}} tk (sys:${{t.system_tk||0}} + warm:${{t.warm_tk||0}} + hot:${{t.hot_tk||0}} + user:${{t.user_tk||0}} + over:${{t.overhead_tk||0}})</span>
        <span>Baseline: ${{t.baseline_tk}} tk</span>
        <span style="color:${{t.acervo_tk < t.baseline_tk ? '#3fb950' : '#f85149'}}">
          ${{t.baseline_tk > 0 ? ((1 - t.acervo_tk/t.baseline_tk)*100).toFixed(0) : 0}}% saved
        </span>
        <span>Context: ${{t.hit ? 'HIT' : 'MISS'}}</span>
        <span>Nodes: ${{t.nodes}}</span>
        ${{t.topic ? `<span>Topic: ${{t.topic}}</span>` : ''}}
      </div>
      <div class="turn-detail" style="display:none;margin-top:0.5rem;">
        <div class="msg msg-user"><div class="msg-label">User</div>${{t.user}}</div>
        ${{t.warm ? `<div class="msg msg-context"><div class="msg-label">Acervo injected context (warm layer)</div>${{t.warm}}</div>` : ''}}
        ${{t.assistant ? `<div class="msg msg-assistant"><div class="msg-label">Assistant</div>${{t.assistant}}</div>` : ''}}
        ${{t.entities.length ? `<div style="margin-top:0.5rem;"><span class="msg-label">Entities extracted: </span><span class="entity-grid">${{t.entities.map(e => `<span class="entity-tag">${{e}}</span>`).join('')}}</span></div>` : ''}}
      </div>
    </div>
  `).join('');
}}
if (evidenceNames.length) renderEvidence();

// ── Prompt comparison ──
if (promptData) {{
  const section = document.getElementById('prompt-section');
  section.innerHTML = '<h2>Prompt Variant Comparison</h2><p style="color:#8b949e;margin-bottom:1rem;">Same scenario run with different extraction prompts. Shows how prompt engineering affects entity recall without breaking infrastructure.</p>';

  const variants = Object.keys(promptData);
  let html = '<table><thead><tr><th>Metric</th>' + variants.map(v => `<th>${{v}}</th>`).join('') + '</tr></thead><tbody>';
  const metrics = [
    ['Nodes', 'node_count'], ['Edges', 'edge_count'], ['Entity Recall', 'entity_recall'],
    ['Phantoms', 'phantom_count'], ['Token Savings', 'avg_savings_pct'],
    ['Context Hits', 'context_hit_rate'], ['Soft Pass Rate', 'soft_pass_rate']
  ];
  metrics.forEach(([label, key]) => {{
    html += `<tr><td>${{label}}</td>`;
    const values = variants.map(v => promptData[v][key]);
    const best = key === 'phantom_count' ? Math.min(...values) : Math.max(...values);
    variants.forEach(v => {{
      let val = promptData[v][key];
      const isBest = val === best;
      if (['entity_recall','context_hit_rate','soft_pass_rate'].includes(key))
        val = (val * 100).toFixed(0) + '%';
      else if (key === 'avg_savings_pct')
        val = val.toFixed(1) + '%';
      html += `<td style="${{isBest ? 'color:#3fb950;font-weight:bold' : ''}}">${{val}}</td>`;
    }});
    html += '</tr>';
  }});
  html += '</tbody></table>';

  html += '<h3>Entity Coverage per Variant</h3><div class="grid">';
  variants.forEach(v => {{
    html += `<div class="card"><h3>${{v}}</h3><div class="entity-grid">`;
    (promptData[v].entities||[]).forEach(e => {{ html += `<span class="entity-tag">${{e}}</span>`; }});
    html += '</div></div>';
  }});
  html += '</div>';
  section.innerHTML += html;
}}
</script>
</body>
</html>"""


# ── Narrative Report ──


def html_report_narrative(
    results: list["ScenarioResult"],
    narratives: list["ScenarioNarrative"],
    prompt_data: dict | None = None,
    version: str = "",
) -> str:
    """Clean data report with clear titles, consolidated charts, and summary cards.

    Structure:
    1. Summary cards (turns, savings, hits, phantoms, entities)
    2. Scissor chart — longest scenario showing Acervo vs full history over turns
    3. Scorecard — grouped bar comparing savings % and context hit % across all scenarios
    4. Token anatomy — sampled stacked bar (every 10th turn) for the longest scenario
    5. Per-scenario breakdown table
    6. Cost estimation table
    7. Notable moments (story beats) as a compact table
    8. Conversation evidence (collapsible)
    """
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    total_turns = sum(r.total_turns for r in results)
    avg_savings = sum(r.avg_savings_pct for r in results) / max(len(results), 1)
    avg_hits = sum(r.context_hit_rate for r in results) / max(len(results), 1) * 100
    total_nodes = sum(r.final_node_count for r in results)
    total_phantoms = sum(len(r.phantom_entities) for r in results)

    # Find longest scenario for the scissor chart
    longest = max(results, key=lambda r: r.total_turns)
    last_turn = longest.turns[-1] if longest.turns else None
    scissor_savings = last_turn.savings_pct if last_turn else 0
    scissor_acervo = last_turn.acervo_tokens if last_turn else 0
    scissor_baseline = last_turn.baseline_tokens if last_turn else 0

    # Scenario data for scorecard
    scenarios_json = json.dumps([
        {
            "name": _scenario_display_name(r),
            "turns": r.total_turns,
            "savings": round(r.avg_savings_pct, 1),
            "hits": round(r.context_hit_rate * 100, 1),
            "recall": round(r.entity_recall * 100, 1),
            "phantoms": len(r.phantom_entities),
            "nodes": r.final_node_count,
            "edges": r.final_edge_count,
        }
        for r in results
    ])

    # Scissor chart data (longest scenario)
    scissor_json = json.dumps({
        "acervo": [t.acervo_tokens for t in longest.turns],
        "baseline": [t.baseline_tokens for t in longest.turns],
    })

    # Token anatomy — sampled every 10 turns for longest scenario
    sample_indices = list(range(0, len(longest.turns), max(1, len(longest.turns) // 10)))
    if sample_indices[-1] != len(longest.turns) - 1:
        sample_indices.append(len(longest.turns) - 1)
    anatomy_json = json.dumps({
        "labels": [f"Turn {longest.turns[i].turn_number}" for i in sample_indices],
        "baseline": [longest.turns[i].baseline_tokens for i in sample_indices],
        "warm": [longest.turns[i].warm_tokens for i in sample_indices],
        "hot": [longest.turns[i].hot_tokens for i in sample_indices],
        "system_user": [longest.turns[i].system_tokens + longest.turns[i].user_tokens + longest.turns[i].overhead_tokens for i in sample_indices],
    })

    # Per-turn data for all scenarios (for evidence section)
    all_turn_data = {}
    for r in results:
        all_turn_data[r.name] = {
            "acervo": [t.acervo_tokens for t in r.turns],
            "baseline": [t.baseline_tokens for t in r.turns],
        }
    all_turns_json = json.dumps(all_turn_data)

    # Detail data for evidence
    detail_data = {}
    for r in results:
        detail_data[r.name] = [
            {
                "turn": t.turn_number, "phase": t.phase, "desc": t.description,
                "user": t.user_msg, "assistant": t.assistant_msg, "warm": t.warm_context,
                "entities": t.entities_extracted, "topic": t.topic,
                "acervo_tk": t.acervo_tokens, "baseline_tk": t.baseline_tokens,
                "warm_tk": t.warm_tokens, "hot_tk": t.hot_tokens,
                "system_tk": t.system_tokens, "user_tk": t.user_tokens,
                "hit": t.context_hit, "nodes": t.node_count,
            }
            for t in r.turns if t.user_msg
        ]
    detail_json = json.dumps(detail_data, ensure_ascii=False)

    # Cost estimation
    cost_models = {
        "GPT-4o": {"input": 2.50, "output": 10.00},
        "Claude Sonnet": {"input": 3.00, "output": 15.00},
        "GPT-4o-mini": {"input": 0.15, "output": 0.60},
    }
    cost_rows = []
    for r in results:
        row = {"name": _scenario_display_name(r)}
        for model_name, prices in cost_models.items():
            total_a = sum(t.acervo_tokens for t in r.turns)
            total_b = sum(t.baseline_tokens for t in r.turns)
            a_cost = (total_a * prices["input"] + total_a * 0.3 * prices["output"]) / 1_000_000
            b_cost = (total_b * prices["input"] + total_b * 0.3 * prices["output"]) / 1_000_000
            row[model_name] = {"baseline": round(b_cost, 4), "acervo": round(a_cost, 4), "saved": round(b_cost - a_cost, 4)}
        cost_rows.append(row)
    cost_json = json.dumps({"models": list(cost_models.keys()), "rows": cost_rows})

    # Story beats
    beats_json = json.dumps([
        {
            "scenario": n.persona_name or n.category,
            "turn": b.turn_number,
            "type": b.beat_type.replace("_", " ").title(),
            "headline": b.headline,
        }
        for n in narratives for b in n.key_beats
    ], ensure_ascii=False)

    version_str = f" &mdash; {version}" if version else ""
    longest_name = _scenario_display_name(longest)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Acervo Benchmark Report{' — ' + version if version else ''}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #ffffff; color: #24292f; line-height: 1.6; padding: 2rem; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{ color: #0969da; font-size: 2rem; margin-bottom: 0.3rem; }}
  h2 {{ color: #0969da; font-size: 1.3rem; margin: 2.5rem 0 0.3rem; border-bottom: 1px solid #d0d7de; padding-bottom: 0.4rem; }}
  .subtitle {{ color: #656d76; margin-bottom: 2rem; font-size: 0.9rem; }}
  .section-desc {{ color: #656d76; font-size: 0.9rem; margin-bottom: 1rem; }}

  /* Summary cards */
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin: 1.5rem 0; }}
  .card {{ background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 8px; padding: 1.2rem; text-align: center; }}
  .card .val {{ font-size: 2rem; font-weight: 800; }}
  .card .val.green {{ color: #1a7f37; }}
  .card .val.blue {{ color: #0969da; }}
  .card .val.red {{ color: #cf222e; }}
  .card .lbl {{ color: #656d76; font-size: 0.85rem; }}

  /* Charts */
  .chart-box {{ background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 8px; padding: 1.5rem; margin: 1rem 0; }}
  .chart-box h3 {{ color: #24292f; font-size: 1.1rem; margin-bottom: 0.3rem; }}
  .chart-box p {{ color: #656d76; font-size: 0.85rem; margin-bottom: 0.8rem; }}
  .chart-callouts {{ display: flex; justify-content: center; gap: 3rem; margin-top: 1rem; flex-wrap: wrap; }}
  .chart-callout {{ text-align: center; }}
  .chart-callout .val {{ font-size: 1.6rem; font-weight: 800; }}
  .chart-callout .lbl {{ color: #656d76; font-size: 0.8rem; }}

  /* Tables */
  table {{ width: 100%; border-collapse: collapse; margin: 0.5rem 0; }}
  th, td {{ padding: 0.6rem 0.8rem; text-align: left; border-bottom: 1px solid #d0d7de; }}
  th {{ background: #f6f8fa; color: #656d76; font-weight: 600; font-size: 0.8rem; text-transform: uppercase; }}
  td {{ font-size: 0.9rem; }}
  tr:hover td {{ background: #f6f8fa; }}
  .good {{ color: #1a7f37; }}
  .bad {{ color: #cf222e; }}
  .saved {{ color: #1a7f37; font-weight: bold; }}

  /* Evidence */
  .evidence-select {{ background: #f6f8fa; border: 1px solid #d0d7de; color: #24292f; padding: 0.4rem;
                      border-radius: 6px; font-size: 0.9rem; }}
  .turn-card {{ background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 8px; padding: 0.8rem; margin: 0.4rem 0; }}
  .turn-header {{ display: flex; justify-content: space-between; align-items: center; }}
  .turn-meta {{ display: flex; gap: 0.8rem; font-size: 0.8rem; color: #656d76; flex-wrap: wrap; margin-top: 0.3rem; }}
  .turn-meta span {{ background: #eaeef2; padding: 0.1rem 0.5rem; border-radius: 4px; }}
  .msg {{ padding: 0.4rem 0.6rem; border-radius: 5px; margin: 0.2rem 0; font-size: 0.85rem; }}
  .msg-user {{ background: #ddf4ff; border-left: 3px solid #0969da; }}
  .msg-assistant {{ background: #dafbe1; border-left: 3px solid #1a7f37; }}
  .msg-context {{ background: #fff8c5; border-left: 3px solid #bf8700; font-size: 0.8rem; }}
  .msg-label {{ font-size: 0.7rem; color: #656d76; text-transform: uppercase; margin-bottom: 0.1rem; }}
  .entity-tag {{ background: #ddf4ff; border: 1px solid #54aeff66; color: #0969da;
                 padding: 0.1rem 0.5rem; border-radius: 4px; font-size: 0.75rem; display: inline-block; margin: 0.1rem; }}
  .badge {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 10px; font-size: 0.75rem; font-weight: 600; background: #0969da; color: #fff; }}
  .btn {{ background: #f6f8fa; border: 1px solid #d0d7de; color: #656d76; padding: 0.3rem 0.7rem;
          border-radius: 6px; cursor: pointer; font-size: 0.8rem; }}
  .btn:hover {{ background: #eaeef2; color: #24292f; }}
  .collapse-btn {{ background: none; border: none; color: #0969da; cursor: pointer; font-size: 0.8rem; }}

  .footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #d0d7de; color: #656d76; font-size: 0.8rem; text-align: center; }}
</style>
</head>
<body>
<div class="container">
  <h1>Acervo Benchmark Report</h1>
  <p class="subtitle">{generated}{version_str} &mdash; {len(results)} scenarios, {total_turns} turns</p>

  <!-- ─── Summary Cards ─── -->
  <div class="cards">
    <div class="card"><div class="val green">{avg_savings:.0f}%</div><div class="lbl">Average savings</div></div>
    <div class="card"><div class="val blue">{total_turns}</div><div class="lbl">Total turns tested</div></div>
    <div class="card"><div class="val green">{avg_hits:.0f}%</div><div class="lbl">Avg context hits</div></div>
    <div class="card"><div class="val {'good' if total_phantoms == 0 else 'red'}">{total_phantoms}</div><div class="lbl">Phantom entities</div></div>
    <div class="card"><div class="val blue">{total_nodes}</div><div class="lbl">Entities extracted</div></div>
  </div>

  <!-- ─── Scissor Chart ─── -->
  <h2>Token usage over time &mdash; {longest_name} ({longest.total_turns} turns)</h2>
  <p class="section-desc">Without Acervo (red), every message adds to the context. With Acervo (green), the knowledge graph keeps token usage flat regardless of conversation length.</p>
  <div class="chart-box">
    <canvas id="scissorChart" height="70"></canvas>
    <div class="chart-callouts">
      <div class="chart-callout"><div class="val green">{scissor_savings:.0f}%</div><div class="lbl">Savings at turn {longest.total_turns}</div></div>
      <div class="chart-callout"><div class="val green">~{scissor_acervo} tk</div><div class="lbl">Acervo context</div></div>
      <div class="chart-callout"><div class="val" style="color:#cf222e;">{scissor_baseline:,} tk</div><div class="lbl">Baseline at turn {longest.total_turns}</div></div>
    </div>
  </div>

  <!-- ─── Scorecard ─── -->
  <h2>Scorecard &mdash; all {len(results)} scenarios</h2>
  <p class="section-desc">Token savings and context hit rate side by side. Green = savings, blue = context hit rate (% of turns where Acervo injected relevant context).</p>
  <div class="chart-box">
    <canvas id="scorecardChart" height="70"></canvas>
  </div>

  <!-- ─── Token Anatomy ─── -->
  <h2>Token anatomy &mdash; what Acervo sends vs what it would cost</h2>
  <p class="section-desc">Sampled every ~10 turns from {longest_name}. The stacked bars show Acervo's composition (warm context from graph, hot layer, system+user). The red bars show what full history would cost.</p>
  <div class="chart-box">
    <canvas id="anatomyChart" height="70"></canvas>
  </div>

  <!-- ─── Per-Scenario Table ─── -->
  <h2>Per-scenario breakdown</h2>
  <table>
    <thead><tr><th>Scenario</th><th>Turns</th><th>Savings</th><th>Context Hits</th><th>Entity Recall</th><th>Nodes</th><th>Edges</th><th>Phantoms</th></tr></thead>
    <tbody id="scenario-tbody"></tbody>
  </table>

  <!-- ─── Cost Estimation ─── -->
  <h2>Estimated API cost per scenario</h2>
  <p class="section-desc">Based on published API pricing (per 1M tokens). Output estimated at 30% of input tokens.</p>
  <div id="cost-table"></div>

  <!-- ─── Notable Moments ─── -->
  <h2>Notable moments</h2>
  <p class="section-desc">Auto-detected events: context restored after long gaps, cost milestones, peak compression, graph growth.</p>
  <table id="beats-table">
    <thead><tr><th>Scenario</th><th>Turn</th><th>Type</th><th>What happened</th></tr></thead>
    <tbody></tbody>
  </table>

  <!-- ─── Per-Scenario Token Charts ─── -->
  <h2>Token trends per scenario</h2>
  <p class="section-desc">Acervo (green) vs full history (red) for each scenario individually.</p>
  <div id="per-scenario-charts"></div>

  <!-- ─── Conversation Evidence ─── -->
  <h2>Conversation evidence</h2>
  <p class="section-desc">Actual messages, injected context, and extracted entities for each turn.</p>
  <div style="margin-bottom:0.8rem;">
    <select id="evidence-select" class="evidence-select" onchange="renderEvidence()"></select>
    <button class="btn" onclick="toggleAll()" style="margin-left:0.4rem;">Expand/Collapse All</button>
  </div>
  <div id="evidence-section"></div>

  <div class="footer">
    <p>Acervo &mdash; Graph-based context compression for AI conversations.</p>
  </div>
</div>

<script>
const scenarios = {scenarios_json};
const scissorData = {scissor_json};
const anatomyData = {anatomy_json};
const allTurnData = {all_turns_json};
const detailData = {detail_json};
const costInfo = {cost_json};
const beats = {beats_json};

// ─── Scissor Chart ───
new Chart(document.getElementById('scissorChart'), {{
  type: 'line',
  data: {{
    labels: scissorData.acervo.map((_, i) => i + 1),
    datasets: [
      {{ label: 'With Acervo (graph context)', data: scissorData.acervo,
         borderColor: '#238636', backgroundColor: '#23863622', fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2 }},
      {{ label: 'Without Acervo (full history)', data: scissorData.baseline,
         borderColor: '#f85149', backgroundColor: '#f8514922', fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2 }},
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#656d76' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#656d76', maxTicksLimit: 15 }}, grid: {{ color: '#d0d7de44' }}, title: {{ display: true, text: 'Conversation turn', color: '#656d76' }} }},
      y: {{ ticks: {{ color: '#656d76', callback: v => v >= 1000 ? (v/1000).toFixed(0)+'k' : v }}, grid: {{ color: '#d0d7de44' }}, title: {{ display: true, text: 'Tokens sent to LLM', color: '#656d76' }} }},
    }}
  }}
}});

// ─── Scorecard Bar Chart ───
new Chart(document.getElementById('scorecardChart'), {{
  type: 'bar',
  data: {{
    labels: scenarios.map(s => s.name),
    datasets: [
      {{ label: 'Token savings %', data: scenarios.map(s => s.savings), backgroundColor: '#238636' }},
      {{ label: 'Context hit rate %', data: scenarios.map(s => s.hits), backgroundColor: '#1f6feb' }},
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#656d76' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#656d76' }}, grid: {{ display: false }} }},
      y: {{ ticks: {{ color: '#656d76' }}, grid: {{ color: '#d0d7de44' }}, min: 0, max: 100, title: {{ display: true, text: '%', color: '#656d76' }} }},
    }}
  }}
}});

// ─── Token Anatomy ───
new Chart(document.getElementById('anatomyChart'), {{
  type: 'bar',
  data: {{
    labels: anatomyData.labels,
    datasets: [
      {{ label: 'Full history (baseline)', data: anatomyData.baseline, backgroundColor: '#f8514966', borderColor: '#f85149', borderWidth: 1 }},
      {{ label: 'Warm layer (graph context)', data: anatomyData.warm, backgroundColor: '#3fb95088', stack: 'acervo' }},
      {{ label: 'Hot layer (current topic)', data: anatomyData.hot, backgroundColor: '#d2992288', stack: 'acervo' }},
      {{ label: 'System + user message', data: anatomyData.system_user, backgroundColor: '#8b949e88', stack: 'acervo' }},
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#656d76' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#656d76' }}, grid: {{ display: false }} }},
      y: {{ ticks: {{ color: '#656d76', callback: v => v >= 1000 ? (v/1000).toFixed(0)+'k' : v }}, grid: {{ color: '#d0d7de44' }}, title: {{ display: true, text: 'Tokens', color: '#656d76' }} }},
    }}
  }}
}});

// ─── Scenario Table ───
const stbody = document.getElementById('scenario-tbody');
scenarios.forEach(s => {{
  stbody.innerHTML += `<tr>
    <td>${{s.name}}</td><td>${{s.turns}}</td>
    <td class="good">${{s.savings}}%</td>
    <td>${{s.hits}}%</td>
    <td>${{s.recall}}%</td>
    <td>${{s.nodes}}</td><td>${{s.edges}}</td>
    <td class="${{s.phantoms === 0 ? 'good' : 'bad'}}">${{s.phantoms}}</td>
  </tr>`;
}});

// ─── Cost Table ───
const costEl = document.getElementById('cost-table');
let ch = '<table><thead><tr><th>Scenario</th>';
costInfo.models.forEach(m => {{ ch += `<th colspan="3">${{m}}</th>`; }});
ch += '</tr><tr><th></th>';
costInfo.models.forEach(() => {{ ch += '<th>Baseline</th><th>Acervo</th><th>Saved</th>'; }});
ch += '</tr></thead><tbody>';
costInfo.rows.forEach(r => {{
  ch += `<tr><td>${{r.name}}</td>`;
  costInfo.models.forEach(m => {{
    const d = r[m];
    ch += `<td>$${{d.baseline.toFixed(4)}}</td><td>$${{d.acervo.toFixed(4)}}</td><td class="saved">$${{d.saved.toFixed(4)}}</td>`;
  }});
  ch += '</tr>';
}});
// Totals
ch += '<tr style="border-top:2px solid #d0d7de;font-weight:bold;"><td>TOTAL</td>';
costInfo.models.forEach(m => {{
  let tb=0, ta=0;
  costInfo.rows.forEach(r => {{ tb += r[m].baseline; ta += r[m].acervo; }});
  ch += `<td>$${{tb.toFixed(4)}}</td><td>$${{ta.toFixed(4)}}</td><td class="saved">$${{(tb-ta).toFixed(4)}}</td>`;
}});
ch += '</tr>';
// Projected
ch += '<tr style="color:#9a6700;"><td>Projected: 1,000 conversations</td>';
costInfo.models.forEach(m => {{
  let tb=0, ta=0;
  costInfo.rows.forEach(r => {{ tb += r[m].baseline; ta += r[m].acervo; }});
  const s = 1000 / costInfo.rows.length;
  ch += `<td>$${{(tb*s).toFixed(2)}}</td><td>$${{(ta*s).toFixed(2)}}</td><td class="saved">$${{((tb-ta)*s).toFixed(2)}}</td>`;
}});
ch += '</tr></tbody></table>';
costEl.innerHTML = ch;

// ─── Beats Table ───
const btbody = document.querySelector('#beats-table tbody');
beats.forEach(b => {{
  btbody.innerHTML += `<tr><td>${{b.scenario}}</td><td>${{b.turn}}</td><td>${{b.type}}</td><td>${{b.headline}}</td></tr>`;
}});
if (!beats.length) btbody.innerHTML = '<tr><td colspan="4" style="color:#656d76;">No notable beats detected.</td></tr>';

// ─── Per-Scenario Charts ───
const perEl = document.getElementById('per-scenario-charts');
Object.entries(allTurnData).forEach(([name, data]) => {{
  const div = document.createElement('div');
  div.className = 'chart-box';
  div.style.marginBottom = '0.5rem';
  const canvas = document.createElement('canvas');
  canvas.height = 45;
  div.appendChild(canvas);
  perEl.appendChild(div);
  new Chart(canvas, {{
    type: 'line',
    data: {{
      labels: data.acervo.map((_, i) => i + 1),
      datasets: [
        {{ label: name.replace(/_/g, ' ') + ' \u2014 Acervo', data: data.acervo,
           borderColor: '#238636', fill: false, tension: 0.3, pointRadius: 0, borderWidth: 2 }},
        {{ label: name.replace(/_/g, ' ') + ' \u2014 Baseline', data: data.baseline,
           borderColor: '#f85149', fill: false, tension: 0.3, pointRadius: 0, borderWidth: 2 }},
      ]
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ labels: {{ color: '#656d76' }} }} }},
      scales: {{
        x: {{ ticks: {{ color: '#656d76', maxTicksLimit: 15 }}, grid: {{ display: false }} }},
        y: {{ ticks: {{ color: '#656d76' }}, grid: {{ color: '#d0d7de44' }} }},
      }}
    }}
  }});
}});

// ─── Evidence ───
const evSelect = document.getElementById('evidence-select');
Object.keys(detailData).filter(k => detailData[k].length > 0).forEach(name => {{
  const opt = document.createElement('option');
  opt.value = name; opt.textContent = name.replace(/_/g, ' ');
  evSelect.appendChild(opt);
}});
let allExpanded = false;
function toggleAll() {{ allExpanded = !allExpanded; document.querySelectorAll('.turn-detail').forEach(el => el.style.display = allExpanded ? 'block' : 'none'); }}
function renderEvidence() {{
  const turns = detailData[evSelect.value] || [];
  const el = document.getElementById('evidence-section');
  if (!turns.length) {{ el.innerHTML = '<p style="color:#656d76;">No data.</p>'; return; }}
  el.innerHTML = turns.map(t => `
    <div class="turn-card">
      <div class="turn-header">
        <span><strong>Turn ${{t.turn}}</strong> \u2014 ${{t.desc||''}}
          <button class="collapse-btn" onclick="this.closest('.turn-card').querySelector('.turn-detail').style.display=this.closest('.turn-card').querySelector('.turn-detail').style.display==='none'?'block':'none'">[toggle]</button>
        </span>
        <span class="badge">${{t.phase}}</span>
      </div>
      <div class="turn-meta">
        <span>Acervo: ${{t.acervo_tk}} tk</span>
        <span>Baseline: ${{t.baseline_tk}} tk</span>
        <span style="color:${{t.acervo_tk < t.baseline_tk ? '#1a7f37' : '#cf222e'}}">${{t.baseline_tk > 0 ? ((1-t.acervo_tk/t.baseline_tk)*100).toFixed(0) : 0}}% saved</span>
        <span>Context: ${{t.hit ? 'HIT' : 'MISS'}}</span>
        <span>Nodes: ${{t.nodes}}</span>
      </div>
      <div class="turn-detail" style="display:none;margin-top:0.4rem;">
        <div class="msg msg-user"><div class="msg-label">User</div>${{t.user}}</div>
        ${{t.warm ? `<div class="msg msg-context"><div class="msg-label">Injected context</div>${{t.warm}}</div>` : ''}}
        ${{t.assistant ? `<div class="msg msg-assistant"><div class="msg-label">Assistant</div>${{t.assistant}}</div>` : ''}}
        ${{t.entities.length ? `<div style="margin-top:0.3rem;">${{t.entities.map(e => `<span class="entity-tag">${{e}}</span>`).join('')}}</div>` : ''}}
      </div>
    </div>
  `).join('');
}}
if (evSelect.options.length) renderEvidence();
</script>
</body>
</html>"""


def _scenario_display_name(r: "ScenarioResult") -> str:
    """Convert scenario name to display format."""
    name = r.name.replace("_", " ").title()
    return f"{name} ({r.total_turns} turns)"


# ── Quick Regression Report ──


def html_report_quick(results: list["ScenarioResult"]) -> str:
    """Compact regression dashboard for quick dev iteration.

    Shows pass/fail per scenario, key metrics table, and token trend charts.
    Designed to be scanned in 5 seconds.
    """
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    total_turns = sum(r.total_turns for r in results)
    avg_savings = sum(r.avg_savings_pct for r in results) / max(len(results), 1)

    scenarios_json = json.dumps([
        {
            "name": r.name.replace("_", " ").title(),
            "turns": r.total_turns,
            "savings": round(r.avg_savings_pct, 1),
            "recall": round(r.entity_recall * 100, 1),
            "hits": round(r.context_hit_rate * 100, 1),
            "phantoms": len(r.phantom_entities),
            "nodes": r.final_node_count,
        }
        for r in results
    ])

    turn_data = {}
    for r in results:
        turn_data[r.name] = {
            "acervo": [t.acervo_tokens for t in r.turns],
            "baseline": [t.baseline_tokens for t in r.turns],
        }
    turns_json = json.dumps(turn_data)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Acervo Quick Check</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0d1117; color: #c9d1d9; line-height: 1.6; padding: 2rem; }}
  .container {{ max-width: 1000px; margin: 0 auto; }}
  h1 {{ color: #58a6ff; font-size: 1.8rem; margin-bottom: 0.3rem; }}
  .subtitle {{ color: #8b949e; margin-bottom: 1.5rem; font-size: 0.9rem; }}
  .status {{ font-size: 2.5rem; font-weight: 800; text-align: center; padding: 1.5rem;
             border-radius: 12px; margin-bottom: 2rem; }}
  .status.pass {{ background: #238636; color: #fff; }}
  .status.warn {{ background: #9e6a03; color: #fff; }}
  .status.fail {{ background: #da3633; color: #fff; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
  th, td {{ padding: 0.6rem 1rem; text-align: left; border-bottom: 1px solid #30363d; }}
  th {{ background: #161b22; color: #8b949e; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; }}
  .good {{ color: #3fb950; }}
  .warn {{ color: #d29922; }}
  .bad {{ color: #f85149; }}
  .chart-box {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1rem; margin: 1rem 0; }}
</style>
</head>
<body>
<div class="container">
  <h1>Acervo Quick Check</h1>
  <p class="subtitle">{generated} &mdash; {total_turns} turns across {len(results)} scenarios</p>

  <div id="overall-status" class="status"></div>

  <table>
    <thead>
      <tr><th>Scenario</th><th>Turns</th><th>Savings</th><th>Recall</th><th>Context Hits</th><th>Phantoms</th><th>Nodes</th></tr>
    </thead>
    <tbody id="metrics-table"></tbody>
  </table>

  <div id="charts"></div>
</div>

<script>
const scenarios = {scenarios_json};
const turnData = {turns_json};

// Overall status
const avgSavings = scenarios.reduce((s, r) => s + r.savings, 0) / scenarios.length;
const totalPhantoms = scenarios.reduce((s, r) => s + r.phantoms, 0);
const minRecall = Math.min(...scenarios.map(s => s.recall));
const statusEl = document.getElementById('overall-status');
if (avgSavings >= 50 && totalPhantoms === 0 && minRecall >= 50) {{
  statusEl.className = 'status pass';
  statusEl.textContent = 'PASS \u2014 ' + avgSavings.toFixed(0) + '% avg savings';
}} else if (avgSavings >= 30) {{
  statusEl.className = 'status warn';
  statusEl.textContent = 'WARN \u2014 ' + avgSavings.toFixed(0) + '% avg savings' + (totalPhantoms > 0 ? ', ' + totalPhantoms + ' phantoms' : '');
}} else {{
  statusEl.className = 'status fail';
  statusEl.textContent = 'FAIL \u2014 ' + avgSavings.toFixed(0) + '% avg savings';
}}

// Metrics table
const tbody = document.getElementById('metrics-table');
scenarios.forEach(s => {{
  const savCls = s.savings >= 50 ? 'good' : s.savings >= 30 ? 'warn' : 'bad';
  const recCls = s.recall >= 70 ? 'good' : s.recall >= 50 ? 'warn' : 'bad';
  const hitCls = s.hits >= 70 ? 'good' : s.hits >= 50 ? 'warn' : 'bad';
  const phCls = s.phantoms === 0 ? 'good' : 'bad';
  tbody.innerHTML += `<tr>
    <td>${{s.name}}</td><td>${{s.turns}}</td>
    <td class="${{savCls}}">${{s.savings}}%</td>
    <td class="${{recCls}}">${{s.recall}}%</td>
    <td class="${{hitCls}}">${{s.hits}}%</td>
    <td class="${{phCls}}">${{s.phantoms}}</td>
    <td>${{s.nodes}}</td>
  </tr>`;
}});

// Token charts
const chartsEl = document.getElementById('charts');
Object.entries(turnData).forEach(([name, data]) => {{
  const div = document.createElement('div');
  div.className = 'chart-box';
  const canvas = document.createElement('canvas');
  canvas.height = 50;
  div.appendChild(canvas);
  chartsEl.appendChild(div);
  new Chart(canvas, {{
    type: 'line',
    data: {{
      labels: data.acervo.map((_, i) => i + 1),
      datasets: [
        {{ label: name.replace(/_/g, ' ') + ' \u2014 Acervo', data: data.acervo,
           borderColor: '#238636', fill: false, tension: 0.3, pointRadius: 0, borderWidth: 2 }},
        {{ label: name.replace(/_/g, ' ') + ' \u2014 Baseline', data: data.baseline,
           borderColor: '#f85149', fill: false, tension: 0.3, pointRadius: 0, borderWidth: 2 }},
      ]
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ labels: {{ color: '#656d76' }} }} }},
      scales: {{
        x: {{ ticks: {{ color: '#656d76' }}, grid: {{ display: false }} }},
        y: {{ ticks: {{ color: '#656d76' }}, grid: {{ color: '#d0d7de44' }} }},
      }}
    }}
  }});
}});
</script>
</body>
</html>"""
