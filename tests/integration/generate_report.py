"""Unified Benchmark Report Generator — collects all test layers into one report.

Reads results from:
  - Layer 1:  pipeline_diagnostic.json
  - Layer 1b: graph_quality.json
  - Layer 2:  benchmark_public.json + benchmark_diagnostic.json
  - Layer 3:  conversation_c1/c2/c3.json

Generates:
  - benchmark_unified.json (machine-readable)
  - benchmark_unified.md (human-readable)
  - benchmark_report.html (publication quality)

Usage:
    python -m tests.integration.generate_report
    python -m tests.integration.generate_report --version v0.5.0
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

_REPORTS = Path(__file__).parent / "reports"


def _load_json(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def generate_unified(version: str = "v0.5.0") -> dict:
    """Collect all test results into a unified report."""
    version_dir = _REPORTS / version

    # Load available results
    l1 = _load_json(version_dir / "pipeline_diagnostic.json")
    l1b = _load_json(version_dir / "graph_quality.json")
    l2_public = _load_json(version_dir / "benchmark_public.json")
    l2_diag = _load_json(version_dir / "benchmark_diagnostic.json")

    c1 = _load_json(version_dir / "conversation_c1_multi_project.json")
    c2 = _load_json(version_dir / "conversation_c2_personal_knowledge.json")
    c3 = _load_json(version_dir / "conversation_c3_progressive_building.json")

    # Compute summary
    summary: dict = {
        "version": version,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    # L2 scores
    if l2_public:
        summary["category_scores"] = l2_public.get("scores", {})
        summary["total_indexed_turns"] = l2_public.get("total_turns", 0)

    # L1b quality
    if l1b:
        projects = l1b.get("projects", [])
        total_checks = sum(p["total"] for p in projects)
        passed_checks = sum(p["passed"] for p in projects)
        summary["graph_quality"] = {
            "total_checks": total_checks,
            "passed": passed_checks,
            "pass_rate": round(passed_checks / total_checks * 100) if total_checks else 0,
        }

    # L3 conversations
    conv_results = [r for r in [c1, c2, c3] if r]
    if conv_results:
        total_conv_turns = sum(r.get("total_turns", 0) for r in conv_results)
        passed_conv_turns = sum(r.get("passed_turns", 0) for r in conv_results)
        summary["conversation"] = {
            "scenarios": len(conv_results),
            "total_turns": total_conv_turns,
            "passed_turns": passed_conv_turns,
            "pass_rate": round(passed_conv_turns / total_conv_turns * 100) if total_conv_turns else 0,
        }

    unified = {
        "summary": summary,
        "layers": {
            "L1_pipeline": l1,
            "L1b_graph_quality": l1b,
            "L2_indexed_retrieval": l2_public,
            "L2_diagnostic": l2_diag,
            "L3_conversations": {
                "c1_multi_project": c1,
                "c2_personal_knowledge": c2,
                "c3_progressive_building": c3,
            },
        },
    }

    return unified


def write_unified_json(unified: dict, version: str = "v0.5.0") -> None:
    version_dir = _REPORTS / version
    version_dir.mkdir(parents=True, exist_ok=True)
    (version_dir / "benchmark_unified.json").write_text(
        json.dumps(unified, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def write_unified_md(unified: dict, version: str = "v0.5.0") -> None:
    version_dir = _REPORTS / version
    s = unified.get("summary", {})

    lines = [
        f"# Acervo {version} — Unified Benchmark Report",
        f"Generated: {s.get('timestamp', '')}",
        "",
    ]

    # Category scores
    scores = s.get("category_scores", {})
    if scores:
        lines.append("## Category Scores (Layer 2)")
        lines.append("| Category | Score |")
        lines.append("|----------|-------|")
        for cat, score in scores.items():
            lines.append(f"| {cat} | {score}% |")
        lines.append("")

    # Graph quality
    gq = s.get("graph_quality", {})
    if gq:
        lines.append(f"## Graph Quality (Layer 1b)")
        lines.append(f"**{gq['passed']}/{gq['total_checks']} checks passed ({gq['pass_rate']}%)**")
        lines.append("")

    # Conversations
    conv = s.get("conversation", {})
    if conv:
        lines.append(f"## Conversation Scenarios (Layer 3)")
        lines.append(f"**{conv['passed_turns']}/{conv['total_turns']} turns passed "
                      f"({conv['pass_rate']}%) across {conv['scenarios']} scenarios**")
        lines.append("")

    (version_dir / "benchmark_unified.md").write_text("\n".join(lines), encoding="utf-8")


def write_html_report(unified: dict, version: str = "v0.5.0") -> None:
    """Generate publication-quality HTML report."""
    version_dir = _REPORTS / version
    s = unified.get("summary", {})
    scores = s.get("category_scores", {})
    gq = s.get("graph_quality", {})
    conv = s.get("conversation", {})

    # Compute headline numbers
    avg_score = round(sum(scores.values()) / len(scores)) if scores else 0
    total_turns = s.get("total_indexed_turns", 0) + conv.get("total_turns", 0)

    # C1 evolution data for chart
    c1_data = unified.get("layers", {}).get("L3_conversations", {}).get("c1_multi_project", {})
    c1_turns = c1_data.get("turns", []) if c1_data else []

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Acervo {version} Benchmark Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Inter', -apple-system, sans-serif; background: #ffffff; color: #0f172a; line-height: 1.6; }}
  .container {{ max-width: 900px; margin: 0 auto; padding: 2rem 1.5rem; }}
  h1 {{ font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; }}
  h2 {{ font-size: 1.25rem; font-weight: 600; margin: 2rem 0 1rem; color: #334155; }}
  .subtitle {{ color: #64748b; font-size: 1rem; margin-bottom: 2rem; }}
  .hero-stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 2rem 0; }}
  .stat-card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.5rem; text-align: center; }}
  .stat-value {{ font-size: 2rem; font-weight: 700; color: #2563eb; }}
  .stat-label {{ font-size: 0.8rem; color: #64748b; margin-top: 0.25rem; }}
  .scores-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.75rem; margin: 1rem 0; }}
  .score-card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; text-align: center; }}
  .score-value {{ font-size: 1.5rem; font-weight: 700; }}
  .score-pass {{ color: #16a34a; }}
  .score-warn {{ color: #d97706; }}
  .score-fail {{ color: #dc2626; }}
  .score-label {{ font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }}
  .chart-container {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.5rem; margin: 1.5rem 0; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.9rem; }}
  th {{ background: #f1f5f9; padding: 0.75rem; text-align: left; font-weight: 600; border-bottom: 2px solid #e2e8f0; }}
  td {{ padding: 0.75rem; border-bottom: 1px solid #e2e8f0; }}
  .pass {{ color: #16a34a; font-weight: 600; }}
  .fail {{ color: #dc2626; font-weight: 600; }}
  .badge {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 500; }}
  .badge-pass {{ background: #dcfce7; color: #16a34a; }}
  .badge-fail {{ background: #fee2e2; color: #dc2626; }}
  .footer {{ margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid #e2e8f0; color: #94a3b8; font-size: 0.85rem; text-align: center; }}
  @media (max-width: 600px) {{ .hero-stats {{ grid-template-columns: repeat(2, 1fr); }} .scores-grid {{ grid-template-columns: repeat(3, 1fr); }} }}
</style>
</head>
<body>
<div class="container">

<h1>Acervo {version} Benchmark Report</h1>
<p class="subtitle">Semantic compression for AI agents — one knowledge graph, two input sources</p>

<div class="hero-stats">
  <div class="stat-card">
    <div class="stat-value">{avg_score}%</div>
    <div class="stat-label">Avg Category Score</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{total_turns}</div>
    <div class="stat-label">Turns Tested</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{gq.get('pass_rate', 0)}%</div>
    <div class="stat-label">Graph Quality</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{conv.get('pass_rate', 0)}%</div>
    <div class="stat-label">Conversation Acc</div>
  </div>
</div>

<h2>Category Scores (5 categories, {s.get('total_indexed_turns', 55)} indexed turns)</h2>
<div class="scores-grid">
"""

    for cat, score in scores.items():
        css = "score-pass" if score >= 90 else "score-warn" if score >= 70 else "score-fail"
        html += f"""  <div class="score-card">
    <div class="score-value {css}">{score}%</div>
    <div class="score-label">{cat}</div>
  </div>
"""

    html += "</div>\n"

    # Graph evolution chart (C1)
    if c1_turns:
        nodes_data = [t.get("graph_node_count", 0) for t in c1_turns]
        edges_data = [t.get("graph_edge_count", 0) for t in c1_turns]
        warm_data = [t.get("warm_tokens", 0) for t in c1_turns]
        labels = [f"T{t.get('turn', i+1)}" for i, t in enumerate(c1_turns)]

        html += f"""
<h2>Graph Evolution — Conversation Scenario C1</h2>
<div class="chart-container">
  <canvas id="evolutionChart" height="200"></canvas>
</div>
<script>
new Chart(document.getElementById('evolutionChart'), {{
  type: 'line',
  data: {{
    labels: {json.dumps(labels)},
    datasets: [
      {{ label: 'Nodes', data: {json.dumps(nodes_data)}, borderColor: '#2563eb', fill: false, tension: 0.3 }},
      {{ label: 'Edges', data: {json.dumps(edges_data)}, borderColor: '#16a34a', fill: false, tension: 0.3 }},
      {{ label: 'Warm Tokens', data: {json.dumps(warm_data)}, borderColor: '#d97706', fill: false, tension: 0.3, yAxisID: 'y1' }}
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ position: 'bottom' }} }},
    scales: {{
      y: {{ title: {{ display: true, text: 'Count' }} }},
      y1: {{ position: 'right', title: {{ display: true, text: 'Warm Tokens' }}, grid: {{ drawOnChartArea: false }} }}
    }}
  }}
}});
</script>
"""

    # Graph quality table
    if gq:
        l1b = unified.get("layers", {}).get("L1b_graph_quality", {})
        projects = l1b.get("projects", []) if l1b else []
        html += """
<h2>Graph Quality (Layer 1b)</h2>
<table>
  <tr><th>Project</th><th>Passed</th><th>Failed</th><th>Entities</th><th>Nodes</th><th>Edges</th></tr>
"""
        for p in projects:
            badge = "badge-pass" if p["failed"] == 0 else "badge-fail"
            html += f"""  <tr>
    <td>{p['project']}</td>
    <td><span class="badge {badge}">{p['passed']}/{p['total']}</span></td>
    <td>{p['failed']}</td>
    <td>{p['entity_count']}</td>
    <td>{p['node_count']}</td>
    <td>{p['edge_count']}</td>
  </tr>
"""
        html += "</table>\n"

    # Conversation scenarios table
    l3 = unified.get("layers", {}).get("L3_conversations", {})
    conv_list = [l3.get("c1_multi_project"), l3.get("c2_personal_knowledge"), l3.get("c3_progressive_building")]
    conv_list = [r for r in conv_list if r]
    if conv_list:
        html += """
<h2>Conversation Scenarios (Layer 3)</h2>
<table>
  <tr><th>Scenario</th><th>Turns</th><th>Passed</th><th>Graph</th><th>Entity Acc</th></tr>
"""
        for r in conv_list:
            badge = "badge-pass" if r.get("pass_rate", 0) >= 70 else "badge-fail"
            html += f"""  <tr>
    <td>{r.get('name', '?')}</td>
    <td>{r.get('total_turns', 0)}</td>
    <td><span class="badge {badge}">{r.get('passed_turns', 0)}/{r.get('total_turns', 0)}</span></td>
    <td>{r.get('final_graph', {}).get('nodes', 0)}n / {r.get('final_graph', {}).get('edges', 0)}e</td>
    <td>{r.get('avg_entity_accuracy', 0)}%</td>
  </tr>
"""
        html += "</table>\n"

    html += f"""
<div class="footer">
  <p>Acervo {version} — Apache 2.0 — Generated {s.get('timestamp', '')}</p>
</div>
</div>
</body>
</html>"""

    (version_dir / "benchmark_report.html").write_text(html, encoding="utf-8")


def main():
    version = sys.argv[1] if len(sys.argv) > 1 else "v0.5.0"
    print(f"Generating unified report for {version}...")

    unified = generate_unified(version)
    write_unified_json(unified, version)
    write_unified_md(unified, version)
    write_html_report(unified, version)

    s = unified.get("summary", {})
    print(f"  Category scores: {s.get('category_scores', {})}")
    print(f"  Graph quality: {s.get('graph_quality', {})}")
    print(f"  Conversations: {s.get('conversation', {})}")
    print(f"  Reports written to tests/integration/reports/{version}/")


if __name__ == "__main__":
    main()
