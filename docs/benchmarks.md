# Benchmark Reports

Acervo runs a full benchmark suite on every release to measure context efficiency,
token savings, and entity extraction quality. Each report covers 6 scenarios
(360 turns) testing different knowledge domains.

## Reports by Version

| Version | Date | Scenarios | Turns | Avg Savings | Report |
|---------|------|-----------|-------|-------------|--------|
| **v0.2.2-3** | 2026-03-27 | 6 | 360 | 76.1% | [Full Report](benchmarks/v0.2.2-3/) |
| v0.2.2-2 | 2026-03-27 | 6 | 360 | 76.1% | [Full Report](benchmarks/v0.2.2-2/) |
| v0.2.2-1 | 2026-03-27 | 6 | 360 | 76.1% | [Full Report](benchmarks/v0.2.2-1/) |

## Scenarios

| # | Scenario | Turns | Description |
|---|----------|-------|-------------|
| 1 | Developer Workflow | 60 | Programming questions, debugging, code review |
| 2 | Literature & Comics | 60 | Character tracking, plot analysis, cross-references |
| 3 | Academic Research | 60 | Citations, methodology, multi-domain synthesis |
| 4 | Mixed Domains | 60 | Rapid topic switching across unrelated subjects |
| 5 | SaaS Founder (100t) | 60 | Long-form business context, metrics, strategy |
| 6 | Product Manager | 60 | Real-world PM workflow, stakeholder tracking |

## How to Read the Reports

Each report includes:

- **Summary cards** -- total turns, token savings, entity counts
- **Token usage charts** -- context vs naive accumulation per turn
- **Scorecard** -- per-scenario pass/fail on key metrics
- **Entity extraction** -- precision, recall, and F1 for the knowledge graph
- **Turn-by-turn evidence** -- expandable details for every conversation turn

## Generating New Reports

```bash
# Run full benchmark suite
python -m tests.integration.run_benchmarks --format html

# Export from existing JSON results
python -m tests.integration.export_report --tier full --open
```

After generating, copy the report to `docs/benchmarks/vX.Y.Z/index.html` and update this page.
