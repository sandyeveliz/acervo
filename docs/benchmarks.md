# Benchmark Reports

Acervo runs benchmarks on every release to measure context quality, token efficiency,
and entity extraction. Two benchmark types:

- **Conversation benchmarks** (v0.2-v0.3): 360 turns across 6 scenarios measuring token savings and context hit rates.
- **Indexed project benchmarks** (v0.4+): 55 turns across 3 projects with 5-category scoring (RESOLVE/GROUND/RECALL/FOCUS/ADAPT) and agent efficiency comparison.

## Reports by Version

| Version | Date | Type | Turns | Key Result | Report |
|---------|------|------|-------|------------|--------|
| **v0.4.0** | 2026-04-01 | Indexed project | 55 | 100% RESOLVE, 12.1x efficiency | [Full Report](benchmarks/v0.4.0/) |
| v0.2.2-3 | 2026-03-27 | Conversation | 360 | 76.1% savings | [Full Report](benchmarks/v0.2.2-3/) |
| v0.2.2-2 | 2026-03-27 | Conversation | 360 | 76.1% savings | [Full Report](benchmarks/v0.2.2-2/) |
| v0.2.2-1 | 2026-03-27 | Conversation | 360 | 76.1% savings | [Full Report](benchmarks/v0.2.2-1/) |

## v0.4.0 — Indexed Project Benchmarks

### Category Scores

| Category | What it proves | Score |
|----------|---------------|-------|
| RESOLVE  | Answers questions requiring project context | 100% |
| GROUND   | Prevents hallucination with verified data | 92% |
| RECALL   | Remembers user-stated facts across turns | 67% |
| FOCUS    | Sends only relevant context, respects budget | 100% |
| ADAPT    | Handles topic changes cleanly | 100% |

### Approach Comparison (RESOLVE, 13 turns)

| Approach | Can Answer | Avg Input Tokens | Avg Steps |
|----------|-----------|-----------------|-----------|
| Stateless LLM | 8% | -- | -- |
| Agent + Tools | 100% | 7,462 | 2.8 |
| **Acervo** | **100%** | **616** | **0** |

> **12.1x fewer tokens** than an agent approach for the same questions.

### Component Health

| Component | Score |
|-----------|-------|
| S1 Intent | 78% |
| S2 Activation | 56% |
| S3 Budget | 32% |
| S3 Quality | 81% |

### Test Projects

| Project | Domain | Content |
|---------|--------|---------|
| P1 — TODO App | Source code | 31 TypeScript/React files |
| P2 — Literature | Prose | Sherlock Holmes epub (public domain) |
| P3 — PM Docs | Project management | 11 markdown files |

For detailed methodology, see the [Benchmark Guide](benchmark-guide.md).

## v0.2.x — Conversation Benchmarks

| # | Scenario | Turns | Description |
|---|----------|-------|-------------|
| 1 | Developer Workflow | 60 | Programming questions, debugging, code review |
| 2 | Literature & Comics | 60 | Character tracking, plot analysis, cross-references |
| 3 | Academic Research | 60 | Citations, methodology, multi-domain synthesis |
| 4 | Mixed Domains | 60 | Rapid topic switching across unrelated subjects |
| 5 | SaaS Founder (100t) | 60 | Long-form business context, metrics, strategy |
| 6 | Product Manager | 60 | Real-world PM workflow, stakeholder tracking |

## Generating Reports

```bash
# Indexed project benchmarks (v0.4+, requires LM Studio + Ollama)
pytest tests/integration/test_benchmarks.py -v -s

# Conversation benchmarks (v0.2-v0.3)
python -m tests.integration.run_benchmarks --format html
python -m tests.integration.export_report --tier full --open
```

After generating, copy reports to `docs/benchmarks/vX.Y.Z/` and update this page.
