# Benchmark Reports

Acervo runs benchmarks on every release to measure context quality, token efficiency,
and entity extraction. Three benchmark types:

- **Conversation benchmarks** (v0.2-v0.3): 360 turns across 6 scenarios measuring token savings and context hit rates.
- **Indexed project benchmarks** (v0.4+): 55 turns across 3 projects with 5-category scoring (RESOLVE/GROUND/RECALL/FOCUS/ADAPT) and agent efficiency comparison.
- **Conversation scenario tests** (v0.5+): 24 turns across 3 scenarios testing real-time graph construction from conversation.

## Reports by Version

| Version | Date | Type | Turns | Key Result | Report |
|---------|------|------|-------|------------|--------|
| **v0.5.0** | 2026-04-06 | Indexed + Conversation | 79 | 100% GROUND, 21.3x efficiency, BFS layers | [Full Report](benchmarks/v0.5.0/) |
| v0.4.0 | 2026-04-01 | Indexed project | 55 | 100% RESOLVE, 12.1x efficiency | [Full Report](benchmarks/v0.4.0/) |
| v0.2.2-3 | 2026-03-27 | Conversation | 360 | 76.1% savings | [Full Report](benchmarks/v0.2.2-3/) |
| v0.2.2-2 | 2026-03-27 | Conversation | 360 | 76.1% savings | [Full Report](benchmarks/v0.2.2-2/) |
| v0.2.2-1 | 2026-03-27 | Conversation | 360 | 76.1% savings | [Full Report](benchmarks/v0.2.2-1/) |

## v0.5.0 — Hexagonal Architecture + BFS Semantic Layers

### Category Scores

| Category | What it proves | v0.4 | v0.5 |
|----------|---------------|------|------|
| RESOLVE  | Answers questions requiring project context | 100% | 85% |
| GROUND   | Prevents hallucination with verified data | 92% | **100%** |
| RECALL   | Remembers user-stated facts across turns | 67% | 67% |
| FOCUS    | Sends only relevant context, respects budget | 100% | 100% |
| ADAPT    | Handles topic changes cleanly | 100% | 89% |

### Efficiency vs Agent (21.3x improvement)

| Approach | Can Answer | Avg Input Tokens | Avg Steps |
|----------|-----------|-----------------|-----------|
| Stateless LLM | 8% | -- | -- |
| Agent + Tools | 100% | 7,462 | 2.8 |
| **Acervo** | **100%** | **~350** | **0** |

> **21.3x fewer tokens** than an agent approach. Up from 12.1x in v0.4.

### Graph Quality (85/85 checks)

| Project | Checks | Entities | Nodes | Edges |
|---------|--------|----------|-------|-------|
| P1 Code (Todo App) | 28/28 ✓ | 7 | 231 | 1,109 |
| P2 Literature (Sherlock Holmes) | 21/21 ✓ | 5 | 40 | 307 |
| P3 PM Docs | 32/32 ✓ | 6 | 108 | 331 |

### Conversation Scenarios (NEW in v0.5)

| Scenario | Turns | Passed | Graph | Entity Accuracy |
|----------|-------|--------|-------|----------------|
| C1: Multi-project portfolio | 10 | 7/10 | 13n / 27e | 72% |
| C2: Personal knowledge | 6 | 3/6 | 5n / 4e | 60% |
| C3: Progressive building | 8 | 7/8 | 6n / 5e | 83% |

### What's new in v0.5

- **BFS semantic layers** — S2 does breadth-first traversal: HOT (depth 0), WARM (depth 1), COLD (depth 2)
- **Compressed context format** — XML-delimited (`<hot>`, `<warm>`), ~50% fewer tokens
- **Conversation pipeline** — Graph grows in real time from chat. warm_tokens > 0 on 80%+ of retrieval turns (was 0% in v0.4)
- **Hexagonal architecture** — facade.py (1,848 LOC) → domain/pipeline.py (~200 LOC)
- **Graph quality specs** — Automated checks for required/forbidden entities

For the full interactive report with charts, see the [v0.5.0 Benchmark Report](benchmarks/v0.5.0/).

## v0.4.0 — Indexed Project Benchmarks

| Category | Score |
|----------|-------|
| RESOLVE  | 100% |
| GROUND   | 92% |
| RECALL   | 67% |
| FOCUS    | 100% |
| ADAPT    | 100% |

> **12.1x fewer tokens** than an agent approach (avg 616 tokens vs 7,462).

For details, see the [v0.4.0 Report](benchmarks/v0.4.0/).

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
# Run all integration tests (requires Ollama with acervo-extractor-v3)
pytest tests/integration/ -v -s

# Generate unified report (JSON + MD + HTML)
python tests/integration/generate_report.py v0.5.0

# Copy HTML report to docs for publishing
cp tests/integration/reports/v0.5.0/benchmark_report.html docs/benchmarks/v0.5.0/index.html
```

For detailed methodology, see the [Benchmark Guide](benchmark-guide.md).
