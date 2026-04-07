# v0.5.0 — "Clean Architecture"

> Hexagonal pipeline refactor. Each stage is standalone and testable.
> S2 works identically for conversation and indexed projects.
> The foundation for MCP, SDK, and multi-project support.

---

## M1 — Architecture Refactor ✅

Split the 1,848-line God Module (`facade.py`) into:

```
ports/           → Protocol interfaces (LLM, Embedder, VectorStore, GraphStore)
domain/          → Pipeline stages (S1, S2, S3, S1.5) + orchestrator
graph/           → Knowledge graph (TopicGraph, Layer, ontology)
context/         → Context management (ContextIndex, TopicDetector, synthesizer)
extraction/      → Entity extractors (Conversation, Text, Search, RAG)
indexing/        → File indexation pipeline (unchanged internals)
adapters/        → Concrete implementations (OpenAI client, ChromaDB)
```

Key results:
- `facade.py` → thin wrapper delegating to `Pipeline`
- S2 Activator: ONE code path for all modes
- S3 Assembler: intent controls budget, not whether to inject
- 178/185 tests pass, 0 regressions

---

## M2 — Pipeline Bug Fixes ✅

Critical fixes discovered during architecture audit:
- JSON parser: 3-level repair for malformed model output
- Nested relations: extracted from entity objects
- Relation IDs: resolved to entity labels (not model-generated IDs)
- Entity expansion: S2 traverses ALL entity edges
- Topic drift: limited PREVIOUS ASSISTANT influence
- Description chunks: included in S2 context building
- S3 unified: no intent-based chunk filtering

---

## M3 — Ollama Migration ✅

- Default provider: Ollama (port 11434) replaces LM Studio (port 1234)
- Modelfile: Qwen3.5 chat template for fine-tuned extractor
- `acervo up --dev` works from any directory

---

## M4 — Studio Features ✅

- Telemetry: per-turn annotation with actual vs expected
- Ollama monitor: live VRAM/GPU/RAM
- Per-project telemetry persistence
- Annotation export (JSONL training data)

---

## What's left for v0.5.0 release

1. **Verify Turn 4** — restart proxy, confirm warm_tokens > 0 for overview intent
2. **Documentation** — update README, ARCHITECTURE.md, CLAUDE.md
3. **Commit & tag** — clean commit history, tag v0.5.0

---

## Deferred to v0.6.0

The original v0.5.0 milestones (MCP Server, TypeScript SDK, LangChain integration)
are deferred. The architecture refactor was prerequisite work that expanded scope.

- **MCP Server** — `acervo-mcp` package (now easier with clean Pipeline)
- **TypeScript SDK** — `acervo-client` npm package
- **LangChain/LlamaIndex** — drop-in memory/retriever
- **S1 intent improvement** — RECALL benchmark (67% → target 85%)
- **S1.5 refactor** — extract from s1_5_graph_update.py into domain layer (currently re-export only)
- **Proxy API dedup** — unify Anthropic/OpenAI code paths
- **Auto-save graph** — remove manual `graph.save()` calls (15 call sites)
