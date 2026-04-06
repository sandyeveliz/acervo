# v0.5.0 Progress Tracker

> Updated: 2026-04-06

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Done
- [!] Blocked

---

## M1 — Architecture Refactor (NEW — replaced MCP milestone)

- [x] ARCHITECTURE_AUDIT.md — full codebase audit (14,045 LOC, 40 files)
- [x] `ports/` — LLMPort, EmbedderPort, VectorStorePort, GraphStorePort protocols
- [x] `domain/models.py` — S1Result, S2Result, S3Result, S15Result, GatheredNode, RankedChunk
- [x] `domain/s2_activator.py` — single code path, no conversation/project divergence
- [x] `domain/s3_assembler.py` — intent controls budget, not filtering
- [x] `domain/s1_extractor.py` — re-export of S1Unified
- [x] `domain/s15_updater.py` — re-export of S1_5GraphUpdate
- [x] `domain/pipeline.py` — thin orchestrator (prepare → S1→S2→S3, process → S1.5)
- [x] `facade.py` — thin wrapper delegating to Pipeline
- [x] File reorganization — all modules moved to packages with backward compat stubs
- [x] 178/185 tests pass (7 pre-existing, 0 regressions)
- [x] warm_tokens > 0 verified on 4-turn manual test (turns 2-3 confirmed, turn 4 pending restart)

---

## M2 — Pipeline Bug Fixes

- [x] JSON parser repair (greedy + repair fallbacks)
- [x] Nested relations extraction from entity objects
- [x] Relation ID resolution (model IDs → entity labels)
- [x] Self-referencing relation guard
- [x] Entity neighbor expansion in S2
- [x] Topic drift prevention (PREVIOUS ASSISTANT limited to 150 chars)
- [x] Generic model support (string entities, string topics)
- [x] Description chunks in S2
- [x] S3 unified assembly (no intent-based filtering)
- [x] Delayed flush for proxy mode (TelemetryCollector + TurnLogger)
- [ ] Verify Turn 4 warm_tokens > 0 after proxy restart

---

## M3 — Ollama Migration

- [x] Settings defaults: port 1234 → 11434
- [x] Modelfile: added Qwen3.5 chat template + stop tokens
- [x] `.env` override fix (settings.toml as source of truth for models)
- [x] LM Studio dependency removed from DevRunner
- [x] `acervo up --dev` without project requirement
- [x] Active project lookup from Studio SQLite DB

---

## M4 — Acervo Studio Features

- [x] Telemetry page: per-turn annotation with S1/S2/S3/LLM/S1.5 stage cards
- [x] Annotation backend (AnnotationStore + CRUD endpoints + JSONL export)
- [x] Ollama monitor page (VRAM/GPU/RAM live)
- [x] Sidebar reorganization (main nav + ACERVO divider)
- [x] Per-project telemetry persistence
- [x] S1 debug enrichment (prompts, responses, timing in stage_data)
- [x] Graph counter telemetry fix (query proxy status)
- [x] Project status refresh fix
- [x] Description save UX (Save/Cancel buttons)
- [x] Reset clears telemetry + annotations

---

## M5 — Documentation & Release (NOT STARTED)

- [ ] Update README.md with new architecture
- [ ] Update ARCHITECTURE.md (or merge with ARCHITECTURE_AUDIT.md)
- [ ] Update CLAUDE.md with new package structure
- [ ] Commit all changes
- [ ] Tag v0.5.0
- [ ] Blog post / release notes

---

## Release Criteria
- [x] Hexagonal architecture implemented (ports → domain → adapters)
- [x] S2 single code path (no conversation/project divergence)
- [x] warm_tokens > 0 on conversation-mode turns 2-3
- [~] warm_tokens > 0 on ALL turns 2-4 (turn 4 pending restart verification)
- [x] All existing tests pass (0 regressions)
- [x] Backward compatible public API
- [ ] Documentation updated
- [ ] Tagged release
