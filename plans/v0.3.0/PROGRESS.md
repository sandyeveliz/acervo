# v0.3.0 Progress Tracker

> Updated: 2026-03-27

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Done
- [!] Blocked

---

## M1 — Foundation (Week 1-2)

### 1.1 Context builder fallback fix (BLOCKER) — Claude ✓ (v0.2.2)
- [x] Eliminate full-history fallback path — removed `if not has_context: return body` guard in `proxy.py:606`
- [x] History window enforced ALWAYS (with or without graph context) — OpenAI + Anthropic
- [x] Added `_window_history_anthropic()` — Anthropic API had no windowing at all
- [x] `history_window` config already existed (default: 2) — no change needed
- [ ] Implement rolling summary when no graph context (deferred — windowing alone may be sufficient, evaluate after trace data)
- [ ] Test: 100-turn conversation, tokens ≤ 600 always (needs trace M1.2 first)

### 1.2 Structured trace per turn — Claude ✓
- [x] Extended `TurnMetric` in `metrics.py` — added compression_ratio, tokens_without_acervo, topic_action, prepare_ms, process_ms, user_message_preview, context_preview
- [x] JSONL persistence — `SessionMetrics._persist_turn()` appends to `.acervo/traces/{session_id}.jsonl`
- [x] Instrument `prepare()` — timing, compression ratio, tokens_without_acervo
- [x] Instrument `process()` — timing (process_ms)
- [x] Trace path init — `.acervo/traces/` auto-created on first write
- [x] `GET /acervo/traces` — list available trace sessions
- [x] `GET /acervo/traces/{id}` — return all turns as JSON array
- [x] `GET /acervo/traces/{id}/summary` — aggregated metrics (avg tokens, compression, timing)
- [x] CLI: `acervo trace show [--session ID]` — table with tokens, ratio, timing per turn
- [ ] CLI: `acervo trace compare` (deferred — needs two sessions to compare)

### 1.3 E2E integration tests — Claude ✓
- [x] YAML scenario framework — 4 scenarios × 50 turns (programming, literature, academic, mixed)
- [x] Baseline comparison — with vs without Acervo (51-78% token savings)
- [x] Token stability — sub-linear growth verified (graph compression works)
- [x] Topic change → context changes correctly (94% context hit rate)
- [x] Return to previous topic → graph restores (verified in return phases)
- [x] Small talk → no phantom entities (0 across all scenarios)
- [x] Cross-session persistence test — graph survives instance destruction + reload
- [x] Prompt variation testing — 4 prompt variants compared (strict wins: 78% recall, 0 phantoms)
- [x] Cost simulation — projected API savings across GPT-4o, Claude Sonnet, GPT-4o-mini
- [x] Multi-format reports: console, markdown, JSON, HTML (interactive dashboard with Chart.js)
- [x] Conversation evidence export — actual messages, context, and entities per turn
- [x] Standalone CLI runner: `python -m tests.integration.run_benchmarks`
- [x] HTML export: `python -m tests.integration.export_report --open`

### 1.3b Warm budget compression + report overhaul — Claude ✓
- [x] Warm token budget capped at 400 (was 1500) — configurable via `Acervo(warm_token_budget=400)`
- [x] Query-relevance boost — entities mentioned in user message get +0.3 score, enter budget first
- [x] Compact chunk format — groups facts by entity on one line, ~40% fewer tokens
- [x] Two real-world scenarios added: 05_saas_founder (101 turns), 06_product_manager (59 turns) — 360 total turns
- [x] Two-tier testing: `quick` mark (01-04, ~15 min) and `full` mark (05-06, ~1 hour)
- [x] Narrative HTML report (`--tier full`) — scissor chart, scorecard bar, token anatomy, cost table, story beats, evidence viewer
- [x] Quick regression dashboard (`--tier quick`) — pass/fail status, compact metrics table, token trends
- [x] Versioned report archive — `reports/archive/v0.2.2-N/` with meta.json (version, git SHA, timestamp)
- [x] Story beat auto-detection (narrative.py) — context resurrections, cost crossover, compression climax, graph milestones, small talk resilience
- [x] Persona metadata in YAML scenarios — persona, persona_role, narrative_hook fields

---

## M2 — User Experience (Week 2-3)

### 2.1 `acervo up` — Sandy ✓
- [x] `acervo up` — proxy foreground with dependency health check (replaces `acervo serve` as primary)
- [x] `acervo up --dev` — full dev stack with multiplexed tagged logs (proxy + studio + web + ollama)
- [x] Auto-detect Ollama, LM Studio, Acervo Studio (binary detection + health checks + sibling dir scan)
- [x] `ServicesConfig` in config.toml: ports + studio_path
- [x] `acervo status` shows dependency health when services configured
- [x] `acervo serve` kept as backwards-compatible alias
- [~] ~~`acervo down`~~ — removed (not needed, both modes run foreground with Ctrl+C)
- [~] ~~First-run wizard~~ — removed (config defaults are sufficient, use `acervo config set`)

### 2.2 Graph inspection CLI — Claude ✓
- [x] `acervo graph show` — list nodes with table (ID, Label, Type, Kind, Layer, Facts, Edges)
- [x] `acervo graph show <id>` — full node detail (facts, edges, linked files, attributes)
- [x] `acervo graph search <query>` — text search across labels and fact content
- [x] `acervo graph delete <id>` — delete node + edges with confirmation prompt (--yes to skip)
- [x] `acervo graph merge <id1> <id2>` — merge two nodes with preview and confirmation
- [x] REST endpoints: GET /acervo/graph/nodes, GET /nodes/{id}, GET /search, DELETE /nodes/{id}, POST /merge
- [x] --json flag on all read commands for piping
- [x] --kind filter on show and search

---

## M3 — Document Ingestion (Week 3-4)

### 3.1-3.3 Document chunk ingestion — Claude ✓
- [x] `graph.py`: `chunk_ids` field on all node types (entity, file, symbol, section) + `link_chunks()`, `get_chunks_for_node()`, `clear_chunks()`, `get_nodes_with_chunks()` methods. Migration adds `chunk_ids: []` to legacy nodes.
- [x] `semantic_enricher.py`: already returns chunk IDs with embeddings (no changes needed)
- [x] `indexer.py`: replaced broken `_store_embeddings()` (only last chunk survived) with `_store_and_link_chunks()` — stores all chunks in one call, links chunk_ids to file + section nodes by line range
- [x] `structural_parser.py`: .md chunking by heading already works (verified by tests)
- [x] `vector_store.py`: extended `index_file_chunks()` (accepts chunk_ids, pre-computed embeddings, extra_metadata), added `search_by_chunk_ids()` (cosine similarity ranking), added `remove_by_chunk_ids()`
- [x] `facade.py`: node-scoped chunk retrieval in S2 Gather — collects chunk_ids from activated nodes, searches within them via `search_by_chunk_ids()`, boosts score +0.15. Added `index_document()` and `delete_document()` methods. Added chunk trace metrics to debug dict.
- [x] Proxy: `POST /acervo/documents` — upload .md + full index pipeline
- [x] Proxy: `GET /acervo/documents` — list documents with chunk counts
- [x] Proxy: `GET /acervo/documents/{id}` — detail with sections and chunk_ids
- [x] Proxy: `DELETE /acervo/documents/{id}` — remove doc + chunks + nodes
- [x] Proxy: re-enabled embeddings + vector store in `_init_acervo()` (gated on config)
- [x] CLI: `acervo index` now initializes vector store and reports chunk linkage
- [x] Test: index .md → node with chunk_ids created (18 tests, all passing)
- [x] Test: search_by_chunk_ids returns scoped results
- [x] Test: delete document → node + chunks + embeddings gone
- [ ] Test: question about .md content → chunks in context (needs live LLM, integration test)

---

## M4 — Demonstration (Week 4-5)

### 4.1 README "Acervo in Action" — Claude ✓
- [x] `scripts/generate_demo.py` — extracts real turn data from benchmark JSONs, generates formatted tables (console + markdown)
- [x] 50-turn scenarios across 4 domains (M1.3 covers this)
- [x] 2 real-world scenarios: 101-turn SaaS founder + 59-turn product manager (M1.3b)
- [x] Summary tables with tokens/turn, compression ratio (HTML report)
- [x] Conversation evidence with actual messages (HTML report)
- [x] Publishable narrative report with scissor chart, scorecard, anatomy, costs (M1.3b)
- [x] "Acervo in Action" section in README with real benchmark data: turn-by-turn walkthrough (developer scenario), 101-turn progression table (SaaS founder), cross-scenario summary table (6 scenarios, 360 turns)
- [x] Updated project status table (M3 features + benchmarks marked as working)

### 4.2 Benchmark script — ✓ (completed in M1.3)
- [x] `tests/integration/run_benchmarks.py` — standalone CLI runner
- [x] Aggregate metrics from scenario runs
- [x] Baseline comparison (with vs without Acervo)
- [x] Console table + JSON + Markdown + HTML output
- [x] Cost estimation (GPT-4o, Claude Sonnet, GPT-4o-mini pricing)
- [x] Versioned archive for tracking improvements across releases (M1.3b)

---

## M5 — Model & Retrieval (Week 5-6)

### 5.1 Chunk-aware retrieval — Claude ✓
- [x] Specificity classifier (regex + keywords) — `acervo/specificity.py`, 15 specific + 8 conceptual patterns
- [x] Filter vector search by activated node (not global) — gated behind classifier in `facade.py`
- [x] Limit to top 3 chunks per node — `n_results=3` in `search_by_chunk_ids()`
- [x] Tests: 31 tests (15 specific, 12 conceptual, 4 edge cases) — all passing
- [x] Trace metrics: `query_specificity` field in debug dict

### 5.2 Fine-tune improvements — Sandy
- [ ] 100 "don't extract" examples
- [ ] 50 dedup examples
- [ ] 50 chunk-aware examples
- [ ] Multi-turn training conversations
- [ ] Spanish rioplatense domain
- [ ] Benchmark before/after retrain

---

## M6 — Polish (Week 6-7)

### 6.1 Logs — Claude ✓
- [x] Configurable log levels — `--log-level trace|debug|info|warning|error`
- [x] Default: one line per turn (INFO: `prepare done: topic=X tokens=Y Zms`)
- [x] Debug: entities + topic decisions (`S1 topic: action=X`, `S2 gather: N nodes`)
- [x] Trace: custom TRACE=5 level, full prompts + timestamps
- [x] Colors with `--no-color` flag — `ColorFormatter` with ANSI codes
- [x] `-v` shorthand for `--log-level debug`
- [x] `log_config.py` module: `setup_logging(level, color)`, quiets noisy third-party loggers

### 6.2 Error handling — Claude ✓
- [x] LLM down → graceful degradation (already in S1 `_fallback_result`, extractors return empty, proxy returns 502)
- [x] Corrupt graph → `acervo graph repair` — fixes missing fields, orphan edges, duplicates
- [x] Invalid extractor JSON → retry with temperature=0.0 in `s1_unified.py`
- [x] Configurable timeouts — `TimeoutsConfig` in config.toml, `OllamaEmbedder` and `OpenAIClient` accept `timeout` param

### 6.3 Documentation — Claude ✓
- [x] README with "Acervo in Action" (done in M4)
- [x] Getting Started with `acervo up` — updated `docs/getting-started.md`
- [x] Trace format docs — `docs/traces.md`
- [x] Graph CLI docs — `docs/graph-cli.md`
- [x] Document ingestion docs — `docs/document-ingestion.md`
- [x] Benchmark reports on GitHub Pages — `docs/benchmarks.md` index + `docs/benchmarks/v{X}/index.html` static reports served at `sandyeveliz.github.io/acervo/benchmarks/`
- [ ] Changelog (final — needs all milestones complete)

---

## Release Criteria
- [ ] Constant-tokens test passes at 50 turns
- [ ] `acervo up` works on Linux/macOS
- [ ] `acervo index --path file.md` indexes + links chunks to graph
- [ ] Question about indexed .md → node-scoped chunks in context
- [ ] README has "Acervo in Action" with real data
- [ ] `acervo benchmark` produces reproducible results
- [ ] Zero crashes in 100-turn session
- [ ] Complete changelog
- [ ] Published on PyPI
- [ ] Benchmark report published on GitHub Pages for this tag
