# v0.3.0 Changelog (draft)

> Working notes for the final changelog. Each entry summarizes what was done.

---

## M1.1 — Context builder fallback fix (2026-03-25)

### Fixed
- **Eliminated full-history fallback** — removed the `if not has_context: return body` guard in `proxy.py:606` that bypassed the sliding window when the graph had no context, causing unbounded token growth in early turns.
- **History window enforced ALWAYS** — both OpenAI and Anthropic API paths now apply the sliding window (default: 2 turn pairs) regardless of graph context state.
- **Added `_window_history_anthropic()`** — Anthropic API path had no windowing at all; now uses the same `history_window` config as OpenAI.

### Result
- Token count stays bounded from turn 1, even before the graph has accumulated knowledge.

---

## M1.2 — Structured trace per turn (2026-03-25)

### Added
- **Extended `TurnMetric`** — added `compression_ratio`, `tokens_without_acervo`, `topic_action`, `prepare_ms`, `process_ms`, `user_message_preview`, `context_preview`.
- **JSONL persistence** — `SessionMetrics._persist_turn()` appends one JSON line per turn to `.acervo/traces/{session_id}.jsonl`. Auto-creates trace directory.
- **Timing instrumentation** — `prepare()` and `process()` are timed with `time.perf_counter()`.
- **REST endpoints** — `GET /acervo/traces` (list sessions), `GET /acervo/traces/{id}` (all turns), `GET /acervo/traces/{id}/summary` (aggregated metrics).
- **CLI** — `acervo trace show [--session ID]` displays table with tokens, compression ratio, timing per turn.

### Result
- Every turn is auditable. Trace files are the source of truth for benchmarking and debugging.

---

## M1.3 — E2E integration test framework (2026-03-26)

### Added

- **YAML scenario framework** — declarative test scenarios with per-turn expectations (entities, relations, facts, context checkpoints, max_new_nodes guard).

- **4 scenarios × 50 turns** covering programming, literature/comics, academic/NLP, and mixed domains (cooking, travel, sports, music, work). 200 total turns tested.

- **Results across all scenarios:**

  | Scenario | Token Savings | Nodes | Edges | Phantoms | Context Hits |
  |----------|:---:|:---:|:---:|:---:|:---:|
  | Programming | 50.9% | 21 | 46 | 0 | 94% |
  | Literature | 78.1% | 43 | 35 | 0 | 94% |
  | Academic | 69.2% | 15 | 24 | 0 | 84% |
  | Mixed | 77.9% | 23 | 31 | 0 | 82% |

- **Hard vs soft assertions** — hard assertions verify infrastructure (token stability, graph integrity, no phantom entities, trace persistence). Soft assertions measure model+prompt extraction quality (entity recall, context mentions) with a 30% pass threshold to account for LLM non-determinism.

- **Cross-session persistence test** — creates graph in session 1, destroys instance, loads from same persist_path in session 2, verifies graph intact + context recall works.

- **Prompt variation testing** — compares 4 extraction prompt variants (default, strict, verbose, structured) on the same scenario. `strict` prompt won with 78% entity recall and 0 phantoms, adopted as new default S1 prompt.

- **Cost simulation** — projects API savings based on GPT-4o ($2.50/1M), Claude Sonnet ($3/1M), GPT-4o-mini ($0.15/1M) pricing. Shows per-scenario and extrapolated (1000 conversations) costs.

- **Multi-format reports:**
  - Console — compact CI-friendly table
  - Markdown — per-scenario detailed report
  - JSON — machine-readable per-turn data with messages, context, entities
  - HTML — interactive dashboard with Chart.js charts, cost tables, conversation evidence viewer, image export buttons

- **Standalone CLI runner** — `python -m tests.integration.run_benchmarks [--scenario X] [--format html]`
- **HTML export from existing data** — `python -m tests.integration.export_report --open`

### Files created
- `tests/integration/framework.py` — YAML loader, ScenarioRunner, hard assertions
- `tests/integration/metrics.py` — TurnResult, ScenarioResult, EntityExpectation, Checkpoint dataclasses
- `tests/integration/reporter.py` — console, markdown, JSON, HTML report generators
- `tests/integration/test_scenarios.py` — pytest parametrized over discovered YAML scenarios
- `tests/integration/test_prompts.py` — prompt variant comparison tests
- `tests/integration/test_persistence.py` — cross-session graph persistence test
- `tests/integration/run_benchmarks.py` — standalone CLI runner (no pytest required)
- `tests/integration/export_report.py` — generate HTML from existing JSON results
- `tests/integration/scenarios/01_programming.yaml` — 50-turn developer workflow
- `tests/integration/scenarios/02_literature.yaml` — 50-turn comics/manga/literature
- `tests/integration/scenarios/03_academic.yaml` — 50-turn NER thesis/databases
- `tests/integration/scenarios/04_mixed.yaml` — 50-turn multi-domain chaos test
- `tests/integration/scenarios/prompt_test.yaml` — 10-turn prompt comparison scenario

### Files modified
- `acervo/s1_unified.py` — updated S1 system prompt to `strict` variant (78% recall, 0 phantoms)
- `pyproject.toml` — added `pyyaml` to dev dependencies, `benchmark` pytest marker
- `tests/integration/conftest.py` — `e2e_memory` fixture with proper `.acervo/` directory structure

### Files deleted
- `tests/integration/test_pipeline.py` — replaced by YAML framework
- `tests/integration/test_tui.py` — obsolete TUI tests

### Design decisions
- **YAML over Python** — scenarios are data, not code. Easier to add new scenarios, review in PRs, and maintain.
- **Baseline = cumulative full-history tokens** — no LLM call, just `count_tokens()` on what would be sent without Acervo. Fair comparison.
- **Soft threshold at 30%** — the fine-tuned model is deliberately conservative (rejects inferred entities). 30% catches catastrophic regressions while tolerating LLM non-determinism.
- **Token stability check** — `last_third_avg < max(first_third_avg, 150) * 4` — floor of 150tk prevents false positives from near-zero early turns.
- **prompt_test.yaml excluded from main tests** — short scenario only used for prompt comparison, not full 50-turn assertions.

---

## M1.3b — Warm budget compression + report overhaul (2026-03-27)

### Changed

- **Warm token budget capped at 400** (was hardcoded 1500) — configurable via `Acervo(warm_token_budget=400)`. Forces `select_chunks_by_budget()` to pick only the most relevant facts.
- **Query-relevance boost** — when the user mentions an entity by name, its chunks get +0.3 score (cap 1.0), ensuring directly-referenced entities enter the budget first.
- **Compact chunk format** — `format_chunks_compact()` groups facts by entity on a single line (`**Sandy**: programmer; lives in Cipolletti`) instead of one block per fact. ~40% fewer tokens for the same information.

### Added

- **Two real-world scenarios:**
  - `05_saas_founder_100turns.yaml` — Carlos building Menuboard (SaaS), 101 turns across 3 simulated weeks. Tests long-term recall, tech pivots, personal tangents.
  - `06_product_manager_real.yaml` — Ana at Enviotech (logistics), 59 turns. Tests non-technical business entities (companies, metrics, people) in prose-heavy conversation.

- **Two-tier testing:**
  - `pytest -m "integration and quick"` — runs scenarios 01-04 (~15 min) for fast dev iteration.
  - `pytest -m "integration and full"` — runs scenarios 05-06 (~40 min) for publishable reports.
  - `pytest -m integration` — runs all (backward compatible).

- **Narrative HTML report** (`--tier full`) — clean data report with:
  - Summary cards (savings, turns, hits, phantoms, entities)
  - Scissor chart — longest scenario showing Acervo vs full history divergence over turns
  - Scorecard — grouped bar comparing savings % and context hit % across all scenarios
  - Token anatomy — sampled stacked bar (every ~10 turns) showing warm/hot/system vs baseline
  - Per-scenario breakdown table + per-scenario token trend charts
  - Cost estimation table (GPT-4o, Claude Sonnet, GPT-4o-mini) with projected x1000 conversations
  - Notable moments table — auto-detected story beats
  - Conversation evidence — collapsible per-turn details with messages, context, entities

- **Quick regression dashboard** (`--tier quick`) — pass/fail status with traffic light, compact metrics table, token trend charts. Designed to scan in 5 seconds.

- **Versioned report archive:**
  - `python -m tests.integration.export_report --tier full` auto-archives to `reports/archive/v{version}-{seq}/`
  - Each archive contains: report.html, source JSONs, meta.json (version, git SHA, timestamp, avg_savings)
  - Auto-version from pyproject.toml + sequence number (v0.2.2-1, v0.2.2-2, ...)

- **Story beat detection** (`narrative.py`) — pure Python analysis over turn data, no LLM calls:
  - Context resurrection: topic dormant for 5+ turns, then context_hit=True on return
  - Cost crossover: turn where cumulative Acervo cost drops below 50% of baseline
  - Compression climax: turn with highest savings_pct
  - Graph milestone: node_count crossing 10/25/50/100 thresholds
  - Small talk resilience: 3+ consecutive tangent turns with clean graph

- **Persona metadata** in YAML scenarios — optional `persona`, `persona_role`, `narrative_hook` fields for richer report display.

### Files created
- `tests/integration/narrative.py` — StoryBeat/ScenarioNarrative dataclasses, detect_beats(), build_narrative()
- `tests/integration/scenarios/05_saas_founder_100turns.yaml` — 101-turn real-world SaaS scenario
- `tests/integration/scenarios/06_product_manager_real.yaml` — 59-turn non-technical PM scenario

### Files modified
- `tests/integration/reporter.py` — added `html_report_narrative()`, `html_report_quick()`
- `tests/integration/export_report.py` — added `--tier`, `--version`, `--no-archive`, archive logic, scenario meta loading
- `tests/integration/test_scenarios.py` — added `test_scenario_quick`, `test_scenario_full` with quick/full marks
- `tests/integration/metrics.py` — added `persona`, `persona_role`, `narrative_hook` to Scenario dataclass
- `tests/integration/framework.py` — load persona fields from YAML
- `tests/integration/scenarios/01-04` — added persona/narrative_hook metadata
- `pyproject.toml` — registered `quick` and `full` pytest markers
- `acervo/facade.py` — warm_token_budget param, query-relevance boost, compact formatter
- `acervo/context_builder.py` — added `format_chunks_compact()`

### Result
- Token savings improved from ~55% to ~76% average across all scenarios.
- 360 total turns tested (was 200) with 6 scenarios (was 4).
- Reports are versioned and archivable for tracking improvements across releases.
- Two report tiers: quick for dev iteration, full for sharing.

### Usage
```bash
# Quick check
pytest tests/integration/test_scenarios.py -m "integration and quick" -v -s
python -m tests.integration.export_report --tier quick --open

# Full publishable report
pytest tests/integration/test_scenarios.py -m integration -v -s
python -m tests.integration.export_report --tier full --open
```

---

## M2.1 — `acervo up` (2026-03-26)

### Added

- **`acervo up`** command — starts the Acervo proxy in foreground with a dependency health check banner showing Ollama and LM Studio status before the proxy starts. Same behavior as `acervo serve` but with better DX.

- **`acervo up --dev`** — dev mode that starts all services (Ollama, proxy, Studio backend, Studio frontend) in a single terminal with multiplexed tagged logs (`[ollama]`, `[proxy]`, `[studio]`, `[web]`). Like docker-compose: Ctrl+C stops everything. Uses `asyncio.create_subprocess_exec` with stream piping.

- **Service auto-detection** — detects Ollama binary (`shutil.which`), LM Studio (HTTP health check on `/v1/models`), Acervo Studio (config path > `importlib.find_spec` > sibling directory scan for `AVS-Agents/` or `acervo-studio/`), npm for frontend.

- **`ServicesConfig`** in `config.py` — flat dataclass with `ollama_port`, `lmstudio_port`, `studio_path`, `studio_port`, `frontend_port`. Loaded from `[acervo.services]` section in config.toml.

- **`acervo status`** now shows dependency health (Ollama/LM Studio) when `[acervo.services]` is configured.

### Files created
- `acervo/services.py` — `DevRunner`, `check_health()`, `check_dependencies()`, `detect_binary()`, `detect_studio_path()`, `format_dep_check()`

### Files modified
- `acervo/config.py` — added `ServicesConfig` dataclass, TOML load/save for `[acervo.services]`
- `acervo/cli.py` — added `cmd_up` with `--dev` flag, updated `cmd_status` to show dep health
- `acervo/project.py` — added `run/` to `.gitignore` template

### Design decisions
- **No `acervo down`** — both modes run in foreground, Ctrl+C is sufficient. No PID files needed.
- **No wizard** — config defaults work out of the box. Studio path auto-detected or set via `acervo config set services.studio_path`.
- **`acervo serve` kept** — backwards compatible, no dep check banner. `acervo up` is the new recommended way.
- **Windows-first** — handles `.cmd` scripts (npm.cmd), DETACHED_PROCESS flags, cp1252 encoding (ASCII-only output).
- **stdlib only** — no new dependencies. Health checks use `urllib.request`, process management uses `asyncio.create_subprocess_exec`.

---

## M2.2 — Graph inspection & editing (2026-03-26)

### Added

- **`acervo graph show`** — list all nodes in table format (ID, Label, Type, Kind, Layer, Facts count, Edges count). Supports `--kind` filter and `--json` output.

- **`acervo graph show <id>`** — full detail view of a single node: facts with source/date, edges with direction, linked files, attributes.

- **`acervo graph search <query>`** — text search across node labels and fact content. Shows match location (label vs specific fact text).

- **`acervo graph delete <id>`** — delete a node and all its edges. Shows preview and asks confirmation (bypass with `--yes`).

- **`acervo graph merge <id1> <id2>`** — merge two nodes (keep first, absorb second's facts + edges). Preview + confirmation.

- **REST endpoints** for Studio integration:
  - `GET /acervo/graph/nodes` — list (query: `kind`, `limit`)
  - `GET /acervo/graph/nodes/{id}` — detail with edges + linked files
  - `GET /acervo/graph/search?q=...&kind=...` — search
  - `DELETE /acervo/graph/nodes/{id}` — delete
  - `POST /acervo/graph/merge` — body: `{"keep": "id1", "absorb": "id2"}`

### Files created
- `acervo/graph_cli.py` — CLI command implementations + formatters (show, search, delete, merge)

### Files modified
- `acervo/cli.py` — added `graph` subcommand with `show`, `search`, `delete`, `merge` sub-subcommands
- `acervo/proxy.py` — added 5 REST endpoints for graph CRUD

### Design decisions
- **All graph operations already existed in TopicGraph** — this was purely a wiring task. No new graph logic.
- **Separate `graph_cli.py` module** — keeps cli.py manageable (it was already 500+ lines).
- **ASCII-only output** — no Unicode box-drawing to avoid Windows cp1252 encoding issues.
- **Confirmation prompts** — delete and merge require `y` confirmation (or `--yes` flag) to prevent accidents.

---

## M3 — Document ingestion with chunks linked to graph (2026-03-27)

### Added

- **`chunk_ids` field on all graph nodes** — entities, files, symbols, and sections can now track linked document chunks. Migration adds `chunk_ids: []` to legacy nodes on load.

- **Graph chunk methods** — `link_chunks(node_id, chunk_ids)`, `get_chunks_for_node(node_id)`, `clear_chunks(node_id)`, `get_nodes_with_chunks()`.

- **Node-scoped chunk retrieval** in `prepare()` S2 — when activated nodes have `chunk_ids`, searches within those chunks (via `search_by_chunk_ids`) instead of global RAG. Scoped results get +0.15 score boost. ~200tk context instead of ~2500tk global RAG.

- **`search_by_chunk_ids(chunk_ids, query_embedding, n_results)`** — new vector store method that retrieves specific chunks and ranks by cosine similarity.

- **`remove_by_chunk_ids(chunk_ids)`** — targeted chunk deletion for document management.

- **Document management API:**
  - `POST /acervo/documents` — upload .md file → full index pipeline → chunks linked to graph
  - `GET /acervo/documents` — list indexed documents with chunk counts
  - `GET /acervo/documents/{id}` — document detail with sections and chunk_ids
  - `DELETE /acervo/documents/{id}` — remove document + chunks from ChromaDB + graph nodes

- **`Acervo.index_document(file_path)`** — single-file indexer pipeline wrapping parse → enrich → store → link.

- **`Acervo.delete_document(document_id)`** — removes document, chunks, and child nodes.

- **Chunk trace metrics** — debug dict now includes `chunks.documents_with_chunks_activated`, `chunks_retrieved`, `chunks_total_on_activated_nodes`, `retrieval_scope`.

- **Re-enabled embeddings + vector store in proxy** — gated on config (`embeddings.url` + `embeddings.model`).

- **CLI vector store support** — `acervo index` now initializes ChromaDB vector store when embedder is available, reports chunk linkage stats.

### Fixed

- **Critical bug: only last chunk survived indexing** — `_store_embeddings()` called `index_file_chunks(file_path, [chunk.content])` once per chunk, but each call first removed ALL existing chunks for that file. Replaced with `_store_and_link_chunks()` that stores all chunks in one call.

- **`index_file_chunks()` extended** — now accepts `chunk_ids` (custom IDs), `embeddings` (pre-computed, skip re-embedding), and `extra_metadata`. Returns list of stored IDs. Backward compatible.

- **numpy array truthiness in ChromaDB results** — fixed `ValueError` when checking embeddings returned by ChromaDB (numpy arrays don't support Python truthiness).

### Files modified
- `acervo/graph.py` — chunk_ids field, migration, 4 new methods, _remove_file_children cleanup
- `acervo/vector_store.py` — extended index_file_chunks, search_by_chunk_ids, remove_by_chunk_ids, _cosine_similarity
- `acervo/indexer.py` — _store_and_link_chunks replaces broken _store_embeddings
- `acervo/facade.py` — node-scoped retrieval in S2, index_document(), delete_document(), chunk trace metrics
- `acervo/proxy.py` — 4 document endpoints, re-enabled embeddings
- `acervo/cli.py` — vector store init in cmd_index, chunk linkage reporting

### Files created
- `tests/test_document_ingestion.py` — 18 tests (graph chunk_ids, vector store operations, indexer linkage, deletion)

---

## M4.1 — README "Acervo in Action" (2026-03-27)

### Added

- **"Acervo in Action" section in README** — real benchmark data from 360 turns across 6 scenarios, replacing abstract comparisons with actual turn-by-turn data:
  - Developer workflow walkthrough (turns 1, 10, 25, 35, 50) showing token counts, graph growth, topic switching, and context restoration
  - 101-turn SaaS founder progression table (turn 1 → 100, 17tk → 5,157tk baseline vs 6tk → 490tk Acervo)
  - Cross-scenario summary table (6 scenarios: 67-82% savings, 84-98% context hit rate)

- **`scripts/generate_demo.py`** — generates README demo tables from existing benchmark JSON data. Supports `--scenario`, `--turns`, `--summary-only`, `--list` flags. ASCII-only output for Windows compatibility.

- **Updated project status table** — M3 features (document ingestion, node-scoped retrieval, document API) and benchmarks marked as working. Moved progressive retrieval and Docker to v0.4.

### Files created
- `scripts/generate_demo.py` — demo data extraction and formatting script

### Files modified
- `README.md` — added "Acervo in Action" section, updated project status table

---

## M5.1 — Chunk-aware retrieval (2026-03-27)

### Added

- **Specificity classifier** (`acervo/specificity.py`) — regex + keyword heuristic that classifies user messages as "specific" (needs chunks) or "conceptual" (summary only). 15 specific patterns (code, numbers, dates, "show me", errors, config) and 8 conceptual patterns (explain, why, overview, compare, opinion).

- **Gated node-scoped retrieval** — chunk retrieval in `facade.py` S2 Gather is now conditional on the specificity classifier. Conceptual queries skip chunk retrieval entirely, keeping context lean (~100 tokens). Specific queries fetch top 3 chunks from activated nodes (~400 tokens).

- **Trace metric: `query_specificity`** — debug dict now includes the classifier's decision for each turn.

- **31 tests** (`tests/test_specificity.py`) — 15 specific queries, 12 conceptual queries, 4 edge cases.

### Files created
- `acervo/specificity.py` — specificity classifier module
- `tests/test_specificity.py` — 31 parametrized tests

### Files modified
- `acervo/facade.py` — import classifier, gate chunk retrieval, add `query_specificity` to trace

---

## M6.1 — Configurable logs (2026-03-27)

### Added

- **`--log-level` flag** — global CLI flag accepting `trace|debug|info|warning|error`. Default: `warning`.
- **Custom TRACE level** (level 5, below DEBUG) — for full prompt/response logging.
- **`ColorFormatter`** — ANSI color-coded log output (green=INFO, blue=DEBUG, cyan=TRACE, yellow=WARNING, red=ERROR). Auto-detects TTY, with `--no-color` override.
- **`-v` shorthand** — equivalent to `--log-level debug`.
- **Structured log messages in facade.py**:
  - INFO: `prepare done: topic=X tokens=Y (warm=W hot=H) Zms`
  - DEBUG: `S1 topic: action=X label=Y`, `S1 extraction: N entities, N relations, N facts`, `S2 gather: N nodes, N chunk_ids, specificity=X`
- **Third-party logger quieting** — httpcore, httpx, uvicorn.access, chromadb are silenced unless TRACE level.

### Files created
- `acervo/log_config.py` — `setup_logging(level, color)`, TRACE level, ColorFormatter

### Files modified
- `acervo/cli.py` — replaced `logging.basicConfig` with `setup_logging`, added `--log-level`, `--no-color` flags
- `acervo/facade.py` — added INFO/DEBUG log lines at key pipeline stages

---

## M6.2 — Error handling (2026-03-27)

### Added

- **`acervo graph repair`** — CLI command + `TopicGraph.repair()` method. Detects and fixes:
  - Nodes missing required fields (id, label, type, kind, facts, chunk_ids)
  - Edges referencing non-existent nodes
  - Duplicate edges (same source + target + relation)
  - Saves graph after repairs.

- **S1 Unified JSON retry** — when `_parse_s1_response()` fails to parse JSON, retries the LLM call once with `temperature=0.0` before falling back to empty result.

- **Configurable timeouts** — `TimeoutsConfig` dataclass with per-phase defaults:
  - `llm_chat: 120`, `embedding: 30`, `s1_unified: 60`, `s1_5_update: 60`, `vector_search: 10`
  - Loadable from `[acervo.timeouts]` in config.toml
  - `OllamaEmbedder` and `OpenAIClient` accept `timeout` parameter (was hardcoded 120s)

### Files modified
- `acervo/graph.py` — added `repair()` method
- `acervo/cli.py` — added `graph repair` subcommand
- `acervo/s1_unified.py` — JSON parse retry with lower temperature
- `acervo/config.py` — added `TimeoutsConfig`, loaded from `[acervo.timeouts]`
- `acervo/openai_client.py` — `timeout` parameter on `OllamaEmbedder` and `OpenAIClient`

---

## M6.3 — Documentation (2026-03-27)

### Added

- **Updated `docs/getting-started.md`** — rewritten for `acervo up` workflow, SDK using `prepare()/process()` API, logging section, prerequisites, core concepts.
- **`docs/graph-cli.md`** — full graph CLI reference (show, search, delete, merge, repair) + REST API table.
- **`docs/traces.md`** — trace format reference (JSONL fields), CLI usage, REST API, debug dict structure.
- **`docs/document-ingestion.md`** — ingestion pipeline, CLI usage, REST API, specificity classifier, configuration.

### Files created
- `docs/graph-cli.md`
- `docs/traces.md`
- `docs/document-ingestion.md`

### Files modified
- `docs/getting-started.md` — full rewrite

---

## M6.3b — Benchmark reports on GitHub Pages (2026-03-27)

### Added

- **Benchmark reports published on GitHub Pages** — versioned HTML reports served as static files alongside the MkDocs documentation site at `sandyeveliz.github.io/acervo/benchmarks/`.

- **`docs/benchmarks.md`** — index page with version table (date, scenarios, turns, avg savings, link), scenario descriptions, and instructions for generating new reports.

- **Versioned report folders** — `docs/benchmarks/v{X.Y.Z-N}/index.html` with self-contained HTML reports (Chart.js via CDN). Clean URLs: `/benchmarks/v0.2.2-3/` serves the report directly.

- **`mkdocs.yml`** — added "Benchmarks" to navigation. MkDocs copies static HTML files to the site output automatically.

- **Convention: publish benchmarks per release tag** — from v0.3.0 onward, every tagged release includes a benchmark report on GitHub Pages for public reference and blog linking.

### Report versions published
| Version | Turns | Scenarios | Avg Savings |
|---------|-------|-----------|-------------|
| v0.2.2-3 | 360 | 6 | 76.1% |
| v0.2.2-2 | 360 | 6 | 76.1% |
| v0.2.2-1 | 360 | 6 | 76.1% |

### Files created
- `docs/benchmarks.md` — index page with version table and scenario descriptions
- `docs/benchmarks/v0.2.2-1/index.html` — archived report
- `docs/benchmarks/v0.2.2-2/index.html` — archived report
- `docs/benchmarks/v0.2.2-3/index.html` — current full benchmark report (418KB)
- `docs/benchmarks/v0.2.2-1/meta.json` — version metadata
- `docs/benchmarks/v0.2.2-2/meta.json` — version metadata

### Files modified
- `mkdocs.yml` — added Benchmarks to nav
