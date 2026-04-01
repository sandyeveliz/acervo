# Changelog

All notable changes to this project will be documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

## [0.4.0] - 2026-04-01

### Architecture
- **Intent-aware context pipeline** — S1 classifies user intent as `overview`, `specific`, or `chat`. Overview questions use synthesis/entity nodes only (no section/symbol noise). Specific questions use vector search + section nodes. Chat messages get minimal or no context.
- **Curate + Synthesize pipeline** — Two new post-indexing stages: `curate` extracts entities from indexed content via LLM batch analysis, `synthesize` generates a project overview node summarizing the entire project. Both persist to the knowledge graph.
- **S1.5 memory extraction** — `process()` extracts entities and facts from user messages into the graph for later recall. Runs after every turn to build persistent memory.
- **Context builder fallback fix** — Fixed `UnboundLocalError` on `conversation_chunks`/`verified_chunks` that silently crashed the context pipeline when the graph had data but intent classification diverged.
- **Proxy graph reload** — Added `/acervo/reload-graph` endpoint to prevent the proxy's in-memory graph from overwriting disk state after external indexing.

### Ingestion
- **`.epub` ingestion** — Parses epub files via ebooklib, extracts chapters as sections, creates graph nodes with summaries. Optional dependency: `pip install acervo[epub]`.
- **`.pdf` ingestion** — Parses PDF files via PyMuPDF, extracts pages as sections. Optional dependency: `pip install acervo[pdf]`.
- **`.txt` ingestion** — Parses plain text files with paragraph-based sectioning.
- **Paragraph-based prose chunking** — `_chunk_prose()` splits large sections (epub chapters, PDF pages) into ~500-token paragraph clusters instead of storing entire chapters as single chunks.
- **`acervo chunks` CLI** — `stats` (chunk count, avg size, distribution), `list` (all chunks with metadata), `show <id>` (full chunk content), `search <query>` (semantic search across chunks).

### Curation & Synthesis
- **`acervo curate`** — LLM analyzes indexed files in batches, extracts entities (people, technologies, concepts) with types from a controlled ontology, and creates relationship edges between entities and files.
- **`acervo synthesize`** — LLM generates a `synthesis:project_overview` node summarizing the entire project. Used as the primary context for overview-intent questions.
- **Curation system prompt** — Explicit system prompt for the fine-tuned model with structured JSON output format, requesting 3-5 entities per batch with type classification.
- **Entity type mapping** — `map_extractor_type()` converts lowercase extractor output to capitalized ontology types (e.g., "technology" → "Technology").

### Context Pipeline
- **Intent keyword fallback** — Overview detection uses keyword matching ("how many files", "what is this project") when the LLM intent classification is ambiguous.
- **Overview intent filtering** — Skips vector search, skips section nodes, filters to verified chunks only. Prevents section-level noise from flooding overview answers.
- **Node-scoped retrieval** — Chunk retrieval scoped to activated graph nodes instead of global RAG search. Only chunks belonging to relevant nodes are retrieved.
- **Debug key alignment** — Renamed `s1_unified` → `s1_detection` in trace output for frontend compatibility.
- **S1/S2 log level upgrade** — Pipeline step logs promoted from DEBUG to INFO for visibility in traces.

### Benchmarks
- **5-category benchmark system** — RESOLVE (enables impossible answers), GROUND (prevents hallucination), RECALL (persistent memory), FOCUS (efficient context), ADAPT (topic switching). 55 turns across 3 test projects.
- **Test fixtures** — P1: TODO App (31 TypeScript/React files), P2: Sherlock Holmes epub (public domain, Project Gutenberg), P3: PM docs (11 markdown files with roadmaps, sprints, issues, ADRs).
- **Pipeline validation tests** — 26 tests inspecting graph state per project (index/curate/synthesize structural checks, phantom entity detection, entity relation validation).
- **Agent comparison scorecard** — Each RESOLVE/GROUND turn includes estimated costs for Stateless LLM, Agent+Tools, and Acervo approaches. Reports show efficiency ratios (12.1x fewer tokens than agent approach).
- **Per-turn component diagnostics** — S1 intent accuracy, S2 activation accuracy, S3 budget compliance, S3 content quality. Cross-matrix shows which categories are affected by which component failures.
- **Version-tracked results** — Each benchmark run appends to `version_history.json` for cross-version regression detection.
- **Auto-indexing fixtures** — `conftest.py` runs the full init→index→curate→synthesize pipeline on first run for each test project.

### Acervo Studio
- **Multi-project support** — `ProjectContext` (React context) replaces window events. `select()` uses 2 API calls (was 4). Project switching with loading spinner.
- **Project config UI** — Read/write `.acervo/config.toml` from the Settings tab.
- **Indexation tab auto-load** — File status loads automatically when the Indexation tab opens. Operation timestamps display.
- **Session export** — Copy/Markdown/JSON export buttons in the chat toolbar.
- **Pipeline trace updates** — S1 Unified fields (topic_action, intent, entities), chunk text previews in S2, collapsible "all available chunks" section.
- **Chat auto-clear on project switch** — Chat messages clear when switching projects via `useEffect` on `activeId`.
- **Export toolbar layout fix** — Fixed flex layout broken by the export toolbar wrapper.

### CLI
- **`acervo up --dev`** — Starts Ollama, proxy, Studio backend, and Studio frontend in one terminal with multiplexed tagged logs.
- **`acervo chunks stats|list|show|search`** — Chunk inspection CLI for debugging and validating ingestion.
- **`acervo graph repair`** — Detects and fixes missing fields, orphan edges, duplicate edges.
- **`acervo graph show|search|delete|merge`** — Full graph inspection and editing from the terminal.
- **Configurable logging** — `--log-level trace|debug|info|warning|error`, `--no-color`, `-v` shorthand.

### Bug Fixes
- **Context builder `UnboundLocalError`** — Variables `conversation_chunks`/`verified_chunks` only defined in `else` branch but used unconditionally in debug dict. Fixed by initializing before if/elif/else.
- **Proxy graph overwrite after indexing** — Indexer creates its own TopicGraph and saves to disk. Proxy's in-memory graph (empty) overwrites on next save. Fixed with `/acervo/reload-graph` endpoint called after indexing.
- **Curation producing 0 entities** — No system prompt + vague user prompt for the fine-tuned model. Fixed with explicit curation system prompt and richer file context.
- **Entity types "Unknown"** — `map_extractor_type()` not called; lowercase "technology" didn't match capitalized registry. Fixed by routing through ontology mapper.
- **Unicode encoding error on Windows** — `─` character in print() caused cp1252 error. Fixed by using ASCII `-`.
- **Prose under-chunking** — Entire book chapters stored as single 20k+ char chunks. Fixed with `_chunk_prose()` paragraph-cluster splitting.
- **S1/S2/S3 invisible in trace** — Debug key mismatch `s1_unified` vs `s1_detection`. Fixed by renaming key.
- **Chat layout broken by export toolbar** — Export toolbar wrapper broke flex layout. Fixed by adding `flex flex-col` to ChatArea container.
- **FileTools removed from pipeline** — FileTools was wired into the pipeline but belonged in S2 gather. Removed entirely; file reading handled via S2 node activation.

### Removed
- **FileTools** — Removed from pipeline. File reading is handled by S2 node activation, not runtime tool calls.
- **Old benchmark files** — Replaced v0.3 conversation benchmark framework with 5-category benchmark system.

### Benchmark Results (v0.4.0 baseline)

| Category | Score |
|----------|-------|
| RESOLVE  | 100%  |
| GROUND   | 92%   |
| RECALL   | 67%   |
| FOCUS    | 100%  |
| ADAPT    | 100%  |

Component health: S1 Intent 78%, S2 Activation 56%, S3 Budget 32%, S3 Quality 81%.

RESOLVE efficiency: **12.1x fewer tokens** than agent-with-tools approach (avg 616 tokens vs 7,462 tokens per turn).

## [0.2.2] - 2026-03-25

### Fixed
- **History windowing always applies** — Fixed critical bug where conversation history grew unbounded when no graph context was found. Previously, `_window_history_openai()` skipped windowing when `has_context=False`, sending the full history to the LLM. Now windowing applies regardless — tokens stay constant at any conversation length.
- **Anthropic history windowing** — Added `_window_history_anthropic()` so both API formats enforce the history window.

### Changed
- README rewrite with updated architecture, features, and clearer getting started guide.

## [0.2.1] - 2026-03-25

### Changed
- **Disabled entity-level embeddings in chat flow** — S1 fine-tuned model handles entity extraction; graph activation + traversal covers retrieval without per-turn embedding calls. Removes Ollama as a runtime dependency. Embedding code preserved for future document chunk indexing.
- **Graceful Ollama health check** — Proxy verifies Ollama connectivity on startup before enabling embeddings. Falls back cleanly if unavailable.
- **Background fact indexing** — Vector store fact indexing moved to `asyncio.create_task()` so it doesn't block LLM response streaming.
- **Removed startup sync** — `sync_vector_store()` no longer blocks proxy startup; facts are indexed incrementally.

## [0.2.0] - 2026-03-24

### Architecture
- **S1 Unified pipeline** — Combined topic classification + entity extraction in a single LLM call, replacing the separate L3 topic classifier and ConversationExtractor. Runs ALWAYS on every turn; L1/L2 are hints, not gates.
- **S1.5 Graph Update** — Async post-response graph curation that merges duplicates, corrects types, discards garbage entities, creates missing relations, and extracts knowledge from assistant responses.
- **Single model architecture** — One fine-tuned model (`acervo-extractor-qwen3.5-9b`) handles both chat and extraction, differentiated by system prompt. Halves VRAM usage from ~14GB to ~6GB. Separate extractor model still configurable via `[acervo.models.extractor]`.
- **Trace persistence** — Pipeline events persist to JSONL file and restore on frontend reconnection.
- **History windowing** — Conversation history trimmed to N recent turns when graph context is present (configurable `history_window`).
- **Tool-continuation context** — Proxy re-injects enriched context across multi-turn tool call sequences.

### Fine-tuned Model
- Published `acervo-extractor-qwen3.5-9b` on Hugging Face.
- Trained on 612 examples across 5 domains (software, business, literature, personal, academic).
- 85% extraction accuracy, 100% JSON parse rate.
- Supports English and Spanish input natively.
- Training code and datasets in `acervo-models` repository.

### Data Model
- **Unified entity types**: 8 types (person, organization, project, technology, place, event, document, concept).
- **Unified relation types**: 15 controlled relations (uses_technology, part_of, maintains, deployed_on, etc.).
- **Entity layering**: PERSONAL (user owns it) vs UNIVERSAL (public knowledge) classification per entity.
- **Fact dedup**: Jaccard word-overlap similarity (threshold 0.65, was 0.9) catches more paraphrases.
- **Edge dedup + upgrade**: Prevents duplicate edges; generic relations (`related_to`) auto-upgrade to specific ones (`uses_technology`).
- **Type migration on graph load**: Framework → technology, Backend_service → technology, etc.
- **Garbage entity filter**: Blocks jargon like index names, SQL terms, file extensions.

### Storage
- **ChromaVectorStore** — Activated embedding-based semantic search over graph facts and file chunks. Optional dependency (`pip install acervo[vector]`).
- **Embedding optimization** — Batch embedding via Ollama's multi-input API. User text embedding deduplicated (topic L2 + vector search share one call). ~10 sequential HTTP calls → 2 per turn.
- **Data directory** moved to `.acervo/` in project workspace (follows `.git/` pattern).
- `/acervo/clear` endpoint — Atomic reset that drops ChromaDB client before deleting vectordb (fixes Windows file lock issues).

### Prompts
- New `s1_unified.txt` — Minimal system prompt for the fine-tuned extraction model.
- Updated `s1_5_graph_update.txt` — Aligned with new type/relation vocabulary, added valid relations list.
- Updated `extractor_search.txt` — Added layer guards (UNIVERSAL only), aligned entity schema with `id` field.
- Prompt files loaded from `.acervo/prompts/` with built-in defaults as fallback.

### Proxy
- **Graceful client disconnect handling** — Catches `ConnectionResetError` and `ClientConnectionResetError` during streaming instead of crashing.
- **Provider-friendly error messages** — Connection errors show the configured provider name.
- **Per-role model support** — `[acervo.models.extractor]` and `[acervo.models.summarizer]` for different models per pipeline step.

### Bug Fixes
- Fixed clear button bringing back old data — proxy's in-memory facade survived disk deletion; now fully reinitializes on reset.
- Fixed S1.5 using wrong model (default instead of fine-tuned extractor).
- Fixed S1.5 LLM params (temperature 0.0→0.1, max_tokens 1000→2048).
- Fixed search extractor using fine-tuned model instead of default LLM.
- Fixed double embedding in entity enrichment — `facade.py` called `embed()` redundantly before `index_facts()`.
- Fixed stale vector DB data surviving graph deletion.
- Fixed ChromaDB file locks on Windows during data clear.
- Fixed proxy crash on client disconnect during streaming.

### Removed
- `query_planner.py` — Logic folded into S1 Unified + deterministic context gathering.
- `topic_classifier.txt` and `extractor_conversation.txt` — Replaced by S1 Unified.
- `_extract_label_via_llm()` and `_match_known_topic()` — Replaced by S1 Unified.
- L3 LLM topic classification branch — S1 makes the final topic decision.
- Automatic `co_mentioned` edge creation — Too noisy, replaced by S1.5 explicit relations.
- `cycle_status()` — Replaced by topic-aware `update_status()`.

## [0.1.2] - 2026-03-18

### Changed
- README rewrite with Mermaid architecture and knowledge graph diagrams
- Documentation rewrite to reflect only working v0.1.1 features
- Planned features clearly badged as "Planned" throughout docs

### Added
- Roadmap page with planned features and their status
- Mermaid diagrams across docs (pipeline flow, graph structure, context stack, layers)
- CHANGELOG backfill for v0.1.0 and v0.1.1

## [0.1.1] - 2026-03-18

### Added
- Published to PyPI (`pip install acervo`)
- GitHub Actions workflow for automatic PyPI publishing on tag push
- `.env.example` file

## [0.1.0] - 2026-03-17

### Added
- Knowledge graph with JSON persistence (nodes.json, edges.json)
- Two-layer architecture: UNIVERSAL (world knowledge) and PERSONAL (user-specific)
- `prepare()` / `process()` high-level context proxy API
- `commit()` / `materialize()` lower-level graph API
- Auto-registering ontology: LLM creates new entity types and relations dynamically
- Semantic relations: IS_A, CREATED_BY, ALIAS_OF, PART_OF, SET_IN, DEBUTED_IN, PUBLISHED_BY
- Topic detector with 3-level cascade (keywords, embeddings, LLM classification)
- Query planner: LLM decides tool (GRAPH_ALL, GRAPH_SEARCH, WEB_SEARCH, READY)
- Context index with 3-layer stack (system, warm, hot) and token budgeting
- ConversationExtractor and SearchExtractor with conservative extraction policy
- Built-in OpenAIClient (zero external deps beyond pydantic)
- LLMClient and Embedder protocols for provider-agnostic integration
- Web search result extraction and graph persistence
- 56 unit tests, 8 integration tests
- MkDocs Material documentation site
