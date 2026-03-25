# Changelog

All notable changes to this project will be documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

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
