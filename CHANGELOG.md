# Changelog

All notable changes to this project will be documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-03-19

### Added
- **Session metrics**: `Acervo.metrics` property with per-turn tracking (tokens, graph size, extraction counts, context hit rate) and aggregate stats. `metrics.export_json()` for serialization, `metrics.summary()` for human-readable output.
- **Graph import/export**: `TopicGraph.export_json()` and `TopicGraph.import_json(data, mode="merge"|"replace")` for graph portability and backup.
- **Edge validation**: Self-referential edges rejected, `co_mentioned` weight capped at 10.0, `last_active` timestamp added to edges.
- 14 new tests (70 total): metrics, import/export roundtrip, edge validation, weight capping.

### Changed
- **All prompts switched to English**: extractor, planner, topic detector, summarizer, synthesizer — Acervo is now an English-first library.
- **Entity types renamed to English**: Person, Character, Organization, Place, Technology, Work, Project, Universe, Publisher, Document, Rule. Legacy Spanish type names still accepted via extractor mapping.
- **Relations renamed to English**: works_at, lives_in, owns, belongs_to, etc. Legacy Spanish relations still accepted.
- Entity blacklist updated to English stopwords.
- Context markers: `[VERIFIED CONTEXT]`/`[END CONTEXT]`, `[WEB SEARCH RESULTS]`/`[END RESULTS]`.

### Migration notes
- Existing graphs with Spanish type names (Persona, Lugar, etc.) are still accepted by the extractor's type mapping. New extractions will use English types.
- If you need to update existing graph data, use `export_json()` → modify types → `import_json(data, mode="replace")`.

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
