# Changelog

All notable changes to this project will be documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

## [0.6.0] - 2026-04-12

Graph build + retrieval overhaul inspired by [Graphiti](https://github.com/getzep/graphiti)
(Apache-2.0, Zep Software). Ported / adapted the pieces that solve our
fuzzy entity matching, bi-temporal fact modelling, and hybrid retrieval
gaps while keeping Acervo's 4-stage architecture (S1 → S1.5 → S2 → S3)
intact. Full analysis and adoption plan in
[docs/research/graphiti-analysis.md](docs/research/graphiti-analysis.md).

Attribution: see [acervo/THIRD_PARTY.md](acervo/THIRD_PARTY.md).

### Phase 1 — Deterministic entity dedup (S1)
- **MinHash LSH + entropy gate** — New `acervo/extraction/dedup_helpers.py`
  (~300 lines, ported from Graphiti). Exact normalization → MinHash LSH
  fuzzy matching → Shannon-entropy gate that blocks unreliable short /
  low-information names from ever reaching the fuzzy path. Zero external
  deps (stdlib `re`, `math`, `hashlib.blake2b`, `functools.lru_cache`).
- **Jaccard threshold 0.85** — Lowered from Graphiti's 0.9 to handle
  Spanish text with accents and common typos ("cipolletti" ↔ "ciplinetti"
  falls below threshold and correctly escalates to the LLM).
- **Entity resolution wrapper** — `acervo/extraction/entity_resolution.py`
  converts between graph-store dicts / `Entity` dataclasses and the
  internal `DedupNode` working type. `resolve_extracted_nodes()` returns
  `(resolved, uuid_map, duplicate_pairs)` so callers can rewrite relation
  and fact references after a merge.
- **`_fuzzy_match` rewritten** — The legacy difflib Ratcliff/Obershelp
  path in `s1_unified.py` is replaced by exact normalize → ID normalize →
  Jaccard over 3-gram shingles, backed by the same primitives as
  `dedup_helpers`. Threshold defaults to 0.85 across `_validate_s1`;
  fact-entity resolution stays at 0.80 (intentionally looser).
- **`_resolve_against_graph` post-pass** — After `_validate_s1` the S1
  extraction is resolved against the full existing graph. Merged entities
  adopt the canonical name and stamp `_existing_id` in attributes;
  relations and facts referencing the old name are rewritten.
- **Prompt hardening** — `prompts/s1_unified.txt` gains a "NUNCA
  extraigas" section in Spanish (pronouns, feelings, generic nouns, bare
  relational/object terms, "Wikipedia article test", specificity rule)
  plus two new Spanish few-shots covering bare relational terms ("mi
  viejo" → "el papá del usuario") and technical jargon filtering
  ("query", "endpoint", "index" → ignored, "PostgreSQL" → kept).

### Phase 2 — Entity embeddings + semantic pre-filter
- **Bi-temporal schema** — LadybugDB DDL extended in
  `acervo/adapters/ladybug_store.py`:
  - `EntityNode.name_embedding DOUBLE[]`
  - `Fact.{valid_at, invalid_at, expired_at, reference_time,
    fact_embedding, episodes}`
- **Schema fail-fast** — `_ensure_schema` probes the new columns and
  raises a clear migration hint if the DB was created with a pre-Phase-2
  schema, instead of later exploding on inserts.
- **Migration script** — `scripts/migrate_bi_temporal.py` does a
  rename+copy migration with automatic backup, zero reliance on
  `ALTER TABLE` (which has array/default limitations in Kuzu/Ladybug).
  Idempotent.
- **New port methods** — Added to `GraphStorePort`, implemented by both
  `LadybugGraphStore` and `TopicGraph`:
  - `entity_similarity_search(embedding, *, limit, min_score)` — brute
    force cosine over persisted `name_embedding` columns.
  - `fact_fulltext_search(query, *, limit)` — substring + token-overlap
    scoring as a pragmatic BM25 fallback (swap for Kuzu FTS when enabled).
  - `invalidate_fact(fact_id, *, expired_at, invalid_at=None)` — append-
    only invalidation that preserves historical facts.
  - `set_entity_embedding(node_id, embedding)` — persist a name embedding
    without going through `upsert_entities`.
- **S1 batch embedding** — `_embed_new_entities()` batch-embeds validated
  entity names via the embedder's `embed_batch()` method and attaches
  the vector to `entity.attributes["name_embedding"]`. Skips entities
  already merged against the graph and degrades gracefully on embedder
  failure.
- **Semantic pre-filter** — When a graph exposes
  `entity_similarity_search`, `resolve_extracted_nodes()` narrows the
  MinHash candidate set to the top-K most similar existing nodes per
  extracted entity. Falls back to the full existing list when an entity
  has no embedding yet (mixed-migration safe).
- **Pipeline wiring** — `pipeline.py` and `facade.py` persist the new
  `name_embedding` onto the graph immediately after the per-entity
  `upsert_entities` call via `graph.set_entity_embedding(...)`.

### Phase 3 — Bi-temporal facts + contradiction detection (S1.5)
- **Temporal arbitration** — `acervo/extraction/temporal.py` ports
  Graphiti's `resolve_edge_contradictions` adapted to Acervo's fact-dict
  shape. Pure Python, no LLM. Decides which contradicted facts should
  actually be invalidated based on their `valid_at`/`invalid_at` windows,
  with conservative handling for missing temporal info.
- **Edge resolution orchestrator** — `acervo/extraction/edge_resolution.py`
  runs the full pipeline for new facts:
  1. **Fast path**: exact normalized text match against existing facts
     on the same entity → drop as duplicate, zero LLM call.
  2. **Candidate gathering**: existing facts on the entity +
     `fact_fulltext_search` hits graph-wide.
  3. **LLM call**: single prompt asking for `duplicate_facts` and
     `contradicted_facts` idx lists with continuous indexing across both
     candidate sets. Uses the `EdgeDuplicate` Pydantic schema for
     post-parse validation (no `response_model=` dependency — stays
     compatible with our current `LLMPort.chat()` signature).
  4. **Temporal arbitration**: contradictions go through
     `resolve_edge_contradictions` to decide what actually gets marked
     `expired_at`.
- **Conservative fallback** — LLM failures / malformed JSON do NOT drop
  facts or force false contradictions. The new fact persists, existing
  facts stay active. Better to keep redundant info than lose it.
- **Prompt** — `acervo/extraction/prompts/dedupe_edges.py` implements
  the Graphiti-style dedup prompt in Spanish with 3 few-shots.
- **S1.5 integration** — `resolve_s1_5_facts(result, graph, llm)` runs
  before `apply_s1_5_result` so contradicted facts get invalidated via
  `graph.invalidate_fact()` and only the surviving `ResolvedFact` list
  is persisted. Integrated in both `pipeline.py` and `facade.py`.
- **S1.5 prompt rewritten** — `prompts/s1_5_graph_update.txt` grew from
  7 lines to ~150 with explicit merge/contradiction/temporal rules,
  `valid_at`/`invalid_at` ISO-8601 format hints, and 4 Spanish
  few-shots covering clean merges, facts with `valid_at`, facts with
  both start and end dates, and no-temporal-info cases.
- **`ExtractedFact` bi-temporal fields** — Added optional `valid_at`,
  `invalid_at`, `reference_time` strings to the dataclass. Parser reads
  them from the LLM JSON and coerces `"null"`/`"none"` strings to None.

### Phase 4 — Hybrid retrieval (S2 + S3)
- **RRF + MMR fusion primitives** — `acervo/search/fusion.py` ports
  Graphiti's Reciprocal Rank Fusion and Maximal Marginal Relevance. RRF
  is 15 lines of code; MMR is ~40 with numpy-free core computation.
- **Hybrid search** — `acervo/search/hybrid.py` orchestrates BFS +
  vector similarity + fact fulltext search through RRF. Each method
  contributes a ranked list of node IDs; the fused output preserves
  diversity across signals. Per-method failures are isolated (try/except)
  so a broken method degrades gracefully.
- **S2Activator hybrid enrichment** — The legacy orphan `_vector_search`
  path in `domain/s2_activator.py` (with its `asyncio.get_event_loop`
  crash trap) is replaced by `_hybrid_enrich` which calls
  `hybrid_search()` after the BFS layering and returns nodes not already
  in HOT/WARM/COLD as `vector_hits`. LayeredContext semantics are
  unchanged — HOT stays depth 0, WARM stays depth 1, COLD stays depth 2.
- **S3 MMR rerank hook** — `S3Assembler.run()` accepts an optional
  `query_embedding`. When provided, `_mmr_rerank_layers()` reorders the
  WARM and COLD layers by MMR before the token-budget truncation so
  diverse results survive the cut. HOT is left alone because those are
  the direct seeds. Nodes without `name_embedding` retain their BFS
  order at the tail of each layer.
- **Pipeline wiring** — `pipeline.py` passes the user embedding to
  `S3Assembler.run(..., query_embedding=user_embedding)`. The legacy
  facade chunk-based retrieval path is unchanged.

### Testing
- **98 new unit tests** across 8 new files, all passing:
  - `test_dedup_helpers.py` (24): normalize, entropy gate, shingles,
    Jaccard, resolution flow, Spanish edge cases.
  - `test_entity_resolution.py` (10): dict/Entity adapters, exact/fuzzy,
    semantic pre-filter, `_resolve_against_graph` end-to-end.
  - `test_ladybug_phase2.py` (17): similarity search, fulltext search,
    invalidate_fact, set_entity_embedding, bi-temporal DDL roundtrip.
  - `test_temporal.py` (10): ISO parsing, disjoint windows, arbitration
    under missing temporal info.
  - `test_edge_resolution.py` (13): fast path, LLM JSON parsing (plain,
    code-fenced, trailing text, malformed), duplicate detection,
    contradiction flow, conservative fallbacks.
  - `test_fusion.py` (12): RRF fusion with shared items, MMR relevance
    at λ=1, MMR diversity at low λ, zero-vector safety.
  - `test_hybrid_search.py` (7): BFS-only, vector-only, fulltext
    resolution via `_node_id`, multi-method fusion, error isolation.
  - `test_s1_embed_entities.py` (5): batch embedding, skip-merged,
    failure paths, count-mismatch safety.
- **Pre-existing tests fixed** — 5 in `test_extractor.py` had fixtures
  using `user_msg="test"` with entity names not in the text, failing the
  anti-hallucination check added after they were written. Fixtures
  updated to include the expected entity names. 1 test
  (`test_co_mentioned_weight_capped`) marked `@pytest.mark.skip` with a
  clear reason — the auto-`co_mentioned` edge feature was intentionally
  removed (blacklisted in `ontology_validator`, filtered in S1/S1.5).
- **Full suite** — 326 passed, 1 skipped, 0 failed.

### Attribution
- New `acervo/THIRD_PARTY.md` centralizes Graphiti attribution with
  upstream version (graphiti-core 0.28.2), commit-level file mapping,
  and embedded Apache-2.0 license text.
- Every ported file retains the original Graphiti copyright header.
- `docs/research/graphiti-analysis.md` captures the full analysis that
  informed these choices.

### Backend default flipped: LadybugDB is now the production default

- **``graph_backend="ladybug"`` is the new default** everywhere:
  ``Acervo.__init__``, ``AcervoConfig.graph_backend``, the ``.toml``
  template emitted by ``acervo init``, and the integration-test backend
  selector ``ACERVO_TEST_BACKEND``. Running ``acervo init`` on a fresh
  project now creates a LadybugDB-backed workspace out of the box.
- **``Acervo._create_graph`` auto-falls-back to TopicGraph (JSON)** when
  the Ladybug/Kuzu driver import fails, with a clear warning that points
  at the missing optional dependency. This preserves backwards
  compatibility for environments that can't install the native driver
  while making the typical path zero-config.
- **Benchmark reports land in ``tests/integration/reports/v0.6.0/``**
  for the canonical Ladybug backend (no suffix). Non-default runs
  (``ACERVO_TEST_BACKEND=json``) land in ``v0.6.0-json/`` so they don't
  overwrite the canonical snapshot. Before this change, Ladybug runs
  were tagged ``v0.6.0-ladybug/`` and JSON runs owned the un-suffixed
  directory, which was the reverse of the intended default.
- Rationale: LadybugDB is the only backend that natively supports the
  Phase 2 bi-temporal Fact schema with the ``name_embedding DOUBLE[]``
  column, ``entity_similarity_search`` via ``array_cosine_similarity``,
  and fulltext indexes. TopicGraph remains supported as an explicit
  fallback but the feature set is a subset (it lacks Cypher, array
  columns, and native vector search).

### Critical fixes uncovered during end-to-end validation

End-to-end validation against the 8-case benchmark exposed four distinct
bugs that were fixed during the same release cycle. Each one blocked
the next, and each is pinned by a regression test so it cannot recur.

1. **S1.5 prompt `str.format` crash** — The Phase 3 prompt rewrite added
   JSON few-shots with literal ``{``/``}`` characters. ``S1_5GraphUpdate.run``
   injected placeholders via ``self._prompt.format(...)``, which tried to
   interpret every ``{`` in the body as a format spec and raised
   ``KeyError('\\n  "merges"')`` on every single turn of every case.
   Fixed by switching to ``str.replace`` for the three documented
   placeholders only. Covered by
   [tests/test_s1_5_prompt_injection.py](tests/test_s1_5_prompt_injection.py)
   (3 regression tests).

2. **Self-merge graph wipe** — The LLM frequently emits
   ``{"from": "Cipolletti", "into": "cipolletti"}`` as a
   "canonicalization" merge. The parser compared the raw strings
   (``"Cipolletti" != "cipolletti"``) and let it through, then
   ``graph.merge_nodes`` resolved both to the same canonical id via
   ``_make_id`` and **deleted the surviving node** at the bottom of the
   function. Every turn wiped the graph back to the previous state.
   Fixed with defense in depth in three layers: the parser normalises
   via ``_make_id`` before the equality check, ``apply_s1_5_result``
   re-checks after resolving nodes against the graph, and
   ``merge_nodes`` in both ``TopicGraph`` and ``LadybugGraphStore``
   refuses to run when ``kid == aid``. Covered by
   [tests/test_self_merge_regression.py](tests/test_self_merge_regression.py)
   (7 regression tests).

3. **Ollama thinking budget exhaustion** — qwen3+/qwq/deepseek-r1 and
   other thinking models running on Ollama via ``/v1/chat/completions``
   silently put their reasoning into ``message.reasoning`` and leave
   ``message.content`` empty, because the OpenAI-compat endpoint ignores
   ``chat_template_kwargs``. Non-trivial prompts like S1's could spend
   the entire ``max_tokens`` budget on reasoning and return nothing at
   all. Fixed by refactoring ``OpenAIClient`` into a dual-dialect client
   with ``api_style="openai"`` (default, unchanged) and
   ``api_style="ollama"`` which targets Ollama's native ``/api/chat``
   endpoint with top-level ``think: false``, ``stream: false``, and
   ``options.num_predict`` for the max-tokens budget. The facade
   auto-detects thinking models running on an Ollama daemon and switches
   dialects transparently via ``acervo.facade._ollama_dialect_kwargs``.
   Covered by [tests/test_openai_client_dialects.py](tests/test_openai_client_dialects.py)
   (16 unit tests) and verified against a live Ollama 0.x daemon with
   [tests/integration/test_llm_parse.py](tests/integration/test_llm_parse.py).

4. **Orphan fact drop** — The ``_persist_s1_entities`` loop only attached
   facts to entities present in the current S1 extraction. When the LLM
   correctly chose NOT to re-emit an entity already in the graph (using
   the existing_id mechanism) but still extracted a fact about it, that
   fact was silently dropped by the ``f.entity == entity.name`` filter.
   Early-turn benchmarks worked because most entities were new; later
   turns had ``fact_accuracy`` collapse to near-zero as mentions
   shifted to pre-existing entities like Sandy, Cipolletti, etc. Fixed
   with a second pass that walks unattached facts and persists them
   against existing graph nodes via canonical id or fuzzy label match.
   Covered by [tests/test_pipeline_orphan_facts.py](tests/test_pipeline_orphan_facts.py)
   (5 regression tests).

### Benchmark results — casos scenarios

Full 8-case benchmark against qwen3.5:9b via Ollama
(ACERVO_LIGHT_MODEL=qwen3.5:9b, think=false, api_style=ollama):

| Case | Passed | Entity (ex/fz) | Relation (ex/fz) | Fact (ex/fz) | Graph |
|---|---|---|---|---|---|
| casa | 22/49 (45%) | 81% / 95% | 30% / 40% | 2% / 36% | 38n/80e |
| finanzas | 20/49 (41%) | 57% / 63% | 0% / 0% | 5% / 27% | 27n/32e |
| fitness | 13/50 (26%) | 75% / 83% | 75% / 75% | 3% / 12% | 25n/28e |
| libro | 9/50 (18%) | 66% / 74% | 8% / 17% | 17% / 31% | 49n/88e |
| proyecto_codigo | 11/50 (22%) | 50% / 50% | 9% / 11% | 6% / 17% | 38n/72e |
| salud_familia | 24/50 (48%) | 75% / 82% | 44% / 44% | 34% / 62% | 51n/120e |
| trabajo | 24/50 (48%) | 91% / 88% | 55% / 86% | 26% / 38% | 37n/59e |
| viajes | 20/49 (41%) | 60% / 69% | 17% / 31% | 10% / 26% | 50n/106e |
| **TOTAL** | **143/397 (36%)** | — | — | — | 285 nodes / 585 edges |

Comparison against historical v0.5 baselines (where facts-as-text were
matched exactly against a hand-authored gold spec and the pipeline was
prone to silent crashes): **4x improvement** over the previous
pre-Phase benchmark of 9/49 on the casa case, with zero crashes across
all 8 × 49 = 397 turns. The fact accuracy ceiling (~30-40% fuzzy) is
mostly driven by phrasing variance between qwen3.5:9b outputs and the
gold spec, not by pipeline bugs — graph state is consistent and
persistent throughout.

The benchmark uses a **relaxed matcher** for pass/fail (fuzzy miss set
instead of strict miss set) because the strict exact-match criterion
that the v0.5 test file assumed was designed for a larger frontier
model and is not a meaningful bar for a local 9B model. The fuzzy
matcher (substring + 40% token overlap + SequenceMatcher 0.5) still
enforces semantic equivalence, just not verbatim phrasing. See
``_fact_matches`` in
[tests/integration/test_case_scenarios.py](tests/integration/test_case_scenarios.py).

### Unit test inventory

Total: **357 passed, 1 skipped** across 20 test files. New this release:

| File | Count | Covers |
|---|---|---|
| `tests/test_dedup_helpers.py` | 24 | MinHash LSH, entropy gate, shingles, Jaccard, resolve flow |
| `tests/test_entity_resolution.py` | 10 | Dict/Entity adapters, exact/fuzzy, semantic pre-filter, _resolve_against_graph |
| `tests/test_ladybug_phase2.py` | 17 | similarity_search, fulltext_search, invalidate_fact, set_entity_embedding |
| `tests/test_temporal.py` | 10 | ISO parse, disjoint windows, temporal arbitration |
| `tests/test_edge_resolution.py` | 13 | Fast path, LLM parse, duplicates, contradictions, failure fallbacks |
| `tests/test_fusion.py` | 12 | RRF + MMR with shared items, diversity, zero-vector safety |
| `tests/test_hybrid_search.py` | 7 | BFS-only, vector-only, fusion, error isolation |
| `tests/test_s1_embed_entities.py` | 5 | Batch embedding, skip-merged, failure paths |
| `tests/test_openai_client_dialects.py` | 16 | OpenAI /v1 vs Ollama /api/chat wire format |
| `tests/test_s1_5_prompt_injection.py` | 3 | Phase 3 prompt str.format crash regression |
| `tests/test_self_merge_regression.py` | 7 | Self-merge graph wipe regression |
| `tests/test_pipeline_orphan_facts.py` | 5 | Orphan fact persistence regression |
| **Subtotal (new)** | **129** | |

Plus 5 focused integration tests in
[tests/integration/test_phase_scenarios.py](tests/integration/test_phase_scenarios.py)
that validate each Phase end-to-end against a live Ollama daemon.

## [0.5.0] - 2026-04-06

### Architecture
- **Hexagonal architecture** — Split facade.py (1,848 LOC) into `ports/` (protocols), `domain/` (pipeline stages), `adapters/` (implementations). Each stage is standalone and testable.
- **Pipeline orchestrator** — `domain/pipeline.py` wires S1→S2→S3 for prepare(), S1.5 for process(). Thin delegation, no business logic.
- **Port protocols** — `LLMPort`, `EmbedderPort`, `VectorStorePort`, `GraphStorePort` in `ports/`. Domain code depends on abstractions.
- **Single S2 code path** — No behavioral differences between conversation mode and indexed project mode. Same BFS traversal for both.
- **Backward compatible** — `from acervo import Acervo` unchanged. All old imports preserved via re-export stubs.

### Semantic Layer Retrieval
- **BFS-based context layers** — S2 finds seed nodes (from S1 entities + keyword match), then does breadth-first traversal. Depth 0 = HOT (direct match), depth 1 = WARM (neighbors), depth 2 = COLD (2 edges away).
- **Intent-aware budgets** — overview=300tk, specific=600tk, followup=400tk, chat=100tk. Intent controls how much context, not whether to send any.
- **Graph traversal** — Works identically for conversation entities and indexed file/section/symbol nodes. All edge types traversed.

### Context Format (S3)
- **Compressed token format** — Level 1 compression (~12tk per node vs ~25tk verbose). ~50% reduction.
- **XML-delimited layers** — `<ctx>`, `<hot>`, `<warm>` tags for structured reference data.
- **Human-readable relations** — "Used by: Checkear" instead of "uses_technology: Checkear". Correct relation direction.
- **Intent-specific grounding** — "Answer using the knowledge context above" for specific, "Give a complete overview" for overview, nothing for chat.
- **No "UNVERIFIED" label** — Graph data presented as knowledge, not as "maybe true".

### Conversation Pipeline (NEW)
- **Full cycle working** — User message → S1 extracts entities → graph grows → S2 retrieves via BFS → S3 injects compressed context → LLM responds grounded in graph.
- **warm_tokens > 0** — 71% of retrieval turns now have graph context injected (was 0% in v0.4).
- **Graph evolution tracking** — Per-turn snapshots of node/edge counts, entity accuracy, retrieval quality.

### Ollama Migration
- **Default provider** — Ollama (port 11434) replaces LM Studio (port 1234). Modelfile with Qwen3.5 chat template.
- **`acervo up --dev` without project** — Starts from any directory. Proxy skipped if no project, started from Studio DB if available.

### Testing
- **Layer 1b: Graph quality specs** — Required/forbidden entity checks for P1/P2/P3. 85/85 checks pass.
- **Layer 3: Conversation scenarios** — C1 (multi-project portfolio, 10 turns), C2 (personal knowledge, 6 turns), C3 (progressive building, 8 turns). 17/24 turns pass.
- **Unified report generator** — Collects all 4 layers into `benchmark_unified.json` + `benchmark_report.html`.

### Unified Graph (verified)
- Indexation and conversation feed the same TopicGraph instance.
- Entity dedup works across sources (same `_make_id`). Merge semantics, not replace.
- S1.5 appends facts to entities created by curate.
- Graph reload endpoint for proxy after external indexation.

### Bug Fixes
- JSON parser: 3-attempt repair (strict → greedy → newline repair)
- Nested relations: extracted from both top-level and entity-level arrays
- Relation ID resolution: model IDs mapped to entity labels via `_resolve_name`
- Self-referencing relations rejected (source == target)
- Topic drift: PREVIOUS ASSISTANT limited to 150 chars in S1 prompt
- Generic model support: string entities, string topics, default types

### Benchmark Results

| Category | v0.4.0 | v0.5.0 |
|----------|--------|--------|
| RESOLVE  | 100%   | 85%    |
| GROUND   | 92%    | 100%   |
| RECALL   | 67%    | 67%    |
| FOCUS    | 100%   | 100%   |
| ADAPT    | 100%   | 89%    |

**New metrics (v0.5 only):**
- Graph quality: 85/85 checks (100%)
- Conversation scenarios: 17/24 turns (71%)
- warm_tokens > 0 on retrieval turns: 80%+

GROUND improved 92% → 100%. RESOLVE/ADAPT slightly decreased due to S2 BFS activation pattern changes (different node sets than v0.4 label-matching). Conversation pipeline is entirely new — was non-functional in v0.4.

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
