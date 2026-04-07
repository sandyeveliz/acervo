# Architecture Audit — Acervo v0.4

Generated: 2026-04-05
Codebase: 14,045 LOC across 40 .py files

---

## 1. Dependency Map

### Module count by layer

| Layer | Modules | LOC |
|-------|---------|-----|
| Config & Project | config, project, log_config | 788 |
| Knowledge Graph | graph, layers, ontology, topic_node | 1,398 |
| Structural Analysis | structural_parser, dependency_resolver, file_ingestor | 1,636 |
| Semantic Enrichment | semantic_enricher, indexer, reindexer | 1,197 |
| Extraction & Curation | extractor, curator, s1_unified, s1_5_graph_update | 1,806 |
| Context Management | context_builder, context_index, synthesizer, topic_detector, specificity | 970 |
| Facade & Protocols | facade, llm, openai_client | 2,057 |
| Proxy & Services | proxy, services, infra_prompt | 1,724 |
| CLI & Utilities | cli, graph_cli, chunks_cli, metrics, token_counter, _text | 1,498 |
| Stubs | graph_worker, session_summarizer, topic_node | 55 |

### Top files by LOC

1. facade.py — 1,848
2. proxy.py — 1,303
3. structural_parser.py — 1,286
4. graph.py — 1,069
5. cli.py — 787

### Circular dependencies

**None found.** The dependency graph is a DAG. facade.py is the hub.

### Mermaid graph

See `acervo_dependency_graph.mermaid`

---

## 2. Module Audit

### acervo/facade.py (1,848 LOC) — GOD MODULE

- **Responsibility:** Main Acervo class — prepare(), process(), S1/S2/S3 pipeline, entity persistence, context assembly, node activation, graph traversal, metrics, reindexing, web facts, vector store sync, document management
- **Key classes:** `Acervo`, `PrepareResult`
- **Depends on:** graph, extractor, context_builder, context_index, synthesizer, s1_unified, topic_detector, specificity, reindexer, layers, ontology, metrics, llm
- **Depended on by:** proxy, cli, __init__
- **Issues:**
  - **God class.** 1,848 lines, 40+ methods. prepare() alone is 548 lines (603-1151). Mixes pipeline orchestration, entity persistence, context assembly, node activation, graph traversal, vector store management, metrics, file management, and web fact extraction.
  - **Two mega-methods.** prepare() (548 lines) and process() (109 lines) are the only entry points. All logic is inline, not delegated to stage-specific modules.
  - **S2 activation logic is embedded.** `_find_active_node_ids()` (130 lines) and `_gather_graph_nodes()` (62 lines) are private methods of Acervo, not a separate module. This makes testing and reuse impossible.
  - **Project overview building is inline.** `_build_project_overview()` and `_build_file_list()` are 80 lines that could be in context_builder.

### acervo/proxy.py (1,303 LOC)

- **Responsibility:** Transparent LLM proxy with context enrichment. Handles both Anthropic and OpenAI API formats.
- **Key classes:** `AcervoProxy`
- **Depends on:** config, infra_prompt, facade (via self._acervo)
- **Depended on by:** cli
- **Issues:**
  - **Dual API format handling.** Every operation (turn detection, enrichment, history windowing, streaming, tool-call detection) has Anthropic AND OpenAI code paths. ~60% duplication.
  - **Silent process() failure.** Line 1268: `except Exception as e: log.warning(...)` — if S1.5 crashes, entities are lost silently.
  - **Post-stream extraction is fragile.** Line 1063-1064: only calls process() if `accumulated_text and not accumulated_tool_calls`. Race conditions possible.

### acervo/graph.py (1,069 LOC)

- **Responsibility:** Knowledge graph persistence. Nodes, edges, facts, dedup.
- **Key classes:** `TopicGraph`
- **Depends on:** layers, ontology
- **Depended on by:** facade, cli, graph_cli, curator, s1_5_graph_update, indexer, synthesizer, reindexer, file_ingestor, context_index
- **Issues:**
  - **Manual save() calls.** Graph mutations don't auto-persist. 15 call sites across 8 files. Missing a save() = lost data.
  - **No transaction semantics.** Crash between upsert() and save() = orphan state.
  - **Dedup is Jaccard-based.** Purely string similarity, no semantic dedup. Two nodes "Next.js" and "NextJS" won't merge automatically.

### acervo/s1_unified.py (552 LOC)

- **Responsibility:** Combined topic classifier + knowledge extractor (S1 stage).
- **Key classes:** `S1Unified`, `S1Result`, `TopicResult`
- **Depends on:** extractor, _text, ontology, llm, topic_detector
- **Depended on by:** facade
- **Issues:**
  - **Parser tolerates but doesn't handle nested relations.** Fixed in this session but was a silent data loss for months.
  - **Validation rejects entities not in text.** Line 445: `if name_lower not in conv_lower` — rejects entities the model invented. Good for hallucination prevention, but also rejects legitimate entities spelled differently.
  - **Entity blacklist is hardcoded.** `_GARBAGE_ENTITY_EXACT` and `_GARBAGE_ENTITY_PATTERNS` can't be configured per project.

### acervo/s1_5_graph_update.py (359 LOC)

- **Responsibility:** Post-response graph curation and assistant entity extraction (S1.5 stage).
- **Key classes:** `S1_5GraphUpdate`, `S1_5Result`
- **Depends on:** extractor, graph, layers, llm, ontology
- **Depended on by:** facade
- **Issues:**
  - **Depends on S1 having extracted entities.** If S1 returns 0 entities (parser bug), S1.5 has empty `new_entities_json` and can't curate.
  - **save() is called inside apply_s1_5_result.** This is one of the few places where graph persistence is coupled with business logic.

### acervo/context_index.py (249 LOC)

- **Responsibility:** 3-layer context stack management. Builds system + warm + hot + user message arrays.
- **Key classes:** `ContextIndex`
- **Depends on:** graph, llm, synthesizer, token_counter, _text
- **Depended on by:** facade
- **Issues:**
  - **Warm content comes from facade, not from the graph.** ContextIndex receives `warm_override` from prepare() — it doesn't query the graph itself. If prepare() sends empty warm, context is empty.
  - **Compaction uses LLM.** `maybe_compact()` calls the utility model to summarize. This can fail silently.

### acervo/extractor.py (580 LOC)

- **Responsibility:** Entity/fact/relation extractors. 4 variants: Conversation, Text, Search, RAG.
- **Key functions:** `_parse_first_json`, `_clean_response`
- **Depends on:** llm, _text, ontology
- **Depended on by:** facade, s1_unified, s1_5_graph_update
- **Issues:**
  - **_parse_first_json was too strict.** Fixed in this session (greedy + repair fallbacks).
  - **4 extractor classes share no base class.** ConversationExtractor, TextExtractor, SearchExtractor, RAGExtractor have similar structure but no shared interface.

---

## 3. Data Flow Traces

### Scenario A: Conversation turn (no indexed project)

```
proxy._handle_openai(request)                          [proxy.py:225]
  │
  ├─ _compose_system_message_openai(body)               [proxy.py:233]
  │    └─ Adds ACERVO INFRA block to system message
  │
  ├─ _is_new_user_turn_openai(body)                     [proxy.py:697]
  │    └─ Returns True if last message is role="user"
  │
  ├─ _enrich_openai(body)                               [proxy.py:796]
  │    │
  │    ├─ if not self._acervo: RETURN (pass-through)    [proxy.py:802]
  │    │
  │    ├─ self._acervo.prepare(user_text, history)      [proxy.py:822]
  │    │    │
  │    │    ├─ S1: s1_unified.run()                     [facade.py:665]
  │    │    │    ├─ Embeds user text (if embedder)       [facade.py:631]
  │    │    │    ├─ Topic hints (L1/L2)                  [facade.py:639]
  │    │    │    ├─ Graph summary for context            [facade.py:642]
  │    │    │    ├─ LLM call to extractor model          [s1_unified.py:238]
  │    │    │    └─ Parse JSON, validate entities        [s1_unified.py:248-271]
  │    │    │
  │    │    ├─ Persist S1 entities to graph              [facade.py:711-756]
  │    │    │    └─ GATED: if s1_extraction.entities     [facade.py:711]
  │    │    │
  │    │    ├─ S2: _find_active_node_ids()              [facade.py:705]
  │    │    │    └─ Text matching on all graph nodes     [facade.py:1628]
  │    │    │    └─ Entity neighbor expansion            [facade.py:1726]
  │    │    │
  │    │    ├─ S2: _gather_graph_nodes(active_ids)      [facade.py:774]
  │    │    │    └─ Collect nodes + relations + neighbors
  │    │    │    └─ GATE: neighbors only if has facts    [facade.py:1490]
  │    │    │
  │    │    ├─ S2: Vector search                        [facade.py:854]
  │    │    │    └─ GATED: if vector_store AND not skip  [facade.py:854]
  │    │    │    └─ Conversation mode: vector_store=None → SKIP
  │    │    │
  │    │    ├─ Chunk ranking + scoring                  [facade.py:873-940]
  │    │    │    └─ Nodes without facts/relations/summary → SKIP [facade.py:886]
  │    │    │
  │    │    ├─ S3: Project overview                     [facade.py:958]
  │    │    │    └─ If no files and no description → ""
  │    │    │    └─ Conversation mode: has description only
  │    │    │
  │    │    ├─ S3: Chunk selection by budget            [facade.py:987]
  │    │    │    └─ Verified vs conversation separation
  │    │    │
  │    │    └─ S3: build_context_stack()                [facade.py:1008]
  │    │         └─ warm_override + hot layer + user msg
  │    │
  │    ├─ Inject context as [VERIFIED CONTEXT] block    [proxy.py:840-865]
  │    └─ Store for re-injection on tool continuations  [proxy.py:858]
  │
  ├─ Forward to upstream LLM (stream)                   [proxy.py:259]
  │
  └─ After stream completes:
       _process_response_text(body, accumulated_text)    [proxy.py:1064]
         │
         └─ self._acervo.process()                      [proxy.py:1252]
              │
              ├─ Skip if "no data" response              [facade.py:1178]
              ├─ Build S1 entities JSON                   [facade.py:1195]
              ├─ S1.5: S1_5GraphUpdate.run()             [facade.py:1210]
              │    └─ LLM call to extractor model
              │    └─ Parse merges, corrections, new entities
              ├─ apply_s1_5_result() → graph.save()      [facade.py:1218]
              ├─ Enrich new entities                      [facade.py:1242]
              └─ Mark touched nodes                       [facade.py:1250]
```

### Scenario B: Conversation turn WITH indexed project

Same as A except:

| Step | Difference |
|------|-----------|
| S2 Vector search | `vector_store` exists → performs semantic search |
| S2 File gathering | `workspace_path` exists → reads linked file contents |
| S2 Symbol injection | section/symbol nodes get summary/content as pseudo-facts |
| S3 Project overview | Has synthesis node overview + file listing |
| S3 Chunk selection | Has verified chunks (from indexed files) separate from conversation chunks |
| Post-LLM reindex | `reindexer` exists → checks for stale files |

### Scenario C: Index → Curate → Synthesize

```
CLI: acervo index                                       [cli.py:98]
  ├─ Create Indexer(graph, llm, embedder, parser)       [cli.py:120]
  ├─ Phase 1: Structural parse (tree-sitter)            [indexer.py:182]
  │    └─ graph.upsert_file_structure() per file
  │    └─ graph.save()                                   [indexer.py:227]
  ├─ Phase 2: Semantic enrichment                        [indexer.py:229]
  │    └─ SemanticEnricher.enrich_file() per file
  │    └─ Embed chunks → vector_store
  │    └─ graph.save()                                   [indexer.py:274]
  └─ DependencyResolver → cross-file edges

CLI: acervo curate (MANUAL, separate command)            [cli.py not shown]
  ├─ Create batches of file groups                       [curator.py:109]
  ├─ LLM-powered relationship discovery per batch        [curator.py:123]
  ├─ Apply: merges, type corrections, facts              [curator.py:270]
  └─ graph.save()                                        [curator.py:160]

CLI: acervo synthesize (MANUAL, separate command)        [cli.py:267]
  ├─ Group files by directory                            [graph_synthesizer.py:118]
  ├─ Generate project overview (LLM)                     [graph_synthesizer.py:136]
  ├─ Generate module summaries (LLM)                     [graph_synthesizer.py:156]
  └─ graph.save()                                        [graph_synthesizer.py:212]
```

---

## 4. The S2 Problem

### All node activation/gathering functions

| Function | File:Line | When Called | Traverses Edges? | Does Vector Search? |
|----------|-----------|------------|-------------------|---------------------|
| `_find_active_node_ids()` | facade.py:1628 | Every prepare() call | Yes (contains + entity) | No |
| `_gather_graph_nodes()` | facade.py:1453 | After _find_active_node_ids | Yes (1-level neighbors) | No |
| Vector search | facade.py:854 | After gather, if vector_store exists | No (returns scored hits) | Yes |
| `materialize()` | facade.py:1430 | Manual API call | Via _find_active_node_ids | No |

### Why S2 failed in conversation mode

1. **Entity expansion was missing** (fixed this session) — entities didn't traverse edges to find neighbors
2. **Neighbor expansion gated on facts** — `if nbr.get("facts"):` at facade.py:1490 skips neighbors without facts. Project nodes created by S1 often have 0 facts → skipped
3. **But `_relations` are still attached** — even without facts, nodes get `_relations` from _gather_graph_nodes, and chunks are built from relations (line 920-926). So the fix should work.
4. **No vector store in conversation mode** — only keyword/label matching. No semantic search fallback.

### The real gap

Even after the entity expansion fix, there's a subtlety: `_gather_graph_nodes` neighbor expansion (line 1481-1494) only adds neighbors WITH FACTS. But the expansion we added in `_find_active_node_ids` adds ALL entity neighbors to the active set. So the active set is correct, and `_gather_graph_nodes` will fetch those nodes directly (not via neighbor expansion). They'll get `_relations` attached, and chunks will be built from those relations.

---

## 5. Architecture Smells

### 5.1 God Module
facade.py is 1,848 lines with 40+ methods. It handles: pipeline orchestration, entity persistence, context assembly, node activation, graph traversal, metrics recording, reindexing, vector store sync, web fact extraction, document management. Should be split into at least 4 modules: pipeline orchestrator, node activator, context assembler, entity persister.

### 5.2 Dual Code Paths
**Intent-based:** overview vs specific vs chat vs followup — 4 different behaviors in S2 activation (line 1656-1741) and S3 assembly (line 966-1004).

**Workspace-based:** 3 gates on `self._workspace_path` (file gathering, symbol injection, reindexing). Conversation mode hits NONE of these.

**Vector store-based:** 3 gates on `self._vector_store`. Conversation mode has no vector store → no semantic search.

### 5.3 Silent Failures
- proxy.py:1268 — process() failure swallowed
- proxy.py:791-792 — enrichment crash swallowed (Anthropic path)
- facade.py:798 — hash check failure silent (no log)
- graph.py:139-148 — JSON decode failure defaults to empty (no warning)

### 5.4 State Management
4 sources of truth for graph state:
1. JSON files on disk (nodes.json, edges.json)
2. In-memory TopicGraph instance (in proxy's facade)
3. Studio's understanding via `/acervo/status` API
4. Telemetry collector's `_prev_graph` counter

When graph changes (S1.5 persists), only #1 and #2 update synchronously. #3 updates on next poll (5s). #4 updates on next GraphUpdated event (may never fire in proxy mode).

### 5.5 Telemetry Accuracy
- `graph.node_count/edge_count` was always 0 in proxy mode — fixed by querying `/acervo/status` after process().
- `s1.entities_extracted` was 0 when parser failed — root cause was JSON parser bug, now fixed.
- `s1_detection` in stage_data showed 0 entities even when raw_response had 10 — because detection reports POST-validation counts, not pre-validation.

---

## 6. Architecture Proposal

### 6.1 Clean Architecture (target)

```
acervo/
├── core/
│   ├── graph.py          # TopicGraph (unchanged)
│   ├── layers.py         # Layer enum (unchanged)
│   ├── ontology.py       # Type registry (unchanged)
│   └── token_counter.py  # Token estimation (unchanged)
├── pipeline/
│   ├── orchestrator.py   # prepare() + process() — thin orchestrator only
│   ├── s1_extractor.py   # S1 Unified (from s1_unified.py)
│   ├── s2_activator.py   # Node activation + gathering (from facade.py)
│   ├── s3_assembler.py   # Context assembly (from facade.py + context_builder.py)
│   ├── s15_curator.py    # S1.5 Graph Update (from s1_5_graph_update.py)
│   └── entity_persister.py # Graph write operations (from facade.py)
├── context/
│   ├── context_index.py  # 3-layer stack (unchanged)
│   ├── synthesizer.py    # Node rendering (unchanged)
│   └── topic_detector.py # Topic detection (unchanged)
├── indexing/
│   ├── indexer.py        # Structural + semantic indexing
│   ├── curator.py        # Batch curation
│   ├── synthesizer.py    # Project overview generation
│   └── ...
├── proxy/
│   ├── proxy.py          # HTTP proxy (simplified, delegates to pipeline)
│   └── infra_prompt.py
└── ...
```

### 6.2 Minimal Refactor (fix current bugs)

**Don't restructure.** The fixes applied this session (JSON parser, nested relations, entity expansion, graph counter) address the immediate bugs. The remaining issue is:

1. **Verify the entity expansion fix works end-to-end** — run the 4-turn test
2. **If warm_tokens is still 0**, the issue is in chunk building (line 886: skip nodes without facts/relations/summary). Nodes need `_relations` attached by `_gather_graph_nodes` to survive this filter.

### 6.3 Proper Refactor (v0.5)

1. **Split facade.py** into pipeline stages (orchestrator, activator, assembler, persister)
2. **Unify Anthropic/OpenAI paths** in proxy.py (abstract message format layer)
3. **Auto-save graph** on mutation (or at minimum, on prepare/process boundaries)
4. **Single S2 code path** — no behavioral differences between conversation and project mode. Graph traversal should work the same regardless.

### 6.4 Technical Debt (ordered by impact)

| Priority | Debt | Location | Impact | Fix |
|----------|------|----------|--------|-----|
| 1 | God module | facade.py | Every bug requires reading 1,848 lines | Split into pipeline stages |
| 2 | Dual API format | proxy.py | 60% code duplication, bugs in one path | Abstract message format |
| 3 | Manual graph.save() | 15 call sites | Missing save = data loss | Auto-save on mutation |
| 4 | Silent failures | proxy.py, facade.py | Bugs disappear silently | Log all errors at ERROR level |
| 5 | No vector store in conversation | facade.py | No semantic search for chat-only mode | Initialize basic vector store always |
| 6 | 4 extractor classes | extractor.py | No shared interface | Create base class |
| 7 | Hardcoded entity blacklist | s1_unified.py | Can't customize per project | Move to config |
