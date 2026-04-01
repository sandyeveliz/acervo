# Acervo Pipeline — Complete Technical Documentation

> Last updated: 2026-03-31 (v0.4.0)

---

## Overview

Acervo is a context proxy that enriches LLM conversations with knowledge from indexed project files. The pipeline has two main phases:

1. **Offline**: Index → Curate → Synthesize (builds the knowledge graph)
2. **Online**: S1 → S2 → S3 → LLM → Process (enriches each conversation turn)

---

## 1. Indexation Pipeline (Phase 1 + Phase 2 + Phase 3)

**Entry point:** `Indexer.index()` in `acervo/indexer.py`

### Phase 1: Structural Analysis (no LLM, fast)

**Input:** workspace directory + file extensions + excludes

**Processing:**
1. **Scan files** — recursively walk workspace, filter by extension, skip excludes
2. **Parse each file** via `StructuralParser.parse()`:
   - Markdown: split by heading hierarchy with parent tracking
   - Code (Python/TS/JS): tree-sitter AST if available, else regex fallback
   - EPUB: extract spine-ordered HTML documents, convert to markdown with headings
   - PDF: extract text per page via PyMuPDF
   - Plaintext: split by double blank lines into ~50-line sections
3. **Hash check**: skip file if content_hash unchanged from existing node
4. **Upsert to graph** via `graph.upsert_file_structure()`:
   - File node (kind="file")
   - Folder nodes (kind="folder") with hierarchy
   - Section/Symbol nodes (kind="section"/"symbol") per structural unit
   - "contains" edges (folder→file, file→section)
5. **Dependency resolution**: extract imports → "imports" edges
6. **Save**: `nodes.json` + `edges.json`

**Output:** Graph with file/folder/section/symbol nodes + structural edges

### Phase 2: Semantic Enrichment (optional, requires LLM + Embedder)

**Processing:**
1. **Chunking** via `SemanticEnricher._create_chunks()`:
   - Code: one chunk per function/class, large entities split at blank lines
   - Prose (epub/pdf/txt): paragraph-cluster splitting (~500 tokens per chunk)
   - Markdown: one chunk per heading, large sections split at paragraphs
2. **Parallel enrichment**:
   - `_generate_embeddings()` → embed each chunk (prepend file context) → ChromaDB
   - `_generate_summaries()` → LLM 3B → `{summary, topics[], implicit_relations[]}`
3. **Link chunks** to graph nodes via `graph.link_chunks(node_id, chunk_ids)`
4. **Attach summaries** to node attributes (summary, topics, implicit_relations)
5. **Semantic edges** (within-file): nodes sharing topics → "related_to" (weight 0.6)

### Phase 3: Cross-file Semantic Edges

- Group all nodes by topic across entire graph
- For topics in 2-50 nodes: create "related_to" edges (weight 0.5) between nodes in different files
- Max 20 pairs per topic

---

## 2. Curation Pipeline

**Entry point:** `curate_graph()` in `acervo/curator.py`
**Trigger:** Manual (Studio UI button or CLI)

**Processing:**
1. **Batch files** (~10 per batch, grouped by directory)
2. **LLM analysis per batch**: discover series ordering, shared characters, thematic connections
3. **Apply results**: create entity nodes (source="curation"), relation edges, facts

**LLM prompt asks for:**
- `entities[]` — discovered meta-concepts (e.g., "Harry Potter Saga", "Authentication Module")
- `relations[]` — part_of, sequel_of, created_by, shares_characters_with, shares_theme_with
- `facts[]` — textual facts about discovered entities

**Output:** New entity nodes + relation edges enriching the graph

---

## 3. Synthesis Pipeline

**Entry point:** `synthesize_graph()` in `acervo/graph_synthesizer.py`
**Trigger:** Manual (Studio UI button or CLI)

**Processing:**
1. **Detect content type**: scan file extensions → "prose" (>70% epub/md/txt) or "code"
2. **Project Overview** (Tier 1):
   - LLM generates 200-word overview from file list + curation entities
   - Code prompt: tech stack, architecture, key components, patterns
   - Prose prompt: authors, genre, time period, characters, themes, relationships
   - Creates `synthesis:project_overview` node (kind="synthesis")
3. **Module Summaries** (Tier 2):
   - Group files by top-level directory
   - LLM generates 100-word summary per module
   - Creates `synthesis:module_{name}` nodes

**Output:** Synthesis nodes used by `_build_project_overview()` in prepare()

---

## 4. prepare() Pipeline (S1 → S2 → S3)

**Entry point:** `Acervo.prepare(user_text, history)` in `acervo/facade.py`

### S1 — Topic Detection + Entity Extraction

1. **L1/L2 hints** (no LLM):
   - L1: keyword match against current topic + active nodes
   - L2: cosine similarity from user embedding to topic embedding
2. **Graph summary**: rank top-20 existing nodes by relevance
3. **S1 Unified LLM call** (fine-tuned 9B extractor):
   - Input: existing nodes + topic hint + previous assistant + user message
   - Output: `{intent, topic: {action, label}, entities[], relations[], facts[]}`
   - Intent: "overview" | "specific" | "chat"
4. **Keyword fallback**: promote to "overview" if keywords match ("cuantos", "how many", "structure")
5. **Persist extraction**: upsert entities/relations/facts to graph (sync, so S2 sees them)

### S2 — Gather

1. **Activate nodes** via `_find_active_node_ids()`:
   - Overview intent: ALL file + folder + synthesis nodes (skip sections/symbols)
   - Specific intent: text-match on labels (prefix, substring, multi-word)
2. **Gather graph nodes** via `_gather_graph_nodes()`:
   - Fetch active nodes with relations
   - Expand: folders → files, 1-level neighbors
3. **Gather file contents**: read linked files (line-precise for symbols)
4. **Node-scoped chunk search** (if specificity=="specific"): search ChromaDB within activated chunk_ids
5. **Global vector search** (if intent != "overview"): top-5 semantic matches
6. **Convert to RankedChunks**: facts, summaries, relations → scored chunks
   - Verified sources: `verified_fact`, `verified_summary`, `verified_file`
   - Conversation sources: `conversation_fact`, `conversation_relation`
   - Vector hits: `vector`

### S3 — Context Assembly

1. **Project overview**: `_build_project_overview()` — synthesis node text + file tree
2. **Intent-based selection**:
   - Overview: verified chunks only (no vector noise)
   - Chat: overview only (lightweight)
   - Specific: all chunks, budget-gated
3. **Budget selection**: `select_chunks_by_budget(chunks, 400 tokens)`
4. **Build context stack**: `[system] + [warm: verified context] + [hot: last N messages] + [user]`

**Output:** `PrepareResult` with context_stack ready for LLM

---

## 5. process() Pipeline (Post-LLM)

**Entry point:** `Acervo.process()` in `acervo/facade.py`

1. **Skip check**: if "I don't have information" + no web results → skip
2. **Web facts**: persist web-sourced entities/facts (layer=UNIVERSAL, source="web")
3. **S1.5 Graph Update**: LLM deduplicates entities, fixes types, extracts assistant-spoken facts
4. **Enrich nodes**: promote placeholder→enriched, background-index to vector store
5. **Mark touched nodes**: update last_active, increment session_count
6. **Deferred reindex**: re-parse files marked stale during S2

---

## 6. Complete Data Flow

```
[acervo index ./project]
    │
    ├── Phase 1: Structural Parsing
    │   ├── StructuralParser.parse() → FileStructure (units + hash)
    │   ├── graph.upsert_file_structure() → file/folder/section/symbol nodes
    │   ├── DependencyResolver → import edges
    │   └── graph.save() → nodes.json + edges.json
    │
    ├── Phase 2: Semantic Enrichment (if LLM + Embedder configured)
    │   ├── _create_chunks() → paragraph-cluster chunks (~500 tokens)
    │   ├── _generate_embeddings() → ChromaDB vectors
    │   ├── _generate_summaries() → LLM 3B → summary/topics/relations
    │   ├── graph.link_chunks() → chunk_ids on nodes
    │   └── _create_semantic_edges() → related_to edges (within-file)
    │
    └── Phase 3: Cross-file Edges
        └── Topic-grouped related_to edges (weight 0.5, cross-file)

[curate] (manual trigger)
    ├── Batch files (~10 per batch)
    ├── LLM discovers: series, characters, themes, authorship
    └── graph.upsert_entities() → entity nodes + relation edges

[synthesize] (manual trigger)
    ├── LLM project overview → synthesis:project_overview node
    └── LLM module summaries → synthesis:module_* nodes

[user asks a question]
    │
    ├── S1: L1/L2 hints + S1 Unified LLM
    │   └── Output: intent, topic, entities, relations, facts
    │
    ├── S2: Gather
    │   ├── Activate nodes (intent-aware)
    │   ├── Gather graph nodes + file contents
    │   ├── Vector search (skip for overview)
    │   └── Convert to RankedChunks
    │
    ├── S3: Context Assembly
    │   ├── Project overview (synthesis node)
    │   ├── Intent-based chunk selection
    │   ├── Budget-gated selection (~400tk)
    │   └── Build context stack
    │
    ├── LLM receives: [system + [VERIFIED CONTEXT] + history + message]
    │
    └── process(): S1.5 graph update, fact extraction, node enrichment
```

---

## 7. Current State of Test Projects

| Project | Nodes | Edges | Files | Sections | Symbols | Folders | Entities | Synthesis |
|---------|-------|-------|-------|----------|---------|---------|----------|-----------|
| P1 TODO App | 222 | 284 | 31 | 38 | 134 | 17 | 2 | 0 |
| P2 Books | 250 | 1,358 | 7 | 240 | 0 | 1 | 1 | 1 |
| P4 PM Docs | 102 | 349 | 11 | 84 | 0 | 3 | 0 | 4 |

**P1** — Code project. 134 symbols (functions, classes) from tree-sitter parsing. No synthesis node (not yet synthesized). 2 entities from conversation testing.

**P2** — Prose project. 240 sections (epub chapter headings). 1 synthesis node with Harry Potter collection summary. 1,358 edges (mostly cross-chapter semantic links).

**P4** — Markdown docs. 84 sections (markdown headings). 4 synthesis nodes (project overview + 3 module summaries). 349 edges.

---

## 8. Key Observations

1. **Two-phase indexation**: Phase 1 (parsing) is fast and LLM-free. Phase 2 (enrichment) is expensive but optional.

2. **Intent drives everything in S3**: Overview queries get clean project structure + synthesis. Specific queries get vector chunks + detailed content. Chat queries get minimal context.

3. **Graph IS the memory**: Conversation history is temporary (hot layer, last 2 turns). The graph persists everything. If it's not in the graph, the LLM says "I don't have verified information."

4. **Chunk linking is bidirectional**: Chunks link to multiple nodes (file + matching symbols by line range). Node-scoped retrieval searches only within activated chunks.

5. **Synthesis is the key to overview quality**: Without synthesis, overview questions only get file tree structure. With synthesis, they get a rich project-level summary.

6. **Vector search is skipped for overview**: This prevents section-level noise (random chapter summaries) from polluting high-level answers.

7. **Curation creates meta-knowledge**: It discovers relationships between files that structural parsing can't see (e.g., "these 7 epubs are a book series by the same author").

8. **S1.5 is the conversation-graph bridge**: After each LLM response, S1.5 deduplicates entities, fixes types, and extracts new facts from what the assistant said.
