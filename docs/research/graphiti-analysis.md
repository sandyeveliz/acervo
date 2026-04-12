# Graphiti Deep Analysis — Adoption Plan for Acervo

**Repo:** https://github.com/getzep/graphiti (getzep/graphiti)
**Version analyzed:** graphiti-core 0.28.2
**License:** Apache-2.0 — copy with attribution OK
**Local clone:** `d:/Development/graphiti-analysis/`

---

## TL;DR — Top 5 takeaways

1. **Graphiti's Kuzu driver is our easiest integration point.** LadybugDB (our store) is a Kuzu fork. Graphiti already has a full Kuzu driver (`graphiti_core/driver/kuzu_driver.py`) with Cypher schema, vector search, fulltext search, and full CRUD. This means most of Graphiti's persistence layer is *already aligned* with our stack.
2. **Hybrid dedup (determinístico → LLM) is the single biggest idea to adopt.** Their `dedup_helpers.py` does exact-normalize → MinHash+LSH fuzzy → entropy gate → only escalates to LLM when unreliable. It's ~300 lines, pure Python stdlib, zero external deps, and solves exactly the "fuzzy entity matching" problem you just committed to (`634cc6c`).
3. **Bi-temporal edges with LLM-driven contradiction detection.** `EntityEdge` has `valid_at`, `invalid_at`, `expired_at`, `reference_time` + a two-phase resolution: search existing edges between the same node pair → LLM marks duplicates vs contradictions → deterministic time-range arbitration invalidates stale facts. This is way more principled than what we have.
4. **LLM-free retrieval exists via RRF-fused BM25 + vector + BFS.** Retrieval is 100% LLM-free by default. `rrf()` is 15 lines of code. The heavy lifting happens in the DB via fulltext/vector indexes, which Kuzu (== Ladybug) supports natively.
5. **The extraction prompts are worth stealing wholesale.** Their entity extraction prompt (`extract_nodes.py`) has extensive "NEVER extract X" guardrails (pronouns, bare relational terms, generic nouns) that directly solve the kind of garbage-extraction problems we see in `tests/integration/test_case_scenarios.py`. The bare-term disambiguation rule alone ("dad" → "Nisha's dad") is gold.

---

## 1. Architecture overview

### Pipeline diagram (`add_episode`)

```
add_episode(name, body, reference_time, ...)
  │
  ├─ retrieve_episodes(last_n=10)              ── DB read, no LLM
  │
  ├─ extract_nodes(clients, episode, prev)     ── 1 LLM call
  │    └─ prompt_library.extract_nodes.extract_message|text|json
  │    └─ structured output: ExtractedEntities (pydantic)
  │    └─ _collapse_exact_duplicate_extracted_nodes()
  │
  ├─ resolve_extracted_nodes(clients, nodes)   ── Hybrid dedup
  │    ├─ _semantic_candidate_search()         ── vector search per node (no LLM)
  │    ├─ _build_candidate_indexes()           ── MinHash+LSH index (in-memory)
  │    ├─ _resolve_with_similarity()           ── exact-name + fuzzy deterministic
  │    │    └─ escalates unresolved to LLM
  │    └─ _resolve_with_llm()                  ── 1 LLM call (batched) for ambiguous
  │        └─ prompt_library.dedupe_nodes.nodes
  │
  ├─ _extract_and_resolve_edges(episode, nodes, ...)
  │    ├─ extract_edges(...)                   ── 1 LLM call
  │    │    └─ prompt_library.extract_edges.edge
  │    │    └─ structured output: ExtractedEdges with valid_at/invalid_at
  │    │    └─ validate entity names are in nodes list (reject fabrications)
  │    └─ resolve_extracted_edges(...)         ── Bi-temporal dedup
  │         ├─ EntityEdge.get_between_nodes()  ── exact endpoint match (no LLM)
  │         ├─ search(EDGE_HYBRID_SEARCH_RRF)  ── BM25+vector, no LLM
  │         ├─ resolve_extracted_edge() per edge:
  │         │    ├─ fast path: exact fact text reuse existing
  │         │    └─ 1 LLM call per edge (batched) → duplicate + contradicted
  │         └─ resolve_edge_contradictions()   ── deterministic temporal arbitration
  │
  ├─ extract_attributes_from_nodes(...)        ── 1 LLM call per node w/ custom type
  │                                               + 1 batched LLM call for summaries
  │
  └─ _process_episode_data() + save to graph   ── DB write
```

**LLM calls per turn (baseline, no custom types):**
- 1 extract_nodes
- 0–1 dedupe_nodes (only if deterministic dedup fails for any node)
- 1 extract_edges
- N resolve_extracted_edge (one per edge, parallelized)
- 1 batched summaries (if any nodes need re-summary)

→ **Typical: 3–5 LLM calls per turn**, most with small context. Retrieval (search) is always 0 LLM calls unless you opt into `cross_encoder` reranker.

### Layer map

| Directory | Role |
|---|---|
| `graphiti_core/graphiti.py` | Main orchestrator (`Graphiti` class, `add_episode`, `search`) — 1722 lines, god-class like our `facade.py` |
| `graphiti_core/nodes.py` | `EntityNode`, `EpisodicNode`, `CommunityNode`, `SagaNode` pydantic models + save/load |
| `graphiti_core/edges.py` | `EntityEdge`, `EpisodicEdge`, etc. pydantic models + save/load |
| `graphiti_core/prompts/` | All LLM prompts as pure-python functions returning `list[Message]` |
| `graphiti_core/utils/maintenance/` | Graph construction ops: `node_operations`, `edge_operations`, `dedup_helpers`, `community_operations` |
| `graphiti_core/search/` | Retrieval: `search.py` orchestrator, `search_utils.py` (low-level DB queries + reranking), `search_config.py` + recipes |
| `graphiti_core/driver/` | Pluggable DB layer: Neo4j, FalkorDB, **Kuzu**, Neptune |
| `graphiti_core/llm_client/` | Pluggable LLM layer: OpenAI, Anthropic, Gemini, Groq, **OpenAIGenericClient** (custom base_url) |
| `graphiti_core/embedder/` | OpenAI, Azure, Gemini, Voyage |

---

## 2. Component-by-component analysis

### 2.1 Entity Extraction

**How it works:**
- Single LLM call per episode. Three prompt variants by source type: `extract_message` (conversation), `extract_text` (prose), `extract_json` (structured data). Selected in [node_operations.py:160-178](../../../graphiti-analysis/graphiti_core/utils/maintenance/node_operations.py#L160-L178).
- Structured output via `ExtractedEntities` (pydantic). Each entity has a `name` + `entity_type_id` (integer ref into a passed entity-types catalog).
- Ontology is **configurable by the user**: user passes `entity_types: dict[str, type[BaseModel]]`, each pydantic model with a `__doc__` as the type description. The catalog is serialized into the prompt with IDs 0..N and the model picks an ID. `Entity` (ID 0) is the fallback. See [node_operations.py:109-138](../../../graphiti-analysis/graphiti_core/utils/maintenance/node_operations.py#L109-L138).
- Post-extraction, `_collapse_exact_duplicate_extracted_nodes()` merges same-message duplicates by normalized name, preferring the more specific label.

**The prompt itself is the gold here.** See [`prompts/extract_nodes.py:78-190`](../../../graphiti-analysis/graphiti_core/prompts/extract_nodes.py#L78-L190). Key rules that we should adopt verbatim:
- Explicit "NEVER extract" list: pronouns, feelings, generic common nouns, generic media/event nouns, broad institutional nouns, sentence fragments, **bare relational terms** ("dad" unless qualified as "Nisha's dad"), **bare generic objects** ("supplies").
- "Could this have its own Wikipedia article?" as the specificity test.
- "Always use the most specific form" rule: "road cycling" not "cycling", "wool coat" not "coat".
- 4 high-quality few-shot examples inline in the prompt.

**Recommendation: ADOPT (prompt) + ADAPT (code).**
- Copy the `extract_message` / `extract_text` prompts into our `acervo/extraction/prompts/` directory with attribution. Our current `s1_unified.py` extraction is probably producing exactly the kind of garbage these rules block.
- Replace our ad-hoc entity extraction with the pydantic-BaseModel + `__doc__` pattern for ontology. It's clean, matches what we already do with pydantic-ai, and makes custom types per-project trivial.
- Don't copy `ExtractedEntity.entity_type_id: int` — requires the LLM to pick an integer from a lookup. Our approach of `entity_type: str` with name-based validation is easier for local models. Qwen2.5:7b may struggle with the integer indirection.

### 2.2 Entity Resolution (dedup)

**This is the single most sophisticated and copy-worthy file in the repo.** [`utils/maintenance/dedup_helpers.py`](../../../graphiti-analysis/graphiti_core/utils/maintenance/dedup_helpers.py) (296 lines, zero external deps — only stdlib `re`, `math`, `hashlib.blake2b`, `functools.lru_cache`).

**The algorithm** (from [node_operations.py:490-571](../../../graphiti-analysis/graphiti_core/utils/maintenance/node_operations.py#L490-L571)):

1. **Semantic candidate search** — for each extracted node, do a cosine similarity search (`node_similarity_search`) against existing nodes, returning top-15 with `min_score=0.6`. This limits the dedup universe to plausible candidates and avoids quadratic comparisons.
2. **Build in-memory indexes** for those candidates:
   - `normalized_existing: dict[str, list[EntityNode]]` — exact-name (lowercased, whitespace-collapsed) lookup.
   - `shingles_by_candidate: dict[uuid, set[str]]` — 3-gram character shingles.
   - `lsh_buckets: dict[(band_idx, band), list[uuid]]` — MinHash LSH: 32 permutations, 4-permutation bands → 8 bands.
3. **Try exact match first** — normalized name hit → if exactly 1 candidate, accept; if >1, escalate to LLM; if 0, try fuzzy.
4. **Entropy gate** — compute Shannon entropy over characters. Names shorter than 6 chars AND < 2 tokens, or entropy < 1.5, are considered unreliable for fuzzy matching → escalate to LLM. This prevents "Java" from matching "Java" across different meanings via fuzzy.
5. **MinHash LSH fuzzy match** — compute signature, collect candidates from overlapping buckets, compute Jaccard similarity on shingles, accept if ≥ 0.9.
6. **Escalate unresolved to LLM** — single batched call with all unresolved nodes + all their candidates. Prompt: [`prompts/dedupe_nodes.py`](../../../graphiti-analysis/graphiti_core/prompts/dedupe_nodes.py). LLM returns `duplicate_candidate_id: int` or `-1`.
7. **Promote labels** — `_promote_resolved_node()` upgrades generic `Entity` label to the more specific type from the duplicate when one has it. Smart.

**Why this design beats pure-LLM dedup:**
- Deterministic path catches ~80% of cases (exact spelling variants, obvious fuzzy matches like "NYC" vs "New York City" won't pass entropy gate and go to LLM — but "Nisha" vs "Nisha" same exact will resolve deterministically).
- The semantic pre-filter means the LLM only sees ~15 candidates, never the whole graph.
- Entropy gate prevents false positives on short/generic names (the one class of error that's catastrophic).
- MinHash+LSH is O(N) signature construction + O(1) lookup — no pairwise comparisons.

**Recommendation: ADOPT (verbatim copy + attribution).**
- This file is a drop-in replacement for the `fuzzy entity matching` logic you just added in `634cc6c`. It's 296 lines, zero deps, Apache-2.0.
- The `_semantic_candidate_search` step needs our vector store (we can swap `node_similarity_search` for a call to `ChromaStore.similarity_search` or LadybugDB's native vector search if we enable it).
- The LLM escalation step can use our existing `OpenAIClient`/Ollama — the prompt format is generic OpenAI messages.
- **Tunables** (all constants at top of file): `_FUZZY_JACCARD_THRESHOLD=0.9`, `_NAME_ENTROPY_THRESHOLD=1.5`, `_MIN_NAME_LENGTH=6`, `_MINHASH_PERMUTATIONS=32`, `_MINHASH_BAND_SIZE=4`. These are sensible defaults; we may need to lower the Jaccard threshold for Spanish text.

### 2.3 Fact Extraction & Temporal Model

**Facts are edges, not properties.** [`edges.py:263-298`](../../../graphiti-analysis/graphiti_core/edges.py#L263-L298) — `EntityEdge` fields:

```python
class EntityEdge(Edge):
    name: str                          # e.g. "WORKS_AT" — the relation type
    fact: str                          # NL paraphrase: "Alice works at Acme Corp as engineer"
    fact_embedding: list[float] | None
    episodes: list[str]                # source provenance — which episode(s) introduced this
    expired_at: datetime | None        # when we learned it was no longer true (ingestion time)
    valid_at: datetime | None          # when the fact *started being true* (event time)
    invalid_at: datetime | None        # when the fact *stopped being true* (event time)
    reference_time: datetime | None    # episode's valid_at — anchors relative temporal phrases
    attributes: dict[str, Any]         # custom pydantic-schema attributes
```

This is **bi-temporal**: `created_at`/`expired_at` = ingestion time (system clock), `valid_at`/`invalid_at` = event time (when the fact is true in the world). The LLM extracts `valid_at`/`invalid_at` as ISO-8601 strings from the text — see [`prompts/extract_edges.py:130-140`](../../../graphiti-analysis/graphiti_core/prompts/extract_edges.py#L130-L140):

> - Use ISO 8601 with "Z" suffix (UTC)
> - If the fact is ongoing (present tense), set `valid_at` to REFERENCE_TIME.
> - If a change/termination is expressed, set `invalid_at` to the relevant timestamp.
> - Leave both fields `null` if no explicit or resolvable time is stated.

**Extraction prompt** ([prompts/extract_edges.py:64-141](../../../graphiti-analysis/graphiti_core/prompts/extract_edges.py#L64-L141)):
- Takes extracted nodes list as input, prompts LLM to form edges *only between nodes already in the list*.
- Validates every returned `source_entity_name` / `target_entity_name` against the input list → drops fabricated entity names ([edge_operations.py:149-179](../../../graphiti-analysis/graphiti_core/utils/maintenance/edge_operations.py#L149-L179)).
- Drops self-edges.
- Rejects single-entity "state" facts ("Alice feels happy") — requires two distinct entities.

**Contradiction detection & invalidation** ([`edge_operations.py:495-691`](../../../graphiti-analysis/graphiti_core/utils/maintenance/edge_operations.py#L495-L691)):

1. **Fetch candidates** for each new edge:
   - `EntityEdge.get_between_nodes()` — existing edges with *exact same source + target* (duplicates).
   - `search(EDGE_HYBRID_SEARCH_RRF)` scoped by the same group → duplicate candidates.
   - Second `search()` unscoped → invalidation candidates (any edge that might contradict).
2. **Fast path** — if new edge's exact fact text + endpoints already exist, reuse the existing edge (just append episode to `episodes` list). Zero LLM call.
3. **LLM call** ([prompts/dedupe_edges.py](../../../graphiti-analysis/graphiti_core/prompts/dedupe_edges.py)) with continuous indexing across both lists:
   - `duplicate_facts: list[int]` — indices of facts that are semantically identical.
   - `contradicted_facts: list[int]` — indices of facts that the new fact contradicts.
   - An existing fact can be **both** a duplicate AND contradicted (e.g. "Alice is engineer" → "Alice is senior engineer" — same relationship, updated title).
4. **Deterministic temporal arbitration** — `resolve_edge_contradictions()`:
   - If old edge's `valid_at > new edge's valid_at`, the *new* edge gets invalidated (we just learned something older). Otherwise the old edge is invalidated with `invalid_at = new_edge.valid_at` and `expired_at = now`.
   - Edges are never deleted, just marked expired — the graph is append-only.

**Recommendation: ADAPT (heavily).**

- **Schema**: copy the `valid_at`/`invalid_at`/`expired_at`/`reference_time` fields into our LadybugDB `Fact` node. Our current `Fact` schema in [ladybug_store.py:72-80](../acervo/adapters/ladybug_store.py#L72-L80) has `date` and `session` but no bi-temporal structure. This is a concrete gap.
- **Extraction prompt**: copy [`prompts/extract_edges.py`](../../../graphiti-analysis/graphiti_core/prompts/extract_edges.py) wholesale. The entity-name validation loop is a must-have — it fixes the "LLM hallucinates a node we never saw" bug that likely affects our pipeline.
- **Contradiction detection**: adopt the LLM-based `duplicate_facts` + `contradicted_facts` approach for edges where the same node pair has prior facts. For Ollama+qwen2.5:7b the structured output may need more examples — expect to add 2-3 few-shots beyond what Graphiti has.
- **Temporal arbitration**: adopt `resolve_edge_contradictions()` as-is — pure Python, deterministic, no LLM.
- **Skip bulk community detection** — Graphiti has Leiden/community stuff we don't need.

### 2.4 Retrieval (without LLM)

**All retrieval is LLM-free by default.** The `search()` entry point ([search/search.py:98](../../../graphiti-analysis/graphiti_core/search/search.py#L98)) runs:

1. **Parallel scope execution**: edges + nodes + episodes + communities, each with its own `SearchConfig`.
2. **Per scope**: execute configured `search_methods` in parallel:
   - `bm25` → `edge_fulltext_search` / `node_fulltext_search` — delegates to the DB's fulltext index (Neo4j FTS, Kuzu FTS, etc.).
   - `cosine_similarity` → `edge_similarity_search` / `node_similarity_search` — delegates to the DB's vector index.
   - `bfs` → `edge_bfs_search` / `node_bfs_search` — graph traversal from seed UUIDs.
3. **Reranker**: one of `rrf` (default), `mmr`, `node_distance`, `episode_mentions`, `cross_encoder`.

**`rrf()` ([search/search_utils.py:1780-1795](../../../graphiti-analysis/graphiti_core/search/search_utils.py#L1780-L1795))** — 15 lines of code:

```python
def rrf(results: list[list[str]], rank_const=1, min_score: float = 0):
    scores: dict[str, float] = defaultdict(float)
    for result in results:
        for i, uuid in enumerate(result):
            scores[uuid] += 1 / (i + rank_const)
    scored_uuids = [term for term in scores.items()]
    scored_uuids.sort(reverse=True, key=lambda term: term[1])
    sorted_uuids = [term[0] for term in scored_uuids]
    return [uuid for uuid in sorted_uuids if scores[uuid] >= min_score], [...]
```

This is the canonical [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — merging ranked lists from different retrievers by summing `1/(rank + k)`. No tuning, no embeddings needed at fusion time.

**`maximal_marginal_relevance()` ([search/search_utils.py:1901-1939](../../../graphiti-analysis/graphiti_core/search/search_utils.py#L1901-L1939))** — 40 lines using numpy. Selects candidates that are both similar to query AND diverse from each other. Useful for context assembly to avoid redundant facts.

**`node_distance_reranker()` ([search/search_utils.py:1798-1857](../../../graphiti-analysis/graphiti_core/search/search_utils.py#L1798-L1857))** — reranks results by graph distance from a "center node" (e.g. the user entity). Cypher query counts hops. This is conceptually similar to our S2 BFS seed expansion but applied at retrieval time.

**Pre-configured recipes** ([search_config_recipes.py](../../../graphiti-analysis/graphiti_core/search/search_config_recipes.py)):
- `COMBINED_HYBRID_SEARCH_RRF` — BM25+vector for edges/nodes/communities, RRF fused.
- `EDGE_HYBRID_SEARCH_RRF` — edges only, BM25+vector, RRF.
- `NODE_HYBRID_SEARCH_NODE_DISTANCE` — for ranked-by-proximity-to-center-node use cases.
- `COMBINED_HYBRID_SEARCH_CROSS_ENCODER` — the heaviest, cross-encoder rerank (requires a separate model).

**Recommendation: ADOPT (rrf, mmr) + ADAPT (hybrid search orchestration).**

- **Copy `rrf()` and `maximal_marginal_relevance()` verbatim** into `acervo/search/fusion.py`. 55 lines total, numpy-only. These are a strict upgrade over whatever fusion we currently do in S2.
- **Adopt the scope/method/reranker config pattern** — our S2/S3 currently does ad-hoc BFS + topic-layer filtering. A clean `SearchConfig(methods=[bm25, cosine, bfs], reranker=rrf)` gives us a uniform place to experiment.
- **LadybugDB/Kuzu fulltext**: check whether Kuzu FTS is enabled in Ladybug. If yes, adopt `edge_fulltext_search` pattern. If no, we fall back to BM25 in Python over the edge facts (tantivy or `rank-bm25` package).
- **Vector search**: Ladybug supports `FLOAT[]` arrays (your schema at [ladybug_store.py:66](../acervo/adapters/ladybug_store.py#L66) stores `name_embedding FLOAT[]`) but need to verify vector indexing. If not natively supported, keep ChromaDB as the vector adapter.
- **Skip cross_encoder by default** — requires a separate HF sentence-transformers model. Local Ollama doesn't do cross-encoder.
- **Node-distance reranker** is a direct fit for our "center node = active topic" concept.

### 2.5 Graph Construction Pipeline (detailed)

Already diagrammed in §1. Key insights from reading the code:

1. **Everything is parallelized via `semaphore_gather`** ([edge_operations.py:288-293, 315-341](../../../graphiti-analysis/graphiti_core/utils/maintenance/edge_operations.py#L288-L341)) — per-edge LLM calls run concurrently, bounded by a semaphore. This is a huge throughput win.
2. **Pydantic structured output is relied on throughout** — every LLM call passes `response_model=SomePydanticModel`. Their `OpenAIGenericClient` uses OpenAI's structured output feature; Ollama has a compatible `format={"type":"json_object"}` + grammar mode, so **this works with Ollama but with caveats**: Qwen2.5:7b will produce invalid JSON occasionally — need retry with validation.
3. **Graph writes happen last**, after all extraction + resolution. This means a failed LLM call in mid-pipeline doesn't leave the graph in a partially-updated state.
4. **Previous episodes as context** — every prompt gets the last 10 episodes as a `<PREVIOUS_MESSAGES>` section for anaphora resolution ("she" → "Alice"). This is exactly what we need.
5. **No explicit S1/S2/S3 split** — Graphiti is a single pipeline with no separate "activation" phase. The "relevant context for the response" is computed *at retrieval time*, not at ingestion time. We might consider whether our 4-stage split is actually earning its complexity.

### 2.6 Edge Dedup & Conflict Resolution

Covered in §2.3 above. The canonical function is [`resolve_extracted_edge()`](../../../graphiti-analysis/graphiti_core/utils/maintenance/edge_operations.py#L495). Key design notes:

- **Determinism where possible**: exact fact-text match → reuse (no LLM). The LLM only fires when there's semantic overlap but not exact.
- **LLM returns integer indices, not regenerated text** — much more reliable than "summarize the merged edge". Indices into a continuous-indexed list that spans both `existing_edges` and `invalidation_candidates`.
- **Prompt is tiny** ([dedupe_edges.py](../../../graphiti-analysis/graphiti_core/prompts/dedupe_edges.py), 103 lines) — 3 few-shots, explicit duplicate vs contradicted distinction.
- **Temporal arbitration is 100% Python** ([edge_operations.py:457-492](../../../graphiti-analysis/graphiti_core/utils/maintenance/edge_operations.py#L457-L492)) — no LLM involved in deciding *which* edge to invalidate once we know there's a contradiction.

### 2.7 Schema / Data Model — comparison with LadybugDB

**Graphiti's `EntityNode`** (`nodes.py:492-498`):
```python
class EntityNode(Node):
    uuid: str                   # primary key
    name: str
    group_id: str               # partition/namespace
    labels: list[str]           # multi-label (e.g. ['Entity', 'Person'])
    created_at: datetime
    name_embedding: list[float] | None
    summary: str
    attributes: dict[str, Any]  # custom fields from user pydantic types
```

**Acervo's `EntityNode` in Ladybug** ([ladybug_store.py:29-47](../acervo/adapters/ladybug_store.py#L29-L47)):
```sql
CREATE NODE TABLE EntityNode (
    id STRING PRIMARY KEY,
    label STRING, type STRING, kind STRING,
    created_at STRING, last_active STRING,
    session_count INT64, attributes STRING,     -- JSON-encoded
    files STRING[], chunk_ids STRING[],
    layer STRING, owner STRING, source STRING,
    confidence_for_owner DOUBLE,
    status STRING, pending_fields STRING[]
)
```

| Field | Graphiti | Acervo/Ladybug | Gap |
|---|---|---|---|
| PK | `uuid` | `id` | — (same concept) |
| Name | `name` | `label` | naming only |
| Type | `labels: list[str]` | `type: str` + `kind: str` | Graphiti's multi-label is richer; we force single type |
| Partition | `group_id` | `owner` + `layer` + `source` | we encode more partitioning info |
| Embedding | `name_embedding: list[float]` | *(separate Chroma)* | we don't store on node |
| Summary | `summary: str` | *(missing)* | **gap — we have no regional summary** |
| Custom fields | `attributes: dict` | `attributes: str` (JSON) | ours is string-encoded — can't index |
| Temporal | `created_at` only | `created_at`, `last_active`, `session_count` | we track activity; they don't |

**Graphiti's `EntityEdge`** → most critical schema gap. Our [`Fact` node](../acervo/adapters/ladybug_store.py#L72-L80) has `fact_text`, `date`, `session`, `source`, `speaker` — no bi-temporal model, no `expired_at`, no `episodes` provenance list, no `fact_embedding`.

**Recommendation: ADAPT (schema migration).**

- Add `summary` to `EntityNode`/`StructuralNode`.
- Add to `Fact`: `valid_at: STRING`, `invalid_at: STRING`, `expired_at: STRING`, `reference_time: STRING`, `fact_embedding: FLOAT[]`, `episodes: STRING[]` (provenance).
- Consider whether `Fact` should be an edge rather than a node. Graphiti models facts as edges (`EntityEdge`) — the fact text lives *on* the relationship. Our model has Fact as a node + SemanticRel edges linking to EntityNodes. Graphiti's design is simpler but less queryable. This is an open design question and a potential refactor; don't rush it.
- Keep our `labels: list[str]` equivalent by using Graphiti's multi-label trick where needed.

---

## 3. Concrete copy/adapt checklist

### Direct copy (Apache-2.0 attribution only)

| File | Lines | What it gives us | Target location |
|---|---|---|---|
| `graphiti_core/utils/maintenance/dedup_helpers.py` | 296 | Hybrid determinístic+LLM dedup: exact-normalize, MinHash LSH, entropy gate | `acervo/extraction/dedup_helpers.py` |
| `graphiti_core/prompts/extract_nodes.py` (functions `extract_message`, `extract_text`, `extract_json`) | ~240 | Battle-tested entity extraction prompts with rich "NEVER extract" guardrails | `acervo/extraction/prompts/extract_entities.py` |
| `graphiti_core/prompts/extract_edges.py` (function `edge`) | ~80 | Fact extraction prompt with bi-temporal fields + entity validation | `acervo/extraction/prompts/extract_facts.py` |
| `graphiti_core/prompts/dedupe_nodes.py` (function `nodes`) | ~60 | Node dedup prompt (batched, indexed IDs) | `acervo/extraction/prompts/dedupe_nodes.py` |
| `graphiti_core/prompts/dedupe_edges.py` (function `resolve_edge`) | ~60 | Edge dedup + contradiction prompt (continuous-indexed duplicates + contradictions) | `acervo/extraction/prompts/dedupe_edges.py` |
| `graphiti_core/search/search_utils.py::rrf` | 15 | Reciprocal Rank Fusion for hybrid search | `acervo/search/fusion.py` |
| `graphiti_core/search/search_utils.py::maximal_marginal_relevance` | 40 | MMR reranking (diversity) | `acervo/search/fusion.py` |
| `graphiti_core/utils/maintenance/edge_operations.py::resolve_edge_contradictions` | 35 | Deterministic temporal arbitration | `acervo/extraction/temporal.py` |

### Adapt with changes

| Graphiti function | What to adapt | Target in Acervo |
|---|---|---|
| `node_operations.extract_nodes` | Swap their `GraphitiClients` for our `LLMPort` + `EmbedderPort`; replace integer entity_type_id with name-based | `acervo/extraction/entities.py` |
| `node_operations.resolve_extracted_nodes` | Replace their `node_similarity_search` with our ChromaStore or Ladybug vector call | `acervo/extraction/resolution.py` |
| `edge_operations.extract_edges` | Same port swap; add pydantic validation for node-name-exists | `acervo/extraction/facts.py` |
| `edge_operations.resolve_extracted_edge` | Same; reuse our LLM client | `acervo/extraction/resolution.py` |
| `search/search.py::search` | Strip Neo4j-specific reranker paths; keep RRF/MMR; use our stores | `acervo/retrieval/search.py` |

### Ignore (for now)

- `graphiti_core/driver/*` — we keep our LadybugDB adapter. But worth studying `driver/kuzu_driver.py` + `driver/kuzu/operations/*` as a reference for Cypher query patterns against Kuzu/Ladybug.
- `graphiti_core/utils/maintenance/community_operations.py` — Leiden community detection. Nice-to-have, not urgent; we're not at graph size where communities matter yet.
- `graphiti_core/cross_encoder/*` — requires separate HF model, not useful for local-first.
- `graphiti_core/nodes.py::SagaNode`, `HAS_EPISODE`/`NEXT_EPISODE` — their linking layer for conversation threads. We already have our own session/episode concept; no reason to adopt.
- `graphiti_core/server/*` + `mcp_server/*` — we already have our own proxy.

---

## 4. Compatibility with our stack

### Ollama / qwen2.5-7b

- **OpenAI-compatible API**: Graphiti's `OpenAIGenericClient` ([llm_client/openai_generic_client.py](../../../graphiti-analysis/graphiti_core/llm_client/openai_generic_client.py)) uses `AsyncOpenAI` with a custom `base_url`. It is **already designed to talk to Ollama** via `http://localhost:11434/v1`. The Acervo facade's current env vars (`ACERVO_LIGHT_MODEL_URL`) map directly.
- **Structured output**: Graphiti passes `response_model=SomePydanticModel` and the client uses the `response_format={"type":"json_object"}` path. Ollama supports this but has limitations:
  - qwen2.5:7b sometimes returns slightly malformed JSON (trailing commas, escape issues).
  - Graphiti retries twice (`MAX_RETRIES=2`) on validation errors. We may want to bump to 3–4 for 7B local models.
- **Integer-enum outputs are risky** — Graphiti's `ExtractedEntity.entity_type_id: int` pattern (prompt has IDs 0..N, model picks one) is harder for small local models than free-form string output. Recommend switching to `entity_type: str` with post-validation.
- **Context length**: Graphiti's prompts are short (few hundred tokens system + entity list). Qwen2.5:7b's 32K window is more than enough.

### LadybugDB (Kuzu fork)

- **Cypher + embedded + local-first**: Graphiti has a **full Kuzu driver** at [`driver/kuzu_driver.py`](../../../graphiti-analysis/graphiti_core/driver/kuzu_driver.py) + [`driver/kuzu/operations/*`](../../../graphiti-analysis/graphiti_core/driver/kuzu/). The DDL at [kuzu_driver.py:54-95](../../../graphiti-analysis/graphiti_core/driver/kuzu_driver.py#L54-L95) defines `Episodic`, `Entity`, `Community` node tables + `RelatesToNode_` intermediate table for edges.
- **Schema mismatch**: Our Ladybug schema uses `EntityNode`, `StructuralNode`, `Fact`, `SemanticRel`, `StructuralRel`, `EntityToStructural`. Graphiti uses `Entity`, `Episodic`, `Community`, `RELATES_TO`. These are **different ontologies**, so we can't use Graphiti's driver as-is.
- **Vector search**: Graphiti's Kuzu queries use `CAST($search_vector AS FLOAT[dim])` for cosine similarity. Our schema already has `name_embedding FLOAT[]` — this should work directly if Ladybug inherits Kuzu's `array_cosine_similarity` function.
- **Fulltext search**: Graphiti works around Kuzu's limitation (no FTS on edge properties) by wrapping edges as intermediate `RelatesToNode_` tables. Worth knowing if we adopt fact-as-edge modeling.
- **Recommendation**: keep our Ladybug adapter but **borrow the Cypher query patterns** from `driver/kuzu/operations/*` for vector + fulltext search and Kuzu-specific gotchas.

### Local-first constraints

- ✅ Graphiti has **no telemetry we can't disable** (posthog, but optional). Attribution in LICENSE is enough.
- ✅ All LLM/embedder/DB code is pluggable via ABC ports.
- ⚠️ Their default `OPENAI_API_KEY` expectation means we need to set a dummy key when pointing at Ollama.
- ⚠️ They expect a **separate embedder** (OpenAI/Gemini/Voyage). Our Ollama-based embedder (`OllamaEmbedder` adapter) would need to implement their `EmbedderClient` interface if we copy their code directly — but we're copying individual functions, not the whole framework, so this is moot.

### New dependencies if we copy selected code

- `numpy` — already a transitive dep
- `pydantic>=2` — already have
- `tenacity` (for retry) — optional; we can use our own retry

**Zero new runtime deps** for the pieces we'd copy.

---

## 5. Proposed Acervo pipeline after adoption

```
acervo.facade.Acervo.process(message, reference_time)
  │
  ├─ S1 ── Extraction (LLM call #1)
  │    ├─ previous_episodes = graph.retrieve_episodes(last_n=10)
  │    ├─ episode = EpisodicNode(content=message, ...)
  │    ├─ extracted_nodes = extract_nodes(                  ← NEW: copied from Graphiti
  │    │       llm, episode, previous_episodes,
  │    │       entity_types=ACERVO_ENTITY_TYPES,            ← our pydantic ontology
  │    │       prompt=EXTRACT_MESSAGE_PROMPT                ← copied verbatim
  │    │   )
  │    └─ _collapse_exact_duplicate_extracted_nodes()       ← copied verbatim
  │
  ├─ S1-resolve ── Entity resolution (LLM call #2, optional)
  │    ├─ candidates = chroma.similarity_search(node.name, k=15, min=0.6)
  │    ├─ _resolve_with_similarity(extracted, candidates)   ← copied verbatim (MinHash LSH)
  │    └─ _resolve_with_llm(unresolved)                     ← copied verbatim (only if needed)
  │
  ├─ S1.5 ── Facts (LLM call #3) + contradiction resolution (LLM call #4)
  │    ├─ edges = extract_edges(llm, episode, resolved_nodes)  ← copied + adapted
  │    │    └─ validates entity names against nodes list
  │    └─ resolved_edges, invalidated = resolve_extracted_edges(  ← copied + adapted
  │           clients, edges, existing_edges_from_graph
  │       )
  │         ├─ fast path: exact fact-text match
  │         ├─ LLM dedup+contradict call (batched, per edge pair)
  │         └─ resolve_edge_contradictions(temporal arbitration)
  │
  ├─ S1.5-summary ── Node summaries (LLM call #5, batched)
  │    └─ extract_attributes_from_nodes(...)                ← copied + adapted
  │
  ├─ Commit to LadybugDB
  │    ├─ upsert resolved_nodes
  │    ├─ upsert resolved_edges (with new valid_at/invalid_at)
  │    ├─ mark invalidated_edges with expired_at=now
  │    └─ save episode + episodic edges
  │
  └─ S2/S3 ── Context assembly for response (NO LLM calls — all local)
       ├─ search(                                            ← uses our adapted search.py
       │       query=message,
       │       config=EDGE_HYBRID_SEARCH_RRF,                ← copied recipe
       │       center_node_uuid=current_topic_uuid
       │   )
       │     ├─ parallel: edge_fulltext + edge_similarity
       │     ├─ rrf() fusion                                 ← copied verbatim
       │     └─ (optional) mmr() reranking                   ← copied verbatim
       │
       └─ assemble_context_xml(results, token_budget=600)    ← keep our current S3
```

**Total LLM calls per turn**: 3 mandatory (extract_nodes, extract_edges, 1× summary batch) + 0–2 conditional (dedupe_nodes only on ambiguity, resolve_extracted_edge per edge with prior facts — usually 0–2).

**Context window per call**: each prompt is <4K tokens. No multi-shot reasoning needed.

**Deterministic paths**: name normalization, MinHash LSH dedup, exact fact-text reuse, temporal arbitration, RRF, MMR, vector search, fulltext search, BFS. All of these avoid LLM calls.

---

## 6. Known gaps / open questions

1. **Fact-as-edge vs fact-as-node.** Graphiti models facts as edges (`EntityEdge.fact` is text on the relationship). We model facts as `Fact` nodes linked by `SemanticRel`. Their model is simpler but harder to query "all facts from source X". Decide before schema migration.
2. **Single-type vs multi-label entities.** Graphiti allows `labels=['Entity', 'Person', 'Engineer']`; we have `type: str`. Their approach is flexible but requires Kuzu-side handling (since Kuzu node labels are table names). Either we adopt multi-label via `attributes.secondary_types` array, or we stay single-type.
3. **Structured output reliability with qwen2.5:7b.** Before committing to Graphiti's per-edge LLM resolution (up to N calls per turn), benchmark how often qwen2.5:7b fails to return valid JSON on the `EdgeDuplicate` schema. If >10%, we need a local fine-tune or a larger model for that step specifically.
4. **Kuzu FTS availability in Ladybug.** Do we have Kuzu's fulltext search extension? If yes, we can follow Graphiti's `edge_fulltext_search` pattern. If no, we need a Python-side BM25 over fact texts (tantivy or `rank-bm25`).
5. **Summary LLM calls vs our 4-stage architecture.** Graphiti has no separate S2/S3 — it just searches at response time. Our S2 (activator) + S3 (assembler) was designed to pre-compute token-budgeted context. Is that still worth the complexity if Graphiti's approach hits our latency budget? Worth benchmarking after migration.
6. **Communities (Leiden).** Graphiti auto-clusters nodes into communities. We don't. This is a nice-to-have for summarization of big subgraphs but not urgent.

---

## 7. Attribution

All code copied or adapted from Graphiti (Apache-2.0, © Zep Software Inc.) must retain the LICENSE header. Suggested `acervo/extraction/THIRD_PARTY.md`:

```
Portions of this module are adapted from Graphiti
(https://github.com/getzep/graphiti), Copyright 2024 Zep Software, Inc.,
licensed under the Apache License, Version 2.0.

Specific files/functions adapted:
- dedup_helpers.py ← graphiti_core/utils/maintenance/dedup_helpers.py
- prompts/extract_entities.py ← graphiti_core/prompts/extract_nodes.py
- prompts/extract_facts.py ← graphiti_core/prompts/extract_edges.py
- prompts/dedupe_nodes.py ← graphiti_core/prompts/dedupe_nodes.py
- prompts/dedupe_edges.py ← graphiti_core/prompts/dedupe_edges.py
- search/fusion.py ← graphiti_core/search/search_utils.py (rrf, mmr)
- extraction/temporal.py ← graphiti_core/utils/maintenance/edge_operations.py (resolve_edge_contradictions)
```
