# Acervo Pipeline Stages — S1, S2, S3, S1.5

> **⚠️ This is the canonical reference for the 4-stage pipeline.**
> Any change to the flow of S1, S2, S3, or S1.5 — whether it's a
> new LLM call, a new graph mutation, a new input/output field,
> or a re-ordering of internal steps — **MUST** be reflected here
> in the same commit. Treat this doc as part of the surface area,
> not as documentation that drifts. If the doc disagrees with the
> code, the doc is wrong and needs an update.
>
> Last verified against: `release/v0.6.0` after Phase 1-4 adoption
> (see [docs/research/v0.6.0-status-report.md](../research/v0.6.0-status-report.md)).

---

## Table of contents

- [Overall flow per turn](#overall-flow-per-turn)
- [S1 — Unified topic classifier + extractor](#s1--unified-topic-classifier--extractor)
- [S2 — Activator (graph retrieval)](#s2--activator-graph-retrieval)
- [S3 — Assembler (context budgeting)](#s3--assembler-context-budgeting)
- [S1.5 — Graph curation + assistant extraction](#s15--graph-curation--assistant-extraction)
- [How the stages talk to each other](#how-the-stages-talk-to-each-other)
- [LLM calls per turn (worst case)](#llm-calls-per-turn-worst-case)

---

## Overall flow per turn

```
┌──────────────────────────────────────────────────────────────────┐
│  USER → acervo.prepare(user_msg, history)                        │
│                                                                  │
│   1. user_embedding (Embedder.embed)        ── optional          │
│   2. topic_detector.detect_hints (L1/L2)    ── deterministic     │
│   3. S1   ── 1 LLM call (extraction + topic + intent)            │
│           ── _validate_s1                                        │
│           ── _embed_new_entities (batch via Embedder)            │
│           ── _resolve_against_graph (Phase 1 dedup)              │
│   4. S2   ── 0 LLM calls                                         │
│           ── _find_seeds                                         │
│           ── BFS traverse (depth 0/1/2 = HOT/WARM/COLD)          │
│           ── _hybrid_enrich (vector + fulltext fused via RRF)    │
│   5. S3   ── 0 LLM calls                                         │
│           ── (optional MMR rerank if query_embedding present)    │
│           ── budget-bounded XML context block                    │
│           ── grounding instruction                               │
│                                                                  │
│  → returns context_stack to the caller                           │
└──────────────────────────────────────────────────────────────────┘

  caller assembles: [system, ...history, {user}, {context}]
  caller calls the "main" LLM that responds to the user
  caller gets back assistant_text

┌──────────────────────────────────────────────────────────────────┐
│  acervo.process(user_msg, assistant_text)                        │
│                                                                  │
│   6. S1.5 ── 1 LLM call (graph_update prompt)                    │
│           ── resolve_s1_5_facts (Phase 3, conditional LLM call)  │
│           ── apply_s1_5_result                                   │
│              ├─ merges                                           │
│              ├─ type corrections                                 │
│              ├─ discards                                         │
│              ├─ assistant_entities + facts                       │
│              └─ validation log                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## S1 — Unified topic classifier + extractor

**Location:** [`acervo/s1_unified.py::S1Unified.run`](../../acervo/s1_unified.py)
**Stage in cycle:** runs inside `prepare()`, **before** the main LLM responds to the user.
**Always runs.** L1 (keyword) and L2 (embedding) topic detection feed it as hints, never as gates.

### When it runs

`Acervo.prepare(user_msg, history)` → after building previous-assistant
context, the topic hint (L1/L2), and the existing-graph summary,
`pipeline.py:175` calls `self._s1.run(...)`. This happens before S2/S3
and before any user-facing LLM call.

### What it's responsible for

S1 is the **single LLM-powered step in the prepare path**. One call
does five things at once:

1. Topic classification (`same` / `subtopic` / `changed` + label)
2. Intent classification (`overview` / `specific` / `followup` / `chat` + retrieval flag)
3. Entity extraction (with type, layer, optional `existing_id` link)
4. Relation extraction (between extracted entities, ontology-validated downstream)
5. Fact extraction (atomic statements attached to an entity)

Then it runs **deterministic post-processing** that does NOT call the LLM:

6. **Validation** (`_validate_s1`) — drops garbage entities, fuzzy-resolves fact references against valid entities, caps relation count, rejects vague facts.
7. **Phase 2: batch embedding** (`_embed_new_entities`) — when `self._embedder` is set, the validated entity names are sent to `embedder.embed_batch()` and the vectors are attached to `entity.attributes["name_embedding"]`.
8. **Phase 1: graph dedup** (`_resolve_against_graph`) — passes the entities through `entity_resolution.resolve_extracted_nodes()` against the existing graph. Stamps `_existing_id`, rewrites canonical names, and propagates the rewrites to relation source/target and fact entity references.

### Inputs

- `user_msg: str` — the new user turn (truncated to 800 chars in the prompt)
- `prev_assistant_msg: str` — last assistant response (first 150 chars; intentionally short to avoid topic drift)
- `current_topic: str` — current topic from the topic detector
- `topic_hint: str` — L1/L2-derived hint string ("same (high confidence)", "changed", etc.)
- `existing_nodes_summary: str` — JSON of top-N existing graph nodes (built by `build_graph_summary` in `s1_unified.py`; ranked by token overlap + recency boost + status boost, capped at 20 nodes)
- `existing_node_names: set[str]` — flat name set used by the validator for hallucination check
- `existing_nodes: list[dict] | None` — full node dicts; when provided, enables Phase 1 graph dedup
- `graph: GraphStorePort | None` — graph reference; when provided, enables Phase 2 semantic pre-filter (`graph.entity_similarity_search`)

### LLM call

- **One** chat call to `self._llm` (the extractor LLM, normally qwen3.5:9b)
- System prompt: [`acervo/prompts/s1_unified.txt`](../../acervo/prompts/s1_unified.txt) — 153 lines, includes "NEVER extract" guardrails (pronouns, abstract feelings, dates, generic nouns, bare kinship terms)
- User prompt: structured `EXISTING NODES / TOPIC HINT / CURRENT TOPIC / PREVIOUS ASSISTANT / USER` block
- `temperature=0.1`, `max_tokens=2048`
- **Retry once at `temperature=0.0`** if the JSON parse fails on the first attempt

### Output (`S1Result`)

```python
@dataclass
class S1Result:
    topic: TopicResult                 # action + label
    extraction: ExtractionResult       # entities, relations, facts
    intent: str                        # "overview"|"specific"|"followup"|"chat"
    retrieval: str | None              # "summary_only"|"with_chunks"|None
    prompt_sent: str                   # JSON for telemetry
    raw_response: str                  # raw LLM output for telemetry
    raw_entity_count: int              # count BEFORE validation
    raw_relation_count: int
    raw_fact_count: int
    dropped_facts: list[dict]          # validation drop reasons
```

### Failure modes

- **LLM call fails** → `_fallback_result()` returns a default `S1Result` with empty extraction and `topic.action="same"`. Pipeline continues with empty extraction; S2/S3 still run.
- **JSON parse fails twice** (first call + retry at temp=0.0) → falls through to validation with the partially parsed result (usually empty), `prompt_sent`/`raw_response` still attached for telemetry.
- **Embedder unavailable** → `_embed_new_entities` no-ops with a warning. Phase 2 semantic pre-filter degrades to MinHash-only.
- **Graph dedup fails** → `_resolve_against_graph` returns the result unchanged. Logs a warning.

### Side effects

- None directly — S1 does NOT touch the graph. It builds the `S1Result` and returns it. The pipeline persists it later via `_persist_s1_entities` (see "How the stages talk to each other" below).
- When `_persist_s1_entities` runs after S1, it stamps **`source="llm"`** and **`updated_by="llm"`** on every entity, relation, fact, topic edge, and orphan-fact pass-2 attachment (v0.6.1). This makes every LLM-driven write distinguishable from user edits coming through the REST API.

### Phase 1-4 visibility

| Phase | Where it shows in S1 | Log line to look for |
|---|---|---|
| Phase 1 (dedup) | `_resolve_against_graph` post-pass | `entity_resolution: N extracted, M merged against graph, K new (semantic_prefilter=on/off)` |
| Phase 1 (logging) | `_resolve_against_graph` per merge | `S1._resolve_against_graph: merged 'X' -> canonical 'Y'` or `stamped _existing_id=Z on 'X'` |
| Phase 2 (embedding) | `_embed_new_entities` | (no info log; check that node attributes have `name_embedding`) |
| Phase 2 (semantic search) | `entity_resolution` | `semantic_prefilter=on` in the line above |

---

## S2 — Activator (graph retrieval)

**Location:** [`acervo/domain/s2_activator.py::S2Activator.run`](../../acervo/domain/s2_activator.py)
**Stage in cycle:** inside `prepare()`, after S1, before S3.
**Zero LLM calls.** Pure graph reads + fusion math.

### When it runs

`pipeline.py:225` calls `self._s2.run(user_text, s1_result, self._graph, ..., user_embedding=user_embedding)` immediately after S1 returns.

### What it's responsible for

S2 is the "find what's relevant in the graph" step. It does three things:

1. **Seed selection** (`_find_seeds`) — figures out which graph nodes the conversation is touching right now.
2. **BFS traversal** — expands from the seeds outward and bins nodes by depth into the HOT/WARM/COLD layers that S3 expects.
3. **Phase 4 hybrid enrichment** (`_hybrid_enrich`) — runs the same query through vector and fulltext search in parallel, fuses with RRF, and returns extras that BFS missed.

### Seed selection logic

In order of preference:

1. **S1-extracted entities** — for each `entity` in `s1_result.extraction.entities`, look it up via `graph.get_node(_make_id(entity.name))`. This is the strongest signal because it came from the LLM's reading of the user message.
2. **Keyword match against node labels** — for every node in the graph, check if its label appears as a substring in the user message, or if a 4+ char user word is a prefix of the label.
3. **Overview fallback** — if `intent == "overview"` and no seeds found, use ALL entity nodes as seeds. This produces a sweep over the whole graph.
4. **Chat intent special case** — if `intent == "chat"`, ignore the above and only use synthesis nodes whose summary contains a user keyword. Returns minimal/no context for greetings and small talk.

### BFS traversal

Two paths, picked at runtime:

- **Cypher path** (preferred): if `graph.traverse_bfs` exists (LadybugGraphStore has it), call it directly with `seed_ids` and `max_depth=2`. Returns a `dict[depth → list[node]]` that maps directly to the layer structure.
- **Python BFS** (fallback): `_traverse` walks edges via `graph.get_edges_for(nid)` using a `deque`, assigning depth-0 to seeds, depth-1 to their neighbors, etc. Both directions of every edge are followed.

The result is a `LayeredContext`:

```python
@dataclass
class LayeredContext:
    hot: list[dict]        # depth 0 — direct seeds
    warm: list[dict]       # depth 1 — 1-hop neighbors
    cold: list[dict]       # depth 2 — 2-hop neighbors
    seeds_used: list[str]  # debug label list
```

### Phase 4 hybrid enrichment

After BFS, for any non-`chat` intent, `_hybrid_enrich` runs:

```python
hybrid_search(
    graph=graph,
    query=user_text,
    seed_ids=seed_ids,
    query_embedding=user_embedding,
    limit=10,
)
```

Internally that runs three retrieval methods in parallel against the graph:

| Method | Source | Requires |
|---|---|---|
| BFS | `graph.traverse_bfs(seed_ids, max_depth=2)` | seeds |
| Vector | `graph.entity_similarity_search(query_embedding, ...)` | embedder |
| Fulltext | `graph.fact_fulltext_search(query, ...)` | text query |

The three ranked lists are fused via **Reciprocal Rank Fusion** (`acervo/search/fusion.py::rrf`), and the top results are returned. `_hybrid_enrich` then filters out nodes that BFS already found (kept in `existing_ids`) and returns the remaining ones as `vector_hits` on the `S2Result`.

This step has its own degradation: any internal failure logs a warning and returns `[]` (no-op). BFS results survive intact.

### Inputs

- `user_text: str` — full user message, used for keyword seed match + fulltext query
- `s1_result: S1Result` — needed for `s1_result.extraction.entities` (primary seeds)
- `graph: GraphStorePort` — read-only access
- `intent: str` — controls fallback behavior
- `vector_store: Any | None` — **legacy parameter, currently a no-op**. Phase 4 reads vector data through `graph.entity_similarity_search` directly. Kept in the signature for backward compat.
- `user_embedding: list[float] | None` — passed through to hybrid_search for the vector arm

### Output (`S2Result`)

```python
@dataclass
class S2Result:
    layered: LayeredContext        # hot/warm/cold lists (BFS-derived)
    active_node_ids: set[str]      # union of all node IDs in layered
    vector_hits: list[dict]        # Phase 4 hybrid extras (BFS misses)
```

### Failure modes

- **No seeds** → all layers empty → S3 will return an empty context block. Pipeline continues with `has_context=False`, which the caller can use to trigger a tool call.
- **`graph.traverse_bfs` raises** → falls back to Python BFS via `_traverse`. No log.
- **`hybrid_search` raises** → caught in `_hybrid_enrich`, logs warning, returns empty extras list. BFS results unaffected.

### Side effects

- None. S2 is a pure read.

### Phase 4 visibility

```
[acervo] S2 — seeds=N, hot=N, warm=N, cold=N, hybrid_extra=N
```

`hybrid_extra=0` is normal in small graphs because BFS already finds
everything reachable. We expect `hybrid_extra > 0` in graphs >500 nodes
where vector similarity catches semantically-related nodes that aren't
in the BFS frontier.

---

## S3 — Assembler (context budgeting)

**Location:** [`acervo/domain/s3_assembler.py::S3Assembler.run`](../../acervo/domain/s3_assembler.py)
**Stage in cycle:** inside `prepare()`, after S2, last step before returning to the caller.
**Zero LLM calls.** Pure formatting + token math.

### When it runs

`pipeline.py:241` calls `self._s3.run(layered=s2_result.layered, ..., query_embedding=user_embedding)` after S2 returns.

### What it's responsible for

Take the `LayeredContext` (HOT/WARM/COLD node lists) from S2 and turn it into a **token-budgeted XML context block** that the caller can prepend to its system prompt before invoking the user-facing LLM.

It does this in 4 sub-steps:

1. **(Phase 4) MMR rerank** — when `query_embedding` is provided, call `_mmr_rerank_layers(layered, query_embedding)` to reorder WARM and COLD lists by Maximal Marginal Relevance. HOT is intentionally left alone (those are direct seeds, reordering would hide the highest-signal items). Nodes without a stored embedding stay at the tail of each layer in their original BFS order.
2. **Format + budget** (`_build_context_block`) — walk HOT, then WARM, then optionally COLD (only for `specific`/`followup` intents), formatting each node and accumulating tokens until the per-intent budget is hit. Nodes that would push over budget are silently dropped.
3. **Grounding instruction** — append an intent-specific suffix telling the user-facing LLM how to use the context ("Answer using the knowledge context above...").
4. **Context stack assembly** — when a `context_index` is provided, hand off the formatted warm content to `context_index.build_context_stack(history, ..., warm_override=...)` which produces the actual `[{role: system, content: ...}, ...]` list that the caller will use.

### Layer formats (Level 1 compression)

- **HOT** — `_format_hot`: full detail. `Label [Type] — desc | grouped relations | up to 3 facts`. Used for direct seeds; ~25-40 tokens per node.
- **WARM** — `_format_warm`: 1-line summary. `Label [Type]: connection-summary — desc`. Used for 1-hop neighbors; ~10-15 tokens per node.
- **COLD** — flat list. `Label [Type], Label [Type], Label [Type]`. Only for `specific`/`followup` intents. Comma-joined as `Also: X, Y, Z`. ~3-5 tokens per node.

### Token budgets per intent

```python
_BUDGETS = {
    "overview": 300,
    "specific": 600,
    "followup": 400,
    "chat": 100,
}
```

Can be overridden via `warm_budget_override` from the caller (used by some test paths).

### XML structure

```xml
<ctx>
<hot>
  Sandy Veliz [Person] — desc | works at: Altovallestudio | facts: ...
</hot>
<warm>
  Altovallestudio [Organization]: Sandy works here — Spanish dev studio
  Butaco [Project]: maintained by Altovallestudio — ERP for taller mecánico
  Also: Angular [Technology], Firebase [Technology], Capacitor [Technology]
</warm>
</ctx>

Answer using the knowledge context above. If the answer is not in the
context, say so clearly.
```

The `Also:` line for COLD lives **inside** `</warm>` to keep the XML
shallow.

### Inputs

- `layered: LayeredContext` — the output of S2
- `intent: str` — picks the budget
- `graph: GraphStorePort` — passed to formatters so they can read facts/edges per node
- `project_overview: str` — optional prefix that lives before `<ctx>` (built upstream by `_build_project_overview`)
- `context_index: ContextIndex | None` — when provided, builds the full message stack
- `history: list[dict] | None` — conversation history (not the user msg itself)
- `current_topic: str` — passed to `context_index.build_context_stack`
- `warm_budget_override: int | None` — escape hatch for tests
- `query_embedding: list[float] | None` — Phase 4 MMR signal

### Output (`S3Result`)

```python
@dataclass
class S3Result:
    context_stack: list[dict]    # ready-to-send messages list
    warm_content: str            # the XML context block as a string
    warm_tokens: int
    hot_tokens: int
    total_tokens: int
    has_context: bool            # True if warm_tokens > 0
    needs_tool: bool             # True if has_context is False
```

### Failure modes

- **Empty layers** — returns `warm_content=""`, `has_context=False`. The caller can use this as a signal to fall back to a tool (web search, retrieval, etc.).
- **MMR rerank with no embeddings on nodes** — `_node_embedding` returns `None` for all nodes, MMR has nothing to score, keeps original BFS order. Silent fallback.
- **`context_index` not provided** — skips the stack assembly and just counts warm tokens for telemetry.

### Side effects

- None. S3 is a pure transform.

### Phase 4 visibility

```
[acervo] S3 — warm=Ntk hot=Ntk total=Ntk layers(hot=N warm=N cold=N) intent=X
```

When MMR runs, you can verify it indirectly by checking that the warm
nodes order changed between two consecutive turns with the same BFS
seeds but different user embeddings. There's no dedicated MMR log line
yet — could be added if needed.

---

## S1.5 — Graph curation + assistant extraction

**Location:** [`acervo/s1_5_graph_update.py::S1_5GraphUpdate.run`](../../acervo/s1_5_graph_update.py) + `apply_s1_5_result` + `resolve_s1_5_facts` (same file)
**Stage in cycle:** inside `process()`, **after** the user-facing LLM has responded.
**1-2 LLM calls per turn.**

### When it runs

`Acervo.process(user_msg, assistant_text)` is called by the host **after** the user-facing LLM has produced its response. Inside, after the "no data response" guard and any tool-result processing, `S1_5GraphUpdate.run(...)` is invoked.

### What it's responsible for

S1.5 does two distinct things in one stage:

1. **Graph curation** — fix problems S1 left behind:
   - Merge entities that turned out to be duplicates of existing nodes
   - Correct entity types when the assistant clarifies them ("X is actually a person, not a concept")
   - Discard spurious entities (typos, garbage that survived S1 validation)
   - Add new relations between existing entities that the user message implied but S1 didn't extract
2. **Assistant-side extraction** — the user-facing LLM's response is itself a source of knowledge (it often surfaces info from the graph or from tool results that the user didn't say). S1.5 extracts new entities, facts, and relations from that response.

Then it does **post-processing**:

3. **Phase 3 contradiction detection** (`resolve_s1_5_facts`) — runs before the assistant facts are persisted. Drops exact duplicates fast, asks the LLM to mark semantic duplicates and contradictions against existing facts, and applies invalidations via `graph.invalidate_fact()`.
4. **Persistence** (`apply_s1_5_result`) — applies all 7 action types to the graph via the `GraphStorePort` interface.

### Inputs

- `user_text: str` — what the user said this turn (used to build the existing-nodes context summary)
- `assistant_text: str` — what the user-facing LLM responded
- `web_results: str` — optional raw content from a `web_search` tool call

Plus some state cached from `prepare()`:

- `self._last_s1_extraction` — the entities S1 extracted this turn, passed in as `new_entities` context for S1.5

### LLM calls

**Call 1 — `S1_5GraphUpdate.run` (always):**
- One chat call to `self._extractor_llm`
- System prompt: [`acervo/prompts/s1_5_graph_update.txt`](../../acervo/prompts/s1_5_graph_update.txt) — 152 lines after Phase 3 rewrite, with explicit rules for merges, contradictions, temporal extraction, and 4 Spanish few-shots
- User prompt: 3 placeholders injected via `str.replace` (NOT `str.format` — that crashes on JSON few-shots, see Bug 1 in [docs/research/v0.6.0-status-report.md](../research/v0.6.0-status-report.md)):
  - `{new_entities}` — JSON of S1's entities, capped at 1500 chars
  - `{existing_nodes}` — `build_graph_summary` output, capped at 2000 chars
  - `{current_assistant_msg}` — assistant response, capped at 1500 chars
- `temperature=0.1`, `max_tokens=2048`
- Returns a JSON with 7 fields (see "Output schema" below)

**Call 2 — `resolve_s1_5_facts` Phase 3 LLM (conditional):**
- Only fires when there are existing facts on the same entity OR `fact_fulltext_search` returns invalidation candidates
- Uses prompt [`acervo/extraction/prompts/dedupe_edges.py`](../../acervo/extraction/prompts/dedupe_edges.py)
- Returns `{duplicate_facts: [int], contradicted_facts: [int]}` with continuous indexing across both candidate lists
- Failures degrade conservatively: if the LLM call fails or returns malformed JSON, the new fact is kept and no invalidations are applied (better to keep redundant info than lose info)

### Output (`S1_5Result`)

```python
@dataclass
class S1_5Result:
    merges: list[MergeAction]                # {from, into, reason}
    new_relations: list[Relation]            # validated by OntologyValidator later
    type_corrections: list[TypeCorrection]   # {node_id, old_type, new_type, reason}
    discards: list[DiscardAction]            # {node_id, reason}
    assistant_extraction: ExtractionResult   # entities + facts + relations from assistant
    prompt_sent: str                         # telemetry
    raw_response: str                        # telemetry
```

### Persistence (`apply_s1_5_result`)

After `resolve_s1_5_facts` mutates `assistant_extraction.facts` in place to drop duplicates and mark invalidations, `apply_s1_5_result` walks the 7 action types in order:

1. **Merges** — `graph.merge_nodes(into_id, from_id)` for each. Three guards prevent the self-merge bug (parser normalizes via `_make_id`, `apply_s1_5_result` re-checks resolved ids, `merge_nodes` itself refuses if `kid == aid`). See Bug 2 in the v0.6.0 status report.
2. **Type corrections** — `validator.validate_entity_type(...)` then `graph.update_node(node_id, type=new_type)`.
3. **Discards** — `graph.remove_node(node_id)`, but only when the node has ≤2 facts (avoids deleting nodes with valuable accumulated info).
4. **New relations** — validated through `OntologyValidator`; persisted via `graph.upsert_entities([], validated_relations, ...)`.
5. **Assistant entities + their facts** — for each new entity from `assistant_extraction`, attach its facts and `graph.upsert_entities([(name, type)], None, entity_facts, ...)`.
6. **Assistant relations** — same pattern, ontology-validated.
7. **Validation log persistence** — any decisions from `OntologyValidator` (mapped, rejected) get written to the `ValidationLog` table for audit.

### Output schema (LLM JSON)

```json
{
  "merges": [
    {"from": "entity_id", "into": "canonical_id", "reason": "..."}
  ],
  "new_relations": [
    {"source": "entity_id", "target": "entity_id", "relation": "verb_in_snake_case"}
  ],
  "type_corrections": [
    {"id": "entity_id", "old_type": "concept", "new_type": "person", "reason": "..."}
  ],
  "discards": [
    {"id": "entity_id", "reason": "..."}
  ],
  "assistant_entities": [
    {"name": "Display Name", "type": "...", "layer": "PERSONAL|UNIVERSAL", "existing_id": null}
  ],
  "assistant_facts": [
    {"entity": "Display Name", "fact": "...", "valid_at": null, "invalid_at": null}
  ],
  "assistant_relations": [
    {"source": "entity_id", "target": "entity_id", "relation": "verb_in_snake_case"}
  ]
}
```

### Failure modes

- **Skip if "no data" assistant response** — phrases like `"no tengo información"` / `"i don't have data"` / `"querés que busque"` trigger an early return. Empty `ExtractionResult`. Used to avoid extracting facts from refusals.
- **LLM call fails** → logged, `S1_5Result()` empty result returned. Pipeline continues. No graph changes.
- **JSON parse fails** → empty result, same as above.
- **Phase 3 LLM fails** → `resolve_s1_5_facts` falls back conservatively, all assistant facts persist, no invalidations applied.
- **Self-merge through parser** → blocked by `_make_id` normalization in the parser, logged as `dropping self-merge X → Y`.
- **Self-merge through merge_nodes call** → blocked by `kid == aid` guard at the bottom of both backends, logged as `merge_nodes: refusing self-merge of X`.

### Side effects (the only stage that writes)

S1.5 is the **only stage in the prepare/process cycle that mutates the graph** (other than `_persist_s1_entities` which the pipeline runs immediately after S1 to write the user-side facts — that's not S1.5 itself, it's a sibling step in `pipeline.py::process()`). Specifically:

- `graph.merge_nodes(..., updated_by="llm")` — merges
- `graph.update_node(..., updated_by="llm")` — type corrections
- `graph.remove_node(...)` — discards
- `graph.upsert_entities(..., source="llm", updated_by="llm")` — relations + assistant entities/facts (with the orphan fact pass-2 to handle facts about pre-existing entities)
- `graph.set_entity_embedding(...)` — when assistant entities have embeddings (currently they don't, but the path exists)
- `graph.set_fact_embedding(...)` — v0.6.1 Change 3: persists the embedding cached on each kept `ExtractedFact` via `_persist_fact_embeddings_for_entity`, so future dedup passes skip re-embedding.
- `graph.invalidate_fact(...)` — Phase 3 contradiction invalidations
- `graph.update_node(..., status="confirmed", updated_by="system")` — v0.6.1 Change 2: `_auto_promote_pending_entities` promotes low-confidence nodes once they accumulate session_count ≥ 3 or a user edit.

Every S1.5 mutation stamps **`source="llm"`** (or `"system"` for auto-promotions) and **`updated_by`** accordingly, paired with a fresh `updated_at` ISO timestamp (v0.6.1 audit trail — see section below).

**New S1.5 sub-steps in v0.6.1:** between `resolve_s1_5_facts` (Phase 3) and `apply_s1_5_result` the pipeline calls `dedupe_s1_5_facts_by_embedding`, which drops semantically equivalent assistant facts without any extra LLM call. See the "Embedding-based fact dedup (Change 3)" section below.

### Phase 1-3 visibility

| Phase | Where it shows in S1.5 | Log line |
|---|---|---|
| Phase 1 (self-merge guard) | Parser | `S1.5: dropping self-merge X → Y (same canonical id)` |
| Phase 1 (self-merge guard) | apply_s1_5 + merge_nodes | `S1.5 merge skipped (both sides resolve to Z)` or `merge_nodes: refusing self-merge of Z` |
| Phase 3 (edge resolution) | resolve_s1_5_facts | `edge_resolution: N input, M new, K exact-duplicates dropped, Z invalidations` |
| Phase 3 (per-fact dropped) | resolve_s1_5_facts | `edge_resolution: dropped duplicate fact 'X' (matches existing 'Y')` |
| Phase 3 (per-fact invalidated) | resolve_s1_5_facts | `edge_resolution: invalidated fact_id=X (text) — invalid_at=Y` |
| Aggregate per turn | apply_s1_5_result | `[acervo] S1.5 — merges=N, fixes=N, discards=N, entities=N, facts=N, resolved_new=N, dropped=N, invalidated=N` |

---

## How the stages talk to each other

The `Pipeline` class in [`acervo/domain/pipeline.py`](../../acervo/domain/pipeline.py) is the only place that knows about all four stages. It does NOT contain business logic — it just wires data flow.

```
Pipeline.prepare(user_text, history):
    user_embedding = embedder.embed(user_text)              # optional
    detection = topic_detector.detect_hints(...)            # L1/L2 hints
    existing_nodes = graph.get_all_nodes()                  # for S1 dedup
    existing_summary = build_graph_summary(...)             # for S1 prompt context
    s1_result = await s1.run(                               # ◄── S1
        user_text, prev_assistant, current_topic,
        topic_hint, existing_summary,
        existing_node_names, existing_nodes, graph,
    )
    self._last_s1_extraction = s1_result.extraction         # cached for S1.5
    _persist_s1_entities(s1_result.extraction, ...)         # writes user facts/entities to graph
    s2_result = s2.run(                                     # ◄── S2
        user_text, s1_result, graph,
        intent=s1_intent, user_embedding=user_embedding,
    )
    s3_result = s3.run(                                     # ◄── S3
        layered=s2_result.layered,
        intent=s1_intent, graph=graph,
        ...,
        query_embedding=user_embedding,
    )
    return PrepareResult(context_stack, ...)


Pipeline.process(user_text, assistant_text, web_results=""):
    if assistant_text is "no data": return empty
    if web_results: persist_web_facts(...)
    new_entities_json = json(self._last_s1_extraction.entities)
    existing_summary = build_graph_summary(...)
    s15_result = await s15.run(                             # ◄── S1.5
        new_entities_json, existing_summary, assistant_text,
    )
    fact_audit = await resolve_s1_5_facts(                  # Phase 3
        s15_result, graph, llm=extractor_llm,
    )
    audit = apply_s1_5_result(graph, s15_result, owner=...)
    return ExtractionResult
```

### Key shared state

- **`self._last_s1_extraction`** — set by `prepare()`, read by `process()`. This is how S1.5 knows what S1 extracted from the user message in the same turn. If `process()` is called without a preceding `prepare()`, `_last_s1_extraction` is None and S1.5 still works but with less context.
- **`self._graph`** — single `GraphStorePort` instance shared by all stages. Reads from S1/S2/S3, writes from `_persist_s1_entities` and `S1.5::apply_s1_5_result`.
- **`self._embedder`** — shared, optional. When present, S1 batch-embeds entities, the pipeline pre-computes `user_embedding` for S2/S3.
- **`self._extractor_llm`** — used by both S1 and S1.5. Same model, different prompts.

### Entity ID conventions

All entity IDs in the graph use `_make_id(label)` (lowercase + accent strip + non-alnum strip + underscore separator). This is the source of truth for "canonical id" everywhere — parsers, dedup, fact references, edges. Anything that compares entity identity must normalize through `_make_id` first or it's a latent bug (see self-merge bug 2).

---

## v0.6.1 — Confidence scoring + pending_review auto-promotion (Change 2)

The graph now carries forward a `confidence` score on every
`Entity` and `Relation` so the pipeline can decide which extractions
to persist vs. gate behind a review state. v0.6.1 ships the full
plumbing but the **S1 prompt does not instruct the LLM to populate the
field** — a v0.6.1 experiment found that asking qwen3.5:9b for
confidence scores hurt fact extraction (prompt confusion), so the
prompt was reverted to v0.6.0 baseline. The `confidence` field stays in
the dataclasses and the downstream machinery is live, ready to activate
automatically once a stronger model populates it.

**What is active today:**

1. **S1 parser** ([`_parse_s1_response`](../../acervo/s1_unified.py))
   reads a `confidence` field when present on entities or relations,
   clamps to `[0.0, 1.0]`, and defaults to `1.0`. Lives on
   `Entity.confidence` / `Relation.confidence`.
2. **`_validate_s1`** bypasses the garbage-entity filter when
   `confidence < 0.7`. Dead code at the current default of 1.0 but it
   activates automatically for any future model that volunteers
   confidence on short tech jargon.
3. **Pipeline `_persist_s1_entities`** reads `entity.confidence`,
   derives `status="pending_review"` when it's below 0.6, and passes
   both to `upsert_entities(confidence=..., status=...)`.
4. **S1.5 `_auto_promote_pending_entities`** runs at the end of every
   `apply_s1_5_result` and promotes `pending_review` nodes to
   `confirmed` when either:
   - `session_count >= 3` (the node keeps appearing across sessions)
   - `updated_by == "user"` (a human edited it via the REST API)

   Promoted nodes get `confidence_for_owner >= 0.8` (or 1.0 on user
   edit) and are stamped with `updated_by="system"`.

**Literary / kinship relations (v0.6.1):** the S1 prompt's relation
vocabulary was extended from 16 to 21 with five new verbs for book,
film, and family conversations: `appears_in`, `child_of` (already
existed structurally), `married_to`, `set_in`, `narrated_by`. The
`OntologyValidator` has matching synonyms for common variants
(`parent_of` / `father_of` / `mother_of` → `child_of`, `character_in` /
`features` → `appears_in`, etc.). This unblocks the libro benchmark
case which previously had 0-9% relation accuracy because its
character-in-book and family-tree edges had no ontology slot.

**Tests:**
[tests/test_s1_confidence_and_promotion.py](../../tests/test_s1_confidence_and_promotion.py) —
parsing, garbage-filter bypass, pending_review persistence,
auto-promotion rules.
[tests/test_ontology_validator.py](../../tests/test_ontology_validator.py) —
new `test_literary_kinship_*` cases cover the 5 new relations plus 11
common synonym variants.

---

## v0.6.1 — Embedding-based fact dedup (Change 3)

A zero-LLM dedup pass runs inside S1.5 between
`resolve_s1_5_facts` (Phase 3) and `apply_s1_5_result`:

1. **`dedupe_s1_5_facts_by_embedding`** (in
   [acervo/s1_5_graph_update.py](../../acervo/s1_5_graph_update.py))
   groups the assistant facts by canonical entity id and hands the
   bucket to the core function.
2. **`dedupe_facts_by_embedding`** in
   [acervo/extraction/edge_resolution.py](../../acervo/extraction/edge_resolution.py)
   - pulls each node's existing facts via `_load_existing_facts_for_dedup`
     (prefers `graph._get_facts_for` so Ladybug gives us `fact_embedding`
     directly).
   - lazy-embeds any historical facts that don't have an embedding yet
     and persists the result via `graph.set_fact_embedding` so future
     turns skip the cost.
   - batch-embeds the new facts.
   - scores each new fact against every existing one with cosine
     similarity:
     - sim ≥ 0.85 → **drop** (duplicate, don't persist)
     - sim ≥ 0.60 → **flag** (keep but mark as possible duplicate)
     - below → keep with embedding cached on the `ExtractedFact`
3. **`apply_s1_5_result`** calls
   `_persist_fact_embeddings_for_entity` right after each
   `upsert_entities` to look up the newly-written facts by text and
   stamp their `fact_embedding` via `graph.set_fact_embedding`.
4. **REST endpoint** `POST /acervo/graph/deduplicate-facts` runs the
   same algorithm retroactively over any subset of nodes, useful as a
   one-shot "warm up" after deploying v0.6.1 to embed and dedupe a
   graph that was built before per-fact embeddings existed.

**Cost:** 0 LLM calls. One `embed_batch` per node per turn (typically
1-5 new facts). With Ollama + `qwen3-embedding` this is ~500-1000ms on
CPU, ~100ms on GPU.

**Tests:**
[tests/test_fact_dedup_embeddings.py](../../tests/test_fact_dedup_embeddings.py) —
13 tests covering cosine math, drop/flag/keep branches, lazy embedding
persistence, the S1.5 wrapper, and no-op fallbacks.

---

## v0.6.1 — Source tracking and audit trail

Every node, edge, and fact in the graph carries three v0.6.1 audit fields
populated automatically at write time:

| Field | Values | Set by |
|---|---|---|
| `source` | `"llm"`, `"user"`, `"system"` (plus legacy `"world"`, `"user_assertion"`, `"conversation"` on older rows) | The call site: pipeline/S1.5 write `"llm"`; REST handlers in `proxy.py` write `"user"`; batch maintenance jobs write `"system"`. |
| `updated_by` | Same vocabulary as `source`. | Every mutator (`upsert_entities`, `update_node`, `merge_nodes`, `add_edge`, `_add_fact`) accepts an `updated_by=` kwarg that is stamped on the affected row. |
| `updated_at` | ISO-8601 local timestamp | Auto-stamped on every mutation even when the caller forgets. `update_node` has an auto-stamp branch that fills it in if not supplied. |

**Why it matters:**

1. Tells LLM-driven writes apart from user edits, so future retrieval can
   weight or color them differently.
2. Powers the Change 2 auto-promotion rule ("nodes edited by a user become
   `status="confirmed"`").
3. Makes the graph auditable end-to-end without adding a separate
   mutation log table.

**Schema notes:**

- `_ENTITY_NODE_DDL`, `_STRUCTURAL_NODE_DDL`, `_FACT_NODE_DDL` each
  declare `updated_by STRING` and `updated_at STRING`.
- `_REL_DDLS` add the same columns plus a dedicated `source` column
  (distinct from the pre-existing `source_type` column, which still
  carries `"world"` vs `"conversation"`).
- `_ensure_v061_audit_columns` in `ladybug_store.py` runs
  `ALTER TABLE ADD COLUMN IF NOT EXISTS` on startup so databases
  created before v0.6.1 pick up the new columns without a manual
  migration.

---

## LLM calls per turn (worst case)

Per-turn LLM call count for one user message + one assistant response:

| Stage | Call | When | Prompt | Model |
|---|---|---|---|---|
| S1 | extraction + topic + intent | Always | `s1_unified.txt` | extractor (qwen3.5:9b) |
| S1 | retry on parse failure | Conditional (~5% of turns) | same | extractor |
| S2 | — | — | — | — |
| S3 | — | — | — | — |
| S1.5 | curation + assistant extraction | Always (skipped on "no data" responses) | `s1_5_graph_update.txt` | extractor |
| S1.5 (Phase 3) | dedup + contradiction | Conditional (when `existing_facts` or `fact_fulltext_search` candidates) | `dedupe_edges.py` | extractor |

**Typical case:** 2 LLM calls (S1 + S1.5).
**Worst case:** 4 LLM calls (S1 + S1 retry + S1.5 + Phase 3 dedup).
**Skipped cases:** 0 calls when the assistant response matches a "no data" phrase (then `process()` returns immediately).

The user-facing LLM that responds to the user is **separate** from these
— it's the host's responsibility, not Acervo's. Acervo only consumes its
output via `process(assistant_text)`.

---

## Maintenance notes

- **This document is part of the surface area.** If a PR changes the
  flow of S1/S2/S3/S1.5, the doc must be updated in the same PR. A
  CI hook to enforce this would be a nice-to-have but is not in
  place yet.
- **Last verified version:** `release/v0.6.1` (2026-04-13) — all three v0.6.1 changes applied: source tracking + audit trail, embedding-based fact dedup, confidence scoring with pending_review auto-promotion. 391 unit tests passing.
- **Related docs:**
  - [docs/research/graphiti-analysis.md](../research/graphiti-analysis.md) — original Phase 1-4 design
  - [docs/research/v0.6.0-status-report.md](../research/v0.6.0-status-report.md) — full v0.6.0 status, including the 4 critical bug fixes
  - [CHANGELOG.md](../../CHANGELOG.md) — release history
  - [acervo/THIRD_PARTY.md](../../acervo/THIRD_PARTY.md) — Graphiti attribution
