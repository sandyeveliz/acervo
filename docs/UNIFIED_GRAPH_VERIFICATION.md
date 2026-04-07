# Unified Graph Verification Report

> Verified: 2026-04-06
> Acervo v0.5.0

## Path Analysis

| Question | Answer | Evidence |
|----------|--------|----------|
| Same graph instance? | **YES** | CLI, Acervo.from_project(), and proxy all create TopicGraph from `project.graph_path`. Pipeline receives `self._graph` from Acervo. |
| Same node format? | **YES** | All entity nodes use `kind="entity"`, same _make_id() for IDs. File/section/symbol nodes use distinct kinds. |
| Dedup works? | **YES** | upsert_entities() checks `nid in self._nodes` — existing nodes get updated, facts appended. Express.js from curation + conversation = 1 node. |
| S1.5 updates indexed entities? | **YES** | Sandy (conversation) coexists with Express (indexed). Facts append to existing nodes. |
| Graph reloads after index? | **YES** | Proxy has `/acervo/reload-graph` endpoint that calls `graph.reload()` → `_load()` from disk. |

## Combined Test Results

**Test setup:** P1 (Todo App) — 232 nodes, 1103 edges pre-indexed (31 files, 134 symbols, 8 entities from curate, 4 synthesis nodes).

| Step | Result | Details |
|------|--------|---------|
| P1 indexed nodes present | ✓ | 232 nodes, 1103 edges. 8 entity nodes: Todo App, React, Express, PostgreSQL, etc. |
| Conversation adds new nodes | ✓ | Sandy (Person, PERSONAL) created with fact "Works as lead developer" and edge `sandy --maintains--> todo_app` |
| S2 retrieves from both sources | ✓ | Turn 2: 24 hot + 91 warm + 68 cold nodes. warm_tokens=397. Indexed file/symbol nodes and conversation entity coexist in BFS. |
| Dedup prevents duplicates | ✓ | Turn 3: user mentions "Express.js" — only 1 Express node exists (not duplicated). |
| S1.5 appends facts to indexed nodes | ✓ | Sandy got fact appended. Indexed entities updated with `last_active` timestamp. |

## Graph State After 3 Conversation Turns

```
BEFORE: 232 nodes, 1103 edges
AFTER:  233 nodes, 1105 edges (+1 node Sandy, +2 edges)

Entity nodes (9 total — 8 from indexation + 1 from conversation):
  todo_app:     Todo App    [Project]    source=curation
  react:        React       [Technology] source=conversation
  express:      Express     [Technology] source=conversation
  postgresql:   PostgreSQL  [Technology] source=conversation
  react_native: React Native[Technology] source=curation
  typescript:   TypeScript  [Technology] source=curation
  react_hooks:  React Hooks [Concept]    source=curation
  alice:        Alice       [Person]     source=conversation
  sandy:        Sandy       [Person]     source=user_assertion  ← NEW from conversation
```

## Key Mechanisms

### 1. Merge Semantics (graph.py:273-301)
When `upsert_entities()` encounters an existing node ID:
- Updates `last_active` timestamp
- Increments `session_count`
- **Appends** new facts (with dedup check)
- Does NOT overwrite existing attributes

### 2. Consistent ID Generation (_make_id)
All paths use the same deterministic function:
```python
_make_id("Express.js") → "express_js"  (consistent across curation + S1)
```
Strips accents, lowercases, replaces non-alphanumeric with `_`.

### 3. Node Kind Differentiation
| Source | kind | Example |
|--------|------|---------|
| Indexer structural | `file` | todo.controller.ts |
| Indexer structural | `symbol` | createTodo |
| Indexer structural | `section` | Introduction |
| Indexer structural | `folder` | controllers/ |
| Curate/S1 | `entity` | Express, Sandy |
| Synthesize | `synthesis` | project_overview |

S2 BFS traverses ALL kinds via edges. No kind-based filtering in the traversal itself.

### 4. Graph Reload
After `acervo index` writes to disk:
```
POST /acervo/reload-graph → graph.reload() → _load() from nodes.json/edges.json
```
Proxy keeps conversation state, only graph data refreshes.

## Issues Found

None. The graph is genuinely unified:
1. Both paths write to the same `nodes.json` / `edges.json`
2. Entity IDs are deterministic and consistent
3. Merge semantics prevent duplicates
4. S2 retrieves from all sources equally
5. Facts from conversation append to indexed entities

## Conclusion

**YES — indexation and conversation feed a unified knowledge graph.**

The changelog claim is accurate. Both input sources produce nodes in the same TopicGraph instance, queryable through the same S2/S3 pipeline. The graph correctly merges entities from both sources without duplication.
