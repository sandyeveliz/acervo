# Roadmap

Current version: **v0.1.2** ([PyPI](https://pypi.org/project/acervo/) · [Changelog](https://github.com/sandyeveliz/acervo/blob/main/CHANGELOG.md))

---

## What works today

- **Knowledge graph** with JSON persistence (nodes.json, edges.json)
- **Two-layer architecture** — UNIVERSAL (world knowledge) and PERSONAL (user-specific)
- **`prepare()` / `process()` API** — context proxy pipeline
- **`commit()` / `materialize()`** — lower-level graph API
- **Auto-registering ontology** — LLM creates new entity types and relations dynamically
- **Semantic relations** — IS_A, CREATED_BY, ALIAS_OF, PART_OF, SET_IN, DEBUTED_IN, PUBLISHED_BY
- **Topic detector** — 3-level cascade (keywords, embeddings, LLM classification)
- **Query planner** — LLM decides: GRAPH_ALL, WEB_SEARCH, READY
- **Context index** — 3-layer stack (system, warm graph, hot messages) with token budgeting
- **ConversationExtractor + SearchExtractor** — conservative extraction from conversations and web results
- **Built-in OpenAIClient** — zero external deps beyond pydantic
- **LLMClient + Embedder protocols** — bring your own LLM
- **56 unit tests + 8 integration tests**

---

## Planned features

### REST API (`acervo serve`)

Run Acervo as an independent HTTP server. Any language can use it via HTTP.

```
POST /v1/prepare    — enrich context from graph
POST /v1/process    — extract knowledge from response
POST /v1/commit     — store knowledge directly
GET  /v1/materialize — retrieve relevant context
```

**Status:** not started

---

### MCP Server (`acervo mcp`)

[Model Context Protocol](https://modelcontextprotocol.io/) server for Claude Desktop, Cursor, and other MCP-compatible clients. Expose prepare/process as MCP tools.

**Status:** not started

---

### Session Summarizer

End-of-session consolidation: summarize conversation highlights, merge redundant facts, cleanup cold nodes.

**Status:** stub exists (`acervo/session_summarizer.py`)

---

### Vector Search

Semantic similarity search across graph nodes using embeddings. Complement to keyword-based topic detection for better recall on ambiguous queries.

**Status:** not started

---

### Community Knowledge Packs

Pre-built UNIVERSAL layer graphs for specific domains (programming languages, geography, comic book universes, etc.). Installable and shareable.

**Status:** not started — format TBD

---

### `acervo init` — Directory Indexer

Index a codebase or document folder into the knowledge graph. Useful for giving agents project context without manual conversation.

**Status:** not started

---

## Want to help?

Check out [Contributing](contributing.md) or open an [issue](https://github.com/sandyeveliz/acervo/issues) to discuss ideas.
