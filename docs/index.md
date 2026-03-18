# Acervo

**Layered memory for AI agents.** Graph-based context engine with universal knowledge and personal user layers.

---

Every conversation your AI agent has starts from scratch. Every context is forgotten. Your agent asks the same questions, loses the same insights, and has no idea who it's talking to.

Acervo fixes that — not by dumping everything into the context window, but by building a structured, layered memory graph that knows what to retrieve, when to retrieve it, and how much it can trust what it knows.

---

## How it works

Acervo sits between your agent and its LLM. It intercepts every message, extracts knowledge, builds a graph, and assembles only the relevant context before each call — keeping token usage flat no matter how long the conversation runs.

```
Agent sends message
    | Acervo extracts entities + facts -> graph
    | Acervo assembles context (topic-aware, budget-aware)
    | LLM receives a tight, relevant context block
    | Token usage stays stable. Knowledge accumulates.
```

---

## Three ways to use Acervo

### 1 — SDK directo (import Python)

For when you're building your own agent in Python. Install it, import it, use it. No server, no protocol, everything in the same process.

```python
from acervo import Acervo

memory = Acervo(llm=my_client, owner="Sandy")
await memory.commit(user_msg, assistant_msg)
context = memory.materialize("query", token_budget=800)
```

### 2 — REST API

For when your agent isn't in Python, or when you want Acervo running as a separate service. Any language can talk to it via HTTP.

```bash
acervo serve --port 5820

# From any language
POST http://localhost:5820/v1/commit
GET  http://localhost:5820/v1/materialize?query=...
```

### 3 — MCP server

For when you're using Claude Code, Cursor, Windsurf, or any agent that supports MCP. No integration code needed — just config.

```bash
acervo mcp
```

```json
{
  "mcpServers": {
    "acervo": {
      "command": "acervo mcp"
    }
  }
}
```

| Tool | Description |
|------|-------------|
| `mem_commit(text, owner)` | Extract and store knowledge from raw text |
| `mem_materialize(query, budget, owner)` | Retrieve a context string ready for prompt injection |

---

## Two knowledge layers

Acervo separates what the agent knows into two distinct layers:

**Layer 1 — Universal knowledge** (verifiable, shareable, community-built)
Facts about the world: cities, programming languages, frameworks, institutions. Downloadable as community packs. Immutable once verified.

**Layer 2 — Personal context** (user-asserted, real for that user)
What the user tells the agent about themselves: their projects, team, preferences, work. Treated as ground truth within that user's context.

See [Layers](layers.md) for details and code examples.

---

## Why not just use a bigger context window?

Larger context windows delay the problem — they don't solve it. Acervo's context stays stable regardless of conversation length because the graph grows with topics, not with messages.

```
Without Acervo:   turn 1 -> 200tk  |  turn 50 -> 9000tk  |  turn 100 -> limit
With Acervo:      turn 1 -> 200tk  |  turn 50 -> 400tk   |  turn 100 -> 420tk
```

---

## Project status

| Feature | Status |
|---------|--------|
| Graph + topic detection | Working |
| Two-layer architecture | In progress |
| Typed ontology | In progress |
| SDK facade (`commit` / `materialize`) | Working |
| MCP server | Planned |
| REST API | Planned |
| `acervo init` indexer | Planned |
| Community knowledge packs | Planned |

---

## Why Acervo?

In library science, an *acervo* is the complete collection of a library — every book, document, and record it holds, organized so anything can be found when needed.

An agent's memory should work like a well-run library: knowledge organized by subject, filed in the right place, and retrieved by a librarian who knows exactly which shelf to go to — not by someone who reads every book from cover to cover every time you ask a question.

---

## License

Apache 2.0 — see [LICENSE](https://github.com/sandyeveliz/acervo/blob/main/LICENSE).

Copyright 2026 Sandy Veliz
