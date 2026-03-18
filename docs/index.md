# Acervo

**Context proxy for AI agents.** Sits between the user and the LLM — enriches context before, extracts knowledge after.

---

Every conversation your AI agent has starts from scratch. Every context is forgotten. Your agent asks the same questions, loses the same insights, and has no idea who it's talking to.

Acervo fixes that — not by dumping everything into the context window, but by building a structured knowledge graph that knows what to retrieve, when to retrieve it, and how much it can trust what it knows.

---

## How it works

Acervo is a **context proxy**. It intercepts every message to build context from its knowledge graph, and intercepts every response to extract and store new knowledge.

```
User message
    |
acervo.prepare(message, history)
    |  topic detection > query planning > context building
    |  returns context_stack + plan (GRAPH or WEB_SEARCH)
    |
Client calls LLM with enriched context
    |
acervo.process(message, response, web_results?)
    |  entity extraction > fact persistence > graph update
    |
Knowledge accumulates. Token usage stays flat.
```

---

## Quickstart

```python
from acervo import Acervo, OpenAIClient

llm = OpenAIClient(
    base_url="http://localhost:1234/v1",
    model="qwen2.5-3b-instruct",
    api_key="lm-studio",
)

memory = Acervo(llm=llm, owner="Sandy")

# Before LLM call
prep = await memory.prepare(user_text, history)

# After LLM call
await memory.process(user_text, assistant_response)
```

See [Getting Started](getting-started.md) for the full guide.

---

## Project status

| Feature | Status |
|---------|--------|
| Knowledge graph + JSON persistence | Working |
| Two-layer architecture (UNIVERSAL/PERSONAL) | Working |
| Auto-registering ontology | Working |
| Semantic relations (IS_A, CREATED_BY, etc.) | Working |
| prepare()/process() context proxy API | Working |
| Topic detector, query planner, context index | Working |
| 56 unit tests | Passing |
| MCP server | Planned |
| REST API | Planned |
| Vector search | Planned |

---

## Why Acervo?

In library science, an *acervo* is the complete collection of a library — every book, document, and record it holds, organized so anything can be found when needed.

An agent's memory should work like a well-run library: knowledge organized by subject, filed in the right place, and retrieved by a librarian who knows exactly which shelf to go to.
