# acervo

**Context proxy for AI agents.** Sits between the user and the LLM — enriches context before, extracts knowledge after.

---

Every conversation your AI agent has starts from scratch. Every context is forgotten. Your agent asks the same questions, loses the same insights, and has no idea who it's talking to.

Acervo fixes that — not by dumping everything into the context window, but by building a structured knowledge graph that knows what to retrieve, when to retrieve it, and how much it can trust what it knows.

---

## How it works

Acervo is a **context proxy**. It intercepts every message to build context from its knowledge graph, and intercepts every response to extract and store new knowledge.

```
User message
    ↓
acervo.prepare(message, history)
    ↓  topic detection → query planning → context building
    ↓  returns context_stack + plan (GRAPH or WEB_SEARCH)
    ↓
Client calls LLM with enriched context
    ↓
acervo.process(message, response, web_results?)
    ↓  entity extraction → fact persistence → graph update
    ↓
Knowledge accumulates. Token usage stays flat.
```

---

## Quickstart

```bash
pip install acervo
```

```python
from acervo import Acervo, OpenAIClient

# Any OpenAI-compatible LLM (LM Studio, Ollama, OpenAI, etc.)
llm = OpenAIClient(
    base_url="http://localhost:1234/v1",
    model="qwen2.5-3b-instruct",
    api_key="lm-studio",
)

memory = Acervo(llm=llm, owner="Sandy")

# Before LLM call — Acervo builds context from knowledge graph
prep = await memory.prepare(user_text, history)
# prep.context_stack → ready for LLM
# prep.plan.tool → "GRAPH_ALL", "WEB_SEARCH", or "READY"

# After LLM call — Acervo extracts knowledge from the response
await memory.process(user_text, assistant_response)
```

The lower-level API is also available:

```python
# Extract and store knowledge
await memory.commit("Batman was created by Bill Finger in 1939")

# Build context for a query
context = memory.materialize("Batman", token_budget=800)
```

---

## Two knowledge layers

Acervo separates what the agent knows into two layers:

**Layer 1 — Universal** (world knowledge, verifiable)

Facts about the world: cities, characters, technologies, institutions. Detected automatically when the entity type is universal (places, characters, technologies).

**Layer 2 — Personal** (user-asserted, trusted within context)

What the user tells the agent: their projects, preferences, relationships. Treated as ground truth for that user.

```python
from acervo.layers import Layer

# Layer 1 — anyone can verify this
# Detected automatically for Lugar, Personaje, Tecnología types
graph.upsert_entities(
    [("Gotham City", "Lugar")],
    layer=Layer.UNIVERSAL, source="world",
)

# Layer 2 — Sandy says so, real for Sandy
graph.upsert_entities(
    [("Altovallestudio", "Organización")],
    layer=Layer.PERSONAL, source="user_assertion", owner="Sandy",
)
```

---

## Auto-registering ontology

Acervo ships with built-in entity types and relations, but the LLM can create new ones on the fly.

**Built-in types:** `Persona`, `Personaje`, `Organización`, `Proyecto`, `Lugar`, `Tecnología`, `Obra`, `Universo`, `Editorial`

**Built-in relations:** `IS_A`, `CREATED_BY`, `ALIAS_OF`, `PART_OF`, `SET_IN`, `DEBUTED_IN`, `PUBLISHED_BY`, `TRABAJA_EN`, `VIVE_EN`, and more.

When the LLM extracts an entity with a type that doesn't exist, Acervo auto-registers it:

```python
# LLM returns {"name": "Flash", "type": "superhero"}
# → Acervo auto-registers "Superhero" as a new entity type
# → No code changes needed
```

You can also register types explicitly:

```python
from acervo.ontology import register_type, register_relation

register_type("Recipe", ["ingredients", "time", "difficulty"])
register_relation("INSPIRED_BY")
```

---

## Pipeline components

Acervo's `prepare()` runs a full pipeline internally, all behind a single `LLMClient` protocol:

| Component | What it does |
|-----------|-------------|
| **Topic detector** | 3-level cascade: keywords → embeddings → LLM classification |
| **Query planner** | LLM decides: use graph data, search web, or respond directly |
| **Context index** | Builds a 3-layer context stack with sliding window and token budgeting |
| **Extractor** | Extracts entities, relations, and facts from conversations and web results |
| **Synthesizer** | Renders graph nodes into compact text for LLM context injection |

---

## Why not just use a bigger context window?

Larger context windows delay the problem — they don't solve it. Acervo's context stays stable regardless of conversation length because the graph grows with topics, not with messages.

```
Without Acervo:   turn 1 → 200tk  |  turn 50 → 9000tk  |  turn 100 → limit
With Acervo:      turn 1 → 200tk  |  turn 50 → 400tk   |  turn 100 → 420tk
```

---

## Project status

| Feature | Status |
|---------|--------|
| Knowledge graph + JSON persistence | ✅ Working |
| Two-layer architecture (UNIVERSAL/PERSONAL) | ✅ Working |
| Auto-registering ontology | ✅ Working |
| Semantic relations (IS_A, CREATED_BY, etc.) | ✅ Working |
| prepare()/process() context proxy API | ✅ Working |
| Topic detector (3-level cascade) | ✅ Working |
| Query planner (LLM-based tool selection) | ✅ Working |
| Context index with token budgeting | ✅ Working |
| Built-in OpenAIClient (zero deps) | ✅ Working |
| LLMClient + Embedder protocols | ✅ Working |
| 56 unit tests | ✅ Passing |
| MkDocs documentation | ✅ Published |
| MCP server | 📋 Planned |
| REST API (`acervo serve`) | 📋 Planned |
| `acervo init` directory indexer | 📋 Planned |
| Community knowledge packs | 📋 Planned |
| Vector search (embeddings) | 📋 Planned |

---

## Documentation

Full documentation available at **[sandyeveliz.github.io/acervo](https://sandyeveliz.github.io/acervo)**

- [Getting Started](https://sandyeveliz.github.io/acervo/getting-started/)
- [Configuration](https://sandyeveliz.github.io/acervo/configuration/)
- [Knowledge Layers](https://sandyeveliz.github.io/acervo/layers/)

---

## Why Acervo?

In library science, an *acervo* is the complete collection of a library — every book, document, and record it holds, organized so anything can be found when needed.

An agent's memory should work like a well-run library: knowledge organized by subject, filed in the right place, and retrieved by a librarian who knows exactly which shelf to go to — not by someone who reads every book from cover to cover every time you ask a question.

---

## Contributing

Acervo is open source under Apache 2.0. See [CONTRIBUTING.md](./CONTRIBUTING.md) to get started.

---

## License

Apache 2.0 — see [LICENSE](./LICENSE).

Copyright 2026 Sandy Veliz

---

Built by [Sandy Veliz](https://github.com/sandyeveliz) · [github.com/sandyeveliz/acervo](https://github.com/sandyeveliz/acervo)
