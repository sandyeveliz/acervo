<p align="center">
  <img src="assets/images/acervologo.png" alt="Acervo" width="320">
  <br/>
  <strong>Semantic compression layer for AI agents.</strong><br/>Your agent's context window is finite. Acervo makes it infinite.
</p>

<p align="center">
  <a href="https://pypi.org/project/acervo/"><img src="https://img.shields.io/pypi/v/acervo" alt="PyPI"></a>
  <a href="https://github.com/sandyeveliz/acervo/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.11+-green" alt="Python">
  <a href="https://sandyeveliz.github.io/acervo"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-blue" alt="Docs"></a>
</p>

---

## The problem

Every chat application sends the **entire conversation history** to the LLM on every turn. Turn 1 costs 200 tokens. Turn 50 costs 9,000. Turn 100 hits the context window limit and starts losing information.

This is like reading an entire book from page one every time someone asks you a question.

RAG helps, but it's brute force — it searches everything, retrieves long text chunks, and floods the context with tokens that are mostly noise. A 5-chunk retrieval at 500 tokens each costs 2,500 tokens per turn, and most of that text isn't relevant to the current question.

## What Acervo does differently

Acervo builds a **knowledge graph** from every conversation. When a new message arrives, it retrieves only the relevant context — not raw text chunks, but **compressed knowledge nodes**. The graph *is* the summary.

```
Without Acervo:   turn 1 → 200tk  │  turn 50 → 9,000tk  │  turn 100 → context limit
Classic RAG:      turn 1 → 200tk  │  turn 50 → 2,500tk  │  turn 100 → 2,500tk (noisy)
With Acervo:      turn 1 → 200tk  │  turn 50 → ~400tk   │  turn 100 → ~400tk (signal)
```

A question like "What tech does Project Alpha use?" doesn't need 2,500 tokens of retrieved paragraphs. It needs ~80 tokens:

```
Project Alpha: web app, e-commerce platform
  → uses: React, PostgreSQL, Redis
  → maintained by: Alice, Bob
  → deployed on: AWS, web + mobile
```

That's the entire context. Pure signal, zero noise. The graph replaces raw text with structured knowledge that the LLM can consume in a fraction of the tokens.

## How it works

Acervo is a **context proxy** — it sits between your app and the LLM. Two calls: `prepare()` before, `process()` after.

```
User message
     │
     ▼
 prepare()          ← Reads the graph, builds compressed context from relevant nodes
     │
     ▼
 Your app           ← You call the LLM with Acervo's context (you control model, streaming, tools)
     │
     ▼
 process()          ← Extracts entities, relations, facts → updates the graph
     │
     ▼
 Graph grows        ← Next turn has more knowledge available, still constant tokens
```

Acervo does **not** call the LLM itself. Your app controls the model, streaming, and tools. Acervo only enriches context and extracts knowledge.

### The prepare/process cycle

**`prepare()`** — Before the LLM call:
1. **Topic detection**: Identifies what the conversation is about right now
2. **Layer selection**: Loads the hot layer (current topic's nodes) from the graph
3. **Context assembly**: Builds an enriched system message with compressed nodes, using only the last 2 messages instead of full history

**`process()`** — After the LLM responds:
1. **Extract**: Pulls entities, relations, events, and facts from the conversation
2. **Connect**: Links new knowledge to existing nodes in the graph
3. **Curate**: Merges duplicates, corrects types, discards noise
4. **Available next turn**: New knowledge is immediately available for `prepare()`

### Stateless by design

Acervo has no session state. The graph **is** the state. Every turn follows the same pipeline: read from graph → compress → inject → extract → write. If your app crashes and restarts, the next message works identically.

## Quick start

### 1. Install

```bash
pip install acervo
```

### 2. Start a local LLM

Acervo needs a small utility model for topic detection and extraction. Any OpenAI-compatible endpoint works.

**With [LM Studio](https://lmstudio.ai/):**
Load a model like `qwen2.5-3b-instruct` and start the server.

**With [Ollama](https://ollama.ai/):**
```bash
ollama run qwen2.5:3b
```

### 3. Two lines to add memory

```python
import asyncio
from acervo import Acervo, OpenAIClient

async def main():
    llm = OpenAIClient(
        base_url="http://localhost:1234/v1",
        model="qwen2.5-3b-instruct",
    )
    memory = Acervo(llm=llm, owner="demo-user")

    history = [{"role": "system", "content": "You are a helpful assistant."}]

    # Turn 1: user shares information
    user_msg = "I work at Acme Corp, we're building a React app called Beacon with PostgreSQL"
    history.append({"role": "user", "content": user_msg})

    prep = await memory.prepare(user_msg, history)
    # prep.context_stack → enriched messages for the LLM
    # prep.has_context → False (first turn, graph is empty)

    # Call YOUR LLM with prep.context_stack
    assistant_msg = "Got it! Tell me more about Beacon."
    history.append({"role": "assistant", "content": assistant_msg})

    await memory.process(user_msg, assistant_msg)
    # Graph now has: Acme Corp (organization), Beacon (project), React + PostgreSQL (technology)
    # Plus edges: Acme Corp → produces → Beacon, Beacon → uses_technology → React, etc.

    # Turn 2: ask about something stored
    user_msg = "What do you know about our project?"
    history.append({"role": "user", "content": user_msg})

    prep = await memory.prepare(user_msg, history)
    # prep.has_context → True
    # Context includes Beacon's full node: tech stack, relations, facts
    # Only ~80 tokens of graph context — not the raw turn 1 text

    print(prep.context_stack)

asyncio.run(main())
```

### 4. Low-level API

If you don't need the full pipeline, use `commit()` and `materialize()` directly:

```python
# Store knowledge
await memory.commit(
    "Batman was created by Bill Finger and Bob Kane in 1939",
    "He first appeared in Detective Comics #27.",
)

# Retrieve relevant context
context = await memory.materialize("Batman")
# Returns compressed node data, not raw text
```

## Proxy mode (`acervo serve`)

If you don't want to change your app's code, Acervo can run as a **transparent proxy** between your client and the LLM. Point your app's base URL to Acervo; it enriches every request automatically.

```bash
acervo serve --port 9470 --forward-to http://localhost:1234/v1
```

```
Your app                    Acervo proxy (:9470)                LLM server (:1234)
   │                              │                                   │
   ├─ POST /v1/chat/completions ─►│                                   │
   │                              ├─ prepare() (topic + graph)        │
   │                              ├─ inject compressed context         │
   │                              ├─ POST /v1/chat/completions ──────►│
   │                              │◄──────────── stream response ─────┤
   │◄──── stream response ────────┤                                   │
   │                              ├─ process() (extract + persist)    │
   │                              │                                   │
```

**Zero code changes** — just redirect `base_url`. Supports OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) formats, streaming and non-streaming.

```bash
# What Acervo did on the last turn
curl http://localhost:9470/acervo/last-turn

# Graph stats
curl http://localhost:9470/acervo/status
```

## The knowledge graph

As conversations happen, Acervo builds a persistent graph of entities, relations, and facts. Every node is a compressed representation — not raw text, but structured knowledge.

### What gets extracted

**Entities** — real, named things:

```
Alice Chen        → person, PERSONAL
Acme Corp         → organization, PERSONAL
Project Beacon    → project, PERSONAL, purpose: "e-commerce platform"
React             → technology, UNIVERSAL
PostgreSQL        → technology, UNIVERSAL
Sprint review Q1  → event, participants: [Alice, Bob, CTO]
```

**Relations** — precise connections:

```
Alice       → works_at       → Acme Corp
Acme Corp   → produces       → Project Beacon
Beacon      → uses_technology → React
Beacon      → uses_technology → PostgreSQL
Alice       → participated_in → Sprint review Q1
```

**Facts** — specific claims attached to entities:

```
Project Beacon: "In production since March, 50k monthly users"
Alice Chen: "Lead developer, joined in 2024"
```

### Two knowledge layers

**PERSONAL** — User-specific knowledge: projects, preferences, relationships, work context. Scoped to that user.

**UNIVERSAL** — World knowledge: technologies, cities, public figures, concepts. Shareable across users.

The extractor assigns layers automatically. "We use React" creates a PERSONAL edge from your project to the UNIVERSAL React node. A web search about React creates UNIVERSAL nodes without touching your personal graph.

### Events as super-summaries

Events capture *what happened* with *who was involved*. Instead of storing 5,000 tokens of meeting notes, the graph stores:

```
Event: "Sprint review Q1"
  participants: [Alice, Bob, CTO]
  description: "Shipped auth module, discussed perf issues, decided to add Redis cache"
  temporal_marker: "End of Q1 2026"
```

When someone asks "What did we discuss in the sprint review?", the LLM gets this 40-token node — not the full meeting transcript.

## Topic-based context layers

This is what makes Acervo fundamentally different from a cache or a RAG system.

### The layers move with the conversation

Traditional systems use temporal layers — recent things are "hot", old things are "cold". Acervo uses **topic-based layers**: what's relevant depends on what you're talking about *right now*, not when it was mentioned.

```
09:00 — "Let's work on the auth bug in Beacon"
  HOT:  Beacon, React, PostgreSQL, auth module, Alice
  WARM: (empty, first topic)
  COLD: (empty)
  → LLM receives ~200 tokens of graph context

09:45 — "Now let's look at Project Compass, the mobile app"
  HOT:  Compass, React Native, Firebase, push notifications
  WARM: Beacon, React, auth module          ← dropped from hot
  COLD: (empty)
  → LLM receives ~200 tokens about Compass only

14:00 — "Back to the Beacon auth bug, did we fix it?"
  HOT:  Beacon, React, auth module          ← jumps from warm back to hot instantly
  WARM: whatever was just before
  COLD: Compass, and everything else
  → LLM receives Beacon context including this morning's facts. No re-explanation needed.

17:00 — "Switching gears — have you read Dune?"
  HOT:  Dune, Arrakis, Paul Atreides       ← new topic cluster
  COLD: ALL work context (0 tokens)
  → Vector search scoped to "Dune" cluster only
```

### Progressive retrieval

The LLM doesn't receive all layers at once:

1. **Default**: inject only the hot layer (~200-400 tokens)
2. **If the user asks for more** ("What about the other projects?"): bring in warm layer
3. **If needed**: bring in cold layer

In 80% of turns, the hot layer is enough. The context window stays small and relevant.

### The topic graph scopes vector search

When Acervo knows the topic is "Beacon", vector search is filtered to only chunks tagged with that cluster. Instead of searching 10,000 chunks across every topic, it searches ~200 chunks from the relevant cluster. Faster, more precise, smaller results.

## Index a codebase

Acervo can index an entire project directory — parsing source code with tree-sitter, resolving import dependencies, and building a structural knowledge graph without any LLM calls.

```bash
acervo init /path/to/project
acervo index /path/to/project
```

For a 50-file project, structural indexing takes **under 2 seconds**. Each file stores a SHA-256 hash — unchanged files are skipped on re-index.

### What it extracts

**Phase 1 — Structural** (tree-sitter, no LLM):
- Functions, classes, interfaces, types, methods
- Imports/exports with dependency resolution
- Markdown files split by heading hierarchy

**Phase 2 — Semantic** (optional, needs running models):
- Embeddings per code entity for similarity search
- LLM summaries, topic tags, and implicit relation detection

```bash
acervo index /path/to/project \
  --embedding-model nomic-embed-text \
  --embedding-endpoint http://localhost:11434 \
  --llm-model qwen2.5-3b-instruct \
  --llm-endpoint http://localhost:1234/v1
```

### Supported files

| Extension | Parser | Extracts |
|-----------|--------|----------|
| `.py` | tree-sitter | functions, classes, methods, imports, decorators |
| `.ts` `.tsx` | tree-sitter | functions, classes, interfaces, types, imports, exports |
| `.js` `.jsx` | tree-sitter | functions, classes, imports, exports |
| `.html` | regex | component references, element IDs |
| `.css` | regex | selectors, custom properties |
| `.md` | heading parser | sections with hierarchy context |

### Keep it fresh

```bash
acervo reindex    # Only changed files (SHA-256 comparison)
acervo status     # Graph stats
```

## Architecture

### How nodes replace chunks

In a traditional RAG system, a question like "When did Alice and Bob first work together?" retrieves 5 text chunks at ~500 tokens each = 2,500 tokens of raw text. Most of it is irrelevant context around the actual answer.

In Acervo, the graph has:

```
Alice → participated_in → Project Beacon launch (event)
Bob   → participated_in → Project Beacon launch (event)

Event: "Project Beacon launch"
  participants: [Alice, Bob, CTO]
  description: "First joint project, shipped MVP in 2 weeks"
  temporal_marker: "March 2025"
```

The LLM receives ~60 tokens. Same answer quality, 40x fewer tokens.

### Pipeline components

All behind a single `LLMClient` protocol:

| Component | What it does |
|-----------|-------------|
| **Topic detector** | Identifies the current conversation topic. Cascade: keywords → embeddings → LLM |
| **Context index** | Selects hot/warm/cold nodes based on topic proximity, assembles compressed context |
| **Extractor** | Pulls entities, relations, events, and facts from conversations |
| **Graph curator** | Merges duplicates, corrects types, discards noise, connects isolated nodes |

### History windowing

```
Traditional chat (turn 50):
  system + msg1 + msg2 + ... + msg49 + msg50
  = 9,000 tokens and growing

Acervo (turn 50):
  system + [compressed graph context for current topic] + msg49 + msg50
  = ~400 tokens, constant
```

The graph context is **ranked by topic relevance**. Raw history treats every past message equally. Acervo gives you the right 400 tokens of context instead of 9,000 tokens of everything.

### Works with any LLM

Acervo needs one small utility model for extraction and topic detection. Your main LLM can be anything — local or cloud, any size, any provider.

```python
class LLMClient(Protocol):
    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> str: ...
```

## Tested setup

| Component | Tool | Model |
|-----------|------|-------|
| **LLM server** | [LM Studio](https://lmstudio.ai/) | `qwen3.5-9b` (chat + extraction) |
| **Embeddings** | [Ollama](https://ollama.ai/) | `qwen3-embedding` (optional) |
| **Client app** | [AVS-Agents](https://github.com/sandyeveliz/AVS-Agents) | Python TUI + Web UI |

## Project status

v0.1.2 — [Changelog](./CHANGELOG.md)

| Feature | Status |
|---------|--------|
| Knowledge graph (JSON persistence) | ✅ Working |
| UNIVERSAL / PERSONAL layers | ✅ Working |
| `prepare()` / `process()` context proxy API | ✅ Working |
| Topic-based context layers (HOT/WARM/COLD) | ✅ Working |
| Topic detector (keywords → embeddings → LLM) | ✅ Working |
| Context index with token budgeting | ✅ Working |
| History windowing (constant token usage) | ✅ Working |
| Entity + relation + event extraction | ✅ Working |
| Graph curation (dedup, type correction) | ✅ Working |
| `acervo index` — structural + semantic | ✅ Working |
| REST API (`acervo serve`) | ✅ Working |
| Progressive retrieval (hot → warm → cold) | 🔜 Next |
| Topic-scoped vector search | 🔜 Next |
| Fine-tuned extraction model | 🔜 Planned |
| Community knowledge packs | 🔜 Planned |

## Documentation

- **[Tutorial](https://sandyeveliz.github.io/acervo/tutorial/)** — Build a chat with persistent memory in 5 minutes
- **[Getting Started](https://sandyeveliz.github.io/acervo/getting-started/)** — Installation, quick start, LLMClient protocol
- **[Configuration](https://sandyeveliz.github.io/acervo/configuration/)** — SDK parameters, environment variables
- **[Knowledge Layers](https://sandyeveliz.github.io/acervo/layers/)** — UNIVERSAL vs PERSONAL, node lifecycle, topic layers
- **[Roadmap](https://sandyeveliz.github.io/acervo/roadmap/)** — Planned features

## Why "Acervo"?

In library science, an *acervo* is the complete collection of a library — every book, document, and record, organized so anything can be found when needed.

Your agent's memory should work like a library: knowledge organized by subject, retrievable in an instant. Not like someone who reads every book from cover to cover every time you ask a question.

## Contributing

Open source under Apache 2.0. See [CONTRIBUTING.md](./CONTRIBUTING.md).

## License

Apache 2.0 — see [LICENSE](./LICENSE).

---

<p align="center">
  <a href="https://github.com/sandyeveliz/acervo">GitHub</a> · <a href="https://sandyeveliz.github.io/acervo">Docs</a> · <a href="https://pypi.org/project/acervo/">PyPI</a>
</p>