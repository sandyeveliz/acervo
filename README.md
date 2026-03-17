# acervo

**Layered memory for AI agents.** Graph-based context engine with universal knowledge and personal user layers.

---

Every conversation your AI agent has starts from scratch. Every context is forgotten. Your agent asks the same questions, loses the same insights, and has no idea who it's talking to.

Acervo fixes that — not by dumping everything into the context window, but by building a structured, layered memory graph that knows what to retrieve, when to retrieve it, and how much it can trust what it knows.

---

## How it works

Acervo sits between your agent and its LLM. It intercepts every message, extracts knowledge, builds a graph, and assembles only the relevant context before each call — keeping token usage flat no matter how long the conversation runs.

```
Agent sends message
    ↓ Acervo extracts entities + facts → graph
    ↓ Acervo assembles context (topic-aware, budget-aware)
    ↓ LLM receives a tight, relevant context block
    ↓ Token usage stays stable. Knowledge accumulates.
```

---

## Two knowledge layers

Acervo separates what the agent knows into two distinct layers:

**Layer 1 — Universal knowledge** (verifiable, shareable, community-built)
Facts about the world: cities, programming languages, frameworks, institutions. Downloadable as community packs. Immutable once verified.

**Layer 2 — Personal context** (user-asserted, real for that user)
What the user tells the agent about themselves: their projects, team, preferences, work. Treated as ground truth within that user's context — like a new employee trusting what their manager says, even before it's publicly verifiable.

```python
# Layer 1 — anyone can verify this
Node("Cipolletti", type="Place", layer=Layer.UNIVERSAL, source="world")

# Layer 2 — Sandy says so, real for Sandy
Node("Altovallestudio", type="Organization", layer=Layer.PERSONAL,
     owner="Sandy", source="user_assertion", confidence_for_owner=1.0)
```

Nodes that are mentioned but not fully described are stored as `incomplete` with a `pending_fields` list — the agent fills them in naturally as the conversation continues, without interrogating the user.

---

## Quickstart

```bash
pip install acervo
```

```python
from acervo import Acervo

memory = Acervo(owner="Sandy")

# After each message — Acervo extracts and stores
memory.commit("I work at Altovallestudio, we have 4 projects: Butaco, Checkear, Walletfy and the main app")

# Before each LLM call — Acervo assembles relevant context
context = memory.materialize(query="project status", token_budget=800)

# context is a ready-to-use string — inject it into your prompt
```

---

## Run the init indexer on any directory

Point Acervo at a folder and it builds the initial graph from whatever it finds — code, documents, spreadsheets, folder structure. No manual configuration.

```bash
acervo init ./my-project
```

What it extracts **without opening files**: file names, types, folder hierarchy, recency.
What it extracts **when it can open them**: stack from `package.json`/`pyproject.toml`, module structure from source code, text from `.md` and `.txt`, metadata from `.docx` and `.xlsx`.

After init, the agent already knows the shape of your work before you say a word.

---

## MCP server

Acervo exposes itself as an MCP server so any compatible agent — Claude Code, Cursor, Windsurf, or your own — can use it with zero integration code.

```bash
acervo mcp
```

Add to your `claude_desktop_config.json` or Claude Code config:

```json
{
  "mcpServers": {
    "acervo": {
      "command": "acervo mcp"
    }
  }
}
```

Two tools are exposed:

| Tool | Description |
|------|-------------|
| `mem_commit(text, owner)` | Extract and store knowledge from raw text |
| `mem_materialize(query, budget, owner)` | Retrieve a context string ready for prompt injection |

---

## Ontology — extensible entity types

Acervo ships with a base set of entity types. You can register your own.

**Built-in types:** `Person`, `Organization`, `Project`, `Place`, `Technology`, `Document`, `Rule`

**Built-in relations:** `WORKS_AT`, `LIVES_IN`, `OWNS`, `BELONGS_TO`, `USES_TECHNOLOGY`, `HAS_MODULE`, `LIKES`, `RELATED_TO`

```python
from acervo.ontology import register_type, register_relation

# Add your own entity type
register_type(
    name="Recipe",
    attributes=["ingredients", "time", "difficulty"],
    layer_default=Layer.PERSONAL
)

# Add your own relation
register_relation("COAUTHORED_WITH")
```

Entities the extractor can't classify are stored as `type=Unknown` with `status=incomplete` and resolved in future turns.

---

## Why not just use a bigger context window?

Larger context windows delay the problem — they don't solve it. Acervo's context stays stable regardless of conversation length because the graph grows with topics, not with messages. A 50-turn conversation about one project produces roughly the same context block as a 5-turn one.

```
Without Acervo:   turn 1 → 200tk  |  turn 50 → 9000tk  |  turn 100 → limit
With Acervo:      turn 1 → 200tk  |  turn 50 → 400tk   |  turn 100 → 420tk
```

---

## Community knowledge packs (Layer 1)

Community-contributed packs for Layer 1 knowledge — download only what your agent needs.

```bash
acervo install pack javascript
acervo install pack python
acervo install pack geography-argentina
```

Contributions welcome. See `CONTRIBUTING.md` for the pack format spec.

---

## Project status

Acervo is in early development. The core graph and context engine are working. Layers, ontology, MCP server, and init indexer are in progress.

| Feature | Status |
|---------|--------|
| Graph + topic detection | ✅ working |
| Two-layer architecture | 🔧 in progress |
| Typed ontology | 🔧 in progress |
| MCP server | 📋 planned |
| `acervo init` indexer | 📋 planned |
| Community knowledge packs | 📋 planned |

---

## Contributing

Acervo is open source under Apache 2.0. Contributions are welcome — especially:

- New entity types and relations for the base ontology
- Community knowledge packs (Layer 1)
- Parsers for new file types (`.xlsx`, `.docx`, `.pdf`)
- Integrations with other agent frameworks

See `CONTRIBUTING.md` to get started.

---

## License

Apache 2.0 — see [LICENSE](./LICENSE).

Copyright 2026 Sandy Veliz

You can use Acervo freely in personal and commercial projects. If you distribute software that includes Acervo, you must retain the copyright notice and indicate any changes you made.

---

Built by [Sandy Veliz](https://github.com/sandyeveliz) · [github.com/sandyeveliz/acervo](https://github.com/sandyeveliz/acervo)
