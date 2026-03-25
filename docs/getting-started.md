# Getting Started

## Installation

```bash
pip install acervo
```

For development:

```bash
git clone https://github.com/sandyeveliz/acervo.git
cd acervo
pip install -e ".[dev]"
```

---

## Quickstart: SDK

The simplest way to use Acervo — import it into your Python agent.

```python
from acervo import Acervo, OpenAIClient

# Create an LLM client (any OpenAI-compatible endpoint works)
llm = OpenAIClient(
    base_url="http://localhost:1234/v1",
    model="qwen2.5-3b-instruct",
    api_key="lm-studio",
)

# Create the memory instance
memory = Acervo(llm=llm, owner="Sandy")

# After each conversation turn — extract and store knowledge
await memory.commit(
    user_msg="I work at Altovallestudio, we have 4 projects",
    assistant_msg="Tell me more about the projects.",
)

# Before each LLM call — assemble relevant context
context = memory.materialize(query="project status", token_budget=800)
# context is a ready-to-use string — inject it into your prompt

# At the start of each turn — age out context
memory.cycle_status()
```

### Using `from_env()`

If you prefer to configure via environment variables:

```python
from acervo import Acervo

# Reads .env file automatically
memory = Acervo.from_env(owner="Sandy")
```

See [Configuration](configuration.md) for the full list of environment variables.

---

!!! info "Planned: Standalone server and MCP"
    A standalone HTTP server (`acervo serve`) and MCP server (`acervo mcp`) are planned
    for a future release. These will let you use Acervo from any language via HTTP or
    as a tool in MCP-compatible clients like Claude Desktop.

    See the [Roadmap](roadmap.md) for details and status.

---

## Core concepts

### The graph

Acervo stores knowledge as a graph of **nodes** (entities) and **edges** (relations). Each node has:

- **label** — the entity name ("Sandy", "Cipolletti", "React")
- **type** — entity type ("Persona", "Lugar", "Tecnologia")
- **layer** — UNIVERSAL (world knowledge) or PERSONAL (user-specific)
- **facts** — a list of concrete statements about this entity
- **status** — hot / warm / cold (determines retrieval priority)

### The pipeline

Each turn follows this flow:

1. **commit()** — extract entities, relations, and facts from the conversation
2. **materialize()** — assemble a context string from relevant graph nodes
3. **cycle_status()** — age out node statuses (hot -> warm -> cold)

### Conservative extraction

Acervo only stores explicit statements. If the user didn't say it, it doesn't get recorded. An empty node is better than a node with unverified data.

---

## Bringing your own LLM client

Acervo's `LLMClient` protocol is intentionally simple:

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

Messages use the standard `{"role": "user", "content": "..."}` format. The method returns a plain string. Any OpenAI-compatible client can satisfy this with a thin adapter — see the next section.

### Example: wrapping an existing client

```python
from acervo import LLMClient

class MyAdapter:
    def __init__(self, my_client):
        self._client = my_client

    async def chat(self, messages, *, temperature=0.0, max_tokens=500):
        response = await self._client.completions(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.text

# Use it
memory = Acervo(llm=MyAdapter(my_client), owner="Sandy")
```
