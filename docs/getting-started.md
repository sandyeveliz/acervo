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

## Prerequisites

Acervo needs two local model servers:

1. **LM Studio** — runs the fine-tuned `acervo-extractor-qwen3.5-9b` model for chat and extraction
2. **Ollama** — runs `qwen3-embedding` for embeddings

Install both, then pull the embedding model:

```bash
ollama pull qwen3-embedding
```

Load `acervo-extractor-qwen3.5-9b` in LM Studio (or any OpenAI-compatible model).

---

## Quickstart: Proxy Mode

The fastest way to use Acervo — run it as a transparent proxy between your AI client and the LLM.

### 1. Initialize a project

```bash
cd your-project
acervo init
```

This creates `.acervo/` with `config.toml`. Edit the config:

```toml
[acervo]
owner = "Sandy"

[acervo.model]
name = "acervo-extractor-qwen3.5-9b"
url = "http://localhost:1234/v1"

[acervo.embeddings]
url = "http://localhost:11434"
model = "qwen3-embedding"

[acervo.proxy]
port = 9470
target = "http://localhost:1234/v1"
```

### 2. Start the proxy

```bash
acervo up
```

This starts the proxy on port 9470. Point your AI client (Cursor, Continue, etc.) to `http://localhost:9470/v1` instead of the direct LLM endpoint.

For development with all services:

```bash
acervo up --dev
```

This starts the proxy, Acervo Studio web UI, and auto-detects Ollama/LM Studio.

### 3. Use normally

Every conversation turn now flows through Acervo:
- **Before** the LLM call: Acervo injects relevant context from the knowledge graph
- **After** the LLM responds: Acervo extracts entities and facts into the graph

Check the graph:

```bash
acervo status          # overview
acervo graph show      # list all nodes
acervo graph search "React"  # search nodes
```

---

## Quickstart: SDK

Import Acervo into your Python agent for full control.

```python
from acervo import Acervo, OpenAIClient, OllamaEmbedder

# Create clients
llm = OpenAIClient(
    base_url="http://localhost:1234/v1",
    model="acervo-extractor-qwen3.5-9b",
    api_key="lm-studio",
)
embedder = OllamaEmbedder(
    base_url="http://localhost:11434",
    model="qwen3-embedding",
)

# Create the memory instance
memory = Acervo(llm=llm, owner="Sandy", embedder=embedder)

# Before each LLM call — build context from the graph
prep = await memory.prepare(
    user_text="What projects am I working on?",
    history=[],  # conversation history
)
# prep.context_stack is ready for the LLM

# After the LLM responds — extract knowledge
result = await memory.process(
    user_text="I work at Altovallestudio, we have 4 projects",
    assistant_text="Tell me more about the projects.",
)
```

### Bringing your own LLM client

Acervo's `LLMClient` protocol is minimal:

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

Any OpenAI-compatible client can satisfy this with a thin adapter.

---

## Logging

Control log verbosity with `--log-level`:

```bash
acervo up --log-level info      # one line per turn
acervo up --log-level debug     # entities, topics, timing
acervo up --log-level trace     # full prompts and responses
acervo up --no-color            # disable ANSI colors
acervo up -v                    # shorthand for --log-level debug
```

---

## Core Concepts

### The Knowledge Graph

Acervo stores knowledge as a graph of **nodes** (entities) and **edges** (relations). Each node has:

- **label** — the entity name ("Sandy", "Cipolletti", "React")
- **type** — entity type ("Person", "Place", "Technology")
- **layer** — UNIVERSAL (world knowledge) or PERSONAL (user-specific)
- **facts** — concrete statements about this entity
- **status** — hot / warm / cold (determines retrieval priority)
- **chunk_ids** — links to document chunks in the vector store

### The Pipeline

Each turn follows a 3-stage pipeline:

1. **S1 — Topic + Extraction**: Detects topic changes and extracts entities/facts (single LLM call)
2. **S2 — Gather**: Activates relevant graph nodes, retrieves linked chunks via specificity classifier
3. **S3 — Context Assembly**: Budget-aware chunk selection, builds the context stack

### Context Stack

What the LLM actually receives:

```
[system]     Fixed prompt (KV cached)
[user]       [VERIFIED CONTEXT] warm context from graph [END CONTEXT]
[assistant]  "Understood."
[user/asst]  Hot layer: last 2 turn pairs (sliding window)
[user]       Current user message
```

### Conservative Extraction

Acervo only stores explicit statements. If the user didn't say it, it doesn't get recorded. An empty node is better than a node with unverified data.

---

## Next Steps

- [Configuration](configuration.md) — full config reference
- [Graph CLI](graph-cli.md) — inspect and edit the knowledge graph
- [Traces](traces.md) — per-turn trace data for debugging
- [Document Ingestion](document-ingestion.md) — index markdown files into the graph
