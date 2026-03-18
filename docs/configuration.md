# Configuration

Acervo has two modes of operation, each with its own configuration approach.

---

## SDK directo (import Python)

When you use Acervo as a library inside your own agent, **you** control the LLM. Acervo does not read any environment variables related to models — the caller configures and passes its own client.

### `Acervo()` constructor parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `LLMClient` | required | LLM client for extraction tasks. Must implement the `LLMClient` protocol (async `chat()` method). |
| `owner` | `str` | `""` | Identifier for the user whose memory this is. Used for future multi-user support. |
| `persist_path` | `str \| Path` | `"data/graph"` | Directory where graph data (nodes.json, edges.json) is persisted. |

### `LLMClient` protocol

Any object that implements this async method:

```python
async def chat(
    self,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 500,
) -> str
```

Messages use the standard `{"role": "...", "content": "..."}` format. Returns the response content as a plain string.

### Built-in `OpenAIClient`

Acervo includes a zero-dependency client for any OpenAI-compatible API:

```python
from acervo import Acervo, OpenAIClient

llm = OpenAIClient(
    base_url="http://localhost:1234/v1",
    model="qwen2.5-3b-instruct",
    api_key="lm-studio",
)

memory = Acervo(llm=llm, owner="Sandy")
```

Works with LM Studio, Ollama, OpenAI, or any compatible endpoint.

### Example: custom adapter

When your app already has an LLM client, wrap it:

```python
class MyAdapter:
    def __init__(self, router):
        self._router = router

    async def chat(self, messages, *, temperature=0.0, max_tokens=500):
        response = await self._router.chat(messages, temperature=temperature)
        return response.content

memory = Acervo(llm=MyAdapter(my_router), owner="Sandy")
```

!!! note "Acervo never reads LLM env vars in SDK mode"
    When using the SDK directly, all LLM configuration is the responsibility of the caller. The environment variables documented below are **only** for standalone mode (`acervo serve` / `acervo mcp` / `Acervo.from_env()`).

---

## Standalone mode (`acervo serve` / `acervo mcp`)

When Acervo runs as an independent server, it needs its own LLM configuration. This is read from environment variables (or a `.env` file in the working directory).

### Environment variables

#### LLM models

| Variable | Default | Description |
|----------|---------|-------------|
| `ACERVO_LIGHT_MODEL_URL` | `http://localhost:1234/v1` | Base URL for the extraction model (OpenAI-compatible endpoint). |
| `ACERVO_LIGHT_MODEL` | `qwen2.5-3b-instruct` | Model name for extraction and background tasks. A small, fast model is ideal. |
| `ACERVO_LIGHT_API_KEY` | `lm-studio` | API key for the light model endpoint. |
| `ACERVO_MAIN_MODEL_URL` | `http://localhost:1234/v1` | Base URL for the main reasoning model (reserved for future use). |
| `ACERVO_MAIN_MODEL` | `unsloth/qwen3.5-9b` | Model name for reasoning and context assembly (reserved for future use). |
| `ACERVO_MAIN_API_KEY` | `lm-studio` | API key for the main model endpoint (reserved for future use). |

!!! info "Light vs Main model"
    The **light model** is used for extraction (converting text into entities/facts). It should be small and fast — 3B parameters is enough.

    The **main model** is reserved for future features (query planning, reasoning). Currently unused in standalone mode.

#### Storage and logging

| Variable | Default | Description |
|----------|---------|-------------|
| `ACERVO_OWNER` | `"default"` | Default owner identifier for memory isolation. |
| `ACERVO_DATA_DIR` | `"./data"` | Root directory for graph persistence. |
| `ACERVO_LOG_LEVEL` | `"INFO"` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

### Example `.env` file

```bash
# Modelo liviano — extraccion, topic detection, summarizer
ACERVO_LIGHT_MODEL_URL=http://localhost:1234/v1
ACERVO_LIGHT_MODEL=qwen2.5-3b-instruct
ACERVO_LIGHT_API_KEY=lm-studio

# Modelo principal — razonamiento (reservado para uso futuro)
ACERVO_MAIN_MODEL_URL=http://localhost:1234/v1
ACERVO_MAIN_MODEL=unsloth/qwen3.5-9b
ACERVO_MAIN_API_KEY=lm-studio

# Memoria
ACERVO_OWNER=Sandy
ACERVO_DATA_DIR=./data
ACERVO_LOG_LEVEL=INFO
```

### Using `Acervo.from_env()`

The `from_env()` classmethod reads the `.env` file and creates a fully configured instance:

```python
from acervo import Acervo

# Reads .env from current directory
memory = Acervo.from_env(owner="Sandy")

# Or specify a custom .env path
memory = Acervo.from_env(env_file="/path/to/.env", owner="Sandy")

# Or skip .env and rely on existing environment variables
memory = Acervo.from_env(env_file=None, owner="Sandy")
```

---

## Host app configuration (e.g., AVS-Agents)

When Acervo is used as a dependency inside another application, the host app has its own configuration for its own LLM needs. Acervo does not interfere with this.

For example, AVS-Agents has its own `.env` with:

```bash
# AVS-Agents .env — these are for the TUI app, NOT for Acervo
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_MODEL=unsloth/qwen3.5-9b        # main chat model
LMSTUDIO_UTILITY_MODEL=qwen2.5-3b-instruct  # topic detection, planning
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=qwen3-embedding
```

The host app creates an adapter and passes it to Acervo:

```python
from acervo import Acervo
from providers.acervo_adapter import ModelRouterAdapter

adapter = ModelRouterAdapter(my_router)
memory = Acervo(llm=adapter, owner="Sandy")
```

Acervo uses whatever LLM the host gives it. It never reads `LMSTUDIO_*` or `OLLAMA_*` variables — those belong to the host app.
