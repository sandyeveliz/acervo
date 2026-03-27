# Traces

Acervo records per-turn trace data for debugging and performance analysis.

## Trace Format

Each turn is appended as a JSON line to `.acervo/traces/{session_id}.jsonl`:

```json
{
  "turn": 1,
  "timestamp": "2026-03-27T10:30:00",
  "session_id": "abc123",
  "acervo_tokens": 245,
  "baseline_tokens": 1200,
  "compression_ratio": 4.9,
  "tokens_without_acervo": 1200,
  "topic_action": "changed",
  "prepare_ms": 150,
  "process_ms": 80,
  "user_message_preview": "What projects am I...",
  "context_preview": "[VERIFIED CONTEXT] Sandy: works at..."
}
```

### Key Fields

| Field | Description |
|-------|-------------|
| `acervo_tokens` | Total tokens sent to LLM (with Acervo context) |
| `baseline_tokens` | Tokens that would be sent without Acervo (full history) |
| `compression_ratio` | `baseline / acervo` — higher is better |
| `tokens_without_acervo` | Raw history token count |
| `topic_action` | S1 topic decision: `same`, `changed`, `subtopic` |
| `prepare_ms` | Time for prepare() (S1+S2+S3) |
| `process_ms` | Time for process() (extraction) |

## CLI

### `acervo trace show`

Show trace data for the latest session.

```bash
acervo trace show                    # latest session, table format
acervo trace show --session abc123   # specific session
```

Output is a table with columns: Turn, Tokens, Baseline, Ratio, Prep(ms).

## REST API

| Endpoint | Description |
|----------|-------------|
| `GET /acervo/traces` | List available trace sessions |
| `GET /acervo/traces/{id}` | All turns as JSON array |
| `GET /acervo/traces/{id}/summary` | Aggregated metrics |

## Debug Dict

The `PrepareResult.debug` dict contains detailed per-turn diagnostics:

```python
{
    "context": {
        "warm_tokens": 180,
        "hot_tokens": 65,
        "total_tokens": 245,
        "warm_budget": 400,
        "chunks_used": 3,
    },
    "chunks": {
        "documents_with_chunks_activated": 1,
        "chunks_retrieved": 2,
        "chunks_total_on_activated_nodes": 5,
        "retrieval_scope": "node_scoped",
        "query_specificity": "specific",
    },
}
```

The `query_specificity` field shows whether the specificity classifier chose `"specific"` (fetch chunks) or `"conceptual"` (summary only).
