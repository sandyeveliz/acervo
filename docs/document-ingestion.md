# Document Ingestion

Index markdown files into the knowledge graph. Chunks are linked to graph nodes for node-scoped retrieval.

## How It Works

1. **Parse** — structural parser splits `.md` files into sections by heading
2. **Enrich** — semantic enricher generates embeddings for each chunk
3. **Store** — chunks are stored in ChromaDB with their embeddings
4. **Link** — chunk IDs are linked to file and section nodes in the graph

When a user asks a question, Acervo activates relevant graph nodes and retrieves only the chunks linked to those nodes (not a global search across all chunks).

## CLI

### Index a file

```bash
acervo index --path docs/notes.md
```

This runs the full pipeline: parse, enrich, store, link.

### Check indexed documents

```bash
acervo graph show --kind file    # list file nodes
acervo graph show my_notes_md    # show node detail with chunk_ids
```

## REST API

### Upload a document

```
POST /acervo/documents
Content-Type: multipart/form-data

file: notes.md
```

Response:

```json
{
  "document_id": "notes_md",
  "chunk_count": 12,
  "node_id": "notes_md"
}
```

### List documents

```
GET /acervo/documents
```

### Document detail

```
GET /acervo/documents/{id}
```

### Delete a document

```
DELETE /acervo/documents/{id}
```

Removes the file node, section nodes, and all linked chunks from the vector store.

## Specificity Classifier

Not all questions need chunks. Acervo uses a heuristic classifier to decide:

- **Specific queries** (code, numbers, dates, "show me", "exact") — retrieve top 3 chunks from activated nodes
- **Conceptual queries** (explain, why, overview, summarize) — use node summaries only

This keeps conceptual answers concise (~100 tokens of context) while specific questions get detailed chunks (~400 tokens).

## Supported Formats

Currently only `.md` (Markdown) files are supported. The structural parser splits by heading hierarchy.

## Configuration

Embeddings must be configured in `.acervo/config.toml`:

```toml
[acervo.embeddings]
url = "http://localhost:11434"
model = "qwen3-embedding"
```

Vector store data is persisted in `.acervo/data/vectordb/`.
