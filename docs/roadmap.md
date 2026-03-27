# Roadmap

Current version: **v0.3.0** ([PyPI](https://pypi.org/project/acervo/) · [Changelog](https://github.com/sandyeveliz/acervo/blob/main/CHANGELOG.md))

> v0.3.0 proved that conversation memory works: 76% token savings, 94% context hits.
> Now we need to prove that indexation works just as well, make Acervo usable from
> any tool, and then scale.
>
> Small versions, one theme per release, each with publishable benchmarks.

---

## Overview

```
v0.3.0  ========  DONE    Conversation memory works (benchmarks, CLI, .md ingestion)
v0.4.0  ========  NEXT    Indexation works for real (4 domains, formats, fine-tune v2)
v0.5.0  ========          Usable from anywhere (MCP server, TS SDK, integrations)
v0.6.0  ========          Production-ready (Docker, metrics, progressive retrieval)
v0.7.0  ========          Ecosystem (multi-tenant, packs, Studio v2)
```

| Version | What it proves | Evidence produced |
|---------|---------------|-------------------|
| v0.3.0 | Conversation memory | 76% savings, scissors chart |
| v0.4.0 | Document indexation | Indexation scorecard, query recall per domain |
| v0.5.0 | Works in any tool | MCP server downloads, integration demos |
| v0.6.0 | Deploy in production | Docker one-liner, live metrics |
| v0.7.0 | Scales to teams | Multi-user demos, knowledge pack catalog |

---

## What works today (v0.3.0)

- Knowledge graph with JSON persistence
- UNIVERSAL / PERSONAL two-layer architecture
- `prepare()` / `process()` context proxy API
- S1 Unified extraction (topic + entities in one call)
- S1.5 Async graph curation (merges, corrections)
- Fine-tuned extraction model (Qwen 3.5 9B, single model for chat + extraction)
- Topic-based context layers (HOT/WARM/COLD)
- History windowing (constant token usage)
- `acervo serve` / `acervo up` — proxy + dev stack
- `acervo index` — structural + semantic codebase indexing
- Document ingestion with graph-linked chunks (`.md`)
- Node-scoped chunk retrieval with specificity classifier
- Graph inspection CLI (`acervo graph show/search/delete/merge/repair`)
- Configurable logging (`--log-level trace|debug|info`)
- Reproducible benchmarks (360 turns, 6 scenarios, HTML reports)

---

## v0.4.0 — "Indexation works for real"

**Goal:** Same level of evidence for document indexation that we already have for conversation memory.

### New ingestion formats

| Format | Parser | Use case |
|--------|--------|----------|
| `.txt` | Line/paragraph splitter | Literature, books |
| `.pdf` | PyMuPDF / pdfplumber | Academic papers, manuals |
| `.docx` | python-docx | Business docs, specs |
| `.md` | Already works | Technical docs |

### Semantic chunking

Replace fixed-size chunking with embedding-based boundary detection. Consecutive paragraphs are embedded; chunk breaks happen where cosine similarity drops. Hierarchical chunks for long documents (section -> subsection -> semantic chunk).

### 4 domain benchmarks

| Domain | Test material | Key metrics |
|--------|--------------|-------------|
| Code | Small (~20 files) + large (~200 files) project | Node coverage, import accuracy, query recall |
| Literature | Short story + novel | Character coverage, event recall, context tokens |
| Academic | Short paper + thesis (PDF) | Concept extraction, methodology accuracy |
| Multi-project | 3 simultaneous projects | Project isolation, shared UNIVERSAL nodes |

Each domain produces HTML benchmark reports: indexation scorecard, query recall chart, token efficiency comparison (Acervo node-scoped vs global RAG).

### Fine-tune v2

Training data from M3 failure modes. Target: extraction accuracy 85% -> 92%+. New training signal: chunk retrieval decisions (`summary_only` | `with_chunks`) as S1 output.

---

## v0.5.0 — "Usable from anywhere"

### MCP Server

The most important integration. An Acervo MCP server exposing tools (`acervo_prepare`, `acervo_process`, `acervo_index`, `acervo_search`, `acervo_status`) and resources (`acervo://graph`, `acervo://nodes/{id}`, `acervo://traces/latest`).

Compatible with Claude Desktop, Cursor, Windsurf, Continue.dev. Install Acervo, add the MCP server to your config, and you have persistent memory in any tool.

### TypeScript SDK

`npm install acervo-client` — REST API wrapper with full types. Examples with Vercel AI SDK and Next.js.

### Framework integrations

- `AcervoMemory` — LangChain ConversationBufferMemory drop-in
- `AcervoRetriever` — LlamaIndex retriever drop-in

Same interface, but with knowledge graph compression instead of raw chunks.

---

## v0.6.0 — "Production-ready"

### Docker Compose

`docker compose up` -> Acervo proxy + Ollama + model + ChromaDB. GPU passthrough, persistent volumes. "Try Acervo in 30 seconds."

### Progressive retrieval

Hot layer insufficient -> automatically escalate to warm, then cold. Detection via topic confidence and LLM "no info" signals. Configurable budget per layer.

### Runtime metrics

`GET /acervo/metrics` — Prometheus-compatible endpoint. Tokens saved, compression ratio, latency, hit rate. Minimal dashboard in Acervo Studio.

### Advanced error recovery

Automatic graph backup before destructive operations. `acervo graph rollback` to revert. LLM down -> cache last known graph context.

---

## v0.7.0 — "Ecosystem"

### Multi-tenant graphs

UNIVERSAL layer shared across users, PERSONAL scoped per API key. Automatic UNIVERSAL node merging.

### Knowledge packs

Pre-built domain graphs. `acervo pack install javascript`. Export/import graphs. GitHub-based registry.

### Acervo Studio v2

Real-time graph visualization, visual node/relation editor, session comparator, metrics dashboard.

### Model v3+

Extraction accuracy >95%. Multi-language: PT, FR, DE. Document reference extraction. Benchmark vs GPT-4o.

---

## What's NOT planned before v1.0

| Feature | Reason |
|---------|--------|
| Cloud-hosted version | Needs infra + billing |
| Voice/audio ingestion | Text-first, niche |
| Image/multimodal extraction | Requires multimodal model |
| Graph database backend (Neo4j) | JSON + ChromaDB scales enough |
| Real-time collaboration | Multi-tenant first |

---

## Want to help?

Check out [Contributing](contributing.md) or open an [issue](https://github.com/sandyeveliz/acervo/issues) to discuss ideas.
