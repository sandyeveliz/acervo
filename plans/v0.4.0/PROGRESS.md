# v0.4.0 Progress Tracker

> Updated: 2026-03-31

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Done
- [!] Blocked
- [-] Skipped / Deferred

---

## M0 — Validate Current Indexation

- [x] P1 (TODO App) — 220 nodes, 284 edges, conversation test passed
- [x] P2 (Books) — 250 nodes, 1358 edges, epub indexation + synthesis, conversation test passed
- [-] P3 (Academic Papers) — skipped for now
- [x] P4 (PM Docs) — 102 nodes, 349 edges, conversation test passed
- [x] Chunk inspection findings: prose under-chunked (fixed in M2)

## M0.5 — Chunk Inspection CLI

- [x] `acervo chunks stats` / `list` / `show` / `search` commands
- [x] ChromaDB read-only methods: `get_all_file_chunks()`, `get_chunks_by_ids()`, `get_collection_stats()`
- [x] Identified under-chunking in P2 books (chapters as single 20k+ char chunks)

## M0.6 — Multi-Project Support (Acervo Studio)

- [x] Project registry (`data/projects.json`)
- [x] CRUD endpoints: list, add, remove, select, browse, init, check-path
- [x] ProjectSelector dropdown in sidebar
- [x] ProjectsPage with card grid, browse flow, init/index/curate/synthesize
- [x] SSE streaming for indexation/curation/synthesis progress
- [x] Proxy switch-project endpoint (`/acervo/switch-project`)
- [x] Auto-switch proxy graph when selecting project in Studio

## M0.7 — Context Pipeline Fix & Refinement

### Bugs fixed
- [x] `UnboundLocalError` on `conversation_chunks`/`verified_chunks` — root cause of all "no context injected"
- [x] Proxy graph overwrite after indexing — added `/acervo/reload-graph` endpoint
- [x] S1/S2/S3 trace invisible — debug key mismatch `s1_unified` → `s1_detection`
- [x] S2 chunk content not visible in trace — added text previews + "all chunks" section
- [x] Pipeline logs hidden at debug level — upgraded to info with chunk previews
- [x] Old FileTools (v0.1 leftover) — removed from pipeline, deleted `data/workspace/`
- [x] ProjectSelector not syncing — replaced window events with React Context
- [x] Proxy project mismatch — always loaded AVS-Agents graph instead of active project

### Intent-aware context filtering
- [x] Overview intent: skip vector search entirely
- [x] Overview intent: skip section/symbol nodes in `_find_active_node_ids()`
- [x] Overview S3: filter to `verified_*` chunks only (no vector noise)

### UI state management refactor
- [x] `ProjectContext` (`useProject` hook) — single source of truth for active project
- [x] Chat auto-clears on project switch
- [x] Loading spinner during project switching
- [x] Removed delete button from sidebar ProjectSelector
- [x] Replaced all `window.dispatchEvent("project-changed")` with context
- [x] AlertDialog modals (shadcn) instead of browser `confirm()`

---

## M1 — Ingestion Formats

- [x] `.epub` parser (ebooklib + BeautifulSoup) — chapter extraction with heading structure
- [x] `.txt` parser — paragraph-based sectioning (double blank line splits)
- [x] `.pdf` parser (PyMuPDF/fitz) — page-based sectioning with text extraction
- [-] `.docx` parser — deferred (no test data, low priority)
- [x] Added `.txt`, `.pdf` to `DEFAULT_EXTENSIONS` and `_LANG_MAP`
- [x] `pdf` optional dependency in pyproject.toml

## M2 — Semantic Chunking

- [x] `_chunk_prose()` — splits large sections at paragraph boundaries (~2000 chars/chunk)
- [x] Markdown large sections also split at paragraph boundaries
- [x] epub/pdf/plaintext route through `_chunk_prose()` by default
- [x] Before: P2 books had 47 chunks avg 5,547 chars → After: ~2000 chars each
- [-] Embedding-based cosine similarity boundary detection — deferred (paragraph splitting is good enough for now)

## M3 — Domain Benchmarks

- [x] Project benchmark runner (`tests/integration/run_project_benchmarks.py`)
- [x] 3 benchmark suites: P1 (code), P2 (books), P4 (PM docs)
- [x] 12 questions total — **12/12 pass (100%)**
- [x] JSON report output (`tests/integration/reports/project_benchmarks.json`)
- [x] Console report with pass/fail per question
- [-] HTML report — deferred
- [-] Baseline comparison (Acervo vs raw RAG vs no context) — deferred

## M4 — Fine-tune v2

- [-] Deferred — 0 benchmark failures to train from (100% pass rate)
- [-] Will revisit when harder questions or real usage surface failure modes

## M5 — Blog: v0.3.0 Release

- [-] Deferred — non-code work

## M6 — Indexation Tab Persistence & Auto-load

- [x] Auto-scan file status when switching to Indexation tab
- [x] Loading spinner while scanning (replaces empty "click to scan" state)
- [x] `GET /projects/{id}/operations` endpoint (reads timestamps from graph metadata)
- [x] Display last indexed/curated/synthesized timestamps on Indexation tab
- [x] Initialize button on Overview tab for uninitialized projects

## M7 — Project Config UI

- [x] `GET /projects/{id}/config` endpoint (reads `.acervo/config.toml` as JSON)
- [x] `PUT /projects/{id}/config` endpoint (writes JSON back to TOML)
- [x] "Config" tab in ProjectDetail with editable form
- [x] Sections: Model, Embeddings, Context, Indexing extensions, Proxy
- [x] Save button with feedback
- [x] TOML write-back (with `tomli_w` or fallback simple writer)

## M8 — Session Export & Debug Logger

- [x] Export toolbar in chat page (Copy / .md / .json buttons)
- [x] Copy as Markdown to clipboard
- [x] Download as `.md` file with collapsible trace sections
- [x] Download as `.json` with full messages + trace data
- [-] Backend session persistence to `.acervo/data/sessions/` — deferred

---

## Release Criteria

- [x] `.txt`, `.pdf`, `.epub` ingestion working
- [x] Prose chunking splits at paragraph boundaries (not just headings)
- [x] 3 domain benchmarks: 12/12 pass (100%)
- [x] Multi-project support with UI project switching
- [x] Context pipeline: overview vs specific intent filtering
- [x] Project config editable from UI
- [x] Session export for debugging
- [-] Fine-tuned model v2 (deferred — no failures to train from)
- [-] Blog post (deferred — non-code)

## Commits (release/v0.4.0 branch)

| Hash | Description |
|------|-------------|
| `ede19cc` | feat: M0.6/M0.7 — multi-project support, context pipeline fixes, UI state management |
| `b81008a` | feat: M6 — indexation tab auto-load and operation timestamps |
| `f5833d0` | feat: M7 + M8 — project config UI and session export |
| `d737470` | fix: restore chat layout broken by export toolbar wrapper |

## Commits (acervo main branch)

| Hash | Description |
|------|-------------|
| `b198300` | feat: context pipeline fixes, epub support, synthesis, proxy reload |
| `18b8a8e` | feat: M1 — add .txt and .pdf ingestion support |
| `6b3c4b5` | feat: M2 — paragraph-based prose chunking for epub/pdf/txt/markdown |
| `9e80a6e` | feat: M3 — project benchmark framework with 3 test suites |
