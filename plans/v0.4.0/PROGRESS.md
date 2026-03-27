# v0.4.0 Progress Tracker

> Updated: 2026-03-27

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Done
- [!] Blocked

---

## M1 — Ingestion formats

### 1.1 Text file ingestion (.txt)
- [ ] Line/paragraph splitter parser
- [ ] `acervo index --path` accepts .txt
- [ ] REST API accepts .txt uploads
- [ ] Tests: .txt ingestion creates nodes with chunk_ids

### 1.2 PDF ingestion (.pdf)
- [ ] PyMuPDF or pdfplumber text extraction
- [ ] Page-aware chunking (preserve page boundaries)
- [ ] `acervo index --path` accepts .pdf
- [ ] REST API accepts .pdf uploads
- [ ] Tests: .pdf ingestion creates nodes with chunk_ids

### 1.3 DOCX ingestion (.docx)
- [ ] python-docx paragraph extraction
- [ ] Heading-based section detection
- [ ] `acervo index --path` accepts .docx
- [ ] REST API accepts .docx uploads
- [ ] Tests: .docx ingestion creates nodes with chunk_ids

### 1.4 Unified format detection
- [ ] Auto-detect format by extension in indexer
- [ ] Unified error messages for unsupported formats
- [ ] CLI help updated with supported formats

---

## M2 — Semantic chunking

- [ ] Embed consecutive paragraphs
- [ ] Boundary detection (cosine similarity drop)
- [ ] Fallback to fixed size when no clear boundaries
- [ ] Hierarchical chunks (section -> subsection -> semantic chunk)
- [ ] Max chunk size configurable (default 1024 tokens)
- [ ] Tests: semantic chunking vs fixed-size comparison

---

## M3 — 4 test domains with benchmarks

### 3.1 Domain: Code
- [ ] Small project scenario (~20 files)
- [ ] Large project scenario (~200 files)
- [ ] Conversation queries against indexed code
- [ ] Metrics: node coverage, import accuracy, query recall

### 3.2 Domain: Literature
- [ ] Short story scenario (~5,000 words)
- [ ] Novel scenario (~50,000+ words)
- [ ] Conversation queries against indexed literature
- [ ] Metrics: character coverage, event recall, context tokens

### 3.3 Domain: Academic
- [ ] Short paper scenario (~8 pages PDF)
- [ ] Long paper/thesis scenario (~50 pages)
- [ ] Conversation queries against indexed papers
- [ ] Metrics: concept extraction, methodology accuracy

### 3.4 Domain: Multi-project
- [ ] 3 simultaneous projects (frontend, backend, mobile)
- [ ] Project isolation test (no cross-contamination)
- [ ] Shared UNIVERSAL node test (React, PostgreSQL)
- [ ] Cross-project query test

### 3.5 Benchmark CLI + reports
- [ ] `acervo benchmark --domain` CLI command
- [ ] HTML indexation scorecard
- [ ] Query recall chart
- [ ] Token efficiency comparison (Acervo vs global RAG)
- [ ] Versioned report archive

---

## M4 — Fine-tune v2

- [ ] Failure mode analysis per domain
- [ ] 200+ training examples from failure modes
- [ ] Chunk decision training (`summary_only` | `with_chunks`)
- [ ] Indexation-aware training data
- [ ] Retrain model
- [ ] Benchmark: accuracy 85% -> 92%+

---

## M5 — Blog post v0.3.0 + content

- [ ] Blog post with 5 charts + narrative
- [ ] LinkedIn post (hero scissors + stat)
- [ ] X thread (technical highlights)
- [ ] Consistent footer across all posts

---

## Release Criteria
- [ ] `.txt`, `.pdf`, `.docx` ingestion all working
- [ ] Semantic chunking outperforms fixed-size on dense docs
- [ ] 4 domain benchmarks with HTML reports
- [ ] Indexation scorecard equivalent to conversation benchmarks
- [ ] Fine-tuned model v2 extraction accuracy >= 92%
- [ ] Blog post published
