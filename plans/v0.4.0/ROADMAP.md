# v0.4.0 — "Indexation works for real"

> v0.3.0 indexes `.md`. v0.4.0 proves indexation across 4 real domains
> and produces indexation benchmarks equivalent to the conversation ones.

## Goal

Same level of evidence for document indexation that we already have for
conversation memory. Scorecard, scissors chart, accuracy metrics — but
for indexed documents.

---

## M1 — Ingestion formats

Add ingestion support for the formats needed by the 4 test domains:

| Format | Needed for | Parser |
|--------|-----------|--------|
| `.txt` | Literature, books | Line/paragraph splitter |
| `.pdf` | Academic papers, manuals | PyMuPDF / pdfplumber (text extraction) |
| `.docx` | Business docs, specs | python-docx (paragraph extraction) |
| `.md` | Already works (v0.3.0) | Heading parser |

- `acervo index --path` accepts all formats
- REST API accepts uploads of all formats
- Automatic format detection by extension
- Ingestion tests per format

---

## M2 — Semantic chunking

**Problem:** chunking by heading + fixed size works for `.md` but fails
on dense documents (papers, books).

**Implementation:**

```
v0.3.0 (current):
  Split by heading -> paragraphs -> fixed size (512 tokens with overlap)

v0.4.0:
  Split by heading -> embed each paragraph -> detect boundaries
  where similarity between consecutive paragraphs drops -> cut there
```

- Consecutive paragraph embeddings
- Boundary detection: if cosine similarity between paragraph N and N+1
  falls below threshold -> new chunk
- Fallback to fixed size if no clear boundaries
- Hierarchical chunks for long documents:
  section -> subsection -> semantic chunk
- Maximum 1024 tokens per chunk (configurable)

---

## M3 — 4 test domains with benchmarks

Each domain has a test scenario with indexation metrics:

### Domain 1 — Code (small + large project)

```
Small project: ~20 Python files, a REST microservice
  Metrics: nodes created, import relations detected,
  function/class coverage, dependency accuracy

Large project: ~200 files, monorepo with multiple packages
  Metrics: indexation time, total nodes, cross-package
  relation accuracy, query recall
```

- Conversation: "How does the auth module work?" -> brings correct
  auth code chunks?
- Conversation: "What calls the payment service?" -> detects
  dependency chain?

### Domain 2 — Literature

```
Short story: ~5,000 words (e.g., a Borges or Cortazar story)
  Metrics: characters extracted, events detected, narrative arc

Novel/book: ~50,000-100,000 words
  Metrics: character coverage, inter-character relations,
  recall of important events, context tokens per question
```

- Conversation: "What's the relationship between X and Y?" -> graph
  has the correct relation from indexation?
- Conversation: "What happens in chapter 3?" -> brings relevant chunks
  without the whole book?

### Domain 3 — Academic/papers

```
Short paper: ~8 pages, an NLP or ML paper (PDF)
  Metrics: concepts extracted, methodology detected,
  results identified, cross-references

Long paper / thesis: ~50 pages
  Metrics: section hierarchy, terminology accuracy,
  ability to answer methodology questions
```

- Conversation: "What dataset did they use?" -> extracts from paper
  or hallucinates?
- Conversation: "Explain the difference between their approach and
  BERT" -> combines paper knowledge + UNIVERSAL nodes?

### Domain 4 — Multi-project (PM)

```
3 projects indexed simultaneously:
  - Frontend React app
  - Backend API Python
  - Mobile app React Native

  Metrics: project isolation (project A nodes don't mix
  with B), shared tech nodes (React, PostgreSQL correctly
  shared as UNIVERSAL), cross-project queries
  ("which projects use React?")
```

- Conversation: "What's the status of the mobile project?" ->
  only mobile context, not frontend?
- Conversation: "Compare the auth implementation across projects" ->
  brings chunks from all 3 correctly?

### Benchmark deliverables

```bash
acervo benchmark --domain code --project ./my-small-project
acervo benchmark --domain literature --file ./borges-aleph.txt
acervo benchmark --domain academic --file ./transformer-paper.pdf
acervo benchmark --domain multi-project --paths ./frontend,./backend,./mobile
```

HTML reports equivalent to v0.3.0:
- Indexation scorecard (nodes, relations, accuracy per domain)
- Query recall chart (% of questions answered correctly)
- Token efficiency (chunk context vs full document)
- Comparison: Acervo node-scoped vs global RAG on the same docs

---

## M4 — Fine-tune v2 (indexation data)

The 4 domains from M3 will produce failure modes. Those are the dataset.

- Identify failure modes per domain:
  - Code: confuses functions with variables? Loses imports?
  - Literature: creates phantom characters? Confuses narrator with character?
  - Academic: confuses similar concepts? Loses references?
  - Multi-project: contaminates nodes between projects?
- 200+ new training examples from failure modes
- Train model for chunk decisions:
  `"retrieval": "summary_only" | "with_chunks"` as S1 output
- Train with indexation data (model sees nodes with chunk_ids)
- Target: extraction accuracy 85% -> 92%+
- Comparative benchmark before/after retrain

---

## M5 — Blog post v0.3.0 + content

This must ship WITH the v0.3.0 release, not after:

- Blog post: the 5 charts (scissors, scorecard, anatomy,
  savings by phase, three-scenario scissors) with narrative
- LinkedIn: hero scissors + one-line stat + link
- X thread: technical highlights (node-scoped retrieval, specificity
  classifier, 0 phantom entities)
- Consistent footer: all open source, Apache 2.0, 4 repos
