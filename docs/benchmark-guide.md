# Acervo Benchmark Guide

This document describes every test in Acervo's integration benchmark suite:
what it measures, how it works, and how to interpret the results.

---

## Overview

Acervo replaces traditional conversation history with a **persistent knowledge graph**.
Instead of accumulating every message in the context window (which grows linearly and
eventually overflows), Acervo builds a graph of entities, facts, and relationships
from each conversation turn, then injects only the *relevant* subset as context
for the next turn.

The benchmark suite answers two questions:

1. **Does it work?** Can Acervo answer questions that require project knowledge,
   recall user-stated facts, and avoid hallucinating information that doesn't exist?

2. **Is it efficient?** Does Acervo use fewer tokens than alternative approaches
   (raw history accumulation, or an agent with file-reading tools)?

---

## Test Infrastructure

### Fixture Projects

Three real-world projects are used as test data, each representing a different domain:

| Project | Domain | Content | Files |
|---------|--------|---------|-------|
| **P1 — TODO App** | Source code | Full-stack TypeScript app: Express backend, React frontend, SQLite, JWT auth | 31 files (.ts, .tsx, .md, .json, .html, .css) |
| **P2 — Literature** | Prose / epub | "The Adventures of Sherlock Holmes" by Arthur Conan Doyle (1892, public domain, Project Gutenberg) | 1 epub file, 12 stories |
| **P3 — PM Docs** | Project management | Roadmap, sprint plans, issue trackers, architecture decision records | 11 markdown files |

These projects live in `tests/fixtures/` and contain **only source files** (no generated data).

### Auto-Indexing Pipeline

Before tests run, `conftest.py` automatically processes each fixture through
Acervo's full pipeline:

```
init -> index -> curate -> synthesize
```

- **Index**: Parses files, creates graph nodes (file, section, symbol, folder),
  generates embeddings, stores chunks in vector DB.
- **Curate**: LLM analyzes files in batches, extracts entities (people, technologies,
  concepts) and relationships between them.
- **Synthesize**: LLM generates a project overview node that summarizes the entire project.

This happens once. Subsequent runs reuse the existing `.acervo/` state.
Delete `.acervo/` inside a fixture to force re-indexing.

**Requirements**: LM Studio running `acervo-extractor-qwen3.5-9b` and
Ollama running `qwen3-embedding`.

---

## Layer 1: Pipeline Validation (`test_pipeline_validation.py`)

**26 tests** that inspect the graph state produced by index/curate/synthesize.
These tests do NOT call the LLM at runtime -- they read the existing graph and
check structural properties. They run in seconds.

### What it tests per project

#### Index validation
- **File nodes created**: Did the indexer find all source files?
  (P1: >=20 files, P2: >=1 epub, P3: >=5 markdown files)
- **Section nodes created**: Did the parser extract headings/chapters?
  (P2: >=10 story sections, P3: >=30 markdown heading sections)
- **Symbol nodes created**: Did code parsing extract functions and classes?
  (P1: >=50 symbols from TypeScript)
- **Folder nodes created**: Is the directory structure captured?
  (P1: >=5 folders)
- **Chunk IDs linked**: Are text chunks properly associated with nodes?
  (P1: >=100 chunk links, P2: >=50 chunk links)
- **Edges exist**: Are parent-child, import, and contains relationships stored?
  (P1: >=100 edges)

#### Curate validation
- **Entities extracted**: Did the LLM identify real entities?
  (P1: >=3 tech entities like React/Express/SQLite)
  (P2: >=2 character entities like Holmes/Watson)
  (P3: >=1 entity from PM docs)
- **Entity types valid**: Are types from the ontology?
  (person, organization, project, technology, place, event, document, concept, symbol)
- **No phantom entities**: Every entity label must appear in at least one source file.
  Phantoms = hallucinated entities the LLM invented.
- **Entities have relations**: Entities should be connected to file nodes, not orphaned.
- **Character entities** (P2): Does curation extract Sherlock, Watson, Adler, etc?
- **Location entities** (P2): Does curation find Baker Street, London?

#### Synthesize validation
- **Synthesis node exists**: Is there a `synthesis:*overview*` node?
- **Synthesis mentions stack** (P1): Does the overview mention React, Express, etc?
- **Synthesis mentions Sherlock** (P2): Does the overview mention Holmes/Conan Doyle?
- **Module summaries** (P3): Are there module-level synthesis nodes?

#### Diagnostic report
The final test generates a cross-project summary printed to console and saved as
`reports/pipeline_diagnostic.json` and `reports/pipeline_diagnostic.md`.

---

## Layer 2: 5-Category Benchmark (`test_benchmarks.py`)

**55 conversation turns** across 3 projects, organized into 5 capability categories.
Each turn calls `prepare()` (the context injection pipeline) and `process()`
(the memory extraction pipeline), then evaluates component-level checks.

### How a turn works

```
1. User message sent to prepare(user_msg, history)
2. S1 (Intent Detection): Classify as overview/specific/chat
3. S2 (Node Activation): Select relevant graph nodes
4. S3 (Context Assembly): Build warm context within token budget
5. Evaluate per-turn checks against S1/S2/S3 outputs
6. Call process(user_msg, assistant_sim) to update memory graph
7. Record pass/fail for each check
```

The `assistant_sim` is a simulated assistant response (not from an LLM).
This isolates the test to Acervo's context pipeline, not the chat model's quality.

### The 5 Categories

#### RESOLVE (14 turns)

**"Can Acervo enable answers that are impossible without project knowledge?"**

These questions have NO correct answer without access to the project data.
A stateless LLM would have to refuse or hallucinate.

| Turn | Project | Question | Why it's impossible without context |
|------|---------|----------|-------------------------------------|
| P1-1 | TODO App | "How many files does this project have?" | Requires file listing |
| P1-2 | TODO App | "What technologies does this project use?" | Must read package.json/imports |
| P1-5 | TODO App | "What routes does the API define?" | Must read route files |
| P1-13 | TODO App | "How does the frontend communicate with the backend?" | Must trace API client code |
| P1-16 | TODO App | "What data models are defined?" | Must read model files |
| P1-18 | TODO App | "What error handling does the API have?" | Must read middleware code |
| P2-1 | Literature | "What is this book about?" | Must read the epub |
| P2-2 | Literature | "How many stories does it contain?" | Must parse table of contents |
| P2-12 | Literature | "Where does Holmes live?" | Must find address in text |
| P3-1 | PM Docs | "What project is documented here?" | Must read project files |
| P3-2 | PM Docs | "What documents are available?" | Must list document files |
| P3-7 | PM Docs | "What milestones are defined?" | Must read roadmap |
| P3-13 | PM Docs | "Who are the developers working on this?" | Must find names in docs |
| P3-16 | PM Docs | "Which issues are related to the current sprint?" | Must cross-reference sprint + issues |

**Pass criteria**: S3 warm context contains the information needed to answer
(verified by `context_contains` / `context_contains_any` checks).

**Effectiveness check**: `stateless_can_answer: false` marks these as turns
where Acervo is the difference between answering and not answering.

#### GROUND (12 turns)

**"Does Acervo prevent hallucination by grounding answers in verified data?"**

These questions ask about things that may or may not exist in the project.
Acervo should provide context that confirms or denies -- and critically,
should NOT inject context about things that aren't there.

| Turn | Project | Question | What we check |
|------|---------|----------|---------------|
| P1-14 | TODO App | "Does this project use GraphQL?" | Context must NOT contain "graphql" |
| P1-15 | TODO App | "Is there a Python backend?" | Context must NOT contain "python", "django", "flask" |
| P1-20 | TODO App | "This is React and Express with SQLite, right?" | Context MUST contain "react", "express", "sqlite"; must NOT contain "mongodb", "nuxt" |
| P2-3 | Literature | "Who is Sherlock Holmes?" | Context must contain "holmes", "detective", or "baker street" |
| P2-6 | Literature | "Who is Irene Adler?" | Context must contain "adler" or "irene" |
| P2-11 | Literature | "Does Professor Moriarty appear in these stories?" | Moriarty is barely in this collection -- should not hallucinate his presence |
| P2-15 | Literature | "This is by Arthur Conan Doyle, right?" | Context must contain "conan doyle" |
| P3-4 | PM Docs | "What's the tech stack?" | Context must contain "express", "sqlite", "jwt"; must NOT contain "django" |
| P3-12 | PM Docs | "Why was SQLite chosen over PostgreSQL?" | Context must contain "sqlite" and "decision" |
| P3-15 | PM Docs | "Is there a CI/CD pipeline set up?" | Correct answer is "no" -- context should not hallucinate one |
| P3-19 | PM Docs | "Is Ron Weasley part of this project?" | Context must NOT contain "weasley" -- tests negative grounding |

**Pass criteria**: `context_contains` for positive grounding,
`context_not_contains` for negative grounding (absence of noise).

#### RECALL (6 turns)

**"Can Acervo remember user-stated facts across turns?"**

Two phases: the user *states* a fact (S1.5 extracts it into the graph),
then later *asks* about it (S3 should retrieve it).

| Turn | Project | Phase | Message | What we check |
|------|---------|-------|---------|---------------|
| P1-9 | TODO App | Store | "The lead developer is Alice and she started in January" | S1.5 extracts "alice" entity into graph |
| P1-12 | TODO App | Recall | "Who is the lead developer?" | S3 context contains "alice" |
| P2-10 | Literature | Store | "Holmes's deduction method is similar to scientific hypothesis testing" | S1.5 extracts fact about "deduction" |
| P2-13 | Literature | Recall | "What did I say about Holmes's method earlier?" | S3 context contains "deduction", "scientific" |
| P3-8 | PM Docs | Store | "The client wants the MVP ready by end of Q2" | S1.5 extracts "MVP" entity |
| P3-14 | PM Docs | Recall | "What was the deadline I mentioned?" | S3 context contains "q2", "mvp" |

**Pass criteria**: Entity extraction creates the node (store phase),
and later context retrieval includes the fact (recall phase).

#### FOCUS (14 turns)

**"Does Acervo keep context small and relevant?"**

These turns test that Acervo doesn't over-activate nodes or waste the token budget.
Chat messages like "thanks" should produce almost no context.
Specific questions should activate only the relevant files.

| Turn | Project | Question | Budget check |
|------|---------|----------|-------------|
| P1-3 | TODO App | "How does authentication work?" | 100-600 tokens, must contain "auth", must NOT contain "todo.controller" |
| P1-4 | TODO App | "What database does it use?" | <=600 tokens, must contain "sqlite" |
| P1-6 | TODO App | "Interesting, well-structured project" | <=150 tokens (chat -- minimal context) |
| P1-8 | TODO App | "What React hooks does the app use?" | <=600 tokens, <=8 nodes |
| P1-17 | TODO App | "High-level summary of the whole project" | <=300 tokens (synthesis only, no symbols) |
| P1-19 | TODO App | "Ok, I think I understand the project now" | <=150 tokens (chat) |
| P2-4 | Literature | "Tell me about Dr. Watson" | <=600 tokens, <=10 nodes |
| P2-5 | Literature | "What happens in A Scandal in Bohemia?" | 100-600 tokens |
| P2-8 | Literature | "These are great detective stories" | <=150 tokens (chat) |
| P3-3 | PM Docs | "What issues are currently open?" | 100-600 tokens, <=8 nodes, only issue files |
| P3-5 | PM Docs | "What's in the current sprint?" | <=600 tokens, <=6 nodes, sprint files |
| P3-9 | PM Docs | "Thanks for the overview" | <=150 tokens (chat) |
| P3-11 | PM Docs | "What's the auth issue specifically?" | <=600 tokens, <=6 nodes |
| P3-17 | PM Docs | "What progress has been made?" | <=600 tokens, progress files |
| P3-20 | PM Docs | "Give me a final summary" | <=300 tokens, synthesis only |

**Pass criteria**: Token budget within range, node count within limits,
and correct S2 file activation (activated + not-activated checks).

#### ADAPT (8 turns)

**"Can Acervo switch context cleanly when the user changes topic?"**

The user shifts focus mid-conversation. Acervo should activate nodes for the
NEW topic and stop injecting context from the OLD topic.

| Turn | Project | Shift | What we check |
|------|---------|-------|---------------|
| P1-7 | TODO App | Backend -> Frontend | Must activate "component"/"frontend"; must NOT activate "controller", "middleware" |
| P1-10 | TODO App | Frontend -> Config | Must activate "config" files |
| P1-11 | TODO App | Config -> Auth (return) | Must activate "auth", "middleware" |
| P2-7 | Literature | Bohemia -> Red-Headed League | Must contain "red-headed"/"league"; must NOT contain "adler"/"bohemia" |
| P2-9 | Literature | Red-Headed -> Irene Adler (return) | Must contain "adler"/"irene" |
| P2-14 | Literature | Specific story -> Overall themes | <=400 tokens (overview mode) |
| P3-6 | PM Docs | Sprint -> Roadmap | Must activate "roadmap"; must NOT activate "sprint"/"issue" |
| P3-10 | PM Docs | Roadmap -> Issues (return) | Must activate "issue" files |
| P3-18 | PM Docs | Progress -> Architecture decisions | Must activate "decision"; must NOT activate "sprint"/"progress" |

**Pass criteria**: Correct S2 activation for the new topic,
AND absence of noise from the previous topic in S3 context.

---

## Component-Level Checks (Internal Diagnostics)

Each turn can validate specific pipeline components independently:

### S1 — Intent Detection
Checks that the pipeline correctly classifies the user's intent:
- `overview`: broad questions ("what is this project?") -> use synthesis nodes
- `specific`: targeted questions ("how does auth work?") -> use vector search
- `chat`: social messages ("thanks", "ok") -> minimal or no context

### S2 — Node Activation
Checks which graph nodes the pipeline selected:
- `activate_kinds`: which node types should appear (file, section, synthesis, entity)
- `not_activate_kinds`: which node types should NOT appear
- `activate_files_containing`: node labels must include these terms
- `not_activate_files_containing`: node labels must NOT include these terms
- `min_nodes` / `max_nodes`: bounds on how many nodes activate

### S3 — Context Assembly
Checks the warm context text injected into the LLM prompt:
- `warm_tokens_min` / `warm_tokens_max`: token budget bounds
- `context_contains`: ALL these terms must appear in context (case-insensitive)
- `context_contains_any`: at least ONE of these terms must appear
- `context_not_contains`: NONE of these terms should appear (noise check)

### S1.5 — Memory Extraction
Only checked on RECALL-store turns:
- `should_extract_entity`: verify that `process()` created a graph node for this entity

---

## Agent Comparison (Approach Scorecard)

24 turns include `agent_comparison` blocks that estimate what alternative
approaches would cost to answer the same question:

### Three approaches compared

| Approach | Description |
|----------|-------------|
| **Stateless LLM** | Plain model with no project access. Can only use training data. |
| **Agent + Tools** | Model with `list_directory`, `file_search`, `read_file` tools. Must discover and read files on every turn. |
| **Acervo** | Pre-indexed knowledge graph. Context injected from graph in a single step, no tool calls. |

### What gets measured

For each compared turn:

- **Stateless**: `can_answer` (true/false) -- can a plain LLM answer from training alone?
- **Agent + Tools**: `steps` (number of tool calls), `tools` (which tools used),
  `estimated_input_tokens` (total tokens consumed including tool results)
- **Acervo**: `warm_tokens` (actual tokens of warm context injected)

### Reports generated

**RESOLVE Scorecard**: For RESOLVE turns, shows what percentage each approach can answer
and at what token cost. Stateless typically scores 0% (can't answer without data).
Agent scores 100% (can always read files) but at high token cost.
Acervo scores based on effectiveness checks.

**Efficiency Chart** ("killer chart"): Per-turn comparison of agent tokens vs Acervo tokens.
Shows the ratio -- e.g., an agent uses 9,000 tokens where Acervo uses 500 (18x reduction).

Example agent estimates for P1 Turn 16 ("What data models are defined?"):
```
Agent: file_search("model") -> read_file(todo.model.ts) -> read_file(user.model.ts)
       3 steps, ~7,000 input tokens
Acervo: warm context with model summaries
       0 steps, ~400 tokens
       Ratio: 17.5x fewer tokens
```

The key insight: the agent's cost grows with project size (more files to search)
and conversation length (more history to carry). Acervo's cost stays bounded
by the token budget (~2000 tokens max regardless of project size or turn count).

---

## Version History

Each benchmark run appends results to `reports/version_history.json`.
This tracks category scores and component scores across releases,
enabling regression detection and progress tracking.

---

## Running the Tests

```bash
# Layer 1 only (no LLM needed, reads existing graph, ~5 seconds)
pytest tests/integration/test_pipeline_validation.py -v

# Full benchmark (needs LM Studio + Ollama, ~5 minutes)
pytest tests/integration/test_benchmarks.py -v -s

# Single project
pytest tests/integration/test_benchmarks.py -k "p1" -v -s

# Both layers
pytest tests/integration/ -v -s
```

### Output files

After running, find reports in `tests/integration/reports/`:

| File | Content |
|------|---------|
| `benchmark_public.json` | Category scores + scorecard + efficiency chart (machine-readable) |
| `benchmark_public.md` | Category scores + approach comparison tables (human-readable) |
| `benchmark_diagnostic.json` | Full per-turn detail with all component checks |
| `benchmark_diagnostic.md` | Component health + cross-matrix + S1 failures |
| `pipeline_diagnostic.json` | Graph state per project (Layer 1) |
| `pipeline_diagnostic.md` | Pipeline health summary (Layer 1) |
| `version_history.json` | Scores over time for regression tracking |

### Interpreting results

**Category scores** (public view): Percentage of turns where the context
contained the information needed to answer correctly.

- **100%** = every turn had correct context
- **< 80%** = some questions would fail -- investigate which turns and why

**Component scores** (internal view): Where the pipeline breaks down.

- **S1 Intent low** = intent detection misfires (classifying overview as specific or vice versa)
- **S2 Activation low** = wrong nodes selected (wrong files, too many nodes)
- **S3 Budget low** = token budget exceeded (over-stuffing context)
- **S3 Quality low** = context missing key terms or contains noise

**Cross-matrix** shows which categories are affected by which component failures.
For example, if ADAPT has low S2 Activation, the pipeline isn't switching
topics cleanly.

**Efficiency ratio** shows how many fewer tokens Acervo uses compared to a
tool-using agent. Higher is better. Typical: 10-40x fewer tokens per turn.
