# ACERVO v0.4.0 Benchmark

**55 turns** across 3 projects

## Category Scores

| Category | Score |
|----------|-------|
| RESOLVE | 100% |
| GROUND | 92% |
| RECALL | 67% |
| FOCUS | 100% |
| ADAPT | 100% |

## Approach Comparison

How Acervo compares to a stateless LLM and an agent with tools:

### RESOLVE (13 turns)

| Approach | Can Answer | Avg Input Tokens | Avg Steps |
|----------|-----------|-----------------|-----------|
| Stateless LLM | 8% | -- | -- |
| Agent + Tools | 100% | 7,462 | 2.8 |
| **Acervo** | **100%** | **616** | **0** |

> **12.1x** fewer tokens than agent approach

### GROUND (11 turns)

| Approach | Can Answer | Avg Input Tokens | Avg Steps |
|----------|-----------|-----------------|-----------|
| Stateless LLM | 27% | -- | -- |
| Agent + Tools | 100% | 5,500 | 2.3 |
| **Acervo** | **91%** | **600** | **0** |

> **9.2x** fewer tokens than agent approach

## Efficiency Chart

Per-turn token comparison (Agent vs Acervo):

| Turn | Category | Agent Tokens | Acervo Tokens | Ratio | Question |
|------|----------|-------------|--------------|-------|----------|
| 1 | RESOLVE | 3,500 | 828 | 4.2x | How many files does this project have? |
| 2 | GROUND | 8,000 | 801 | 10.0x | What technologies does this project use? |
| 5 | RESOLVE | 6,000 | 833 | 7.2x | What routes does the API define? |
| 13 | RESOLVE | 9,000 | 827 | 10.9x | How does the frontend communicate with the backend? |
| 14 | GROUND | 4,000 | 801 | 5.0x | Does this project use GraphQL? |
| 15 | GROUND | 3,500 | 828 | 4.2x | Is there a Python backend? |
| 16 | RESOLVE | 7,000 | 823 | 8.5x | What data models are defined? |
| 18 | RESOLVE | 6,500 | 826 | 7.9x | What error handling does the API have? |
| 1 | RESOLVE | 12,000 | 413 | 29.1x | What is this book about? |
| 2 | RESOLVE | 15,000 | 485 | 30.9x | How many stories does it contain? |
| 3 | GROUND | 8,000 | 511 | 15.7x | Who is Sherlock Holmes? |
| 6 | GROUND | 10,000 | 478 | 20.9x | Who is Irene Adler? |
| 11 | GROUND | 5,000 | 305 | 16.4x | Does Professor Moriarty appear in these stories? |
| 12 | RESOLVE | 10,000 | 503 | 19.9x | Where does Holmes live? |
| 15 | GROUND | 5,000 | 413 | 12.1x | So this is by Arthur Conan Doyle, right? |
| 1 | RESOLVE | 4,000 | 604 | 6.6x | What project is documented here? |
| 2 | RESOLVE | 2,000 | 526 | 3.8x | What documents are available? |
| 4 | GROUND | 5,000 | 601 | 8.3x | What's the tech stack for this project? |
| 7 | RESOLVE | 6,000 | 528 | 11.4x | What milestones are defined? |
| 12 | GROUND | 5,500 | 604 | 9.1x | Why was SQLite chosen over PostgreSQL? |
| 13 | RESOLVE | 7,000 | 210 | 33.3x | Who are the developers working on this? |
| 15 | GROUND | 3,500 | 656 | 5.3x | Is there a CI/CD pipeline set up? |
| 16 | RESOLVE | 9,000 | 596 | 15.1x | Which issues are related to the current sprint? |
| 19 | GROUND | 3,000 | 601 | 5.0x | Is Ron Weasley part of this project? |