## [0.5.0] - 2026-04-04

### Architecture
- **Ollama as default provider** — All defaults migrated from LM Studio (port 1234) to Ollama (port 11434). Model default changed from `acervo-extractor-v2` to `acervo-extractor-v3-Q4_K_M`. LM Studio health check removed from `acervo up`.
- **`json_mode` for structured LLM calls** — `LLMClient.chat()` protocol gains `json_mode=bool` parameter. When enabled, sends `response_format: {"type": "json_object"}` to the provider. Used by curator and semantic enricher to guarantee parseable output from Ollama/Qwen3.
- **S1 `followup` intent** — New intent classification between `specific` and `chat`: detects when the user continues a previous topic. Pipeline treats followup like specific (retrieves chunks) but preserves topic continuity.
- **S1 `retrieval` hint** — Model now returns `"retrieval": "summary_only" | "with_chunks"` to control chunk retrieval. Replaces the heuristic specificity classifier for v3 models. Falls back to keyword-based `classify_specificity()` when the model doesn't provide the field.
- **Intent-based node cap** — `_activate_nodes()` caps the number of activated nodes by intent: chat gets only synthesis nodes, specific/followup caps at 15 (structural nodes only), overview has no cap.
- **Project overview counts against budget** — Project overview text is now subtracted from the warm token budget instead of being injected on top. Prevents budget overruns on large projects.
- **Chat intent skips project context** — Chat messages no longer receive the project overview, reducing noise for casual conversation.

### Curation
- **Curator prompt rewrite** — Rewrote curation prompt to use the search-extractor format (`═══ RULES ═══` separators, schema with examples, `JSON:` terminator) that the fine-tuned model recognizes. Previous prompt caused the model to continue the file listing instead of extracting entities.
- **`strip_think_blocks` in curator** — Imported the shared `strip_think_blocks()` helper to handle Qwen3 `<think>...</think>` reasoning traces before JSON parsing.
- **Diagnostic logging** — Curator logs raw response (first 500 chars) and parsed entity/relation/fact counts per batch for debugging.

### Ingestion
- **`strip_think_blocks` in semantic enricher** — Enricher now strips think blocks before parsing LLM summaries, fixing parse failures with Qwen3 models.

### Data Model
- **`location` type mapping** — Added `"location" → "Place"` to the ontology type map. Previously, entities typed as "location" by the LLM were unmapped.

### DevRunner
- **`acervo up --dev` without project** — Dev mode no longer requires being inside an Acervo project. If no project is found, starts Studio and lets the user select one from the UI. Proxy is skipped until a project is active.
- **Active project from Studio DB** — DevRunner looks up the active project from Studio's SQLite database when no local project exists, so the proxy starts automatically after first project selection.
- **LM Studio dependency removed** — `check_dependencies()` no longer checks for LM Studio. Only Ollama is required.

### Fine-tuned Model
- **acervo-extractor-v3** — Trained on 1,076 examples (891 v2 + 185 new). New training data focused on S1 intent classification edge cases and followup detection. Model published as `acervo-extractor-v3-Q4_K_M`.

### Benchmark Results (v0.5.0)

| Category | v0.4.0 | v0.5.0 |
|----------|--------|--------|
| RESOLVE  | 100%   | 100%   |
| GROUND   | 92%    | 100%   |
| RECALL   | 67%    | 67%    |
| FOCUS    | 100%   | 100%   |
| ADAPT    | 100%   | 100%   |

Efficiency: **16.9x fewer tokens** than agent-with-tools approach.

GROUND improved from 92% to 100%. RECALL unchanged (S1 intent classification deferred to v0.6).
