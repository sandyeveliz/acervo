## [0.5.0] - 2026-04-06

### Architecture Refactor — Hexagonal Pipeline

- **Facade split into domain stages** — The 1,848-line God Module (`facade.py`) is now a thin wrapper that delegates to `domain/pipeline.py`. Pipeline orchestrates S1→S2→S3 for prepare() and S1.5 for process(). Each stage is a standalone, testable module.
- **New package structure** — `ports/` (protocols), `domain/` (pipeline stages), `graph/` (knowledge graph), `context/` (context management), `extraction/` (extractors), `indexing/` (file indexation), `adapters/` (implementations), `proxy/` (HTTP), `cli/` (commands). All old imports preserved via re-export stubs.
- **S2 Activator: one code path** — `domain/s2_activator.py` replaces the dual conversation/project code paths. Entity neighbor traversal follows ALL edges regardless of mode. Nodes with descriptions (not just facts) are included in context.
- **S3 Assembler: intent controls budget, not filtering** — `domain/s3_assembler.py` uses one `_assemble()` method for all intents. Overview, specific, followup, chat all inject S2 data — intent only changes the token budget (250/600/400/100).
- **Port protocols** — `LLMPort`, `EmbedderPort`, `VectorStorePort`, `GraphStorePort` in `ports/`. Domain code depends on protocols, not implementations.
- **Domain models** — `S1Result`, `S2Result`, `S3Result`, `S15Result`, `GatheredNode`, `RankedChunk` in `domain/models.py`. Typed contracts between stages.
- **Backward compatible** — `from acervo import Acervo`, `from acervo.graph import TopicGraph`, all `__init__.py` exports unchanged. 178/185 tests pass (7 pre-existing failures).

### Pipeline Fixes

- **JSON parser repair** — `_parse_first_json()` now has 3 fallback levels: strict depth tracking → greedy (last closing brace) → repair (`]\n"` → `], "`). Fixes silent entity loss when the fine-tuned model produces slightly malformed JSON.
- **Nested relations** — S1 parser extracts relations from inside entity objects (`entity.relations[]`), not just the top-level `relations[]` array. The fine-tuned model puts relations nested; previously they were silently discarded.
- **Relation ID resolution** — S1 parser builds `model_id → entity_label` map. Relations like `checkear → supabase_db` are resolved to `Checkear → Supabase` matching actual node IDs. Previously, edges referenced non-existent node IDs.
- **Self-referencing relation guard** — `_add_relation()` rejects `source == target` (e.g. `Acervo → part_of → Acervo`).
- **Entity neighbor expansion** — S2 traverses ALL entity edges (not just `contains`). `supabase` → `uses_technology` → `checkear`, `walletfy` now activates neighbors correctly.
- **Topic drift prevention** — `PREVIOUS ASSISTANT` in S1 prompt limited to 150 chars (was 500). Prevents the model from adopting hallucinated topics from long assistant responses.
- **Generic model support** — S1 parser handles string entities (`["Butaco"]`), string topics (`"Proyectos"`), and default types when the model doesn't provide structured objects.
- **Description chunks** — S2 includes `node.attributes.description` in context chunks. Previously only facts and relations were used; conversation-created entities with descriptions were invisible to S3.
- **Delayed flush for proxy mode** — TelemetryCollector and TurnLogger schedule a 2-second flush after `StreamCompleted`. If `GraphUpdated` arrives first (non-proxy mode), it cancels the timer. Fixes 0-span telemetry in proxy mode.

### Ollama Migration

- **Ollama as default provider** — All defaults migrated from LM Studio (port 1234) to Ollama (port 11434/v1). LM Studio health check removed from `acervo up`.
- **Modelfile fix** — `acervo-extractor-v3-Q4_K_M` Ollama Modelfile now includes the Qwen3.5 chat template (`<|im_start|>`/`<|im_end|>`) and stop tokens. Previously used raw template causing infinite generation loops.
- **Settings override fix** — `.env` variables no longer override `settings.toml` model name, allowing the Studio UI to change models dynamically.

### DevRunner

- **`acervo up --dev` without project** — Dev mode no longer requires `.acervo/` in cwd. Proxy is skipped if no project found; Studio and Ollama start normally.
- **Active project from Studio DB** — DevRunner looks up the active project from Studio's SQLite database when no local project exists.

### Acervo Studio

- **Telemetry page: per-turn annotation** — Replaced 4-tab Metrics view with unified turn list + turn detail panel. Each turn shows S1/S2/S3/LLM/S1.5 stage cards with actual vs expected annotation. Export as JSONL (training data) or JSON (full dump).
- **Ollama monitor page** — Live VRAM/GPU/RAM monitoring, loaded/available models, auto-refresh 2s.
- **Sidebar reorganization** — Main nav (Chat, Graph, Agents, Projects, Settings) separated from Acervo debug tools (Metrics, Ollama) by a divider.
- **Per-project telemetry** — Telemetry spans and annotations persist in each project's `.acervo/` directory. Switching projects loads the correct data.
- **Annotation backend** — `AnnotationStore` with JSONL persistence, CRUD REST endpoints, JSONL training export with S1 prompt reconstruction.
- **S1 debug enrichment** — `stage_data` now includes `s1_prompt`, `s1_raw_response`, `s1_latency_ms`, `s15_prompt`, `s15_raw_response`, `s15_actions`. Visible in the telemetry turn detail.
- **Graph counter fix** — Telemetry `graph.node_count` now queries proxy `/acervo/status` after process(), emitting `GraphUpdated` so counters reflect actual state.
- **Project status fix** — Refresh button no longer falsely shows "Not initialized" on network errors.
- **Description save UX** — Project description editor has Save/Cancel buttons (was Enter-only).
- **Reset clears everything** — Trash button clears chat, trace, telemetry JSONL, and annotations.

### Fine-tuned Model

- **acervo-extractor-v3** — Trained on 1,076 examples. S1 intent classification + followup detection. Published as `acervo-extractor-v3-Q4_K_M` for Ollama with correct chat template.

### Benchmark Results (v0.5.0)

| Category | v0.4.0 | v0.5.0 |
|----------|--------|--------|
| RESOLVE  | 100%   | 100%   |
| GROUND   | 92%    | 100%   |
| RECALL   | 67%    | 67%    |
| FOCUS    | 100%   | 100%   |
| ADAPT    | 100%   | 100%   |

178/185 tests pass (7 pre-existing failures, 0 regressions from refactor).

### Manual 4-Turn Test (conversation mode, no indexed project)

```
Turn 1: "Tenemos 4 proyectos: Butaco con Angular y Firebase..."
  S1: 9 entities, 8 relations ✓
  Graph: 9 nodes, 17 edges ✓

Turn 2: "¿Qué proyectos usan Supabase?"
  S2: 3 nodes activated (supabase + checkear + walletfy) ✓
  S3: warm_tokens=103, has_context=True ✓
  Warm content includes Walletfy→Supabase, Checkear→Supabase ✓

Turn 3: "¿Y cuáles usan Firebase?"
  S2: 3 nodes activated ✓
  S3: warm_tokens=102, has_context=True ✓

Turn 4: "¿Cuántos proyectos tenemos?"
  S2: 9 nodes gathered ✓
  S3: warm_tokens>0 (pending proxy restart for S3 fix) ⏳
```
