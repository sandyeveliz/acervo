# Acervo v0.3.0 — Roadmap

> **Codename:** "Proof it works"
>
> v0.2.0 demostró que la arquitectura funciona. v0.3.0 tiene que demostrarlo
> a cualquiera que lo instale en 5 minutos. Cada feature de esta release
> responde a una pregunta: *¿puede alguien que no es nosotros probarlo,
> entenderlo, y verificar que funciona mejor que lo que ya tiene?*

---

## Principios de la release

1. **Nada se rompe en conversaciones largas.** El claim principal de Acervo es
   tokens constantes. Si una conversación de 100 turnos explota en tokens,
   nada más importa.
2. **Un comando para probar.** Si el setup tarda más de 3 minutos, la gente
   no llega a la parte donde funciona.
3. **Evidencia, no promesas.** Cada mejora tiene que ser demostrable con
   datos: tokens por turno, latencia, calidad de respuesta.

---

## Milestone 1 — Fundación (semana 1–2)

> Corregir lo que está roto y construir la infraestructura de medición.
> Sin esto, todo lo demás se construye sobre arena.

### 1.1 Fix: Context builder fallback (BLOQUEANTE)

**Problema:** cuando `prepare()` no detecta contexto relevante en el grafo,
el context builder no genera resumen y envía toda la historia de la
conversación como fallback. En conversaciones largas esto anula el propósito
de Acervo — vuelve al problema de tokens lineales.

**Solución:**

```
ANTES (buggy):
  prepare() → no graph context → fallback → enviar TODA la historia
  Turn 80: ~12,000 tokens enviados

DESPUÉS (correcto):
  prepare() → no graph context → resumir historia + window últimos 2 mensajes
  Turn 80: ~400 tokens (resumen ~200tk + últimos 2 mensajes ~200tk)
```

**Tareas:**

- [ ] En `context_builder.py`: eliminar el path que envía `history` completa
- [ ] Implementar resumen de emergencia: cuando no hay graph context, generar
      un resumen rolling de la conversación usando el utility LLM
- [ ] History window se aplica SIEMPRE, con o sin graph context:
      `system + [graph_context OR rolling_summary] + últimos N mensajes`
- [ ] Configurar `N` en config (default: 2, configurable con `history_window`)
- [ ] Test: conversación de 100 turnos sin entidades extraíbles (puro small
      talk) → verificar que tokens enviados nunca superan threshold
- [ ] Test: conversación que alterna entre temas con contexto y sin contexto
      → verificar transición limpia

**Criterio de éxito:** tokens enviados al LLM ≤ 600 en CUALQUIER turno,
independientemente del largo de la conversación, cuando `history_window=2`.

---

### 1.2 Trace estructurado por turno

**Problema:** no hay forma estandarizada de medir qué hace Acervo en cada
turno. Sin esto, no podemos hacer benchmarks, comparaciones, ni demostrar
mejoras entre versiones.

**Diseño del trace:**

```jsonc
// Un archivo JSONL por conversación: .acervo/traces/{conversation_id}.jsonl
// Una línea por turno:
{
  "turn": 14,
  "timestamp": "2026-03-25T10:30:00Z",
  "phase": "complete",  // "prepare" | "process" | "complete"

  // Métricas de contexto
  "context": {
    "tokens_user_message": 45,
    "tokens_graph_context": 280,
    "tokens_history_window": 150,
    "tokens_total_to_llm": 475,
    "tokens_without_acervo": 8900,  // lo que habría costado sin Acervo
    "compression_ratio": 0.053,     // 475/8900
    "graph_nodes_in_context": 6,
    "layer": "hot",                 // "hot" | "warm" | "cold" | "none"
    "topic": "beacon-auth-bug",
    "topic_action": "same"          // "same" | "subtopic" | "changed"
  },

  // Métricas de extracción
  "extraction": {
    "entities_extracted": 2,
    "entities_merged": 1,
    "relations_created": 3,
    "facts_added": 1,
    "facts_deduped": 0,
    "garbage_filtered": 0
  },

  // Métricas de rendimiento
  "performance": {
    "prepare_ms": 120,
    "llm_call_ms": 2400,     // tiempo del LLM principal (no Acervo)
    "process_ms": 350,
    "total_ms": 2870
  },

  // Contexto para debug
  "debug": {
    "graph_size_nodes": 42,
    "graph_size_edges": 67,
    "user_message_preview": "What tech does Beacon use?",
    "context_preview": "Beacon: web app, e-commerce..."  // primeros 100 chars
  }
}
```

**Tareas:**

- [ ] Crear `tracer.py` con clase `TurnTrace` (dataclass/pydantic)
- [ ] Instrumentar `prepare()` para capturar métricas de contexto
- [ ] Instrumentar `process()` para capturar métricas de extracción
- [ ] Calcular `tokens_without_acervo` (sum de toda la historia)
- [ ] Persistir a JSONL en `.acervo/traces/`
- [ ] Endpoint `GET /acervo/traces/{conversation_id}` en el proxy
- [ ] Endpoint `GET /acervo/traces/{conversation_id}/summary` → resumen
      agregado (avg tokens, compression ratio, etc.)
- [ ] CLI: `acervo trace show` — última conversación
- [ ] CLI: `acervo trace compare` — dos conversaciones side-by-side

**Criterio de éxito:** después de una conversación de 30 turnos, `acervo
trace show` muestra una tabla con tokens por turno y compression ratio.

---

### 1.3 Test de integración end-to-end

**Problema:** el claim "tokens constantes" no tiene test automatizado.

**Diseño:**

```python
# tests/e2e/test_constant_tokens.py

SCRIPTED_CONVERSATION = [
    # 50 turnos que cubren:
    # - Introducción de entidades (turnos 1-10)
    # - Preguntas sobre entidades conocidas (11-20)
    # - Cambio de tema (21-30)
    # - Vuelta al tema original (31-40)
    # - Small talk sin entidades (41-50)
]

async def test_tokens_stay_constant():
    """Tokens enviados al LLM nunca superan 800 después del turno 5."""
    memory = Acervo(llm=mock_llm, owner="test")
    for turn in SCRIPTED_CONVERSATION:
        trace = await run_turn(memory, turn)
        if trace.turn > 5:
            assert trace.context.tokens_total_to_llm <= 800, \
                f"Turn {trace.turn}: {trace.context.tokens_total_to_llm} tokens"
```

**Tareas:**

- [ ] Escribir conversación scripted de 50 turnos (5 dominios)
- [ ] Test: tokens ≤ threshold en todos los turnos post warm-up
- [ ] Test: cambio de tema → contexto cambia, no acumula
- [ ] Test: vuelta a tema anterior → contexto se restaura del grafo
- [ ] Test: small talk → resumen mínimo, sin entidades fantasma
- [ ] Agregar a CI (puede correr con mock LLM o modelo local)

---

## Milestone 2 — Experiencia de usuario (semana 2–3)

> Que alguien pueda instalarlo y probarlo sin fricciones.

### 2.1 `acervo up` — un comando para todo

**Problema actual:** para probar Acervo hay que levantar manualmente:
1. Ollama con embeddings
2. LM Studio (o servidor LLM)
3. Backend Python de Acervo Studio
4. WebUI de Acervo Studio

Eso son 4 terminales y 4 comandos. Para un usuario nuevo es inaceptable.

**Solución para v0.3.0** (sin Electron, pragmática):

```bash
acervo up                          # levanta todo
acervo up --no-ui                  # solo Acervo + modelo, sin Studio
acervo up --model qwen3.5-9b      # especificar modelo
acervo down                        # para todo
acervo status                      # health check de todos los servicios
```

**Diseño interno:**

```
acervo up
  ├── Detectar si Ollama está instalado → arrancar embeddings
  ├── Detectar servidor LLM (LM Studio / Ollama) → arrancar modelo
  ├── Arrancar Acervo proxy (acervo serve)
  ├── Arrancar Acervo Studio backend + UI
  └── Health check → "✓ Everything running at http://localhost:9470"
```

**Implementación:** process manager liviano en Python. No Docker (agrega
dependencia). No docker-compose. Un módulo `acervo/runner.py` que:
- Usa `subprocess.Popen` para cada proceso
- Guarda PIDs en `.acervo/pids.json`
- `acervo down` lee los PIDs y hace shutdown limpio
- `acervo status` hace health check a cada endpoint

**Tareas:**

- [ ] `acervo/cli/up.py` — detección de servicios y arranque
- [ ] `acervo/cli/down.py` — shutdown limpio por PID
- [ ] `acervo/cli/status.py` — health check con output legible
- [ ] Detección automática de Ollama (check `ollama list`)
- [ ] Detección automática de LM Studio (check puerto 1234)
- [ ] Config en `acervo.toml`: puertos, modelos, paths
- [ ] Fallback si Ollama no está: instrucción clara de instalación
- [ ] Primer-run wizard: si no hay config, preguntar qué modelo usar
- [ ] Documentar en README: `pip install acervo && acervo up`

**Criterio de éxito:** usuario hace `pip install acervo && acervo up` y en
< 3 minutos tiene la UI corriendo con un modelo funcional.

**Nota sobre Electron / app nativa:**

Descartado indefinidamente. `acervo up` levantando la web UI existente cubre
el caso de uso sin agregar Node.js al stack. Si en el futuro se necesita
una app de escritorio, evaluar Tauri o Pywebview antes que Electron.

---

### 2.2 Graph inspection y edición básica

**Problema:** si el extractor se equivoca (entidad mal tipada, relación
incorrecta, merge erróneo), el usuario no tiene forma de verlo ni corregirlo
sin borrar todo el grafo.

**Tareas:**

- [ ] CLI: `acervo graph show` — listar nodos con tipo, layer, # facts
- [ ] CLI: `acervo graph show <entity_id>` — detalle de un nodo
- [ ] CLI: `acervo graph search <query>` — buscar en el grafo
- [ ] CLI: `acervo graph delete <entity_id>` — borrar nodo y sus edges
- [ ] CLI: `acervo graph merge <id1> <id2>` — merge manual de dos nodos
- [ ] Endpoint REST para cada operación (para uso desde Studio)
- [ ] Output en formato tabla (terminal) y JSON (para pipes)

**Criterio de éxito:** el usuario puede ver un nodo mal extraído y borrarlo
sin perder el resto del grafo.

---

## Milestone 3 — Document ingestion con chunks en el grafo (semana 3–4)

> La primera demostración real de "knowledge beyond conversation". El usuario
> agrega un archivo y Acervo lo convierte en nodos del grafo con chunks
> recuperables. Scope acotado: solo `.md` por ahora.

### 3.1 Arquitectura: chunks vinculados a nodos del grafo

**El problema con el RAG tradicional:** los chunks viven aislados en un
vector store. No tienen relación con el grafo de conocimiento. Cuando
activás un nodo, no sabés si tiene chunks asociados ni cuáles son relevantes.

**Lo que construimos:**

```
archivo.md
  │
  ├── Chunking por heading/párrafo
  │     chunk_0: "## Auth module\nThe auth module uses JWT..."
  │     chunk_1: "## Database\nPostgreSQL with connection pooling..."
  │     chunk_2: "## Deployment\nAWS ECS with auto-scaling..."
  │
  ├── Embedding (Ollama) → ChromaDB
  │     cada chunk tiene un ID único
  │
  └── Nodo en el grafo
        {
          "id": "beacon-architecture",
          "type": "document",
          "label": "Beacon Architecture",
          "layer": "PERSONAL",
          "chunk_ids": ["beacon_arch_c0", "beacon_arch_c1", "beacon_arch_c2"],
          "attributes": {
            "source_file": "architecture.md",
            "content_hash": "sha256:abc...",
            "chunk_count": 3
          }
        }
```

**Flujo de retrieval (en `prepare()`):**

```
User: "How does Beacon handle auth?"
  │
  ├── S1 extrae topic: "beacon" + "auth"
  ├── Grafo activa nodo "Beacon" (hot layer)
  ├── Nodo "Beacon Architecture" vinculado, tiene chunk_ids
  ├── Heurística: pregunta específica ("how does X handle Y") → traer chunks
  ├── Vector search SOLO en chunk_ids del nodo → top 2 chunks relevantes
  └── Contexto: resumen del nodo (80tk) + chunks relevantes (200tk) = 280tk
      vs RAG global que traería 5 chunks de 500tk = 2,500tk
```

### 3.2 Dos vías de ingesta

**Vía 1 — CLI:**

```bash
acervo index --path ./docs/architecture.md
acervo index --path ./docs/                   # directorio completo (solo .md)
```

Reutiliza el pipeline de `acervo index` existente (structural parser +
semantic enricher), extendido para:
- Parsear `.md` por headings y párrafos (ya funciona)
- Generar chunk IDs y persistirlos en el nodo del grafo (GAP actual)
- Almacenar chunks embedidos en ChromaDB con los IDs del nodo

**Vía 2 — API REST (para Acervo Studio):**

```
POST /acervo/documents
Content-Type: multipart/form-data
file: architecture.md

Response:
{
  "document_id": "beacon-architecture",
  "chunks": 3,
  "node_created": true,
  "chunk_ids": ["beacon_arch_c0", "beacon_arch_c1", "beacon_arch_c2"]
}
```

```
GET /acervo/documents                    # listar documentos indexados
GET /acervo/documents/{id}               # detalle + chunks
DELETE /acervo/documents/{id}            # borrar documento + chunks + nodo
```

### 3.3 Cambios técnicos requeridos

**Archivos a modificar (del planning doc existente):**

| Archivo | Cambio |
|---------|--------|
| `acervo/graph.py` | Agregar campo `chunk_ids: list[str]` en nodos tipo `document`/`file`. Método `get_chunks_for_node(node_id) → list[str]` |
| `acervo/semantic_enricher.py` | Retornar chunk IDs junto con embeddings. Actualmente genera embeddings pero no persiste la referencia en el nodo |
| `acervo/indexer.py` | Después de Phase 2, escribir `chunk_ids` en el nodo correspondiente del grafo |
| `acervo/structural_parser.py` | Extender para `.md` con chunking por heading + párrafo (parcialmente hecho) |
| `acervo/facade.py` | En `gather()`, cuando un nodo activado tiene `chunk_ids`, consultar ChromaDB para traer los chunks relevantes al contexto |
| `acervo/vector_store.py` | Método `get_chunks_by_ids(chunk_ids, query) → list[str]` para retrieval node-scoped |
| `acervo/proxy.py` | Endpoints REST para upload y gestión de documentos |

**Tareas:**

- [ ] `graph.py`: campo `chunk_ids` en nodos, método de acceso
- [ ] `semantic_enricher.py`: retornar chunk IDs al indexer
- [ ] `indexer.py`: persistir chunk IDs en nodo después de Phase 2
- [ ] `structural_parser.py`: verificar que el parser `.md` genera chunks
      con headings como delimitador principal, párrafos como fallback
- [ ] `vector_store.py`: `get_chunks_by_ids()` — retrieval filtrado por IDs
- [ ] `facade.py`: integrar chunk retrieval en el pipeline de `prepare()`
- [ ] Proxy: `POST /acervo/documents` — upload + index pipeline
- [ ] Proxy: `GET /acervo/documents` — listar documentos
- [ ] Proxy: `DELETE /acervo/documents/{id}` — borrar doc + chunks + nodo
- [ ] CLI: `acervo index --path` actualizado para persistir chunk_ids
- [ ] Content hash (SHA-256) para skip de archivos sin cambios (ya existe)
- [ ] Test: indexar `.md` → verificar nodo creado con chunk_ids
- [ ] Test: pregunta sobre contenido del `.md` → chunks relevantes en contexto
- [ ] Test: borrar documento → nodo + chunks + embeddings eliminados
- [ ] Agregar métricas de chunks al trace (1.2):

```jsonc
// En el trace, agregar a "context":
"chunks": {
  "documents_with_chunks_activated": 1,
  "chunks_retrieved": 2,
  "chunks_total_on_node": 5,
  "tokens_from_chunks": 180,
  "retrieval_scope": "node_scoped"   // vs "global" (nunca en v0.3.0)
}
```

### 3.4 Scope acotado — lo que NO entra

| Cosa | Por qué no | Cuándo |
|------|-----------|--------|
| Formatos .pdf, .docx, .txt | Requieren parsers adicionales y chunking distinto | v0.4.0 |
| Chunking semántico inteligente | Por ahora heading + párrafo + tamaño fijo alcanza | v0.4.0 |
| Libros completos | Necesita estrategia de chunking para documentos de 100k+ tokens | v0.5.0 |
| S1 entrenado para document references | Keyword activation cubre la mayoría de casos | v0.4.0 |
| Upload desde la WebUI drag & drop | API REST cubre el caso, UI polish después | v0.4.0 |

**Criterio de éxito:** el usuario indexa un `.md` con reglas de JavaScript,
pregunta "what are the coding rules?", y recibe los chunks relevantes en
~200 tokens de contexto — no el archivo entero.

---

## Milestone 4 — Demostración (semana 4–5)

> Que cualquiera pueda VER que funciona, sin instalarlo.

### 4.1 README: "Acervo in Action" — ejemplos turn-by-turn

**El formato más efectivo para demostrar Acervo:**

No videos, no demos, no GIFs. Texto puro en el README mostrando cada turno
de una conversación real, con side-by-side de:

```
═══════════════════════════════════════════════════════════════
TURN 1 — User: "I work at Acme Corp on Project Beacon, a React app"
═══════════════════════════════════════════════════════════════

┌─────────────────────────────┬─────────────────────────────┐
│     WITHOUT ACERVO          │       WITH ACERVO           │
├─────────────────────────────┼─────────────────────────────┤
│ Tokens sent: 180            │ Tokens sent: 195            │
│ Context: (none)             │ Context: (none, first turn) │
│                             │                             │
│ LLM response:               │ LLM response:               │
│ "Sounds great! Tell me      │ "Sounds great! Tell me      │
│  more about Beacon."        │  more about Beacon."        │
├─────────────────────────────┼─────────────────────────────┤
│ Graph: (none)               │ Graph after turn:           │
│                             │  + Acme Corp (organization) │
│                             │  + Beacon (project)         │
│                             │  + React (technology)       │
│                             │  + Acme → produces → Beacon │
│                             │  + Beacon → uses → React    │
└─────────────────────────────┴─────────────────────────────┘

═══════════════════════════════════════════════════════════════
TURN 15 — User: "What tech does our project use?"
═══════════════════════════════════════════════════════════════

┌─────────────────────────────┬─────────────────────────────┐
│     WITHOUT ACERVO          │       WITH ACERVO           │
├─────────────────────────────┼─────────────────────────────┤
│ Tokens sent: 4,200          │ Tokens sent: 380            │
│ Context: full history       │ Context:                    │
│ (turns 1-14, all messages)  │  Beacon: e-commerce app     │
│                             │   → uses: React, PostgreSQL │
│                             │   → deployed: AWS           │
│                             │   → team: Alice, Bob        │
│                             │  + last 2 messages          │
│                             │                             │
│ LLM response:               │ LLM response:               │
│ "Based on our earlier       │ "Beacon uses React,         │
│  conversation, you          │  PostgreSQL, and is          │
│  mentioned React..."        │  deployed on AWS."           │
│                             │                             │
│ (same quality, 11x tokens)  │ (same quality, 90% fewer)  │
├─────────────────────────────┼─────────────────────────────┤
│ Compression: 1.0x (none)    │ Compression: 11.0x          │
└─────────────────────────────┴─────────────────────────────┘

═══════════════════════════════════════════════════════════════
TURN 40 — User: "Back to Beacon, did we fix the auth bug?"
═══════════════════════════════════════════════════════════════

┌─────────────────────────────┬─────────────────────────────┐
│     WITHOUT ACERVO          │       WITH ACERVO           │
├─────────────────────────────┼─────────────────────────────┤
│ Tokens sent: 11,500         │ Tokens sent: 420            │
│ Context: full history       │ Context: Beacon subgraph    │
│                             │  (restored from graph,      │
│                             │   includes morning facts)   │
│                             │                             │
│ LLM response:               │ LLM response:               │
│ ⚠ Context window limit —   │ "Based on your earlier       │
│ oldest messages truncated.  │  discussion, the auth bug    │
│ Lost turn 1-12 context.     │  was identified in the       │
│ "I don't have context       │  session middleware. Alice    │
│  about an auth bug."        │  was assigned to fix it."    │
│                             │                             │
│ DEGRADED — info lost        │ PERFECT — graph remembers   │
├─────────────────────────────┼─────────────────────────────┤
│ Compression: N/A (broken)   │ Compression: 27.4x          │
└─────────────────────────────┴─────────────────────────────┘
```

**Tareas:**

- [ ] Script `scripts/generate_demo.py` que ejecuta una conversación de 30
      turnos y genera el output formateado automáticamente
- [ ] Conversación de ejemplo cubriendo: setup inicial, preguntas, cambio de
      tema, vuelta a tema anterior, small talk, contexto perdido sin Acervo,
      **pregunta sobre contenido de un .md indexado (document chunks)**
- [ ] Tabla resumen al final:

```
Turn  │ Sin Acervo │ Con Acervo │ Compresión │ Respuesta
──────┼────────────┼────────────┼────────────┼──────────
  1   │    180 tk  │    195 tk  │    0.9x    │ Igual
  5   │  1,200 tk  │    350 tk  │    3.4x    │ Igual
 15   │  4,200 tk  │    380 tk  │   11.0x    │ Igual
 30   │  8,900 tk  │    410 tk  │   21.7x    │ Igual
 40   │  TRUNCADO  │    420 tk  │    ∞       │ Acervo wins
 50   │  TRUNCADO  │    400 tk  │    ∞       │ Acervo wins
```

- [ ] Integrar en README reemplazando los diagramas ASCII actuales de
      comparación (que son buenos pero abstractos)

**Criterio de éxito:** alguien lee el README y en 60 segundos entiende
exactamente qué hace Acervo y por qué es mejor.

---

### 4.2 Benchmark script publicable

**Basado en el trace de 1.2**, crear un script que cualquiera pueda correr:

```bash
acervo benchmark --turns 50 --model qwen3.5-9b
```

Output:

```
Acervo Benchmark — 50 turns, model: qwen3.5-9b
──────────────────────────────────────────────────
Avg tokens/turn (with Acervo):     385
Avg tokens/turn (without Acervo):  5,240
Avg compression ratio:             13.6x
Turns where context was lost:      0 (with) vs 8 (without)
Avg prepare() latency:             95ms
Avg process() latency:             280ms
Extraction accuracy:               87%
JSON parse rate:                   100%
```

**Tareas:**

- [ ] `acervo/cli/benchmark.py` — ejecuta conversación scripted con trace
- [ ] Calcular métricas agregadas del trace
- [ ] Comparar con baseline (sin Acervo, full history)
- [ ] Output en formato tabla (terminal) y JSON (para CI)
- [ ] Incluir en CI como regression check

---

## Milestone 5 — Modelo y retrieval (semana 5–6)

> Mejoras incrementales al modelo y al retrieval.

### 5.1 Chunk-aware retrieval

**Problema actual:** cuando un nodo tiene chunks asociados (de `acervo
index`), el retrieval los trae todos o ninguno. No hay decisión inteligente
de "¿necesito los chunks o basta el resumen del nodo?".

**Diseño:**

```
Pregunta conceptual: "What does Beacon do?"
  → Resumen del nodo basta (80 tokens)
  → NO traer chunks de código

Pregunta específica: "What's the exact API endpoint for auth?"
  → Resumen + chunks relevantes del nodo (200 tokens)
  → RAG filtrado SOLO a chunks del nodo "Beacon > auth"

Pregunta de código: "Show me the auth middleware"
  → Chunks de código del nodo (400 tokens)
  → RAG filtrado a chunks del nodo con tag "middleware"
```

**Implementación incremental:**

1. **Fase 1 — Heurística simple** (v0.3.0):
   - Si la pregunta contiene keywords de especificidad (número, fecha,
     "exact", "show me", "code") → traer chunks
   - Si la pregunta es conceptual → solo resumen
   - Filtrar chunks por el nodo relevante, no búsqueda global

2. **Fase 2 — Entrenado** (v0.4.0):
   - Agregar ejemplos al fine-tune dataset donde el modelo decide
     `"retrieval": "summary_only"` vs `"retrieval": "with_chunks"`
   - El modelo ya recibe los nodos como contexto, solo necesita un
     campo más en el output JSON

**Tareas v0.3.0:**

- [ ] Clasificador de especificidad (regex + keywords)
- [ ] Filtrar vector search por nodo relevante (no global)
- [ ] Si chunk retrieval → limitar a top 3 chunks del nodo, no top 5 global
- [ ] Test: pregunta conceptual → ≤ 100 tokens de contexto
- [ ] Test: pregunta específica → chunks relevantes incluidos, ≤ 400 tokens
- [ ] Agregar métricas de retrieval al trace (chunks_retrieved,
      chunks_from_node vs chunks_global)

### 5.2 Mejoras al fine-tune

**Objetivo:** mejorar extraction accuracy de 85% → 90%+ con ejemplos
adicionales enfocados en los failure modes observados en v0.2.0.

**Áreas de mejora identificadas:**

- [ ] Agregar 100 ejemplos de "no extraer" (small talk, preguntas genéricas,
      agradecimientos) — el modelo a veces crea entidades fantasma
- [ ] Agregar 50 ejemplos de dedup (mismo concepto mencionado de formas
      distintas: "nuestro proyecto", "Beacon", "the app")
- [ ] Agregar 50 ejemplos con chunks: el modelo recibe un nodo con chunks
      y debe decidir si traerlos o no
- [ ] Entrenar con conversaciones de 20+ turnos (no solo pares aislados)
- [ ] Agregar dominio: conversaciones técnicas en español rioplatense
- [ ] Evaluar con el benchmark script de 4.2 antes y después del retrain

---

## Milestone 6 — Polish (semana 6–7)

> Detalles que separan "funciona" de "es usable".

### 6.1 Logs legibles

- [ ] Niveles de log configurables: `--log-level info|debug|trace`
- [ ] Log por defecto (`info`): una línea por turno con tokens y compresión
- [ ] Log verbose (`debug`): incluye entidades extraídas y decisiones de topic
- [ ] Log completo (`trace`): incluye prompts enviados y respuestas raw
- [ ] Colores en terminal (desactivables con `--no-color`)

### 6.2 Error handling y recovery

- [ ] Si el utility LLM no responde → degradar gracefully (enviar últimos N
      mensajes sin graph context, loguear warning)
- [ ] Si el grafo se corrompe → detectar y ofrecer `acervo graph repair`
- [ ] Si el extractor devuelve JSON inválido → retry con temperature más
      baja, loguear el intento fallido
- [ ] Timeout configurable para cada fase (prepare, process)

### 6.3 Documentación actualizada

- [ ] README con la sección "Acervo in Action" (4.1)
- [ ] Actualizar Getting Started con `acervo up`
- [ ] Documentar formato de trace
- [ ] Documentar graph inspection CLI
- [ ] Documentar ingesta de documentos (CLI + API)
- [ ] Actualizar changelog con todos los cambios de v0.3.0

---

## Cronograma resumido

```
Semana 1    ██████████  M1: Context builder fix + trace format
Semana 2    ██████████  M1: E2E tests + M2: acervo up
Semana 3    ██████████  M2: Graph inspection + M3: Document ingestion (graph + chunking)
Semana 4    ██████████  M3: API + CLI ingesta + M4: README examples
Semana 5    ██████████  M4: Benchmark + M5: Chunk-aware retrieval
Semana 6    ██████████  M5: Fine-tune + M6: Polish
Semana 7    ██████████  M6: Docs + release
```

## Lo que NO entra en v0.3.0

| Feature | Por qué no | Cuándo |
|---------|-----------|--------|
| Formatos .pdf, .docx, .txt para ingesta | `.md` cubre el caso inicial, parsers adicionales son scope extra | v0.4.0 |
| Chunking semántico inteligente | Heading + párrafo + tamaño fijo alcanza para `.md` | v0.4.0 |
| Libros y documentos largos (100k+ tokens) | Necesita estrategia de chunking distinta | v0.5.0 |
| S1 entrenado para document references | Keyword activation cubre la mayoría de casos | v0.4.0 |
| Progressive retrieval (hot → warm → cold) | Necesita el trace para medir impacto primero | v0.3.1 |
| Topic-scoped vector search | Depende de chunk-aware retrieval | v0.3.1 |
| Community knowledge packs | Necesita modelo estable primero | v0.5.0 |
| Fine-tuned extraction model v3 | Iterativo, no bloqueante | Continuo |
| Multi-user graph sharing | Arquitectura no lista | v0.5.0 |
| Electron / app de escritorio | Descartado indefinidamente. `acervo up` + web UI cubre el caso | — |

---

## Criterios de release

v0.3.0 se publica cuando:

- [ ] El test de tokens constantes pasa en 50 turnos
- [ ] `acervo up` funciona en Linux y macOS sin intervención manual
- [ ] `acervo index --path archivo.md` indexa y vincula chunks al grafo
- [ ] Pregunta sobre contenido de un `.md` indexado → chunks en contexto (node-scoped)
- [ ] El README tiene la sección "Acervo in Action" con datos reales
- [ ] `acervo benchmark` corre y produce resultados reproducibles
- [ ] Zero crashes en una sesión de 100 turnos con el fine-tuned model
- [ ] Changelog completo
- [ ] Publicado en PyPI

# Acervo v0.3.0 — Roadmap

> **Codename:** "Proof it works"
>
> v0.2.0 demostro que la arquitectura funciona. v0.3.0 tiene que demostrarlo
> a cualquiera que lo instale en 5 minutos. Cada feature de esta release
> responde a una pregunta: *puede alguien que no es nosotros probarlo,
> entenderlo, y verificar que funciona mejor que lo que ya tiene?*

---

## Principios de la release

1. **Nada se rompe en conversaciones largas.** El claim principal de Acervo es
   tokens constantes. Si una conversacion de 100 turnos explota en tokens,
   nada mas importa.
2. **Un comando para probar.** Si el setup tarda mas de 3 minutos, la gente
   no llega a la parte donde funciona.
3. **Evidencia, no promesas.** Cada mejora tiene que ser demostrable con
   datos: tokens por turno, latencia, calidad de respuesta.

---

See full roadmap in the conversation where it was provided (2026-03-25).

Key milestones:
- M1: Foundation (context builder fix, trace, e2e tests)
- M2: User experience (acervo up, graph inspection)
- M3: Document ingestion with chunks linked to graph nodes
- M4: Demonstration (README examples, benchmark)
- M5: Model & retrieval improvements
- M6: Polish (logs, error handling, docs)

Work split: Claude handles M1, M2.2, M3, M5.1, M6.1-6.2.
Sandy handles M2.1, M4, M5.2, M6.3, Acervo Studio.
