# Dataset v3 — Plan primero, no generes nada todavía

Necesito que planifiques el dataset de fine-tune v3 para el extractor de Acervo.
Leé todo antes de responder. Quiero un plan detallado para aprobar ANTES de generar
cualquier ejemplo.

## Contexto del modelo actual

Tenemos acervo-extractor-v2 (Qwen3.5-9B, 582 ejemplos, LoRA sobre base).
Lo comparamos contra qwen2.5:7b (genérico, sin fine-tune) en 397 turns reales
(8 escenarios de conversación) y después en 25 turns problemáticos seleccionados.

### Benchmark completo (8 escenarios, 397 turns, qwen2.5:7b con prompt mejorado)

| Caso             | Raw Facts | Parsed | Drop | Entity% | Relation% | Fact% |
|------------------|-----------|--------|------|---------|-----------|-------|
| casa             | 68        | 66     | 3%   | 78%     | 10%       | 3%    |
| finanzas         | 73        | 65     | 11%  | 58%     | 0%        | 7%    |
| fitness          | 68        | 62     | 9%   | 58%     | 38%       | 0%    |
| libro            | 75        | 72     | 4%   | 55%     | 0%        | 24%   |
| proyecto_codigo  | 68        | 66     | 3%   | 52%     | 9%        | 0%    |
| salud_familia    | 77        | 67     | 13%  | 43%     | 8%        | 24%   |
| trabajo          | 83        | 78     | 6%   | 68%     | 7%        | 12%   |
| viajes           | 77        | 54     | 30%  | 44%     | 0%        | 10%   |

### Comparison en 25 turns problemáticos

| Métrica          | qwen2.5:7b | qwen3.5:9b base |
|------------------|------------|-----------------|
| JSON parse fails | 0          | 5/25 turns      |
| Facts parsed     | 20/28 72%  | 29/36 81%       |
| Drop rate        | 28%        | 19%             |
| BETTER           | 13         | 10              |
| WORSE            | 0          | 2               |

### Problemas concretos encontrados (de 397 turns reales)

**1. Facts con entity_id=null (29 drops)** — El modelo genera el fact con datos correctos
pero no asigna entity_id. Ejemplo: "Sueldo: 4.000.000 ARS/mes" con entity_id: null.
El pipeline descarta estos facts porque no sabe a qué entidad adjuntarlos.

**2. Facts con entity inexistente (21 drops)** — El modelo inventa entity IDs como
"Portafolio de Inversiones", "App Proyecto", "Usuario", "Perfil de Fitness" que no
están en el grafo ni en las entities del turno actual. Debería usar existing_id
para referenciar nodos que ya existen, o crear la entity primero.

**3. JSON parse failures (5/25 turns con 9b base)** — El modelo devuelve texto libre
en vez de JSON válido. Con 7b esto no pasaba (0 failures). Es el problema más grave
porque se pierde toda la extracción del turno.

**4. Relaciones inválidas rechazadas** — El modelo genera: produced_by, has_document,
needs, near, visited, works_with, says, departures_from, arrives_at, implemented,
proposes, supports. Ninguna está en las 16 relaciones válidas. El OntologyValidator
las rechaza correctamente, pero el modelo no debería generarlas.

**5. Fact accuracy vs expectations baja (0-24%)** — Los facts se persisten (drop rate
bajo) pero no matchean lo que los test scenarios esperan. El modelo genera facts con
wording o granularidad diferente. Ejemplo: el test espera "Sueldo: 4.000.000 ARS/mes,
monotributista categoría H" pero el modelo genera "El usuario gana 4 millones".

**6. Relaciones 0% en varios casos** — finanzas, libro y viajes tienen 0% de relation
accuracy. El modelo no conecta entidades entre sí en estos dominios.

## Lo que cambia en v3

### 1. SFT DESDE BASE — no continuación del v2
Qwen3.5-9B base limpio. No cargar LoRA de v2. Train fresco.
Razón: v2 tiene sesgos acumulados (alucinaba entities de código en conversaciones
de finanzas). Más limpio empezar de cero con un dataset que cubre todos los dominios.

### 2. Solo S1 — NO incluir S3
S3 (context assembly) lo hacemos con template code en Python, no con LLM. El formato
comprimido de S3 es determinístico (XML tags, token budget). No vale la pena
fine-tunear para eso. Enfoquemos todo el budget de training en S1 que es donde
el modelo necesita aprender.

### 3. Fixes específicos para JSON discipline

Los ejemplos deben cubrir estos casos explícitamente:
- **existing_nodes presentes** → el modelo DEBE usar existing_id para nodos del grafo,
  NUNCA inventar IDs nuevos para entidades que ya existen
- **Facts SIEMPRE con entity_id válido** → si no hay entity para un fact, crear entity primero
- **Conversación ambigua o smalltalk** → output JSON vacío (arrays vacíos), NUNCA texto libre
- **Mezcla español/inglés** → output siempre en el schema correcto
- **Números/precios/fechas** → SIEMPRE son facts, nunca ignorarlos

## System Prompt exacto de S1 (usado en producción)

```
You are a knowledge graph extractor. You receive a conversation turn and existing graph context, and output ONLY a JSON object. No explanation, no markdown, no preamble.

## Your task

Analyze the conversation and extract:
1. The topic status relative to the previous turn
2. New or updated entities mentioned IN THE CONVERSATION
3. Relations between entities
4. New facts about existing entities
5. The user's intent and retrieval needs

## Output schema

{
  "topic": {
    "action": "same | subtopic | changed",
    "label": "short topic label"
  },
  "intent": {
    "type": "overview | specific | followup | chat",
    "retrieval": "summary_only | with_chunks"
  },
  "entities": [
    {
      "id": "lowercase_snake_case",
      "label": "Display Name",
      "type": "person | organization | project | technology | place | event | document | concept",
      "layer": "PERSONAL | UNIVERSAL",
      "description": "One sentence description",
      "existing_id": null
    }
  ],
  "relations": [
    {
      "source": "entity_id",
      "target": "entity_id",
      "relation": "part_of | created_by | maintains | works_at | member_of | uses_technology | depends_on | alternative_to | located_in | deployed_on | produces | serves | documented_in | participated_in | triggered_by | resulted_in"
    }
  ],
  "facts": [
    {
      "entity_id": "existing_entity_id",
      "text": "The specific fact",
      "speaker": "user | assistant"
    }
  ]
}

## CRITICAL RULES

### Entity rules
- Extract ONLY entities that are explicitly mentioned in the current conversation turn
- DO NOT invent, guess, or hallucinate entities that are not in the text
- DO NOT generate entities from your training data or general knowledge
- ONLY use these types: person, organization, project, technology, place, event, document, concept
- If unsure between types, use "concept"
- Use "existing_id" when the entity already exists in the provided graph context — do NOT create duplicates
- PERSONAL = user owns it, created it, works on it, or it's specific to their life/work
- UNIVERSAL = public knowledge (programming languages, cities, famous people, general concepts)
- Generate "id" as lowercase_snake_case derived from the label
- Do NOT extract the user or assistant themselves as entities

### Fact rules — READ CAREFULLY
- Facts are specific data points, numbers, dates, amounts, decisions, or status updates mentioned in the conversation
- EVERY fact MUST have a valid entity_id — NEVER return entity_id as null, empty, or missing
- If a fact doesn't clearly belong to an existing entity, create a new entity for it first, then attach the fact
- Numeric data is ALWAYS a fact: prices, salaries, dates, percentages, quantities, measurements
- Decisions are facts: "decidimos ir con X", "elegimos Y", "compramos Z"
- Progress updates are facts: "ya tenemos X", "faltan Y", "llevamos Z gastados"
- Keep facts concise but include the specific numbers
- Do NOT duplicate facts already present in the graph context
- Attribute to "user" or "assistant" based on who stated it

### Relation rules
- ONLY use the 16 relations listed in the schema above
- If no listed relation fits, do NOT create the relation — skip it entirely
- Common WRONG relations that you must NOT use: near, visited, has, needs, says, uses, owns, supports, proposes, implemented, works_with, has_document, departures_from, arrives_at
- Source and target must both be entity IDs (either new or existing)
- Do NOT create self-referential relations

### When to return empty arrays
- Greetings, small talk, meta-questions about the AI → entities: [], relations: [], facts: []
- Questions that only READ from the graph without adding info → entities: [], relations: [], facts: []
- Do NOT hallucinate entities to fill the output — empty is better than wrong
```

## User message format (cómo se construye el input al modelo)

El pipeline arma el user message así:
```
EXISTING NODES:
[{"id":"cipolletti","label":"Cipolletti","type":"Place","facts":["Zona de búsqueda de terrenos"]},...]

TOPIC HINT: same (high confidence from keyword match)
CURRENT TOPIC: compra terreno

PREVIOUS ASSISTANT: Entendido, lo tengo registrado.
USER: El arquitecto Carlos Peña nos pasó un presupuesto de 180.000 USD llave en mano
```

Es CRÍTICO que los ejemplos de training incluyan EXISTING NODES con nodos reales
del grafo, porque el modelo debe aprender a:
1. Referenciar nodos existentes con existing_id
2. No duplicar entidades que ya están en el grafo
3. Adjuntar facts a entidades existentes (no crear entidades nuevas para cada fact)

## Schema S1

```json
{
  "topic": {"action": "same|changed|subtopic", "label": "string"},
  "intent": {"type": "overview|specific|chat|followup", "retrieval": "summary_only|with_chunks"},
  "entities": [
    {
      "id": "snake_case_max_3_words",
      "label": "Nombre legible",
      "type": "person|organization|project|technology|place|event|document|concept",
      "layer": "PERSONAL|UNIVERSAL",
      "description": "One sentence",
      "existing_id": "id_del_nodo_si_ya_existe_o_null"
    }
  ],
  "relations": [
    {
      "source": "entity_id",
      "target": "entity_id",
      "relation": "uses_technology|part_of|maintains|works_at|member_of|depends_on|alternative_to|located_in|deployed_on|produces|serves|documented_in|participated_in|triggered_by|resulted_in|created_by"
    }
  ],
  "facts": [
    {"entity_id": "entity_id", "text": "hecho concreto con números", "speaker": "user|assistant"}
  ]
}
```

## Distribución objetivo del dataset

Total: ~600 ejemplos S1

### Por tipo de habilidad (lo que el modelo debe aprender)
- 25% — **JSON discipline + existing_id**: EXISTING NODES presentes, modelo DEBE usar existing_id y adjuntar facts a nodos existentes. Incluir casos donde el modelo en v2 fallaba (entity_id=null, entity inventada).
- 20% — **Facts numéricos**: precios, salarios, fechas, porcentajes, cantidades. El fact SIEMPRE con entity_id válido y el número textual exacto.
- 15% — **Empty output correcto**: smalltalk, preguntas sin entidades, saludos. JSON válido con arrays vacíos.
- 15% — **Relations correctas**: SOLO las 16 válidas. Incluir 20+ ejemplos negativos donde la relación natural sería "near"/"has"/"uses" pero el modelo debe elegir la válida más cercana o no crear relación.
- 10% — **Topic changes y subtopic**: cambio de tema claro, drill-down en subtema, continuación.
- 10% — **Dedup y correcciones**: "cambiamos X por Y", "en realidad era Z no W". El modelo debe actualizar facts, no duplicar.
- 5% — **Multi-entidad compleja**: 3+ entidades con relaciones cruzadas en un solo turno.

### Por dominio (diversidad de contenido)
- Software/código (25%): proyectos, tech stacks, arquitectura, deploys
- Personal/finanzas (25%): sueldos, inversiones, presupuestos, gastos
- Construcción/inmobiliaria (15%): terrenos, presupuestos, obras, materiales
- Literatura/cultura (10%): libros, personajes, autores, temas
- Salud/familia (10%): médicos, tratamientos, turnos, seguros
- Viajes (10%): itinerarios, vuelos, alojamiento, presupuesto
- Trabajo/empresa (5%): equipos, clientes, proyectos, deadlines

### Por idioma
- 50% español (argentino — voseo, ARS, localidades argentinas)
- 50% inglés

## Archivos existentes a leer primero

Antes de planificar, leé:
- `01_dataset/schema.py` — schemas Pydantic actuales y system prompts
- `01_dataset/generate_s1_training.py` — generador v2 para entender el formato
- `training_data/` — si hay JSONL existentes, revisar la estructura de ejemplos

## Lo que quiero en tu respuesta (PLAN SOLAMENTE)

1. **Estructura de archivos** que vas a crear/modificar
2. **El system prompt** exacto que va en cada ejemplo (debe ser el de arriba, NO inventar otro)
3. **Detalle de cada grupo** del dataset: cuántos ejemplos, qué casos cubre,
   2-3 ejemplos representativos por grupo (no todos, solo para validar el approach)
4. **Estrategia para JSON discipline**: cómo vas a asegurar que el modelo aprenda
   a usar existing_id y nunca genere entity_id=null en facts
5. **Estrategia para relaciones**: cómo vas a enseñar las 16 válidas y que NO use las inválidas
6. **Validación del dataset**: qué checks automáticos vas a correr sobre los ejemplos
   generados antes de guardar el JSONL

NO generes el dataset todavía. Solo el plan con los ejemplos representativos.
Cuando apruebe el plan, ahí generamos todo.
