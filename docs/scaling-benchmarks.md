# Scaling Guide: Cómo generar casos de 200-500 turnos para Acervo

## Por qué los casos de 100+ turnos son distintos

El beneficio de Acervo escala superlinealmente con la cantidad de turnos:

| Turnos | Tokens brutos (sin Acervo) | Tokens con Acervo | Ratio |
|--------|---------------------------|-------------------|-------|
| 10     | ~1,000                    | ~600              | 1.7x  |
| 50     | ~9,000                    | ~800              | 11x   |
| 100    | ~20,000                   | ~900              | 22x   |
| 500    | ~100,000+                 | ~1,000            | 100x+ |

## Estrategia para generar 500 turnos

Los casos de 100+ turnos NO son 5x el mismo contenido.
Tienen un patron especifico de crecimiento del grafo:

### Fase 1 — Construccion del grafo (turnos 1-30)
Entidades nuevas en casi cada turno. El grafo crece rapido.
Tokens de Acervo suben levemente hasta que el grafo se estabiliza.

### Fase 2 — Enriquecimiento (turnos 31-100)
Las entidades ya existen; se agregan facts y relaciones.
El grafo crece mas lento. Los tokens de Acervo se estabilizan.
CLAVE PARA TEST: el recall de entidades de la fase 1 se activa.

### Fase 3 — Recall dominante (turnos 101-500)
Casi no hay entidades nuevas. El usuario trabaja con el mismo
contexto desde hace tiempo. Cada turno es recall + nuevo fact.
Tokens brutos siguen creciendo; tokens de Acervo = CONSTANTE.

## Script para generar turnos adicionales

```python
# generate_long_turns.py
# Toma un YAML existente y agrega N turnos más con patron de Fase 3

import yaml
import random

RECALL_PATTERNS = [
    # El usuario menciona algo de los primeros turnos
    ("user_msg", "oye me acorde que {entity_early} {fact_early}. sigue siendo asi?"),
    # Correcciones menores
    ("user_msg", "una correccion: {field} de {entity} no es {old_value} sino {new_value}"),
    # Preguntas de resumen
    ("user_msg", "recapitulame todo sobre {entity}"),
    # Nuevos facts sobre entidades existentes
    ("user_msg", "{entity} {new_fact}"),
    # Cross-references
    ("user_msg", "la relacion entre {entity_a} y {entity_b} cambio: {change}"),
    # Small talk que NO crea entidades
    ("user_msg", "uf que dia"),
    ("user_msg", "bien, a seguir"),
    ("user_msg", "perfecto"),
]

def generate_phase3_turns(base_yaml_path, n_turns=400):
    '''
    Genera turnos de Fase 3 (recall dominante) para testing de escalabilidad.
    Usar como: python generate_long_turns.py 05_saas_founder_100turns.yaml 400
    '''
    with open(base_yaml_path) as f:
        data = yaml.safe_load(f)
    
    entities = extract_entities(data['turns'])
    
    new_turns = []
    for i in range(n_turns):
        turn_type = weighted_choice([
            ('recall', 0.35),
            ('new_fact', 0.25),
            ('small_talk', 0.20),
            ('correction', 0.10),
            ('cross_reference', 0.10),
        ])
        new_turns.append(generate_turn(turn_type, entities, i))
    
    data['turns'].extend(new_turns)
    data['scaling_generated'] = f'{n_turns} turnos adicionales de Fase 3'
    
    output = base_yaml_path.replace('.yaml', f'_500turns.yaml')
    with open(output, 'w') as f:
        yaml.dump(data, f, allow_unicode=True)
    
    print(f"Generado: {len(data['turns'])} turnos totales -> {output}")
```

## Metricas de benchmark recomendadas

Para un reporte con y sin Acervo, medir en cada turno:

```python
metrics = {
    # Token usage
    "input_tokens_raw": "todos los mensajes del historial",
    "input_tokens_acervo": "contexto comprimido del grafo",
    "compression_ratio": "raw / acervo",
    
    # Recall accuracy (solo en turnos con 'checkpoint')
    "context_hit_rate": "menciono la entidad esperada?",
    "context_precision": "cuanto del contexto fue relevante?",
    
    # Graph stats
    "graph_nodes": "entidades unicas en el grafo",
    "graph_edges": "relaciones unicas",
    "graph_facts": "facts almacenados",
    "hot_layer_tokens": "tokens de la capa activa",
}
```

## Checkpoints en los YAML

Busca todos los turnos con `checkpoint:` en los YAMLs — son los puntos donde
el test verifica que Acervo recupero el contexto correcto.

Total de checkpoints en los casos existentes:
- 05_saas_founder_100turns.yaml: 11 checkpoints
- 06_product_manager_real.yaml:  8 checkpoints

Para validar recall: en esos turnos, verificar que el contexto inyectado
por Acervo contiene las entidades listadas en `context_should_mention`.