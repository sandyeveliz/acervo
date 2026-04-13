"""LLM prompt for edge (fact) duplicate + contradiction detection.

Adapted from Graphiti (Apache-2.0, Zep Software). See acervo/THIRD_PARTY.md.

Upstream: graphiti_core/prompts/dedupe_edges.py::resolve_edge

Differences from upstream:
    - Spanish-first instructions (Acervo runs with qwen3.5:9b against
      mostly Spanish content).
    - Returns a plain dict shaped like ``EdgeDuplicate`` (pydantic
      schema validated post-parse by edge_resolution.py).
    - Continuous indexing across the two candidate lists is identical
      to Graphiti so the arbitration logic downstream is unchanged.
"""

from __future__ import annotations

from typing import Any

_SYSTEM = (
    "Sos un asistente de deduplicación y detección de contradicciones de hechos. "
    "NUNCA marques hechos como duplicados si tienen diferencias claves (números, "
    "fechas, o calificadores distintos)."
)


def build_resolve_edge_messages(context: dict[str, Any]) -> list[dict[str, str]]:
    """Build the chat messages for edge resolution.

    ``context`` expects:
        - ``existing_edges``: list[{"idx": int, "fact": str}]  (idx 0..M-1)
        - ``edge_invalidation_candidates``: list[{"idx": int, "fact": str}]
          (idx M..M+N-1, continuous numbering)
        - ``new_edge``: str (the new fact text)
    """
    existing = context.get("existing_edges", [])
    invalidation = context.get("edge_invalidation_candidates", [])
    new_edge = context.get("new_edge", "")

    user = f"""
NUNCA marques hechos como duplicados si tienen diferencias claves, particularmente en valores numéricos, fechas o calificadores.

Restricciones IMPORTANTES:
- duplicate_facts: SÓLO valores idx de EXISTING FACTS (NUNCA incluyas FACT INVALIDATION CANDIDATES)
- contradicted_facts: idx de cualquiera de las dos listas (EXISTING FACTS o FACT INVALIDATION CANDIDATES)
- Los idx son continuos entre ambas listas (INVALIDATION CANDIDATES arranca donde termina EXISTING FACTS)

<EXISTING FACTS>
{existing}
</EXISTING FACTS>

<FACT INVALIDATION CANDIDATES>
{invalidation}
</FACT INVALIDATION CANDIDATES>

<NEW FACT>
{new_edge}
</NEW FACT>

Vas a recibir DOS listas de hechos con idx CONTINUO entre ambas.
Primero EXISTING FACTS, después FACT INVALIDATION CANDIDATES.

1. DETECCIÓN DE DUPLICADOS:
   - Si el NEW FACT expresa exactamente la misma información fáctica que algún hecho de EXISTING FACTS, devolvé esos idx en duplicate_facts.
   - Si no hay duplicados, devolvé lista vacía.

2. DETECCIÓN DE CONTRADICCIONES:
   - Determiná qué hechos contradice el NEW FACT (de cualquiera de las dos listas).
   - Un hecho de EXISTING FACTS puede ser duplicado Y contradictorio a la vez (ej: semánticamente igual pero el nuevo actualiza/reemplaza).
   - Devolvé todos los idx contradichos en contradicted_facts.
   - Si no hay contradicciones, devolvé lista vacía.

Tu respuesta DEBE ser un objeto JSON con exactamente dos campos:
{{"duplicate_facts": [int, ...], "contradicted_facts": [int, ...]}}

<EJEMPLO>
EXISTING FACT: idx=0, "Alice se sumó a Acme Corp en 2020"
NEW FACT: "Alice se sumó a Acme Corp en 2020"
Resultado: {{"duplicate_facts": [0], "contradicted_facts": []}}  (información idéntica)

EXISTING FACT: idx=1, "Alice trabaja en Acme Corp como ingeniera"
NEW FACT: "Alice trabaja en Acme Corp como senior engineer"
Resultado: {{"duplicate_facts": [], "contradicted_facts": [1]}}  (misma relación, título actualizado)

EXISTING FACT: idx=2, "Bob corrió 5km el martes"
NEW FACT: "Bob corrió 3km el miércoles"
Resultado: {{"duplicate_facts": [], "contradicted_facts": []}}  (eventos distintos en días distintos)

EXISTING FACT: idx=0, "Sandy vive en Cipolletti"
INVALIDATION CANDIDATE: idx=1, "Sandy vive en General Roca"
NEW FACT: "Sandy se mudó a Neuquén en marzo de 2026"
Resultado: {{"duplicate_facts": [], "contradicted_facts": [0, 1]}}  (nueva ubicación contradice ambas)
</EJEMPLO>

Respondé SOLAMENTE con el objeto JSON. Sin markdown, sin explicación, sin texto adicional.
"""

    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user.strip()},
    ]


__all__ = ["build_resolve_edge_messages"]
