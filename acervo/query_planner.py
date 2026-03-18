"""Query Planner — uses LLM to decide what information is needed before responding.

Decides which tool to use: GRAPH_ALL, GRAPH_SEARCH, WEB_SEARCH, or READY.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from acervo.llm import LLMClient
from acervo._text import strip_think_blocks

log = logging.getLogger(__name__)

_PLANNER_PROMPT = """Sos un planificador de búsqueda. Analizá la pregunta y decidí qué herramienta usar.

Pregunta: {user_message}
Entidad principal: {entity_name} ({entity_type})
Hechos disponibles: {facts_summary}

Herramientas:
- GRAPH_ALL: traer todos los hechos y conexiones de una entidad del grafo local
- GRAPH_SEARCH: buscar nodos relacionados por tipo o keyword
- WEB_SEARCH: buscar en internet
- READY: no necesita buscar nada

REGLAS DE PRIORIDAD (seguir en orden):
1. Si "Hechos disponibles" tiene datos sobre la entidad → usá GRAPH_ALL
2. Si el usuario dice "buscá", "googleá", "internet" → usá WEB_SEARCH
3. Si no hay hechos disponibles y el usuario pregunta sobre algo → usá WEB_SEARCH
4. Si es saludo o pregunta sin tema ("hola", "cómo estás") → usá READY

Respondé SOLO con un JSON, sin explicación:
{{"tool": "NOMBRE", "entity": "nombre_entidad", "query": "texto de búsqueda"}}

Ejemplos:
- "qué sabés de Batman?" (hechos: "es un superhéroe de DC") → {{"tool": "GRAPH_ALL", "entity": "Batman", "query": ""}}
- "qué sabés de Cipolletti?" (hechos: "Sandy vive en Cipolletti") → {{"tool": "GRAPH_ALL", "entity": "Cipolletti", "query": ""}}
- "qué es X?" (hechos: ninguno) → {{"tool": "WEB_SEARCH", "entity": "X", "query": "X"}}
- "buscá en internet sobre X" → {{"tool": "WEB_SEARCH", "entity": "X", "query": "X"}}
- "hola" → {{"tool": "READY", "entity": "", "query": ""}}
JSON:"""


@dataclass
class PlanResult:
    tool: str  # GRAPH_ALL, GRAPH_SEARCH, WEB_SEARCH, READY
    entity: str
    query: str

    VALID_TOOLS = frozenset({"GRAPH_ALL", "GRAPH_SEARCH", "WEB_SEARCH", "READY"})


class QueryPlanner:
    """Uses LLM to plan what information to retrieve before responding."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def plan(
        self,
        user_message: str,
        entity_name: str,
        entity_type: str,
        facts_summary: str,
    ) -> PlanResult:
        """Ask the LLM what tool to use. Returns PlanResult."""
        try:
            return await self._call_llm(
                user_message, entity_name, entity_type, facts_summary,
            )
        except Exception as e:
            log.warning("Planner failed, falling back to GRAPH_ALL: %s", e)
            return PlanResult(
                tool="GRAPH_ALL",
                entity=entity_name or "",
                query="",
            )

    async def _call_llm(
        self,
        user_message: str,
        entity_name: str,
        entity_type: str,
        facts_summary: str,
    ) -> PlanResult:
        prompt = _PLANNER_PROMPT.format(
            user_message=user_message[:300],
            entity_name=entity_name or "ninguna",
            entity_type=entity_type or "desconocido",
            facts_summary=facts_summary[:500] if facts_summary else "ninguno",
        )

        raw_response = await self._llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )

        raw = strip_think_blocks(raw_response).strip()
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip()

        return self._parse(raw, entity_name)

    @staticmethod
    def _parse(raw: str, fallback_entity: str) -> PlanResult:
        """Parse JSON response from LLM. Fallback to GRAPH_ALL on failure."""
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            log.warning("Planner: no JSON found in response: %s", raw[:100])
            return PlanResult(tool="GRAPH_ALL", entity=fallback_entity, query="")

        try:
            data = json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            log.warning("Planner: invalid JSON: %s", raw[:100])
            return PlanResult(tool="GRAPH_ALL", entity=fallback_entity, query="")

        tool = str(data.get("tool", "GRAPH_ALL")).upper()
        if tool not in PlanResult.VALID_TOOLS:
            tool = "GRAPH_ALL"

        return PlanResult(
            tool=tool,
            entity=str(data.get("entity", fallback_entity or "")),
            query=str(data.get("query", "")),
        )
