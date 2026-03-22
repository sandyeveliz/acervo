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

_PLANNER_PROMPT = """You are a search planner. Analyze the question and decide which tool to use.

Question: {user_message}
Main entity: {entity_name} ({entity_type})
Available facts: {facts_summary}

Tools:
- GRAPH_ALL: retrieve all facts and connections for an entity from the local graph
- GRAPH_SEARCH: search related nodes by type or keyword
- WEB_SEARCH: search the internet
- READY: no search needed

PRIORITY RULES (follow in order):
1. If "Available facts" has data about the entity → use GRAPH_ALL
2. If the user says "search", "google", "look up", "internet" → use WEB_SEARCH
3. If no facts are available and the user asks about something → use WEB_SEARCH
4. If it's a greeting or question without a topic ("hello", "how are you") → use READY

Respond ONLY with a JSON, no explanation:
{{"tool": "NAME", "entity": "entity_name", "query": "search text"}}

Examples:
- "what do you know about Batman?" (facts: "is a DC superhero") → {{"tool": "GRAPH_ALL", "entity": "Batman", "query": ""}}
- "what do you know about Cipolletti?" (facts: "Sandy lives in Cipolletti") → {{"tool": "GRAPH_ALL", "entity": "Cipolletti", "query": ""}}
- "what is X?" (facts: none) → {{"tool": "WEB_SEARCH", "entity": "X", "query": "X"}}
- "search the internet for X" → {{"tool": "WEB_SEARCH", "entity": "X", "query": "X"}}
- "hello" → {{"tool": "READY", "entity": "", "query": ""}}
JSON:"""


@dataclass
class PlanResult:
    tool: str  # GRAPH_ALL, GRAPH_SEARCH, WEB_SEARCH, READY
    entity: str
    query: str

    VALID_TOOLS = frozenset({"GRAPH_ALL", "GRAPH_SEARCH", "WEB_SEARCH", "READY"})


class QueryPlanner:
    """Uses LLM to plan what information to retrieve before responding."""

    def __init__(self, llm: LLMClient, prompt_template: str | None = None) -> None:
        self._llm = llm
        self._prompt = prompt_template or _PLANNER_PROMPT

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
        prompt = self._prompt.format(
            user_message=user_message[:300],
            entity_name=entity_name or "none",
            entity_type=entity_type or "unknown",
            facts_summary=facts_summary[:500] if facts_summary else "none",
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
