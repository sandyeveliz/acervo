"""Extractors — only register what already happened, never generate.

Three specialized extractors by source:
- ConversationExtractor: explicit user/assistant statements (source: user)
- SearchExtractor: facts from web search results (source: web) — future
- RAGExtractor: facts from RAG retrieval (source: rag) — future

Principle: an empty node is better than a node with unverified data.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from acervo.llm import LLMClient
from acervo._text import strip_think_blocks
from acervo.ontology import map_extractor_type

log = logging.getLogger(__name__)


# ── Data types ──

@dataclass
class Entity:
    name: str
    type: str

@dataclass
class Relation:
    source: str
    target: str
    relation: str

@dataclass
class ExtractedFact:
    entity: str
    fact: str
    source: str    # "user", "web", "rag"
    speaker: str = "user"  # "user" | "assistant"

@dataclass
class ExtractionResult:
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    facts: list[ExtractedFact] = field(default_factory=list)


# ── Prompts ──

_CONVERSATION_PROMPT = """Extraer entidades, relaciones y hechos concretos de esta conversación.
Cada hecho debe ser una afirmación concreta SOBRE la entidad, no sobre la acción del usuario.
INCORRECTO: "el usuario menciona que es su ciudad" — eso describe al usuario, no a la entidad.
CORRECTO: "Sandy vive en Cipolletti" o "juega en el Torneo Federal A" — hechos sobre la entidad.
Cada hecho lleva "speaker": "user" o "assistant" según quién lo dijo.
NO agregar conocimiento general. Solo lo explícitamente dicho.
Responder en JSON con "entities", "relations" y "facts". Responder siempre en español.

Tipos de entidad: lugar, persona, organizacion, tecnologia, obra, actividad
"obra" es para libros, películas, series, sagas, videojuegos, etc.
"organizacion" es para empresas, clubes, instituciones.
"tecnologia" es para lenguajes, frameworks, herramientas.
Tipos de relación: ubicado_en, tecnico_de, dirigido_por, parte_de, hincha_de, juega_en, jugó_contra, ganó_a, perdió_contra, pertenece_a, relacionado_con
IMPORTANTE: "le ganó a X" es ganó_a, NO juega_en. "juega_en" es para torneos/ligas, no para partidos contra otro equipo.

Ejemplo:
User: soy hincha de River y vivo en Cipolletti
Assistant: Cipolletti queda en Rio Negro.
JSON: {{"entities":[{{"name":"River Plate","type":"entidad"}},{{"name":"Cipolletti","type":"lugar"}},{{"name":"Rio Negro","type":"lugar"}}],"relations":[{{"source":"Cipolletti","target":"Rio Negro","relation":"ubicado_en"}}],"facts":[{{"entity":"River Plate","fact":"Sandy es hincha de River Plate","speaker":"user"}},{{"entity":"Cipolletti","fact":"Sandy vive en Cipolletti","speaker":"user"}},{{"entity":"Cipolletti","fact":"está en Rio Negro","speaker":"assistant"}}]}}

User: {user_msg}
Assistant: {assistant_msg}
JSON:"""

_SEARCH_PROMPT = """Extract verifiable facts from this search result about "{query}".
Return a JSON array of objects with "entity" and "fact".
Only include facts directly stated in the text. Do NOT infer or add knowledge.
Responder siempre en español.

Search result:
{text}

JSON:"""


# ── Shared JSON parsing ──

def _parse_first_json(text: str, target: str = "object") -> dict | list | None:
    """Find and parse the first JSON object or array in text."""
    open_char = "{" if target == "object" else "["
    close_char = "}" if target == "object" else "]"
    start = text.find(open_char)
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == open_char:
            depth += 1
        elif text[i] == close_char:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _clean_response(content: str) -> str:
    """Strip think blocks and code fences from LLM response."""
    raw = strip_think_blocks(content).strip()
    return re.sub(r"```(?:json)?\s*", "", raw).strip()


# ── Extractors ──

VALID_TYPES = frozenset((
    "lugar", "persona", "entidad", "actividad",
    "organizacion", "organización", "tecnologia", "tecnología", "obra",
))
VALID_RELATIONS = frozenset((
    "ubicado_en", "tecnico_de", "parte_de", "hincha_de",
    "juega_en", "pertenece_a", "relacionado_con", "co_mentioned",
    "jugó_contra", "dirigido_por", "ganó_a", "perdió_contra",
))

# Entities that are noise — roles, pronouns, generic terms, common nouns
_ENTITY_BLACKLIST = frozenset({
    # Roles and pronouns
    "user", "usuario", "assistant", "asistente", "bot", "ia", "ai",
    "yo", "tu", "el", "ella", "nosotros", "ustedes",
    # Temporal
    "hoy", "ayer", "mañana", "ahora", "antes", "después",
    # Meta terms
    "persona", "lugar", "entidad", "actividad", "obra",
    "conversación", "conversacion", "chat", "sesión", "sesion",
    "mensaje", "pregunta", "respuesta", "tema", "topic",
    "información", "informacion", "dato", "datos",
    # Greetings
    "hola", "chau", "adiós", "adios", "gracias", "por favor",
    "buenos días", "buenas tardes", "buenas noches",
    # Geographic generics
    "provincia", "ciudad", "país", "pais", "club", "equipo",
    # Common nouns that should never be entities
    "libro", "libros", "película", "películas", "pelicula", "peliculas",
    "novela", "novelas", "serie", "series", "historia", "historias",
    "personaje", "personajes", "autor", "autora", "escritor", "escritora",
    "mundo", "vida", "muerte", "año", "años", "tiempo", "parte", "partes",
    "tipo", "tipos", "forma", "formas", "cosa", "cosas", "gente",
    "rata", "cueva", "bat", "pino", "casa", "familia",
    "cultura", "popular", "memoria", "internet", "web",
    # Numbers as words
    "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho",
    "nueve", "diez", "cien", "mil",
})


class ConversationExtractor:
    """Extracts entities, relations, and facts from user/assistant dialogue.

    Only registers explicit statements. If the user didn't state a fact,
    it doesn't get recorded. Empty facts is the correct result for most turns.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def extract(self, user_msg: str, assistant_msg: str) -> ExtractionResult:
        try:
            return await self._call_llm(user_msg, assistant_msg)
        except Exception as e:
            log.warning("Conversation extraction failed: %s", e)
            return ExtractionResult()

    async def _call_llm(self, user_msg: str, assistant_msg: str) -> ExtractionResult:
        prompt = _CONVERSATION_PROMPT.format(
            user_msg=user_msg[:500],
            assistant_msg=assistant_msg[:500],
        )

        raw_response = await self._llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500,
        )

        raw = _clean_response(raw_response)
        return self._parse(raw)

    def _parse(self, raw: str) -> ExtractionResult:
        result = ExtractionResult()

        obj = _parse_first_json(raw, "object")
        if not isinstance(obj, dict):
            arr = _parse_first_json(raw, "array")
            if isinstance(arr, list):
                for item in arr:
                    e = self._parse_entity(item)
                    if e:
                        result.entities.append(e)
            return result

        for item in obj.get("entities", []):
            e = self._parse_entity(item)
            if e:
                result.entities.append(e)

        for item in obj.get("relations", []):
            if not isinstance(item, dict):
                continue
            src = str(item.get("source", "")).strip()
            tgt = str(item.get("target", "")).strip()
            rel = str(item.get("relation", "")).strip().lower()
            if src and tgt and rel:
                if rel not in VALID_RELATIONS:
                    rel = "relacionado_con"
                result.relations.append(Relation(source=src, target=tgt, relation=rel))

        for item in obj.get("facts", []):
            if not isinstance(item, dict):
                continue
            entity = str(item.get("entity", "")).strip()
            fact = str(item.get("fact", "")).strip()
            speaker = str(item.get("speaker", "user")).strip().lower()
            if speaker not in ("user", "assistant"):
                speaker = "user"
            if entity and fact:
                result.facts.append(ExtractedFact(
                    entity=entity, fact=fact, source="user", speaker=speaker,
                ))

        return result

    @staticmethod
    def _parse_entity(item) -> Entity | None:
        if not isinstance(item, dict):
            return None
        name = str(item.get("name", "")).strip()
        raw_type = str(item.get("type", "")).strip().lower()
        if not name or raw_type not in VALID_TYPES:
            return None
        name_clean = name.lower().strip("!?.,;:\"'()[]")
        if name_clean in _ENTITY_BLACKLIST:
            return None
        if len(name) <= 3:
            return None
        # Map extractor type to ontology type
        ontology_type = map_extractor_type(raw_type)
        return Entity(name=name, type=ontology_type)


class SearchExtractor:
    """Extracts facts from web search results. Source: web."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def extract(self, query: str, search_text: str) -> list[ExtractedFact]:
        try:
            return await self._call_llm(query, search_text)
        except Exception as e:
            log.warning("Search extraction failed: %s", e)
            return []

    async def _call_llm(self, query: str, search_text: str) -> list[ExtractedFact]:
        prompt = _SEARCH_PROMPT.format(
            query=query,
            text=search_text[:1500],
        )

        raw_response = await self._llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=400,
        )

        raw = _clean_response(raw_response)
        arr = _parse_first_json(raw, "array")
        if not isinstance(arr, list):
            return []

        facts = []
        for item in arr:
            if not isinstance(item, dict):
                continue
            entity = str(item.get("entity", "")).strip()
            fact = str(item.get("fact", "")).strip()
            if entity and fact:
                facts.append(ExtractedFact(entity=entity, fact=fact, source="web"))

        return facts


class RAGExtractor:
    """Extracts facts from RAG retrieval results. Source: rag."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def extract(self, query: str, rag_text: str) -> list[ExtractedFact]:
        try:
            return await self._call_llm(query, rag_text)
        except Exception as e:
            log.warning("RAG extraction failed: %s", e)
            return []

    async def _call_llm(self, query: str, rag_text: str) -> list[ExtractedFact]:
        prompt = _SEARCH_PROMPT.format(
            query=query,
            text=rag_text[:1500],
        )

        raw_response = await self._llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=400,
        )

        raw = _clean_response(raw_response)
        arr = _parse_first_json(raw, "array")
        if not isinstance(arr, list):
            return []

        facts = []
        for item in arr:
            if not isinstance(item, dict):
                continue
            entity = str(item.get("entity", "")).strip()
            fact = str(item.get("fact", "")).strip()
            if entity and fact:
                facts.append(ExtractedFact(entity=entity, fact=fact, source="rag"))

        return facts
