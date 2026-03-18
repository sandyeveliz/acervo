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

_CONVERSATION_PROMPT = """Extraer entidades, relaciones semánticas y hechos de esta conversación.

REGLAS:
- Cada hecho debe ser una afirmación concreta SOBRE la entidad.
- Cada hecho lleva "speaker": "user" o "assistant" según quién lo dijo.
- NO agregar conocimiento general. Solo lo explícitamente dicho.
- Extraer relaciones de categoría y jerarquía cuando apliquen.

Tipos de entidad: lugar, persona, personaje, organizacion, universo, editorial, tecnologia, obra, actividad

Tipos de relación:
- is_a: clasificación (Batman is_a personaje, Gotham is_a lugar)
- created_by: creador (Batman created_by Bill Finger)
- alias_of: identidad alternativa (Batman alias_of Bruce Wayne)
- part_of: pertenencia a universo/grupo (Batman part_of DC Universe)
- set_in: ubicación narrativa (Batman set_in Gotham City)
- debuted_in: primera aparición (Batman debuted_in Detective Comics)
- published_by: editorial (Detective Comics published_by DC Comics)
- ubicado_en, parte_de, pertenece_a, relacionado_con: relaciones generales

Ejemplo:
User: Batman fue creado por Bill Finger en 1939
Assistant: Batman es un personaje de DC Comics, su nombre real es Bruce Wayne.
JSON: {{"entities":[{{"name":"Batman","type":"personaje"}},{{"name":"Bill Finger","type":"persona"}},{{"name":"DC Universe","type":"universo"}},{{"name":"Bruce Wayne","type":"persona"}},{{"name":"Gotham City","type":"lugar"}}],"relations":[{{"source":"Batman","target":"Bill Finger","relation":"created_by"}},{{"source":"Batman","target":"DC Universe","relation":"part_of"}},{{"source":"Batman","target":"Bruce Wayne","relation":"alias_of"}}],"facts":[{{"entity":"Batman","fact":"Fue creado en 1939","speaker":"user"}},{{"entity":"Batman","fact":"Es un personaje de DC Comics","speaker":"assistant"}}]}}

User: {user_msg}
Assistant: {assistant_msg}
JSON:"""

_SEARCH_PROMPT = """Extraer entidades, relaciones y hechos verificables de estos resultados de búsqueda sobre "{query}".

Responder en JSON con "entities", "relations" y "facts".
- entities: lista de {{"name": "...", "type": "..."}} (tipos: lugar, persona, personaje, organizacion, universo, editorial, obra)
- relations: lista de {{"source": "...", "target": "...", "relation": "..."}} (relaciones: is_a, created_by, alias_of, part_of, set_in, debuted_in, published_by, ubicado_en, parte_de)
- facts: lista de {{"entity": "...", "fact": "..."}}

Solo incluir datos explícitos del texto. NO inventar.

Resultados:
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
    "lugar", "persona", "personaje", "entidad", "actividad",
    "organizacion", "organización", "tecnologia", "tecnología", "obra",
    "universo", "editorial", "comic",
))
VALID_RELATIONS = frozenset((
    # Universal semantic relations
    "is_a", "created_by", "alias_of", "part_of", "set_in",
    "debuted_in", "published_by",
    # Domain relations
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
            if src and tgt and rel and len(rel) >= 3:
                if rel not in VALID_RELATIONS:
                    from acervo.ontology import register_relation
                    register_relation(rel)
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
    """Extracts entities, relations, and facts from web search results."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def extract(self, query: str, search_text: str) -> ExtractionResult:
        """Extract structured knowledge from web search results.

        Returns ExtractionResult with entities, relations, and facts.
        """
        try:
            return await self._call_llm(query, search_text)
        except Exception as e:
            log.warning("Search extraction failed: %s", e)
            return ExtractionResult()

    async def _call_llm(self, query: str, search_text: str) -> ExtractionResult:
        prompt = _SEARCH_PROMPT.format(
            query=query,
            text=search_text[:2000],
        )

        raw_response = await self._llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=600,
        )

        raw = _clean_response(raw_response)

        # Try parsing as object first (new format with entities/relations/facts)
        obj = _parse_first_json(raw, "object")
        if isinstance(obj, dict) and ("entities" in obj or "facts" in obj):
            return self._parse_object(obj)

        # Fallback: parse as array (old format with just entity/fact pairs)
        arr = _parse_first_json(raw, "array")
        if isinstance(arr, list):
            result = ExtractionResult()
            for item in arr:
                if not isinstance(item, dict):
                    continue
                entity = str(item.get("entity", "")).strip()
                fact = str(item.get("fact", "")).strip()
                if entity and fact and len(entity) > 3:
                    name_clean = entity.lower().strip("!?.,;:\"'()[]")
                    if name_clean not in _ENTITY_BLACKLIST:
                        result.facts.append(ExtractedFact(
                            entity=entity, fact=fact, source="web",
                        ))
            return result

        return ExtractionResult()

    def _parse_object(self, obj: dict) -> ExtractionResult:
        """Parse structured JSON with entities, relations, and facts."""
        result = ExtractionResult()

        for item in obj.get("entities", []):
            e = ConversationExtractor._parse_entity(item)
            if e:
                result.entities.append(e)

        for item in obj.get("relations", []):
            if not isinstance(item, dict):
                continue
            src = str(item.get("source", "")).strip()
            tgt = str(item.get("target", "")).strip()
            rel = str(item.get("relation", "")).strip().lower()
            if src and tgt and rel and len(rel) >= 3:
                if rel not in VALID_RELATIONS:
                    from acervo.ontology import register_relation
                    register_relation(rel)
                result.relations.append(Relation(source=src, target=tgt, relation=rel))

        for item in obj.get("facts", []):
            if not isinstance(item, dict):
                continue
            entity = str(item.get("entity", "")).strip()
            fact = str(item.get("fact", "")).strip()
            if entity and fact and len(entity) > 3:
                name_clean = entity.lower().strip("!?.,;:\"'()[]")
                if name_clean not in _ENTITY_BLACKLIST:
                    result.facts.append(ExtractedFact(
                        entity=entity, fact=fact, source="web",
                    ))

        return result


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
