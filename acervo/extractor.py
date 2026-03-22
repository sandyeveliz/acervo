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

_CONVERSATION_PROMPT = """You are an entity extractor. Analyze ONLY the conversation below.

CRITICAL RULES:
- Extract ONLY entities that appear in the CONVERSATION below.
- Do NOT invent entities. If a name does not appear in the text, do NOT include it.
- Each fact must be explicitly stated in the conversation.
- Each fact has "speaker": "user" or "assistant".
- The conversation may be in any language. Keep entity names in their original language.
- Common words, greetings, and verbs are NOT entities. Only extract proper nouns.
- If the conversation has no extractable entities, return {{"entities":[],"relations":[],"facts":[]}}

Entity types: place, person, character, organization, technology, work, project, document

Relation types: is_a, created_by, alias_of, part_of, located_in, belongs_to, related_to, works_at, lives_in, uses_technology, has_module, published_by

Output valid JSON only. No explanation.

CONVERSATION TO ANALYZE:
User: {user_msg}
Assistant: {assistant_msg}

JSON:"""

_SEARCH_PROMPT = """Extract entities, relations, and verifiable facts from these search results about "{query}".

Respond in JSON with "entities", "relations", and "facts".
- entities: list of {{"name": "...", "type": "..."}} (types: place, person, character, organization, universe, publisher, work, technology)
- relations: list of {{"source": "...", "target": "...", "relation": "..."}} (relations: is_a, created_by, alias_of, part_of, set_in, debuted_in, published_by, located_in, belongs_to)
- facts: list of {{"entity": "...", "fact": "..."}}

Only include data explicitly stated in the text. Do NOT invent.

Results:
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
    # English types
    "place", "person", "character", "entity", "activity",
    "organization", "technology", "work", "project",
    "universe", "publisher", "comic", "document", "rule",
    # Legacy Spanish types (accepted for backward compatibility)
    "lugar", "persona", "personaje", "entidad", "actividad",
    "organizacion", "organización", "tecnologia", "tecnología", "obra",
    "universo", "editorial",
))
VALID_RELATIONS = frozenset((
    # Universal semantic relations
    "is_a", "created_by", "alias_of", "part_of", "set_in",
    "debuted_in", "published_by",
    # English domain relations
    "works_at", "lives_in", "owns", "belongs_to",
    "uses_technology", "has_module", "likes", "related_to",
    "located_in", "managed_by", "played_for", "played_against",
    "directed_by", "won_against", "lost_to", "co_mentioned",
    # Legacy Spanish relations (accepted for backward compatibility)
    "ubicado_en", "tecnico_de", "parte_de", "hincha_de",
    "juega_en", "pertenece_a", "relacionado_con",
    "jugó_contra", "dirigido_por", "ganó_a", "perdió_contra",
))

# Entities that are noise — roles, pronouns, generic terms, common nouns
_ENTITY_BLACKLIST = frozenset({
    # Roles and pronouns
    "user", "assistant", "bot", "ai",
    "i", "you", "he", "she", "we", "they", "me", "him", "her",
    # Temporal
    "today", "yesterday", "tomorrow", "now", "before", "after",
    # Meta terms
    "person", "place", "entity", "activity", "work",
    "conversation", "chat", "session", "message",
    "question", "answer", "topic", "information", "data",
    # Greetings (EN + ES)
    "hello", "hi", "bye", "goodbye", "thanks", "please",
    "hola", "chau", "gracias", "por favor", "buenas",
    # Spanish common words that are NOT entities
    "necesito", "neecsito", "quiero", "puedo", "tengo",
    "como", "cuando", "donde", "porque", "sobre",
    # Geographic generics
    "province", "city", "country", "state", "club", "team",
    # Common nouns that should never be entities
    "book", "books", "movie", "movies", "film", "films",
    "novel", "novels", "series", "story", "stories",
    "character", "characters", "author", "writer",
    "world", "life", "death", "year", "years", "time", "part", "parts",
    "type", "types", "form", "forms", "thing", "things", "people",
    "culture", "popular", "memory", "internet", "web",
    # Numbers as words
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "hundred", "thousand",
})


class ConversationExtractor:
    """Extracts entities, relations, and facts from user/assistant dialogue.

    Only registers explicit statements. If the user didn't state a fact,
    it doesn't get recorded. Empty facts is the correct result for most turns.
    """

    def __init__(self, llm: LLMClient, prompt_template: str | None = None) -> None:
        self._llm = llm
        self._prompt = prompt_template or _CONVERSATION_PROMPT

    async def extract(self, user_msg: str, assistant_msg: str) -> ExtractionResult:
        try:
            result = await self._call_llm(user_msg, assistant_msg)
            return self._validate(result, user_msg, assistant_msg)
        except Exception as e:
            log.warning("Conversation extraction failed: %s", e)
            return ExtractionResult()

    @staticmethod
    def _validate(
        result: "ExtractionResult",
        user_msg: str,
        assistant_msg: str,
    ) -> "ExtractionResult":
        """Post-extraction validation: reject entities not in the conversation."""
        conversation = f"{user_msg} {assistant_msg}".lower()

        valid_entities: list[Entity] = []
        valid_names: set[str] = set()
        for e in result.entities:
            name_lower = e.name.lower()
            # Entity name must appear in the actual conversation text
            if name_lower not in conversation:
                log.info("Rejected hallucinated entity: %s", e.name)
                continue
            valid_entities.append(e)
            valid_names.add(e.name)

        # Filter relations and facts to only reference valid entities
        valid_relations = [
            r for r in result.relations
            if r.source in valid_names or r.target in valid_names
        ]
        valid_facts = [
            f for f in result.facts
            if f.entity in valid_names
        ]

        return ExtractionResult(
            entities=valid_entities,
            relations=valid_relations,
            facts=valid_facts,
        )

    async def _call_llm(self, user_msg: str, assistant_msg: str) -> ExtractionResult:
        prompt = self._prompt.format(
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


class TextExtractor:
    """Extracts entities, relations, and facts from arbitrary text (files, documents).

    Uses the same prompt as SearchExtractor but with a neutral query.
    """

    def __init__(self, llm: LLMClient, prompt_template: str | None = None) -> None:
        self._llm = llm
        self._prompt = prompt_template or _SEARCH_PROMPT

    async def extract(self, text: str, query: str = "document content") -> ExtractionResult:
        """Extract structured knowledge from a text document."""
        try:
            return await self._call_llm(query, text)
        except Exception as e:
            log.warning("Text extraction failed: %s", e)
            return ExtractionResult()

    async def _call_llm(self, query: str, text: str) -> ExtractionResult:
        prompt = self._prompt.format(
            query=query,
            text=text[:3000],
        )
        raw_response = await self._llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=800,
        )
        raw = _clean_response(raw_response)
        obj = _parse_first_json(raw, "object")
        if isinstance(obj, dict):
            return SearchExtractor._parse_object_static(obj)
        return ExtractionResult()


class SearchExtractor:
    """Extracts entities, relations, and facts from web search results."""

    def __init__(self, llm: LLMClient, prompt_template: str | None = None) -> None:
        self._llm = llm
        self._prompt = prompt_template or _SEARCH_PROMPT

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
        prompt = self._prompt.format(
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
            return self._parse_object_static(obj)

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

    @staticmethod
    def _parse_object_static(obj: dict) -> ExtractionResult:
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

    def __init__(self, llm: LLMClient, prompt_template: str | None = None) -> None:
        self._llm = llm
        self._prompt = prompt_template or _SEARCH_PROMPT

    async def extract(self, query: str, rag_text: str) -> list[ExtractedFact]:
        try:
            return await self._call_llm(query, rag_text)
        except Exception as e:
            log.warning("RAG extraction failed: %s", e)
            return []

    async def _call_llm(self, query: str, rag_text: str) -> list[ExtractedFact]:
        prompt = self._prompt.format(
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
