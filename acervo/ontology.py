"""Extensible registry of entity types and relations.

Usage:
    from acervo.ontology import register_type, get_type, all_types
    from acervo.ontology import BUILTIN_RELATIONS

    # Register a custom type
    register_type("Pet", ["name", "species", "age"])

    # Register a custom relation
    register_relation("HAS_PET")
"""

from __future__ import annotations

import logging

from acervo.layers import Layer

log = logging.getLogger(__name__)

# ── Built-in entity types ──────────────────────────────────────────────

BUILTIN_ENTITY_TYPES: dict[str, list[str]] = {
    # People and characters
    "Person":        ["name", "age", "role", "relation_to_owner"],
    "Character":     ["name", "alias", "creator", "universe"],
    # Organizations and universes
    "Organization":  ["name", "type", "industry", "location"],
    "Universe":      ["name", "publisher", "characters"],
    "Publisher":     ["name", "founded", "country"],
    # Projects and works
    "Project":       ["name", "stack", "scaffold", "architecture", "modules", "status"],
    "Work":          ["name", "type", "author", "genre", "year"],
    # Places
    "Place":         ["name", "type", "region", "country"],
    # Technology
    "Technology":    ["name", "type", "version"],
    # Other
    "Document":      ["name", "type", "path", "summary"],
    "Rule":          ["description", "applies_to", "technology", "severity"],
    # Structural parsing
    "File":          ["path", "language", "content_hash"],
    "Symbol":        ["name", "symbol_type", "signature", "start_line", "end_line", "language"],
    "Section":       ["name", "heading_level", "start_line", "end_line"],
}

_entity_registry: dict[str, list[str]] = dict(BUILTIN_ENTITY_TYPES)

# ── Built-in relations ────────────────────────────────────────────────────

BUILTIN_RELATIONS: set[str] = {
    # Universal semantic relations (knowledge graph standard)
    "IS_A",           # Batman is_a Character
    "CREATED_BY",     # Batman created_by Bill Finger
    "ALIAS_OF",       # Batman alias_of Bruce Wayne
    "PART_OF",        # Batman part_of DC Universe
    "SET_IN",         # Batman set_in Gotham City
    "DEBUTED_IN",     # Batman debuted_in Detective Comics
    "PUBLISHED_BY",   # Detective Comics published_by DC Comics
    # Domain relations
    "WORKS_AT",
    "LIVES_IN",
    "OWNS",
    "BELONGS_TO",
    "USES_TECHNOLOGY",
    "HAS_MODULE",
    "LIKES",
    "RELATED_TO",
    # snake_case variants (from extractor output)
    "is_a", "created_by", "alias_of", "part_of", "set_in",
    "debuted_in", "published_by",
    "works_at", "lives_in", "owns", "belongs_to",
    "uses_technology", "has_module", "likes", "related_to",
    "located_in", "managed_by", "played_for", "played_against",
    "directed_by", "won_against", "lost_to", "co_mentioned",
    # Structural parsing relations
    "CONTAINS", "contains",       # File -> Symbol/Section
    "DEFINED_IN", "defined_in",   # Symbol -> File
    "CHILD_OF", "child_of",       # nested method -> class, subsection -> section
    # Dependency graph relations (from indexer)
    "IMPORTS", "imports",         # File -> File (resolved import)
    "CALLS", "calls",             # Function -> Function
    "EXTENDS", "extends",         # Class -> Class
    "IMPLEMENTS", "implements",   # Class -> Interface
}

_relation_registry: set[str] = set(BUILTIN_RELATIONS)

# ── Registration API ────────────────────────────────────────────────────────

def register_type(
    name: str,
    attributes: list[str],
    layer_default: Layer = Layer.PERSONAL,
) -> None:
    """Register a new entity type in the global registry."""
    _entity_registry[name] = attributes


def register_relation(name: str) -> None:
    """Register a new relation type in the global registry."""
    _relation_registry.add(name)


def get_type(name: str) -> list[str] | None:
    """Return the expected attributes for an entity type, or None if unknown."""
    return _entity_registry.get(name)


def all_types() -> dict[str, list[str]]:
    """Return all registered entity types (built-in + custom)."""
    return dict(_entity_registry)


def all_relations() -> set[str]:
    """Return all registered relations (built-in + custom)."""
    return set(_relation_registry)


def is_known_type(name: str) -> bool:
    """Return True if the entity type is registered."""
    return name in _entity_registry


# ── Type mapping (extractor lowercase → ontology capitalized) ──────────────

_EXTRACTOR_TYPE_MAP: dict[str, str] = {
    # English types (primary)
    "person": "Person",
    "character": "Character",
    "organization": "Organization",
    "universe": "Universe",
    "publisher": "Publisher",
    "project": "Project",
    "work": "Work",
    "place": "Place",
    "technology": "Technology",
    "document": "Document",
    "rule": "Rule",
    "entity": "Unknown",
    "activity": "Project",
    "comic": "Work",
    "file": "File",
    "symbol": "Symbol",
    "section": "Section",
    # Legacy Spanish mappings (for existing graphs)
    "lugar": "Place",
    "persona": "Person",
    "personaje": "Character",
    "entidad": "Unknown",
    "actividad": "Project",
    "tecnologia": "Technology",
    "tecnología": "Technology",
    "documento": "Document",
    "regla": "Rule",
    "organizacion": "Organization",
    "organización": "Organization",
    "proyecto": "Project",
    "obra": "Work",
    "universo": "Universe",
    "editorial": "Publisher",
}


def map_extractor_type(raw_type: str) -> str:
    """Map a raw extractor type (lowercase) to an ontology type.

    If the type is not in the mapping, auto-registers it as a new type
    with capitalized name. The LLM is allowed to create new types.
    """
    clean = raw_type.lower().strip()
    mapped = _EXTRACTOR_TYPE_MAP.get(clean)
    if mapped:
        return mapped

    # Auto-register new type from the LLM
    new_type = clean.capitalize()
    if len(new_type) >= 3 and new_type not in _entity_registry:
        register_type(new_type, [])
        _EXTRACTOR_TYPE_MAP[clean] = new_type
        log.info("Auto-registered new entity type: %s", new_type)
    return new_type if len(new_type) >= 3 else "Unknown"


# ── Universal knowledge detection ──────────────────────────────────────────

_UNIVERSAL_TYPES: frozenset[str] = frozenset({
    "Place", "Technology", "Character", "Universe", "Publisher", "Work",
    "File", "Symbol", "Section",
})


def is_likely_universal(entity_type: str) -> bool:
    """Heuristic: return True if this entity type is typically universal knowledge."""
    return entity_type in _UNIVERSAL_TYPES
