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
    "Persona":       ["nombre", "edad", "rol", "relacion_con_owner"],
    "Personaje":     ["nombre", "alias", "creador", "universo"],
    # Organizations and universes
    "Organización":  ["nombre", "tipo", "industria", "ubicacion"],
    "Universo":      ["nombre", "editorial", "personajes"],
    "Editorial":     ["nombre", "fundacion", "pais"],
    # Projects and works
    "Proyecto":      ["nombre", "stack", "scaffold", "arquitectura", "modulos", "estado"],
    "Obra":          ["nombre", "tipo", "autor", "genero", "año"],
    # Places
    "Lugar":         ["nombre", "tipo", "region", "pais"],
    # Technology
    "Tecnología":    ["nombre", "tipo", "version"],
    # Other
    "Documento":     ["nombre", "tipo", "path", "contenido_resumen"],
    "Regla":         ["descripcion", "aplica_a", "tecnologia", "severity"],
}

_entity_registry: dict[str, list[str]] = dict(BUILTIN_ENTITY_TYPES)

# ── Built-in relations ────────────────────────────────────────────────────

BUILTIN_RELATIONS: set[str] = {
    # Universal semantic relations (knowledge graph standard)
    "IS_A",           # Batman is_a Personaje
    "CREATED_BY",     # Batman created_by Bill Finger
    "ALIAS_OF",       # Batman alias_of Bruce Wayne
    "PART_OF",        # Batman part_of DC Universe
    "SET_IN",         # Batman set_in Gotham City
    "DEBUTED_IN",     # Batman debuted_in Detective Comics
    "PUBLISHED_BY",   # Detective Comics published_by DC Comics
    # Domain relations
    "TRABAJA_EN",
    "VIVE_EN",
    "DUEÑO_DE",
    "PERTENECE_A",
    "USA_TECNOLOGIA",
    "TIENE_MODULO",
    "GUSTA_DE",
    "FAMILIAR_DE",
    "RELACIONADO_CON",
    # Legacy from conversation extractor (snake_case)
    "is_a", "created_by", "alias_of", "part_of", "set_in",
    "debuted_in", "published_by",
    "ubicado_en", "tecnico_de", "parte_de", "hincha_de",
    "juega_en", "pertenece_a", "relacionado_con", "co_mentioned",
    "jugó_contra", "dirigido_por", "ganó_a", "perdió_contra",
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
    "lugar": "Lugar",
    "persona": "Persona",
    "personaje": "Personaje",
    "entidad": "Unknown",
    "actividad": "Proyecto",
    "tecnologia": "Tecnología",
    "tecnología": "Tecnología",
    "documento": "Documento",
    "regla": "Regla",
    "organizacion": "Organización",
    "organización": "Organización",
    "proyecto": "Proyecto",
    "obra": "Obra",
    "universo": "Universo",
    "editorial": "Editorial",
    "comic": "Obra",
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
    "Lugar", "Tecnología", "Personaje", "Universo", "Editorial", "Obra",
})


def is_likely_universal(entity_type: str) -> bool:
    """Heuristic: return True if this entity type is typically universal knowledge."""
    return entity_type in _UNIVERSAL_TYPES
