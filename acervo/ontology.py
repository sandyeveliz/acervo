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

from acervo.layers import Layer

# ── Built-in entity types ──────────────────────────────────────────────

BUILTIN_ENTITY_TYPES: dict[str, list[str]] = {
    "Persona":       ["nombre", "edad", "rol", "relacion_con_owner"],
    "Organización":  ["nombre", "tipo", "industria", "ubicacion"],
    "Proyecto":      ["nombre", "stack", "scaffold", "arquitectura", "modulos", "estado"],
    "Lugar":         ["nombre", "tipo", "region", "pais"],
    "Tecnología":    ["nombre", "tipo", "version"],
    "Documento":     ["nombre", "tipo", "path", "contenido_resumen"],
    "Regla":         ["descripcion", "aplica_a", "tecnologia", "severity"],
    "Obra":          ["nombre", "tipo", "autor", "genero", "año"],
}

_entity_registry: dict[str, list[str]] = dict(BUILTIN_ENTITY_TYPES)

# ── Built-in relations ────────────────────────────────────────────────────

BUILTIN_RELATIONS: set[str] = {
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
    "entidad": "Unknown",  # too broad — let incomplete resolution handle it
    "actividad": "Proyecto",
    "tecnologia": "Tecnología",
    "tecnología": "Tecnología",
    "documento": "Documento",
    "regla": "Regla",
    "organizacion": "Organización",
    "organización": "Organización",
    "proyecto": "Proyecto",
    "obra": "Obra",
}


def map_extractor_type(raw_type: str) -> str:
    """Map a raw extractor type (lowercase) to an ontology type.

    Returns the ontology type name, or "Unknown" if no mapping exists.
    """
    return _EXTRACTOR_TYPE_MAP.get(raw_type.lower().strip(), "Unknown")


# ── Universal knowledge detection ──────────────────────────────────────────

# Entity types that are typically universal (world knowledge)
_UNIVERSAL_TYPES: frozenset[str] = frozenset({"Lugar", "Tecnología"})


def is_likely_universal(entity_type: str) -> bool:
    """Heuristic: return True if this entity type is typically universal knowledge."""
    return entity_type in _UNIVERSAL_TYPES
