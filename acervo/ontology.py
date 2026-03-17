"""Registro extensible de tipos de entidades y relaciones.

Uso:
    from acervo.ontology import register_type, get_type, all_types
    from acervo.ontology import BUILTIN_RELATIONS

    # Registrar tipo personalizado
    register_type("Mascota", ["nombre", "especie", "edad"])

    # Registrar relación personalizada
    register_relation("TIENE_MASCOTA")
"""

from __future__ import annotations

from acervo.layers import Layer

# ── Tipos de entidad built-in ──────────────────────────────────────────────

BUILTIN_ENTITY_TYPES: dict[str, list[str]] = {
    "Persona":       ["nombre", "edad", "rol", "relacion_con_owner"],
    "Organización":  ["nombre", "tipo", "industria", "ubicacion"],
    "Proyecto":      ["nombre", "stack", "scaffold", "arquitectura", "modulos", "estado"],
    "Lugar":         ["nombre", "tipo", "region", "pais"],
    "Tecnología":    ["nombre", "tipo", "version"],
    "Documento":     ["nombre", "tipo", "path", "contenido_resumen"],
    "Regla":         ["descripcion", "aplica_a", "tecnologia", "severity"],
}

_entity_registry: dict[str, list[str]] = dict(BUILTIN_ENTITY_TYPES)

# ── Relaciones built-in ────────────────────────────────────────────────────

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
    # Legado del extractor conversacional (snake_case)
    "ubicado_en", "tecnico_de", "parte_de", "hincha_de",
    "juega_en", "pertenece_a", "relacionado_con", "co_mentioned",
    "jugó_contra", "dirigido_por", "ganó_a", "perdió_contra",
}

_relation_registry: set[str] = set(BUILTIN_RELATIONS)

# ── API de registro ────────────────────────────────────────────────────────

def register_type(
    name: str,
    attributes: list[str],
    layer_default: Layer = Layer.PERSONAL,
) -> None:
    """Registra un nuevo tipo de entidad en el registro global."""
    _entity_registry[name] = attributes


def register_relation(name: str) -> None:
    """Registra un nuevo tipo de relación en el registro global."""
    _relation_registry.add(name)


def get_type(name: str) -> list[str] | None:
    """Retorna los atributos esperados de un tipo de entidad, o None si no existe."""
    return _entity_registry.get(name)


def all_types() -> dict[str, list[str]]:
    """Retorna todos los tipos de entidad registrados (built-in + custom)."""
    return dict(_entity_registry)


def all_relations() -> set[str]:
    """Retorna todas las relaciones registradas (built-in + custom)."""
    return set(_relation_registry)


def is_known_type(name: str) -> bool:
    """Retorna True si el tipo de entidad está registrado."""
    return name in _entity_registry
