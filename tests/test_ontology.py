"""Tests for acervo.ontology — extensible entity type and relation registry."""

from acervo.ontology import (
    all_types,
    all_relations,
    get_type,
    is_known_type,
    register_type,
    register_relation,
    map_extractor_type,
    is_likely_universal,
    BUILTIN_ENTITY_TYPES,
)
from acervo.layers import Layer


class TestBuiltinTypes:
    def test_builtin_types_present(self):
        types = all_types()
        assert "Persona" in types
        assert "Lugar" in types
        assert "Proyecto" in types
        assert "Tecnología" in types
        assert "Organización" in types
        assert "Documento" in types
        assert "Regla" in types

    def test_get_type_returns_attributes(self):
        attrs = get_type("Persona")
        assert attrs is not None
        assert "nombre" in attrs
        assert "rol" in attrs

    def test_get_type_unknown_returns_none(self):
        assert get_type("NonexistentType") is None

    def test_is_known_type(self):
        assert is_known_type("Persona") is True
        assert is_known_type("Unknown") is False


class TestRegistration:
    def test_register_custom_type(self):
        register_type("Recipe", ["ingredients", "time", "difficulty"])
        assert is_known_type("Recipe")
        attrs = get_type("Recipe")
        assert attrs is not None
        assert "ingredients" in attrs
        assert "Recipe" in all_types()

    def test_register_custom_relation(self):
        register_relation("COAUTHORED_WITH")
        assert "COAUTHORED_WITH" in all_relations()


class TestExtractorTypeMapping:
    def test_map_lugar(self):
        assert map_extractor_type("lugar") == "Lugar"

    def test_map_persona(self):
        assert map_extractor_type("persona") == "Persona"

    def test_map_entidad(self):
        assert map_extractor_type("entidad") == "Unknown"

    def test_map_actividad(self):
        assert map_extractor_type("actividad") == "Proyecto"

    def test_map_unknown_auto_registers(self):
        """Unknown types get auto-registered by the LLM."""
        result = map_extractor_type("superhero")
        assert result == "Superhero"
        assert is_known_type("Superhero")

    def test_map_case_insensitive(self):
        assert map_extractor_type("LUGAR") == "Lugar"
        assert map_extractor_type("Persona") == "Persona"


class TestUniversalDetection:
    def test_lugar_is_universal(self):
        assert is_likely_universal("Lugar") is True

    def test_tecnologia_is_universal(self):
        assert is_likely_universal("Tecnología") is True

    def test_persona_is_not_universal(self):
        assert is_likely_universal("Persona") is False

    def test_proyecto_is_not_universal(self):
        assert is_likely_universal("Proyecto") is False


# ── User-requested tests ──


def test_register_custom_type_receta():
    register_type("Receta", ["ingredientes", "tiempo", "dificultad"])
    assert "Receta" in all_types()
    assert "ingredientes" in get_type("Receta")


def test_register_custom_relation_cocino():
    register_relation("COCINÓ")
    assert "COCINÓ" in all_relations()


def test_unknown_type_not_in_registry():
    assert get_type("TipoQueNoExiste") is None
