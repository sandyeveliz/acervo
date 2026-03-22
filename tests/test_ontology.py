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
        assert "Person" in types
        assert "Place" in types
        assert "Project" in types
        assert "Technology" in types
        assert "Organization" in types
        assert "Document" in types
        assert "Rule" in types

    def test_get_type_returns_attributes(self):
        attrs = get_type("Person")
        assert attrs is not None
        assert "name" in attrs
        assert "role" in attrs

    def test_get_type_unknown_returns_none(self):
        assert get_type("NonexistentType") is None

    def test_is_known_type(self):
        assert is_known_type("Person") is True
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
    def test_map_english_types(self):
        assert map_extractor_type("place") == "Place"
        assert map_extractor_type("person") == "Person"
        assert map_extractor_type("character") == "Character"
        assert map_extractor_type("organization") == "Organization"

    def test_map_legacy_spanish_types(self):
        """Legacy Spanish types still map correctly."""
        assert map_extractor_type("lugar") == "Place"
        assert map_extractor_type("persona") == "Person"
        assert map_extractor_type("personaje") == "Character"
        assert map_extractor_type("organización") == "Organization"

    def test_map_entidad(self):
        assert map_extractor_type("entity") == "Unknown"
        assert map_extractor_type("entidad") == "Unknown"

    def test_map_actividad(self):
        assert map_extractor_type("activity") == "Project"
        assert map_extractor_type("actividad") == "Project"

    def test_map_unknown_auto_registers(self):
        """Unknown types get auto-registered by the LLM."""
        result = map_extractor_type("superhero")
        assert result == "Superhero"
        assert is_known_type("Superhero")

    def test_map_case_insensitive(self):
        assert map_extractor_type("PLACE") == "Place"
        assert map_extractor_type("Person") == "Person"


class TestUniversalDetection:
    def test_place_is_universal(self):
        assert is_likely_universal("Place") is True

    def test_technology_is_universal(self):
        assert is_likely_universal("Technology") is True

    def test_person_is_not_universal(self):
        assert is_likely_universal("Person") is False

    def test_project_is_not_universal(self):
        assert is_likely_universal("Project") is False


# ── User-requested tests ──


def test_register_custom_type_recipe():
    register_type("Cooking", ["ingredients", "time", "difficulty"])
    assert "Cooking" in all_types()
    assert "ingredients" in get_type("Cooking")


def test_register_custom_relation():
    register_relation("COOKED")
    assert "COOKED" in all_relations()


def test_unknown_type_not_in_registry():
    assert get_type("TypeThatDoesNotExist") is None
