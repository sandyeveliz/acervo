"""Tests for OntologyValidator — strict ontology enforcement."""

import pytest

from acervo.graph.ontology_validator import (
    ALL_VALID_RELATIONS,
    ALL_VALID_TYPES,
    REJECTED_RELATIONS,
    RELATION_SYNONYMS,
    TYPE_SYNONYMS,
    VALID_ENTITY_TYPES,
    VALID_SEMANTIC_RELATIONS,
    VALID_STRUCTURAL_RELATIONS,
    VALID_STRUCTURAL_TYPES,
    OntologyValidator,
)


class TestValidateEntityType:
    """Test entity type validation."""

    def test_valid_types_accepted(self):
        v = OntologyValidator(source_stage="test")
        for t in VALID_ENTITY_TYPES:
            result = v.validate_entity_type(t)
            assert result.action == "accepted", f"{t} should be accepted"
            assert result.resolved == t

    def test_structural_types_accepted(self):
        v = OntologyValidator(source_stage="test")
        for t in VALID_STRUCTURAL_TYPES:
            result = v.validate_entity_type(t)
            assert result.action == "accepted", f"{t} should be accepted"

    def test_case_insensitive(self):
        v = OntologyValidator(source_stage="test")
        result = v.validate_entity_type("Person")
        assert result.resolved == "person"
        assert result.action == "accepted"

    def test_synonym_framework_to_technology(self):
        v = OntologyValidator(source_stage="test")
        result = v.validate_entity_type("framework")
        assert result.resolved == "technology"
        assert result.action == "mapped"

    def test_synonym_library_to_technology(self):
        v = OntologyValidator(source_stage="test")
        result = v.validate_entity_type("library")
        assert result.resolved == "technology"
        assert result.action == "mapped"

    def test_synonym_publisher_to_organization(self):
        v = OntologyValidator(source_stage="test")
        result = v.validate_entity_type("publisher")
        assert result.resolved == "organization"
        assert result.action == "mapped"

    def test_synonym_universe_to_concept(self):
        v = OntologyValidator(source_stage="test")
        result = v.validate_entity_type("universe")
        assert result.resolved == "concept"
        assert result.action == "mapped"

    def test_spanish_legacy_persona(self):
        v = OntologyValidator(source_stage="test")
        result = v.validate_entity_type("persona")
        assert result.resolved == "person"
        assert result.action == "mapped"

    def test_spanish_legacy_tecnologia(self):
        v = OntologyValidator(source_stage="test")
        result = v.validate_entity_type("tecnología")
        assert result.resolved == "technology"
        assert result.action == "mapped"

    def test_unknown_type_falls_back_to_concept(self):
        v = OntologyValidator(source_stage="test")
        result = v.validate_entity_type("Gadget")
        assert result.resolved == "concept"
        assert result.action == "mapped"
        assert "unknown type" in result.reason

    def test_all_type_synonyms_map_to_valid_types(self):
        """Every synonym must map to a valid type."""
        for source, target in TYPE_SYNONYMS.items():
            assert target in ALL_VALID_TYPES, (
                f"TYPE_SYNONYMS['{source}'] = '{target}' is not a valid type"
            )

    def test_whitespace_stripped(self):
        v = OntologyValidator(source_stage="test")
        result = v.validate_entity_type("  person  ")
        assert result.resolved == "person"
        assert result.action == "accepted"


class TestValidateRelation:
    """Test relation validation."""

    def test_valid_semantic_relations_accepted(self):
        v = OntologyValidator(source_stage="test")
        for r in VALID_SEMANTIC_RELATIONS:
            result = v.validate_relation(r)
            assert result.action == "accepted", f"{r} should be accepted"
            assert result.resolved == r

    def test_valid_structural_relations_accepted(self):
        v = OntologyValidator(source_stage="test")
        for r in VALID_STRUCTURAL_RELATIONS:
            result = v.validate_relation(r)
            assert result.action == "accepted", f"{r} should be accepted"

    def test_rejected_relations(self):
        v = OntologyValidator(source_stage="test")
        for r in REJECTED_RELATIONS:
            result = v.validate_relation(r)
            assert result.action == "rejected", f"{r} should be rejected"
            assert result.resolved is None

    def test_synonym_is_a_to_part_of(self):
        v = OntologyValidator(source_stage="test")
        result = v.validate_relation("is_a")
        assert result.resolved == "part_of"
        assert result.action == "mapped"

    def test_uppercase_created_by_lowercased(self):
        v = OntologyValidator(source_stage="test")
        result = v.validate_relation("CREATED_BY")
        assert result.resolved == "created_by"
        # Lowercased match is accepted (not mapped) since "created_by" is valid
        assert result.action == "accepted"

    def test_synonym_uppercase_is_a_mapped(self):
        v = OntologyValidator(source_stage="test")
        result = v.validate_relation("IS_A")
        assert result.resolved == "part_of"
        assert result.action == "mapped"

    def test_synonym_lives_in_to_located_in(self):
        v = OntologyValidator(source_stage="test")
        result = v.validate_relation("lives_in")
        assert result.resolved == "located_in"
        assert result.action == "mapped"

    def test_synonym_uses_to_uses_technology(self):
        v = OntologyValidator(source_stage="test")
        result = v.validate_relation("uses")
        assert result.resolved == "uses_technology"
        assert result.action == "mapped"

    def test_unknown_relation_rejected(self):
        v = OntologyValidator(source_stage="test")
        result = v.validate_relation("favorite_of")
        assert result.action == "rejected"
        assert result.resolved is None
        assert "unknown relation" in result.reason

    def test_all_relation_synonyms_map_to_valid_or_none(self):
        """Every relation synonym must map to a valid relation or None (rejection)."""
        for source, target in RELATION_SYNONYMS.items():
            if target is not None:
                assert target in ALL_VALID_RELATIONS, (
                    f"RELATION_SYNONYMS['{source}'] = '{target}' is not a valid relation"
                )

    @pytest.mark.parametrize("relation", [
        "appears_in", "child_of", "married_to", "set_in", "narrated_by",
    ])
    def test_literary_kinship_relations_v061(self, relation):
        """v0.6.1 — new literary/kinship relations accepted directly."""
        v = OntologyValidator(source_stage="test")
        result = v.validate_relation(relation, entity_name="test")
        assert result.resolved == relation
        assert result.action == "accepted"

    @pytest.mark.parametrize("variant,canonical", [
        ("parent_of", "child_of"),
        ("father_of", "child_of"),
        ("mother_of", "child_of"),
        ("spouse_of", "married_to"),
        ("husband_of", "married_to"),
        ("wife_of", "married_to"),
        ("character_in", "appears_in"),
        ("features", "appears_in"),
        ("takes_place_in", "set_in"),
        ("told_by", "narrated_by"),
        ("narrator", "narrated_by"),
    ])
    def test_literary_kinship_synonyms_mapped(self, variant, canonical):
        """Common variants of the new relations get mapped to canonical form."""
        v = OntologyValidator(source_stage="test")
        result = v.validate_relation(variant, entity_name="test")
        assert result.resolved == canonical
        assert result.action == "mapped"


class TestValidationLog:
    """Test structured logging of validation decisions."""

    def test_log_accumulated(self):
        v = OntologyValidator(source_stage="s1", session_id="test_session")
        v.validate_entity_type("person", entity_name="Alice")
        v.validate_entity_type("framework", entity_name="React")
        v.validate_relation("RELATED_TO", entity_name="test")
        entries = v.drain_log()
        assert len(entries) == 3

    def test_log_entry_fields(self):
        v = OntologyValidator(source_stage="s1", session_id="s_123")
        v.validate_entity_type("framework", entity_name="Django")
        entries = v.drain_log()
        assert len(entries) == 1
        e = entries[0]
        assert e.input_type == "framework"
        assert e.mapped_type == "technology"
        assert e.action == "mapped"
        assert e.source_stage == "s1"
        assert e.entity_name == "Django"
        assert e.session_id == "s_123"
        assert e.timestamp  # non-empty

    def test_drain_clears_log(self):
        v = OntologyValidator(source_stage="test")
        v.validate_entity_type("person")
        assert len(v.drain_log()) == 1
        assert len(v.drain_log()) == 0

    def test_rejected_relation_logged(self):
        v = OntologyValidator(source_stage="s1_5")
        v.validate_relation("likes", entity_name="user")
        entries = v.drain_log()
        assert len(entries) == 1
        e = entries[0]
        assert e.input_relation == "likes"
        assert e.mapped_relation == ""
        assert e.action == "rejected"

    def test_log_entries_readonly(self):
        v = OntologyValidator(source_stage="test")
        v.validate_entity_type("person")
        entries = v.log_entries
        assert len(entries) == 1
        # log_entries returns a copy, doesn't drain
        assert len(v.log_entries) == 1


class TestOntologyCompleteness:
    """Sanity checks on the ontology definitions."""

    def test_11_entity_types(self):
        assert len(VALID_ENTITY_TYPES) == 11

    def test_4_structural_types(self):
        assert len(VALID_STRUCTURAL_TYPES) == 4

    def test_20_semantic_relations(self):
        # 16 original + 4 v0.6.1 literary/kinship (appears_in, married_to,
        # set_in, narrated_by). child_of is also a v0.6.1 kinship relation
        # but is already in VALID_STRUCTURAL_RELATIONS so it doesn't
        # increment this count.
        assert len(VALID_SEMANTIC_RELATIONS) == 20

    def test_7_structural_relations(self):
        assert len(VALID_STRUCTURAL_RELATIONS) == 7

    def test_no_overlap_entity_structural_types(self):
        overlap = VALID_ENTITY_TYPES & VALID_STRUCTURAL_TYPES
        assert not overlap, f"Overlap: {overlap}"

    def test_no_overlap_semantic_structural_relations(self):
        overlap = VALID_SEMANTIC_RELATIONS & VALID_STRUCTURAL_RELATIONS
        assert not overlap, f"Overlap: {overlap}"
