"""Ontology validator — strict type and relation enforcement.

Sits between model output and graph store. Replaces the auto-registration
behavior in ontology.py with strict validation + structured logging.

Every validation decision (accepted/mapped/rejected) produces a LogEntry
that feeds the ValidationLog table for training data generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

log = logging.getLogger(__name__)


# ── Valid types ──────────────────────────────────────────────────────────────

VALID_ENTITY_TYPES: frozenset[str] = frozenset({
    "person", "character", "organization", "project", "technology",
    "place", "event", "document", "concept", "work", "rule",
})

VALID_STRUCTURAL_TYPES: frozenset[str] = frozenset({
    "file", "folder", "symbol", "section",
})

ALL_VALID_TYPES: frozenset[str] = VALID_ENTITY_TYPES | VALID_STRUCTURAL_TYPES

# ── Valid relations ──────────────────────────────────────────────────────────

VALID_SEMANTIC_RELATIONS: frozenset[str] = frozenset({
    "part_of", "created_by", "maintains", "works_at", "member_of",
    "uses_technology", "depends_on", "alternative_to", "located_in",
    "deployed_on", "produces", "serves", "documented_in",
    "participated_in", "triggered_by", "resulted_in",
})

VALID_STRUCTURAL_RELATIONS: frozenset[str] = frozenset({
    "contains", "defined_in", "child_of", "imports", "calls",
    "extends", "implements",
})

ALL_VALID_RELATIONS: frozenset[str] = VALID_SEMANTIC_RELATIONS | VALID_STRUCTURAL_RELATIONS

# ── Synonym maps ─────────────────────────────────────────────────────────────

TYPE_SYNONYMS: dict[str, str] = {
    # Legacy types → canonical
    "publisher": "organization",
    "universe": "concept",
    # Technology variants
    "framework": "technology",
    "library": "technology",
    "platform": "technology",
    "tool": "technology",
    "database": "technology",
    "language": "technology",
    "runtime": "technology",
    "api": "technology",
    "backend_service": "technology",
    "design_system": "technology",
    # Spanish legacy
    "lugar": "place",
    "persona": "person",
    "personaje": "character",
    "tecnologia": "technology",
    "tecnología": "technology",
    "documento": "document",
    "organización": "organization",
    "organizacion": "organization",
    "proyecto": "project",
    "obra": "work",
    "universo": "concept",
    "editorial": "organization",
    "regla": "rule",
    "actividad": "project",
    "entidad": "concept",
    # English aliases
    "entity": "concept",
    "activity": "project",
    "location": "place",
    "comic": "work",
}

RELATION_SYNONYMS: dict[str, str] = {
    # Semantic mappings
    "is_a": "part_of",
    "set_in": "located_in",
    "debuted_in": "documented_in",
    "published_by": "created_by",
    "lives_in": "located_in",
    "belongs_to": "part_of",
    "has_module": "part_of",
    "managed_by": "maintains",
    "uses": "uses_technology",
    "played_for": "member_of",
    "directed_by": "created_by",
    # UPPERCASE → snake_case normalization
    "IS_A": "part_of",
    "CREATED_BY": "created_by",
    "ALIAS_OF": None,  # rejected — handled as fact
    "PART_OF": "part_of",
    "SET_IN": "located_in",
    "DEBUTED_IN": "documented_in",
    "PUBLISHED_BY": "created_by",
    "WORKS_AT": "works_at",
    "LIVES_IN": "located_in",
    "OWNS": None,  # rejected — too vague
    "BELONGS_TO": "part_of",
    "USES_TECHNOLOGY": "uses_technology",
    "HAS_MODULE": "part_of",
    "LIKES": None,  # rejected
    "RELATED_TO": None,  # rejected
    "CONTAINS": "contains",
    "DEFINED_IN": "defined_in",
    "CHILD_OF": "child_of",
    "IMPORTS": "imports",
    "CALLS": "calls",
    "EXTENDS": "extends",
    "IMPLEMENTS": "implements",
}

REJECTED_RELATIONS: frozenset[str] = frozenset({
    "related_to", "RELATED_TO",
    "likes", "LIKES",
    "owns", "OWNS",
    "associated_with",
    "co_mentioned", "is_related_to", "mentioned_with",
    "alias_of", "ALIAS_OF",
})


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ValidatedType:
    """Result of entity type validation."""
    original: str
    resolved: str
    action: str  # "accepted" | "mapped" | "rejected"
    reason: str = ""


@dataclass
class ValidatedRelation:
    """Result of relation validation."""
    original: str
    resolved: str | None  # None if rejected
    action: str  # "accepted" | "mapped" | "rejected"
    reason: str = ""


@dataclass
class ValidationLogEntry:
    """A single validation decision for persistence."""
    timestamp: str
    input_type: str = ""
    mapped_type: str = ""
    input_relation: str = ""
    mapped_relation: str = ""
    action: str = ""        # accepted | mapped | rejected
    reason: str = ""
    source_stage: str = ""  # s1 | s1_5 | indexing | api
    entity_name: str = ""
    session_id: str = ""


# ── Validator ────────────────────────────────────────────────────────────────

class OntologyValidator:
    """Strict ontology enforcement with structured logging.

    Usage:
        validator = OntologyValidator(source_stage="s1", session_id="s_xxx")
        vt = validator.validate_entity_type("Framework")
        # vt.resolved == "technology", vt.action == "mapped"

        vr = validator.validate_relation("RELATED_TO")
        # vr.resolved is None, vr.action == "rejected"

        # After processing, persist the log
        entries = validator.drain_log()
    """

    def __init__(
        self,
        source_stage: str = "",
        session_id: str = "",
    ) -> None:
        self._source_stage = source_stage
        self._session_id = session_id
        self._log: list[ValidationLogEntry] = []

    def validate_entity_type(
        self, raw_type: str, entity_name: str = "",
    ) -> ValidatedType:
        """Validate an entity type against the ontology.

        Returns ValidatedType with resolved type and action taken.
        Fallback: "concept" for unrecognized types.
        """
        clean = raw_type.lower().strip()

        # 1. Exact match against valid types
        if clean in ALL_VALID_TYPES:
            return self._accept_type(raw_type, clean, entity_name)

        # 2. Synonym match
        if clean in TYPE_SYNONYMS:
            target = TYPE_SYNONYMS[clean]
            return self._map_type(raw_type, target, entity_name,
                                  reason=f"synonym: {clean} → {target}")

        # 3. Capitalized match (e.g., "Technology" → "technology")
        if clean.capitalize().lower() in ALL_VALID_TYPES:
            return self._accept_type(raw_type, clean, entity_name)

        # 4. No match — fallback to concept
        return self._map_type(raw_type, "concept", entity_name,
                              reason=f"unknown type '{raw_type}' → concept fallback")

    def validate_relation(
        self, raw_relation: str, entity_name: str = "",
    ) -> ValidatedRelation:
        """Validate a relation type against the ontology.

        Returns ValidatedRelation. Rejected relations have resolved=None.
        """
        clean = raw_relation.strip()

        # 1. Exact match (case-sensitive for snake_case)
        if clean in ALL_VALID_RELATIONS:
            return self._accept_relation(raw_relation, clean, entity_name)

        # 2. Lowercase match
        lower = clean.lower()
        if lower in ALL_VALID_RELATIONS:
            return self._accept_relation(raw_relation, lower, entity_name)

        # 3. Check rejected set first (before synonyms, since some synonyms map to None)
        if clean in REJECTED_RELATIONS or lower in REJECTED_RELATIONS:
            return self._reject_relation(raw_relation, entity_name,
                                         reason=f"rejected: '{raw_relation}' is too generic")

        # 4. Synonym match
        if clean in RELATION_SYNONYMS:
            target = RELATION_SYNONYMS[clean]
            if target is None:
                return self._reject_relation(raw_relation, entity_name,
                                             reason=f"synonym maps to rejection: {clean}")
            return self._map_relation(raw_relation, target, entity_name,
                                      reason=f"synonym: {clean} → {target}")

        if lower in RELATION_SYNONYMS:
            target = RELATION_SYNONYMS[lower]
            if target is None:
                return self._reject_relation(raw_relation, entity_name,
                                             reason=f"synonym maps to rejection: {lower}")
            return self._map_relation(raw_relation, target, entity_name,
                                      reason=f"synonym: {lower} → {target}")

        # 5. No match — reject
        return self._reject_relation(raw_relation, entity_name,
                                     reason=f"unknown relation '{raw_relation}'")

    def drain_log(self) -> list[ValidationLogEntry]:
        """Return and clear accumulated log entries."""
        entries = self._log
        self._log = []
        return entries

    @property
    def log_entries(self) -> list[ValidationLogEntry]:
        """Read-only access to accumulated log entries."""
        return list(self._log)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _accept_type(
        self, original: str, resolved: str, entity_name: str,
    ) -> ValidatedType:
        result = ValidatedType(original=original, resolved=resolved, action="accepted")
        self._log.append(ValidationLogEntry(
            timestamp=_now(),
            input_type=original,
            mapped_type=resolved,
            action="accepted",
            source_stage=self._source_stage,
            entity_name=entity_name,
            session_id=self._session_id,
        ))
        return result

    def _map_type(
        self, original: str, resolved: str, entity_name: str, reason: str,
    ) -> ValidatedType:
        result = ValidatedType(
            original=original, resolved=resolved, action="mapped", reason=reason,
        )
        self._log.append(ValidationLogEntry(
            timestamp=_now(),
            input_type=original,
            mapped_type=resolved,
            action="mapped",
            reason=reason,
            source_stage=self._source_stage,
            entity_name=entity_name,
            session_id=self._session_id,
        ))
        log.info("OntologyValidator type mapped: %s → %s (%s)", original, resolved, reason)
        return result

    def _accept_relation(
        self, original: str, resolved: str, entity_name: str,
    ) -> ValidatedRelation:
        result = ValidatedRelation(original=original, resolved=resolved, action="accepted")
        self._log.append(ValidationLogEntry(
            timestamp=_now(),
            input_relation=original,
            mapped_relation=resolved,
            action="accepted",
            source_stage=self._source_stage,
            entity_name=entity_name,
            session_id=self._session_id,
        ))
        return result

    def _map_relation(
        self, original: str, resolved: str, entity_name: str, reason: str,
    ) -> ValidatedRelation:
        result = ValidatedRelation(
            original=original, resolved=resolved, action="mapped", reason=reason,
        )
        self._log.append(ValidationLogEntry(
            timestamp=_now(),
            input_relation=original,
            mapped_relation=resolved,
            action="mapped",
            reason=reason,
            source_stage=self._source_stage,
            entity_name=entity_name,
            session_id=self._session_id,
        ))
        log.info("OntologyValidator relation mapped: %s → %s (%s)", original, resolved, reason)
        return result

    def _reject_relation(
        self, original: str, entity_name: str, reason: str,
    ) -> ValidatedRelation:
        result = ValidatedRelation(
            original=original, resolved=None, action="rejected", reason=reason,
        )
        self._log.append(ValidationLogEntry(
            timestamp=_now(),
            input_relation=original,
            mapped_relation="",
            action="rejected",
            reason=reason,
            source_stage=self._source_stage,
            entity_name=entity_name,
            session_id=self._session_id,
        ))
        log.info("OntologyValidator relation rejected: %s (%s)", original, reason)
        return result


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
