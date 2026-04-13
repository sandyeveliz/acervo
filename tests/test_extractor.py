"""Tests for acervo.extractor — ConversationExtractor with mock LLM."""

from __future__ import annotations

import json

import pytest

from acervo.extractor import ConversationExtractor, ExtractionResult
from tests.conftest import make_mock_llm


@pytest.mark.asyncio
async def test_extract_entities_from_json():
    """Mock LLM returns valid JSON -> entities parsed correctly."""
    response = json.dumps({
        "entities": [
            {"name": "Sandy", "type": "persona"},
            {"name": "Cipolletti", "type": "lugar"},
        ],
        "relations": [
            {"source": "Sandy", "target": "Cipolletti", "relation": "ubicado_en"},
        ],
        "facts": [
            {"entity": "Sandy", "fact": "Sandy vive en Cipolletti", "speaker": "user"},
        ],
    })
    llm = make_mock_llm(response)
    extractor = ConversationExtractor(llm)

    result = await extractor.extract("Sandy vive en Cipolletti", "Buena ciudad!")
    assert len(result.entities) == 2
    assert result.entities[0].name == "Sandy"
    assert result.entities[1].name == "Cipolletti"
    assert len(result.relations) == 1
    assert result.relations[0].relation == "ubicado_en"
    assert len(result.facts) == 1
    assert result.facts[0].speaker == "user"


@pytest.mark.asyncio
async def test_extract_maps_types_to_ontology():
    """Extractor maps raw types to ontology: 'lugar' -> 'Place', 'persona' -> 'Person'."""
    response = json.dumps({
        "entities": [
            {"name": "Cipolletti", "type": "lugar"},
            {"name": "Sandy", "type": "persona"},
            {"name": "Altovallestudio", "type": "organizacion"},
            {"name": "Harry Potter", "type": "obra"},
        ],
        "relations": [],
        "facts": [],
    })
    llm = make_mock_llm(response)
    extractor = ConversationExtractor(llm)

    # user_msg must contain every entity name — ConversationExtractor._validate
    # rejects entities whose name doesn't appear in the conversation text.
    user_msg = "Sandy vive en Cipolletti, trabaja en Altovallestudio y le encanta Harry Potter"
    result = await extractor.extract(user_msg, "ok")
    types = {e.name: e.type for e in result.entities}
    assert types["Cipolletti"] == "Place"
    assert types["Sandy"] == "Person"
    assert types["Altovallestudio"] == "Organization"
    assert types["Harry Potter"] == "Work"


@pytest.mark.asyncio
async def test_extract_filters_blacklisted_entities():
    """Blacklisted names like 'user', 'today' are filtered out."""
    response = json.dumps({
        "entities": [
            {"name": "user", "type": "person"},
            {"name": "today", "type": "activity"},
            {"name": "Sandy", "type": "person"},
        ],
        "relations": [],
        "facts": [],
    })
    llm = make_mock_llm(response)
    extractor = ConversationExtractor(llm)

    # Text must contain the entities we expect to survive validation;
    # blacklist filtering happens regardless of what's in the text.
    result = await extractor.extract("Sandy says user and today are irrelevant", "ok")
    names = [e.name for e in result.entities]
    assert "user" not in names
    assert "today" not in names
    assert "Sandy" in names


@pytest.mark.asyncio
async def test_extract_handles_malformed_json():
    """Malformed JSON returns empty ExtractionResult."""
    llm = make_mock_llm("this is not json at all {broken")
    extractor = ConversationExtractor(llm)

    result = await extractor.extract("test", "test")
    assert isinstance(result, ExtractionResult)
    assert result.entities == []


@pytest.mark.asyncio
async def test_extract_handles_empty_response():
    """Empty response returns empty ExtractionResult."""
    llm = make_mock_llm("")
    extractor = ConversationExtractor(llm)

    result = await extractor.extract("test", "test")
    assert result.entities == []
    assert result.relations == []
    assert result.facts == []


@pytest.mark.asyncio
async def test_extract_filters_short_names():
    """Names with 2 or fewer characters are filtered out."""
    response = json.dumps({
        "entities": [
            {"name": "AB", "type": "persona"},
            {"name": "X", "type": "lugar"},
            {"name": "Sandy", "type": "persona"},
        ],
        "relations": [],
        "facts": [],
    })
    llm = make_mock_llm(response)
    extractor = ConversationExtractor(llm)

    # Text must contain Sandy so the anti-hallucination check lets it through;
    # short-name filtering happens independently of text content.
    result = await extractor.extract("Sandy is here", "ok")
    names = [e.name for e in result.entities]
    assert "AB" not in names
    assert "X" not in names
    assert "Sandy" in names


@pytest.mark.asyncio
async def test_extract_new_relation_auto_registered():
    """New relation types from LLM get auto-registered, not normalized."""
    response = json.dumps({
        "entities": [
            {"name": "Sandy", "type": "persona"},
            {"name": "River", "type": "organizacion"},
        ],
        "relations": [
            {"source": "Sandy", "target": "River", "relation": "fan_of"},
        ],
        "facts": [],
    })
    llm = make_mock_llm(response)
    extractor = ConversationExtractor(llm)

    # Both entity names must appear in the conversation text for validation
    # to let them (and the relation that references them) through.
    result = await extractor.extract("Sandy es fan de River", "ok")
    assert result.relations[0].relation == "fan_of"
