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

    result = await extractor.extract("Vivo en Cipolletti", "Buena ciudad!")
    assert len(result.entities) == 2
    assert result.entities[0].name == "Sandy"
    assert result.entities[1].name == "Cipolletti"
    assert len(result.relations) == 1
    assert result.relations[0].relation == "ubicado_en"
    assert len(result.facts) == 1
    assert result.facts[0].speaker == "user"


@pytest.mark.asyncio
async def test_extract_maps_types_to_ontology():
    """Extractor maps raw types to ontology: 'lugar' -> 'Lugar', 'persona' -> 'Persona'."""
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

    result = await extractor.extract("test", "test")
    types = {e.name: e.type for e in result.entities}
    assert types["Cipolletti"] == "Lugar"
    assert types["Sandy"] == "Persona"
    assert types["Altovallestudio"] == "Organización"
    assert types["Harry Potter"] == "Obra"


@pytest.mark.asyncio
async def test_extract_filters_blacklisted_entities():
    """Blacklisted names like 'usuario', 'hoy' are filtered out."""
    response = json.dumps({
        "entities": [
            {"name": "usuario", "type": "persona"},
            {"name": "hoy", "type": "actividad"},
            {"name": "Sandy", "type": "persona"},
        ],
        "relations": [],
        "facts": [],
    })
    llm = make_mock_llm(response)
    extractor = ConversationExtractor(llm)

    result = await extractor.extract("test", "test")
    names = [e.name for e in result.entities]
    assert "usuario" not in names
    assert "hoy" not in names
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

    result = await extractor.extract("test", "test")
    names = [e.name for e in result.entities]
    assert "AB" not in names
    assert "X" not in names
    assert "Sandy" in names


@pytest.mark.asyncio
async def test_extract_invalid_relation_normalized():
    """Invalid relation types get normalized to 'relacionado_con'."""
    response = json.dumps({
        "entities": [
            {"name": "Sandy", "type": "persona"},
            {"name": "River", "type": "entidad"},
        ],
        "relations": [
            {"source": "Sandy", "target": "River", "relation": "invalid_rel"},
        ],
        "facts": [],
    })
    llm = make_mock_llm(response)
    extractor = ConversationExtractor(llm)

    result = await extractor.extract("test", "test")
    assert result.relations[0].relation == "relacionado_con"
