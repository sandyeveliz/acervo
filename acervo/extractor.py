"""Backward compat — re-export from acervo.extraction.extractor."""
from acervo.extraction.extractor import *  # noqa: F401, F403
from acervo.extraction.extractor import (  # noqa: F401
    ConversationExtractor, SearchExtractor, TextExtractor, RAGExtractor,
    ExtractionResult, Entity, Relation, ExtractedFact,
    _parse_first_json, _clean_response,
    VALID_TYPES, VALID_RELATIONS,
)
