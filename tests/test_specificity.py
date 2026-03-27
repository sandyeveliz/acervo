"""Tests for the specificity classifier."""

from __future__ import annotations

import pytest

from acervo.specificity import classify_specificity


class TestSpecificQueries:
    """Queries that should be classified as 'specific' (need chunks)."""

    @pytest.mark.parametrize("query", [
        "show me the code for the login function",
        "what does line 42 say?",
        "how many users signed up last month?",
        "give me the exact error message",
        "what's the function signature for prepare()?",
        "show me the config for the proxy",
        "what date was version 0.2.0 released?",
        "paste the implementation of _store_and_link_chunks",
        "what does the file say about chunk retrieval?",
        "how to set up the development environment step by step?",
        "give me an example of the API response",
        "what error does it throw when the graph is corrupt?",
        "what setting controls the token budget?",
        "what's the return type of search_by_chunk_ids?",
        "how much does GPT-4o cost per token?",
    ])
    def test_specific_queries(self, query: str) -> None:
        assert classify_specificity(query) == "specific"


class TestConceptualQueries:
    """Queries that should be classified as 'conceptual' (summary only)."""

    @pytest.mark.parametrize("query", [
        "what is a knowledge graph?",
        "explain how the context pipeline works",
        "why does Acervo use a graph instead of a vector database?",
        "what's the difference between warm and hot context?",
        "describe the overall architecture",
        "what do you think about using ChromaDB?",
        "summarize the approach to memory management",
        "what are the pros and cons of node-scoped retrieval?",
        "in general, how does topic detection work?",
        "what's the high-level idea behind the extractor?",
        "what's the purpose of the reindexer?",
        "why was the rolling summary deferred?",
    ])
    def test_conceptual_queries(self, query: str) -> None:
        assert classify_specificity(query) == "conceptual"


class TestEdgeCases:
    """Edge cases and tie-breaking behavior."""

    def test_empty_string_defaults_to_specific(self) -> None:
        # No patterns match either way, tie-break favors specific
        assert classify_specificity("") == "specific"

    def test_greeting_defaults_to_specific(self) -> None:
        # Neither specific nor conceptual patterns match
        assert classify_specificity("hello!") == "specific"

    def test_mixed_query_with_more_specific(self) -> None:
        # "show me" + "code" + "function" = 3 specific, "explain" = 1 conceptual
        result = classify_specificity("explain the code and show me the function")
        assert result == "specific"

    def test_short_question(self) -> None:
        result = classify_specificity("what is this?")
        assert result == "conceptual"
