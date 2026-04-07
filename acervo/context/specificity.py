"""Specificity classifier — decides whether a query needs chunk retrieval.

A simple regex + keyword heuristic that classifies user messages as either
needing detailed chunks from the vector store ("specific") or just node
summaries from the graph ("conceptual").

Specific queries ask about concrete details: code, numbers, dates, exact
quotes, step-by-step instructions, file contents. Conceptual queries ask
about high-level ideas, definitions, or opinions.
"""

from __future__ import annotations

import re

# Patterns that signal a specific/detail-oriented question
_SPECIFIC_PATTERNS: list[re.Pattern[str]] = [
    # Code / technical detail requests
    re.compile(r"\b(show\s+me|display|print|output|paste|give\s+me)\b", re.I),
    re.compile(r"\b(code|function|method|class|variable|import|snippet|implementation)\b", re.I),
    re.compile(r"\b(line\s+\d+|lines?\s+\d+[-–]\d+)\b", re.I),
    re.compile(r"\b(syntax|signature|parameter|argument|return\s+type)\b", re.I),
    # Numbers, dates, exact values
    re.compile(r"\b(how\s+many|how\s+much|count|number\s+of|percentage|ratio)\b", re.I),
    re.compile(r"\b(exact|exactly|specific|precisely|literal)\b", re.I),
    re.compile(r"\b(date|timestamp|version|release|deadline)\b", re.I),
    re.compile(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b"),  # date patterns
    # Step-by-step / procedural
    re.compile(r"\b(step[\s-]by[\s-]step|instructions|how\s+to|tutorial|guide)\b", re.I),
    re.compile(r"\b(example|sample|demo|walkthrough)\b", re.I),
    # File / document content
    re.compile(r"\b(file|document|section|chapter|paragraph|page)\b", re.I),
    re.compile(r"\b(says?\s+about|mentions?|states?|contains?|includes?)\b", re.I),
    re.compile(r"\b(quote|excerpt|passage|text)\b", re.I),
    # Configuration / settings
    re.compile(r"\b(config|setting|option|flag|env|environment)\b", re.I),
    # Error / debugging
    re.compile(r"\b(error|exception|traceback|stack\s+trace|bug|issue)\b", re.I),
]

# Patterns that signal a conceptual/summary question
_CONCEPTUAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(what\s+is|what\s+are|define|definition|meaning)\b", re.I),
    re.compile(r"\b(explain|describe|overview|summary|summarize)\b", re.I),
    re.compile(r"\b(why|purpose|reason|motivation|goal|philosophy)\b", re.I),
    re.compile(r"\b(difference\s+between|compare|contrast|versus|vs)\b", re.I),
    re.compile(r"\b(concept|idea|approach|strategy|pattern|paradigm)\b", re.I),
    re.compile(r"\b(opinion|think|feel|recommend|suggest|advice)\b", re.I),
    re.compile(r"\b(pros?\s+and\s+cons?|trade[\s-]?offs?|advantages?|disadvantages?)\b", re.I),
    re.compile(r"\b(in\s+general|broadly|overall|high[\s-]level|big\s+picture)\b", re.I),
]


def classify_specificity(text: str) -> str:
    """Classify a user message as 'specific' or 'conceptual'.

    Returns:
        "specific" — query wants detailed chunks (code, numbers, exact content)
        "conceptual" — query wants high-level summary only
    """
    specific_score = sum(1 for p in _SPECIFIC_PATTERNS if p.search(text))
    conceptual_score = sum(1 for p in _CONCEPTUAL_PATTERNS if p.search(text))

    # Tie-break: if both or neither match, default to specific (safer to include chunks)
    if specific_score >= conceptual_score:
        return "specific"
    return "conceptual"
