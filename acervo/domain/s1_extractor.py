"""S1 Extractor — domain layer re-export of S1 Unified.

The actual implementation lives in acervo.s1_unified (the extractor
model orchestrator). This module provides the domain-layer interface.
"""

from acervo.s1_unified import (  # noqa: F401
    S1Unified,
    S1Result,
    TopicResult,
    build_graph_summary,
    generate_topic_hint,
)
