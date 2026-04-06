"""Context management — context building, topic detection, synthesis."""

from acervo.context.context_builder import (  # noqa: F401
    RankedChunk, select_chunks_by_budget, format_chunks_compact, format_chunks_as_context,
    GatheredInfo, ContextBuilder,
)
from acervo.context.context_index import ContextIndex  # noqa: F401
from acervo.context.synthesizer import synthesize  # noqa: F401
from acervo.context.topic_detector import TopicDetector, TopicVerdict, DetectionResult  # noqa: F401
from acervo.context.specificity import classify_specificity  # noqa: F401
