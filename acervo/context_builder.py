"""Backward compat — re-export from acervo.context.context_builder."""
from acervo.context.context_builder import *  # noqa: F401, F403
from acervo.context.context_builder import (  # noqa: F401
    RankedChunk, select_chunks_by_budget, format_chunks_compact,
    format_chunks_as_context, GatheredInfo, ContextBuilder,
)
