"""Backward-compat re-export.

The canonical implementation lives in ``acervo/openai_client.py``. This
module used to hold its own copy, which drifted from the top-level one
and made the dual-dialect fix (OpenAI /v1 vs Ollama /api/chat) hard to
land consistently. It now simply re-exports the public symbols so both
import paths resolve to the same classes.
"""

from __future__ import annotations

from acervo.openai_client import OllamaEmbedder, OpenAIClient

__all__ = ["OllamaEmbedder", "OpenAIClient"]
