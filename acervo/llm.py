"""LLM client protocols — backward-compatible re-exports from ports/.

New code should import from acervo.ports directly.
"""

from acervo.ports.llm import LLMPort as LLMClient  # noqa: F401
from acervo.ports.embedder import EmbedderPort as Embedder  # noqa: F401
from acervo.ports.vector_store import VectorStorePort as VectorStore  # noqa: F401
