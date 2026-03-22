"""acervo — context proxy for AI agents.

Sits between the user and the LLM:
1. prepare() — enrich context from knowledge graph before LLM call
2. process() — extract knowledge from response after LLM call

Public API (stable — safe for external consumers):
    Acervo, AcervoConfig, PrepareResult
    from_config(), is_installed_in()
    LLMClient, Embedder, VectorStore

Internal API (exposed for backward compat — will be removed):
    TopicGraph, _make_id, synthesize, etc.
"""

from __future__ import annotations

from pathlib import Path

from acervo.config import AcervoConfig, ChangelogConfig
from acervo.facade import Acervo, PrepareResult
from acervo.llm import LLMClient, Embedder, VectorStore
from acervo.openai_client import OpenAIClient, OllamaEmbedder
from acervo.project import AcervoProject, find_project, init_project, load_project

# Internal imports — kept for backward compat, will be removed in future
from acervo.graph import TopicGraph, _make_id, make_symbol_id
from acervo.structural_parser import StructuralParser, StructuralUnit, FileStructure
from acervo.context_builder import ContextBuilder, GatheredInfo
from acervo.extractor import (
    ConversationExtractor,
    SearchExtractor,
    TextExtractor,
    RAGExtractor,
    ExtractionResult,
    Entity,
    Relation,
    ExtractedFact,
)
from acervo.file_ingestor import FileIngestor, IngestResult
from acervo.reindexer import Reindexer
from acervo.synthesizer import synthesize
from acervo.layers import Layer, NodeMeta
from acervo.ontology import register_type, register_relation, get_type, all_types
from acervo.query_planner import QueryPlanner, PlanResult
from acervo.topic_detector import TopicDetector, TopicVerdict, DetectionResult
from acervo.context_index import ContextIndex
from acervo.metrics import SessionMetrics, TurnMetric
from acervo.token_counter import count_tokens


# ── Public helper functions ──


def from_config(config_path: Path | None = None) -> Acervo:
    """Create Acervo from config file.

    If config_path is None, searches up from cwd for .acervo/config.toml.
    Raises FileNotFoundError if no config found.
    """
    if config_path is None:
        found = AcervoConfig.find_config()
        if found is None:
            raise FileNotFoundError(
                "No .acervo/config.toml found. Run 'acervo init' in your project root."
            )
        config_path = found

    project = load_project(config_path.parent)
    return Acervo._from_project(project)


def is_installed_in(path: Path | None = None) -> bool:
    """Check if Acervo is initialized in the given path (or cwd).

    Returns True if .acervo/config.toml exists.
    """
    return AcervoConfig.find_config(path) is not None


__all__ = [
    # Public API (stable)
    "Acervo",
    "AcervoConfig",
    "ChangelogConfig",
    "PrepareResult",
    "from_config",
    "is_installed_in",
    "LLMClient",
    "Embedder",
    "VectorStore",
    "OpenAIClient",
    "OllamaEmbedder",
    # Project management
    "AcervoProject",
    "find_project",
    "init_project",
    "load_project",
    # Internal API (backward compat — will be removed)
    "QueryPlanner",
    "PlanResult",
    "TopicDetector",
    "TopicVerdict",
    "DetectionResult",
    "ContextIndex",
    "SessionMetrics",
    "TurnMetric",
    "count_tokens",
    "TopicGraph",
    "_make_id",
    "make_symbol_id",
    "StructuralParser",
    "StructuralUnit",
    "FileStructure",
    "ContextBuilder",
    "GatheredInfo",
    "ConversationExtractor",
    "SearchExtractor",
    "TextExtractor",
    "RAGExtractor",
    "ExtractionResult",
    "Entity",
    "Relation",
    "ExtractedFact",
    "FileIngestor",
    "IngestResult",
    "Reindexer",
    "synthesize",
    "Layer",
    "NodeMeta",
    "register_type",
    "register_relation",
    "get_type",
    "all_types",
]
