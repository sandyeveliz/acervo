"""acervo — context proxy for AI agents.

Sits between the user and the LLM:
1. prepare() — enrich context from knowledge graph before LLM call
2. process() — extract knowledge from response after LLM call
"""

from acervo.facade import Acervo, PrepareResult
from acervo.graph import TopicGraph, _make_id
from acervo.llm import LLMClient, Embedder
from acervo.openai_client import OpenAIClient
from acervo.extractor import (
    ConversationExtractor,
    SearchExtractor,
    RAGExtractor,
    ExtractionResult,
    Entity,
    Relation,
    ExtractedFact,
)
from acervo.synthesizer import synthesize
from acervo.layers import Layer, NodeMeta
from acervo.ontology import register_type, register_relation, get_type, all_types
from acervo.query_planner import QueryPlanner, PlanResult
from acervo.topic_detector import TopicDetector, TopicVerdict, DetectionResult
from acervo.context_index import ContextIndex
from acervo.token_counter import count_tokens

__all__ = [
    # High-level API
    "Acervo",
    "PrepareResult",
    "LLMClient",
    "Embedder",
    "OpenAIClient",
    # Pipeline components
    "QueryPlanner",
    "PlanResult",
    "TopicDetector",
    "TopicVerdict",
    "DetectionResult",
    "ContextIndex",
    "count_tokens",
    # Graph
    "TopicGraph",
    "_make_id",
    # Extractors
    "ConversationExtractor",
    "SearchExtractor",
    "RAGExtractor",
    "ExtractionResult",
    "Entity",
    "Relation",
    "ExtractedFact",
    # Synthesizer
    "synthesize",
    # Layers & ontology
    "Layer",
    "NodeMeta",
    "register_type",
    "register_relation",
    "get_type",
    "all_types",
]
