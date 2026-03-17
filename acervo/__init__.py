"""acervo — episodic memory graph for AVS-Agents.

Exporta las clases y funciones principales del paquete.
"""

from acervo.graph import TopicGraph, _make_id
from acervo.extractor import (
    ConversationExtractor,
    SearchExtractor,
    RAGExtractor,
    ExtractionResult,
    Entity,
    Relation,
    ExtractedFact,
)
from acervo.layers import Layer
from acervo.ontology import register_type, register_relation, get_type, all_types

__all__ = [
    "TopicGraph",
    "_make_id",
    "ConversationExtractor",
    "SearchExtractor",
    "RAGExtractor",
    "ExtractionResult",
    "Entity",
    "Relation",
    "ExtractedFact",
    "Layer",
    "register_type",
    "register_relation",
    "get_type",
    "all_types",
]


if __name__ == "__main__":
    # Ejemplo de nodos con capas y ontología
    from acervo.layers import Layer
    from acervo.ontology import BUILTIN_ENTITY_TYPES

    print("\n=== acervo — ejemplo de nodos ===\n")

    sandy = {
        "label": "Sandy",
        "type": "Persona",
        "layer": Layer.PERSONAL.name,
        "source": "user_assertion",
        "confidence_for_owner": 1.0,
        "status": "complete",
        "pending_fields": [],
        "facts": [
            {"fact": "Sandy es dueña de AltoValleStudio", "source": "user_assertion"}
        ],
        "relations": [
            {"relation": "DUEÑO_DE", "target": "AltoValleStudio"}
        ],
    }

    cipolletti = {
        "label": "Cipolletti",
        "type": "Lugar",
        "layer": Layer.UNIVERSAL.name,
        "source": "world",
        "confidence_for_owner": 1.0,
        "status": "complete",
        "pending_fields": [],
        "facts": [
            {"fact": "Cipolletti es una ciudad en la provincia de Río Negro, Argentina", "source": "world"}
        ],
    }

    for node in [sandy, cipolletti]:
        capa = f"Capa {'1 (UNIVERSAL)' if node['layer'] == 'UNIVERSAL' else '2 (PERSONAL)'}"
        print(f"  Nodo: {node['label']}")
        print(f"    tipo     : {node['type']}")
        print(f"    capa     : {capa}")
        print(f"    source   : {node['source']}")
        print(f"    status   : {node['status']}")
        print(f"    facts    : {[f['fact'] for f in node['facts']]}")
        if node.get("relations"):
            print(f"    relaciones: {node['relations']}")
        print()
