"""Backward compat — re-export from acervo.graph.ontology."""
from acervo.graph.ontology import *  # noqa: F401, F403
from acervo.graph.ontology import (  # noqa: F401
    map_extractor_type, register_type, register_relation,
    get_type, all_types, all_relations, is_known_type, is_likely_universal,
)
