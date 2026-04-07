"""S1.5 Updater — domain layer re-export of S1.5 Graph Update.

The actual implementation lives in acervo.s1_5_graph_update.
This module provides the domain-layer interface.
"""

from acervo.s1_5_graph_update import (  # noqa: F401
    S1_5GraphUpdate,
    S1_5Result,
    apply_s1_5_result,
    MergeAction,
    TypeCorrection,
    DiscardAction,
)
