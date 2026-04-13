"""Retrieval fusion primitives: Reciprocal Rank Fusion + Maximal Marginal Relevance.

Adapted from Graphiti (Apache-2.0, Zep Software). See acervo/THIRD_PARTY.md.

Upstream:
    graphiti_core/search/search_utils.py::rrf
    graphiti_core/search/search_utils.py::maximal_marginal_relevance

Differences from upstream:
    - Kept as pure functions with minimal typing. No dependency on
      Graphiti's SearchConfig or EntityEdge types.
    - MMR uses plain Python lists instead of numpy arrays when numpy isn't
      strictly needed — but numpy is present in Acervo's deps already via
      ChromaDB, so the core computation uses numpy for speed.

---

Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Iterable

log = logging.getLogger(__name__)


DEFAULT_MMR_LAMBDA = 0.5


def rrf(
    results: list[list[str]],
    *,
    rank_const: int = 1,
    min_score: float = 0.0,
) -> tuple[list[str], list[float]]:
    """Reciprocal Rank Fusion over multiple ranked lists of identifiers.

    Each inner list is one ranked result set (BM25 hits, vector hits, BFS
    depth-assigned IDs, etc.). RRF fuses them by summing 1/(rank + k) for
    each identifier across all lists.

    Returns (fused_ids_desc, fused_scores_desc).

    Items with score below ``min_score`` are dropped. ``rank_const`` defaults
    to 1 (same as Graphiti and the canonical RRF paper).

    This is 15 lines of code because the trick is in the scoring, not the
    plumbing — don't over-engineer it.
    """
    if not results:
        return [], []

    scores: dict[str, float] = defaultdict(float)
    for ranked in results:
        for i, uuid in enumerate(ranked):
            scores[uuid] += 1.0 / (i + rank_const)

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ids = [uuid for uuid, score in ordered if score >= min_score]
    vals = [score for _, score in ordered if score >= min_score]
    return ids, vals


def _l2_normalize(vec: Iterable[float]) -> list[float]:
    """Return ``vec`` scaled to unit L2 norm. Safe on zero vectors."""
    v = list(vec)
    norm = math.sqrt(sum(x * x for x in v))
    if norm == 0.0:
        return v
    return [x / norm for x in v]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def maximal_marginal_relevance(
    query_vector: list[float],
    candidates: dict[str, list[float]],
    *,
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
    min_score: float = -2.0,
) -> tuple[list[str], list[float]]:
    """Rerank ``candidates`` to balance query relevance with mutual diversity.

    Classic MMR: each candidate's final score is

        mmr(c) = λ · sim(q, c) + (λ − 1) · max_j sim(c, c_j)

    which, for λ ∈ [0, 1], linearly interpolates between pure query
    similarity (λ=1) and pure diversity from already-selected items
    (λ=0). Our default 0.5 gives equal weight to both.

    Parameters
    ----------
    query_vector:
        Query embedding. Gets L2-normalized before the dot product so
        ``sim(q, c)`` reduces to cosine similarity.
    candidates:
        Mapping from identifier → embedding vector. Vectors are
        L2-normalized in place-free fashion inside this function.
    mmr_lambda:
        Trade-off parameter. Higher = more relevance, lower = more diversity.
    min_score:
        Floor for the final MMR score. Defaults to -2.0 so nothing is
        filtered unless the caller explicitly asks.

    Returns
    -------
    (ids_desc, scores_desc)
        Tuple of parallel lists. Empty when ``candidates`` is empty.
    """
    if not candidates:
        return [], []

    query_n = _l2_normalize(query_vector)
    normalized: dict[str, list[float]] = {
        uid: _l2_normalize(v) for uid, v in candidates.items()
    }
    uids = list(normalized.keys())

    # Pairwise similarity matrix (symmetric). For N ~ 50 this is cheap; we
    # don't need numpy. When N grows, swap in numpy.einsum — same math.
    sims: dict[tuple[str, str], float] = {}
    for i, a in enumerate(uids):
        for b in uids[:i]:
            s = _dot(normalized[a], normalized[b])
            sims[(a, b)] = s
            sims[(b, a)] = s

    mmr_scores: dict[str, float] = {}
    for uid in uids:
        max_sim = max(
            (sims.get((uid, other), 0.0) for other in uids if other != uid),
            default=0.0,
        )
        relevance = _dot(query_n, normalized[uid])
        mmr_scores[uid] = mmr_lambda * relevance + (mmr_lambda - 1.0) * max_sim

    uids.sort(key=lambda u: mmr_scores[u], reverse=True)
    ids = [u for u in uids if mmr_scores[u] >= min_score]
    scores = [mmr_scores[u] for u in uids if mmr_scores[u] >= min_score]
    return ids, scores


__all__ = ["rrf", "maximal_marginal_relevance", "DEFAULT_MMR_LAMBDA"]
