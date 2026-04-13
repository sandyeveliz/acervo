# Third-party code in Acervo

Acervo borrows and adapts code from other open-source projects. This file lists
every module that contains code originating outside of Acervo's own repository,
the upstream version it was taken from, and the license terms.

## Graphiti (Apache-2.0)

- **Upstream repo:** https://github.com/getzep/graphiti
- **Upstream package:** `graphiti-core`
- **Version analyzed / ported:** 0.28.2
- **License:** Apache License, Version 2.0
- **Copyright holder:** Zep Software, Inc.

### Files adapted from Graphiti

| Acervo file | Upstream file | Nature of adaptation |
|---|---|---|
| [acervo/extraction/dedup_helpers.py](extraction/dedup_helpers.py) | `graphiti_core/utils/maintenance/dedup_helpers.py` | Near-verbatim copy. TYPE_CHECKING import redirected to Acervo's local `DedupNode`. `_FUZZY_JACCARD_THRESHOLD` lowered from `0.9` to `0.85` to handle Spanish text. `_promote_resolved_node` simplified to work with Acervo's single-string `type` field instead of Graphiti's `labels: list[str]`. |

### How Graphiti inspired Acervo

Beyond direct code ports, Acervo borrows several architectural ideas from
Graphiti that are implemented from scratch rather than copied:

- Hybrid deterministic + LLM entity resolution (exact-normalize → MinHash LSH
  → entropy gate → LLM escalation).
- Bi-temporal fact model (`valid_at` / `invalid_at` / `expired_at` /
  `reference_time`) with LLM-driven contradiction detection and deterministic
  temporal arbitration. *(Phase 3)*
- Reciprocal Rank Fusion and Maximal Marginal Relevance for hybrid retrieval
  over multiple signals (BM25, vector, graph BFS). *(Phase 4)*
- Entity extraction prompt guardrails ("NEVER extract X" rules for pronouns,
  bare relational terms, generic nouns) adapted and translated to Spanish.

See [`docs/research/graphiti-analysis.md`](../docs/research/graphiti-analysis.md)
for the full analysis that informed these decisions.

### License text (Apache-2.0)

Each adapted file retains the full Apache-2.0 copyright header from Graphiti.
The license text is reproduced here for convenience:

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
