"""S3 Assembler — context assembly from S2 output.

Ranks chunks by score, selects within token budget by intent,
builds the warm context string, and delegates to ContextIndex
for the final context stack.
"""

from __future__ import annotations

import logging
from typing import Any

from acervo.context_builder import select_chunks_by_budget, format_chunks_compact
from acervo.domain.models import RankedChunk, S3Result
from acervo.token_counter import count_tokens

log = logging.getLogger(__name__)

# Budget per intent
_BUDGETS = {
    "overview": 250,
    "specific": 600,
    "followup": 400,
    "chat": 0,
}


class S3Assembler:
    """Assembles context from S2 chunks within budget constraints."""

    def run(
        self,
        chunks: list[RankedChunk],
        intent: str,
        history: list[dict],
        project_overview: str,
        context_index: Any,  # ContextIndex instance
        current_topic: str = "none",
        warm_budget_override: int | None = None,
    ) -> S3Result:
        """Execute S3: select chunks, build context stack.

        Args:
            chunks: ranked chunks from S2
            intent: S1 intent (overview, specific, chat, followup)
            history: conversation history (for hot layer)
            project_overview: project description text (from graph/config)
            context_index: ContextIndex for building the final stack
            current_topic: current topic label
            warm_budget_override: override default budget for intent
        """
        # Calculate budget — intent controls budget, NOT whether to inject
        warm_budget = warm_budget_override or _BUDGETS.get(intent, 400)
        overview_tokens = count_tokens(project_overview) if project_overview else 0
        chunk_budget = max(warm_budget - overview_tokens, 50)

        # ALWAYS assemble from ALL chunks — no filtering by intent
        warm_override = self._assemble(chunks, chunk_budget, project_overview)

        warm_source = "graph" if warm_override else ""

        # Build context stack via ContextIndex
        context_stack, hot_tk, warm_tk, total_tk = context_index.build_context_stack(
            history, current_topic,
            warm_override=warm_override,
            warm_source=warm_source,
        )

        has_context = warm_tk > 0

        log.info(
            "[acervo] S3 — warm=%dtk hot=%dtk total=%dtk has_context=%s intent=%s",
            warm_tk, hot_tk, total_tk, has_context, intent,
        )

        return S3Result(
            context_stack=context_stack,
            warm_content=warm_override,
            warm_tokens=warm_tk,
            hot_tokens=hot_tk,
            total_tokens=total_tk,
            has_context=has_context,
            needs_tool=not has_context,
        )

    def _assemble(
        self, chunks: list[RankedChunk], budget: int, overview: str,
    ) -> str:
        """Assemble warm context from ALL chunks within budget.

        Intent controls the budget (set by caller), not what gets included.
        Verified and conversation chunks are separated with markers.
        """
        selected, _ = select_chunks_by_budget(chunks, budget)
        verified = [c for c in selected if c.source.startswith("verified")]
        conversation = [c for c in selected if not c.source.startswith("verified")]

        parts: list[str] = []
        if overview:
            parts.append(overview)
        if verified:
            parts.append(format_chunks_compact(verified))
        if conversation:
            if parts:
                parts.append("")
            parts.append("[UNVERIFIED]\n" + format_chunks_compact(conversation))
        return "\n".join(parts) if parts else ""
