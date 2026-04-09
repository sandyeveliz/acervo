"""Context builder — ranked chunk selection for LLM context assembly.

Provides RankedChunk-based budget-aware chunk selection (deterministic, no LLM)
and the legacy GatheredInfo/ContextBuilder classes for backward compatibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from acervo.token_counter import count_tokens

log = logging.getLogger(__name__)


# ── Ranked chunk selection (deterministic, no LLM) ──


@dataclass
class RankedChunk:
    """A single piece of context with a relevance score."""

    text: str
    score: float        # 0.0–1.0 relevance score
    source: str         # "graph_fact", "graph_relation", "vector", "file"
    label: str          # entity name or file path (for debug)
    tokens: int = 0     # pre-counted token count


def select_chunks_by_budget(
    chunks: list[RankedChunk],
    token_budget: int,
) -> tuple[list[RankedChunk], int]:
    """Select top-ranked chunks that fit within token budget.

    Iterates chunks in descending score order. Skips chunks that
    would exceed the budget (allows smaller chunks to still fit).

    Returns (selected_chunks, total_tokens_used).
    """
    chunks_sorted = sorted(chunks, key=lambda c: c.score, reverse=True)
    selected: list[RankedChunk] = []
    used = 0
    for chunk in chunks_sorted:
        if used + chunk.tokens > token_budget:
            continue  # skip this chunk, try smaller ones
        selected.append(chunk)
        used += chunk.tokens
    return selected, used


def format_chunks_as_context(chunks: list[RankedChunk]) -> str:
    """Format selected chunks into the warm context block (verbose, one per line)."""
    if not chunks:
        return ""
    return "\n\n".join(c.text for c in chunks)


def format_chunks_compact(chunks: list[RankedChunk]) -> str:
    """Format chunks grouped by entity label for maximum token efficiency.

    Produces compact output like:
        **Sandy**: programador; vive en Cipolletti
        **Chequear**: SaaS de verificacion con NFC; stack React+Vite+Supabase
        **Sandy** -> located_in: Cipolletti; works_at: Alto Valle Studio

    Instead of verbose one-fact-per-block format.
    """
    if not chunks:
        return ""

    # Group by label, separating facts from relations
    facts_by_label: dict[str, list[str]] = {}
    rels_by_label: dict[str, list[str]] = {}

    for c in chunks:
        if "relation" in c.source:
            # Extract relation text after the arrow
            rel_text = c.text.split("\u2192", 1)[-1].strip() if "\u2192" in c.text else c.text
            # Also try ASCII arrow
            if rel_text == c.text and " → " in c.text:
                rel_text = c.text.split(" → ", 1)[-1].strip()
            rels_by_label.setdefault(c.label, []).append(rel_text)
        else:
            # Extract fact text after the colon (strip **Label**: prefix)
            if "**: " in c.text:
                fact_text = c.text.split("**: ", 1)[-1].strip()
            else:
                fact_text = c.text
            facts_by_label.setdefault(c.label, []).append(fact_text)

    lines: list[str] = []

    # Facts first (higher information density)
    for label, facts in facts_by_label.items():
        lines.append(f"**{label}**: {'; '.join(facts)}")

    # Relations for entities not already shown via facts
    for label, rels in rels_by_label.items():
        if label in facts_by_label:
            # Append relations to the existing fact line
            idx = next(i for i, l in enumerate(lines) if l.startswith(f"**{label}**"))
            lines[idx] += f" | {', '.join(rels)}"
        else:
            lines.append(f"**{label}** -> {', '.join(rels)}")

    return "\n".join(lines)


# ── Legacy: GatheredInfo container (kept for debug display) ──

from acervo.prompts import load_prompt

_CONTEXT_BUILD_PROMPT = load_prompt("context_build")


@dataclass
class GatheredInfo:
    """Container for all information gathered in Stage 2."""

    nodes: list[dict] = field(default_factory=list)
    file_contents: dict[str, str] = field(default_factory=dict)  # path → content
    vector_results: list[dict] = field(default_factory=list)

    def format(self) -> str:
        """Render gathered info as text for the context builder prompt."""
        sections: list[str] = []

        # Graph nodes with facts
        if self.nodes:
            sections.append("### From Knowledge Graph")
            for node in self.nodes:
                label = node.get("label", "Unknown")
                ntype = node.get("type", "")
                facts = node.get("facts", [])
                relations = node.get("_relations", [])
                files = node.get("files", [])

                header = f"**{label}** ({ntype})" if ntype else f"**{label}**"
                lines = [header]
                for f in facts:
                    lines.append(f"- {f.get('fact', '')}")
                for rel in relations:
                    lines.append(f"- {rel}")
                if files:
                    lines.append(f"- Linked files: {', '.join(files)}")
                sections.append("\n".join(lines))

        # File contents
        if self.file_contents:
            sections.append("### From Files")
            for path, content in self.file_contents.items():
                # Truncate very long files
                preview = content[:2000] if len(content) > 2000 else content
                sections.append(f"**{path}:**\n{preview}")

        # Vector search results
        if self.vector_results:
            sections.append("### From Semantic Search")
            for hit in self.vector_results:
                source = hit.get("source", "")
                text = hit.get("text", "")
                score = hit.get("score", 0)
                if source == "fact":
                    sections.append(f"- [fact, score={score:.2f}] {text}")
                elif source == "file":
                    fp = hit.get("file_path", "")
                    sections.append(f"- [file: {fp}, score={score:.2f}] {text[:200]}")

        return "\n\n".join(sections) if sections else "No information found."

    @property
    def is_empty(self) -> bool:
        return not self.nodes and not self.file_contents and not self.vector_results


# ── Legacy: ContextBuilder (kept but no longer called from prepare()) ──


class ContextBuilder:
    """Uses the utility LLM to summarize gathered info into optimal context."""

    def __init__(self, llm, prompt_template: str | None = None) -> None:
        """Args:
            llm: object with async chat(messages, temperature, max_tokens) -> str.
            prompt_template: optional override for the context build prompt.
        """
        self._llm = llm
        self._prompt = prompt_template or _CONTEXT_BUILD_PROMPT

    async def build(
        self,
        user_message: str,
        gathered: GatheredInfo,
        token_budget: int = 1500,
    ) -> str:
        """Summarize gathered information into a context block for the main LLM.

        If gathered info is small enough, skip the LLM call and return formatted text directly.
        """
        formatted = gathered.format()

        if gathered.is_empty:
            return ""

        # If gathered info is already concise, skip LLM summarization
        info_tokens = count_tokens(formatted)
        if info_tokens <= token_budget:
            log.info("Context builder: info fits budget (%d <= %d), passing through", info_tokens, token_budget)
            return formatted

        # Use utility LLM to summarize
        log.info("Context builder: summarizing %d tokens → %d budget", info_tokens, token_budget)
        prompt = self._prompt.format(
            user_message=user_message[:300],
            gathered_info=formatted[:4000],
            token_budget=token_budget,
        )

        try:
            response = await self._llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=token_budget + 200,
            )
            result = response if isinstance(response, str) else response.content
            return result.strip()
        except Exception as e:
            log.warning("Context builder LLM failed, using raw format: %s", e)
            # Fallback: truncate formatted info
            return formatted[:token_budget * 4]  # rough char estimate
