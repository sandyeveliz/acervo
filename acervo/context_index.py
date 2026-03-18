"""Context Index — manages the 3-layer context stack.

Builds a constant-size context window from:
  [System prompt]  — fixed, KV cached
  [Warm layer]     — graph context + compacted topics
  [Hot layer]      — last N message pairs
  [User message]   — always included

The LLM sees ~1.5K-2K tokens regardless of conversation depth.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from datetime import datetime
from pathlib import Path

from acervo.graph import TopicGraph
from acervo.llm import LLMClient
from acervo.synthesizer import synthesize
from acervo.token_counter import count_tokens
from acervo._text import strip_think_blocks

log = logging.getLogger(__name__)

_TOPICS_DIR = Path("data/topics")

_SUMMARIZE_PROMPT = """Resumí los hechos clave de estos mensajes en viñetas.
Solo incluir hechos explícitos dichos por el usuario o confirmados en la conversación.
NO incluir saludos, preguntas ni especulación.
Responder siempre en español.
Output SOLO las viñetas, una por línea, empezando con "- ".

Mensajes:
{messages}

Resumen:"""


class ContextIndex:
    """Builds and manages the context stack sent to the LLM each turn."""

    def __init__(
        self,
        graph: TopicGraph,
        llm: LLMClient,
        hot_layer_max_messages: int = 2,
        hot_layer_max_tokens: int = 500,
        compaction_trigger_tokens: int = 2000,
        topics_dir: str | Path = "data/topics",
    ) -> None:
        self._graph = graph
        self._llm = llm
        self._hot_max_messages = hot_layer_max_messages
        self._hot_max_tokens = hot_layer_max_tokens
        self._compaction_trigger = compaction_trigger_tokens
        self._topics_dir = Path(topics_dir)
        self._topics_dir.mkdir(parents=True, exist_ok=True)
        # Track sliding window for eviction detection
        self._last_included_pairs: int = 0
        self._last_total_pairs: int = 0

    def build_context_stack(
        self,
        history: list[dict],
        current_topic: str,
        warm_override: str = "",
        warm_source: str = "",
    ) -> tuple[list[dict], int, int, int]:
        """Build the filtered context stack for the LLM.

        Args:
            history: list of {"role": str, "content": str} messages
            current_topic: current topic label
            warm_override: pre-built warm content (from executor)
            warm_source: source of warm content ("graph", "web", etc.)

        Returns:
            (context_messages, hot_tokens, warm_tokens, total_tokens)
        """
        if len(history) < 2:
            total = sum(count_tokens(m["content"]) for m in history)
            return list(history), total, 0, total

        system_msg = history[0]
        conversation = history[1:]
        current_user_msg = conversation[-1] if conversation else None

        # Warm layer (graph context)
        user_text = current_user_msg["content"] if current_user_msg else ""
        if warm_override:
            warm_content = warm_override
        else:
            warm_content = synthesize(self._graph, user_text)
        warm_tokens = count_tokens(warm_content) if warm_content else 0

        # Persisted topic .md file
        md_content, md_tokens = self._load_warm_content(current_topic)
        if md_content:
            warm_content = f"{warm_content}\n\n{md_content}" if warm_content else md_content
            warm_tokens += md_tokens

        # Hot layer budget
        target_total = self._hot_max_tokens + warm_tokens + count_tokens(system_msg["content"])
        target_total = max(target_total, 2000)
        system_tk = count_tokens(system_msg["content"])
        user_tk = count_tokens(user_text) if current_user_msg else 0
        overhead = 10
        hot_budget = max(target_total - system_tk - warm_tokens - user_tk - overhead, 200)

        prev_messages = conversation[:-1] if len(conversation) > 1 else []

        # Collect user/assistant pairs backwards
        pairs: list[tuple[dict, dict]] = []
        i = len(prev_messages) - 1
        while i >= 1:
            if prev_messages[i]["role"] == "assistant" and prev_messages[i - 1]["role"] == "user":
                pairs.append((prev_messages[i - 1], prev_messages[i]))
                i -= 2
            else:
                i -= 1

        hot_messages: list[dict] = []
        hot_tokens = 0
        for user_msg, asst_msg in pairs[:self._hot_max_messages]:
            pair_tk = count_tokens(user_msg["content"]) + count_tokens(asst_msg["content"])
            if hot_tokens + pair_tk > hot_budget:
                break
            hot_messages = [user_msg, asst_msg] + hot_messages
            hot_tokens += pair_tk

        self._last_included_pairs = len(hot_messages) // 2
        self._last_total_pairs = len(pairs)

        # Build stack
        stack: list[dict] = [system_msg]

        if warm_content:
            if warm_source == "web":
                ctx_block = (
                    "[RESULTADOS DE BÚSQUEDA WEB — datos reales y actualizados]\n"
                    f"{warm_content}\n"
                    "[FIN RESULTADOS]\n"
                    "Usá estos resultados para responder al usuario. Podés citar las fuentes."
                )
            else:
                ctx_block = f"[CONTEXTO VERIFICADO]\n{warm_content}\n[FIN CONTEXTO]"
            stack.append({"role": "user", "content": ctx_block})
            stack.append({"role": "assistant", "content": "Entendido."})

        stack.extend(hot_messages)

        if current_user_msg:
            stack.append(current_user_msg)

        total_tokens = (
            system_tk + warm_tokens + hot_tokens
            + (count_tokens(user_text) if current_user_msg else 0)
            + (count_tokens("Entendido.") if warm_content else 0)
        )

        return stack, hot_tokens, warm_tokens, total_tokens

    async def maybe_compact(
        self,
        history: list[dict],
        current_topic: str,
    ) -> bool:
        """If hot layer exceeds threshold, compact overflow to warm layer."""
        if current_topic == "none" or len(history) < 2:
            return False

        conversation = history[1:]
        if len(conversation) <= self._hot_max_messages:
            return False

        overflow = conversation[:-self._hot_max_messages]
        overflow_tokens = sum(count_tokens(m["content"]) for m in overflow)

        if overflow_tokens < self._compaction_trigger:
            return False

        summary = await self._summarize_messages(overflow)
        if summary:
            self._update_topic_file(current_topic, summary)
            return True
        return False

    def _load_warm_content(self, current_topic: str) -> tuple[str, int]:
        if current_topic == "none":
            return "", 0
        topic_id = _make_topic_id(current_topic)
        md_path = self._topics_dir / f"{topic_id}.md"
        if md_path.exists():
            content = md_path.read_text(encoding="utf-8").strip()
            if content:
                return content, count_tokens(content)
        return "", 0

    async def _summarize_messages(self, messages: list[dict]) -> str:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        prompt = _SUMMARIZE_PROMPT.format(messages=text[:2000])
        try:
            raw = await self._llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=400,
            )
            raw = strip_think_blocks(raw).strip()
            raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
            return raw
        except Exception as e:
            log.warning("Summarization failed: %s", e)
            return ""

    def _update_topic_file(self, topic_label: str, summary: str) -> None:
        topic_id = _make_topic_id(topic_label)
        md_path = self._topics_dir / f"{topic_id}.md"
        now = datetime.now().strftime("%Y-%m-%d")

        if md_path.exists():
            existing = md_path.read_text(encoding="utf-8")
            if "## Hechos conocidos" in existing:
                dated_facts = "\n".join(
                    f"{line} [{now}]" if not line.endswith("]") else line
                    for line in summary.split("\n")
                    if line.strip().startswith("- ")
                )
                if dated_facts:
                    existing = existing.rstrip() + "\n" + dated_facts + "\n"
                md_path.write_text(existing, encoding="utf-8")
            else:
                existing = existing.rstrip() + f"\n\n## Hechos conocidos\n{summary}\n"
                md_path.write_text(existing, encoding="utf-8")
        else:
            content = f"# {topic_label}\n\n**Última actividad:** {now}\n\n## Hechos conocidos\n{summary}\n"
            md_path.write_text(content, encoding="utf-8")


def _make_topic_id(label: str) -> str:
    nfkd = unicodedata.normalize("NFKD", label.lower())
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", ascii_str).strip("_")
