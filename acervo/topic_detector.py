"""Topic change detection — 2 levels (L1 keywords + L2 embeddings).

L1: Keyword matching (free, always runs)
L2: Embedding cosine similarity (optional, needs Embedder)

Topic classification is handled by S1 Unified (LLM). L1/L2 only produce
hints that S1 uses as input — they are not gates.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

from acervo.llm import Embedder
from acervo._text import strip_think_blocks

log = logging.getLogger(__name__)


class TopicVerdict(Enum):
    SAME = "same"
    SUBTOPIC = "subtopic"
    CHANGED = "changed"


@dataclass
class DetectionResult:
    verdict: TopicVerdict
    level: int                  # 0=unresolved, 1=keyword, 2=embedding
    confidence: float
    current_topic: str
    detected_topic: str | None
    keyword: str | None = None       # L1
    similarity: float | None = None  # L2
    detail: str = ""


# --- Level 1: keyword patterns ---

_CHANGE_PATTERNS_ES = [
    r"cambiando de tema", r"cambiemos de tema", r"otra cosa",
    r"ahora sobre", r"hablando de otra cosa", r"pasemos a",
    r"dejando eso", r"volviendo a", r"te quiero preguntar sobre",
    r"te pregunto otra cosa", r"nada que ver pero",
]

_CHANGE_PATTERNS_EN = [
    r"changing topic", r"on another note", r"switching to",
    r"let'?s talk about", r"moving on to", r"by the way",
    r"unrelated,? but", r"different question", r"going back to",
]

_CHANGE_RE = re.compile(
    "|".join(_CHANGE_PATTERNS_ES + _CHANGE_PATTERNS_EN),
    re.IGNORECASE,
)


class TopicDetector:
    """Produces topic hints via L1 (keywords) + L2 (embeddings).

    S1 Unified makes the final topic decision. This class only provides
    fast, cheap signals to guide S1.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        embed_threshold: float = 0.65,
    ) -> None:
        self._embedder = embedder
        self._embed_threshold = embed_threshold
        self._current_topic: str = "none"
        self._current_topic_embedding: list[float] | None = None

    @property
    def current_topic(self) -> str:
        return self._current_topic

    @current_topic.setter
    def current_topic(self, label: str) -> None:
        self._current_topic = label
        self._current_topic_embedding = None

    async def detect_hints(
        self, message: str, pre_embedding: list[float] | None = None,
    ) -> DetectionResult:
        """Run L1+L2 only — returns hints for S1 Unified (no LLM call).

        The result provides keyword/similarity data that S1 Unified uses
        as a topic hint. S1 makes the final topic decision.
        """
        clean_msg = strip_think_blocks(message)

        # Short messages — likely follow-ups
        words = clean_msg.split()
        if len(words) < 5 and self._current_topic != "none":
            return DetectionResult(
                verdict=TopicVerdict.SAME, level=1, confidence=0.9,
                current_topic=self._current_topic, detected_topic=None,
                detail="short_message_followup",
            )

        # L1: keywords
        result = self._level1_keywords(clean_msg)
        if result is not None:
            return result

        # No prior topic — first message
        if self._current_topic == "none":
            return DetectionResult(
                verdict=TopicVerdict.CHANGED, level=1, confidence=1.0,
                current_topic="none", detected_topic=None,
            )

        # L2: embeddings (if available)
        if self._embedder:
            try:
                result = await self._level2_embeddings(clean_msg, pre_embedding=pre_embedding)
                if result is not None:
                    return result
            except Exception:
                pass

        # No L3 — return unresolved so S1 Unified classifies itself
        return DetectionResult(
            verdict=TopicVerdict.SAME, level=0, confidence=0.0,
            current_topic=self._current_topic, detected_topic=None,
            detail="unresolved",
        )

    # Backward compat alias
    async def detect(self, message: str) -> DetectionResult:
        """Alias for detect_hints(). Kept for backward compatibility."""
        return await self.detect_hints(message)

    def _level1_keywords(self, message: str) -> DetectionResult | None:
        match = _CHANGE_RE.search(message)
        if match:
            return DetectionResult(
                verdict=TopicVerdict.CHANGED, level=1, confidence=1.0,
                current_topic=self._current_topic, detected_topic=None,
                keyword=match.group(),
            )
        return None

    async def _level2_embeddings(
        self, message: str, pre_embedding: list[float] | None = None,
    ) -> DetectionResult | None:
        if not self._embedder:
            return None

        if self._current_topic_embedding is None:
            self._current_topic_embedding = await self._embedder.embed(self._current_topic)

        msg_emb = pre_embedding or await self._embedder.embed(message)
        similarity = _cosine_similarity(self._current_topic_embedding, msg_emb)

        if similarity >= 0.80:
            return DetectionResult(
                verdict=TopicVerdict.SAME, level=2, confidence=similarity,
                current_topic=self._current_topic, detected_topic=None,
                similarity=similarity,
            )
        if similarity < self._embed_threshold:
            return DetectionResult(
                verdict=TopicVerdict.CHANGED, level=2, confidence=1.0 - similarity,
                current_topic=self._current_topic, detected_topic=None,
                similarity=similarity,
            )
        return None  # ambiguous range — S1 Unified will decide


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
