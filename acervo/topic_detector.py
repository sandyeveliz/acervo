"""Topic change detection — 3 levels of ascending cost.

Level 1: Keyword matching (free)
Level 2: Embedding cosine similarity (optional, needs Embedder)
Level 3: LLM classification (needs LLMClient)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from enum import Enum

from acervo.llm import LLMClient, Embedder
from acervo._text import strip_think_blocks

log = logging.getLogger(__name__)


class TopicVerdict(Enum):
    SAME = "same"
    SUBTOPIC = "subtopic"
    CHANGED = "changed"


@dataclass
class DetectionResult:
    verdict: TopicVerdict
    level: int                  # 1, 2, or 3
    confidence: float
    current_topic: str
    detected_topic: str | None
    keyword: str | None = None       # L1
    similarity: float | None = None  # L2
    answer: str | None = None        # L3
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

# --- Level 3: classification prompt ---

_CLASSIFY_PROMPT = """Given the current topic and a new message, classify the relationship.

Current topic: {topic}
New message: {message}

Respond ONLY with a single letter:
a) Same topic
b) New subtopic
c) Different topic

Answer:"""

# --- Stopwords for keyword extraction ---

_STOPWORDS = frozenset(
    "de del la las el los un una uno unas unos al lo le les se nos me te "
    "y e o u a ante con contra en entre para por sin sobre tras desde hacia "
    "que qué es son ser estar fue era soy eres está como cómo más pero si "
    "no ya yo tú él ella eso esto esa ese mi tu su muy hay hoy bien mal "
    "todo toda todos todas otro otra otros otras este esta estos estas "
    "quiero quieres puede pueden hacer hablemos hablamos hablar decir decime "
    "vamos voy tengo tiene tenemos algo nada mucho poco donde cuando ahora "
    "the a an and or but in on at to for of is am are was were be been being "
    "do does did have has had will would shall should can could may might "
    "i you he she it we they me him her us them my your his its our their "
    "this that these those what which who whom how when where why not no "
    "so if up out about into with from by very just also some any all "
    "let lets talk about tell me know want need think".split()
)


@dataclass
class _KnownTopic:
    label: str
    embedding: list[float]


class TopicDetector:
    """Detects topic changes using a 3-level cascade.

    - L1 (keywords): free, always runs
    - L2 (embeddings): fast, runs when embedder is provided
    - L3 (LLM classification): only when L2 is ambiguous or unavailable
    """

    def __init__(
        self,
        llm: LLMClient,
        embedder: Embedder | None = None,
        embed_threshold: float = 0.65,
        classify_prompt: str | None = None,
    ) -> None:
        self._llm = llm
        self._embedder = embedder
        self._embed_threshold = embed_threshold
        self._classify_prompt = classify_prompt or _CLASSIFY_PROMPT
        self._current_topic: str = "none"
        self._current_topic_embedding: list[float] | None = None
        self._known_topics: list[_KnownTopic] = []

    @property
    def current_topic(self) -> str:
        return self._current_topic

    @current_topic.setter
    def current_topic(self, label: str) -> None:
        self._current_topic = label
        self._current_topic_embedding = None

    @property
    def known_topics(self) -> list[str]:
        """Return labels of all topics seen so far."""
        return [kt.label for kt in self._known_topics]

    async def extract_topic_label(self, message: str) -> str:
        """Extract a topic label from a message."""
        clean = strip_think_blocks(message)

        # Strategy 1: match against known topics via embeddings
        if self._known_topics and self._embedder:
            try:
                label = await self._match_known_topic(clean)
                if label:
                    return label
            except Exception:
                pass

        # Strategy 2: use LLM for consistent labeling
        try:
            label = await self._extract_label_via_llm(clean)
            if label:
                if self._embedder:
                    try:
                        emb = await self._embedder.embed(label)
                        self._known_topics.append(_KnownTopic(label=label, embedding=emb))
                    except Exception:
                        pass
                return label
        except Exception:
            pass

        # Strategy 3: fallback to keyword extraction
        return _extract_keywords(clean)

    async def _extract_label_via_llm(self, message: str) -> str:
        prompt = (
            "Extract the main topic of this message in 2-5 words. "
            "Respond ONLY with the topic name, nothing else.\n\n"
            f"Message: {message[:300]}\n\nTopic:"
        )
        raw = await self._llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20,
        )
        raw = strip_think_blocks(raw).strip()
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
        label = raw.strip('"').strip("'").strip()
        label = re.sub(r"[^\w\s\-áéíóúñüÁÉÍÓÚÑÜ]", "", label).strip()
        return label if label and len(label) < 60 else ""

    async def _match_known_topic(self, message: str) -> str | None:
        if not self._embedder:
            return None
        msg_emb = await self._embedder.embed(message)

        best_label: str | None = None
        best_sim = 0.0
        for known in self._known_topics:
            sim = _cosine_similarity(msg_emb, known.embedding)
            if sim > best_sim:
                best_sim = sim
                best_label = known.label

        if best_sim >= self._embed_threshold and best_label:
            return best_label
        return None

    async def detect(self, message: str) -> DetectionResult:
        """Run the 3-level detection cascade."""
        clean_msg = strip_think_blocks(message)

        # Short messages (< 5 words) like "si", "sus peliculas", "cuantas tiene?"
        # are almost always follow-ups, not topic changes
        words = clean_msg.split()
        if len(words) < 5 and self._current_topic != "none":
            return DetectionResult(
                verdict=TopicVerdict.SAME, level=1, confidence=0.9,
                current_topic=self._current_topic, detected_topic=None,
                detail="short_message_followup",
            )

        # Level 1: keywords
        result = self._level1_keywords(clean_msg)
        if result is not None:
            return result

        # No prior topic — first message
        if self._current_topic == "none":
            return DetectionResult(
                verdict=TopicVerdict.CHANGED, level=1, confidence=1.0,
                current_topic="none", detected_topic=None,
            )

        # Level 2: embeddings (if available)
        if self._embedder:
            try:
                result = await self._level2_embeddings(clean_msg)
                if result is not None:
                    return result
            except Exception:
                pass

        # Level 3: LLM classification
        try:
            return await self._level3_llm(clean_msg)
        except Exception:
            return DetectionResult(
                verdict=TopicVerdict.SAME, level=1, confidence=0.0,
                current_topic=self._current_topic, detected_topic=None,
            )

    def _level1_keywords(self, message: str) -> DetectionResult | None:
        match = _CHANGE_RE.search(message)
        if match:
            return DetectionResult(
                verdict=TopicVerdict.CHANGED, level=1, confidence=1.0,
                current_topic=self._current_topic, detected_topic=None,
                keyword=match.group(),
            )
        return None

    async def _level2_embeddings(self, message: str) -> DetectionResult | None:
        if not self._embedder:
            return None

        if self._current_topic_embedding is None:
            self._current_topic_embedding = await self._embedder.embed(self._current_topic)

        msg_emb = await self._embedder.embed(message)
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
        return None  # ambiguous → fall through to L3

    async def _level3_llm(self, message: str) -> DetectionResult:
        prompt = self._classify_prompt.format(
            topic=self._current_topic,
            message=message[:500],
        )
        answer = await self._llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,
        )
        answer = strip_think_blocks(answer).strip().lower()

        if answer.startswith("b"):
            verdict = TopicVerdict.SUBTOPIC
        elif answer.startswith("c"):
            verdict = TopicVerdict.CHANGED
        else:
            verdict = TopicVerdict.SAME

        return DetectionResult(
            verdict=verdict, level=3, confidence=0.8,
            current_topic=self._current_topic, detected_topic=None,
            answer=answer[:10],
        )


def _extract_keywords(text: str, max_words: int = 3) -> str:
    words = re.findall(r"[a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+", text.lower())
    keywords = [w for w in words if w not in _STOPWORDS and len(w) > 2]
    if not keywords:
        keywords = [w for w in words if len(w) > 2][:max_words]
    return " ".join(keywords[:max_words]) if keywords else "sin topic"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
