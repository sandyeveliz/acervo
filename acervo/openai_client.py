"""Built-in OpenAI-compatible LLM and embedding clients for standalone mode.

Works with any OpenAI-compatible API: LM Studio, Ollama, OpenAI, etc.
No external dependencies — uses stdlib urllib only.
"""

from __future__ import annotations

import json
import logging
from urllib.request import Request, urlopen

log = logging.getLogger(__name__)


class OllamaEmbedder:
    """Embedder using Ollama's /api/embed endpoint (stdlib only).

    Usage:
        embedder = OllamaEmbedder(
            base_url="http://localhost:11434",
            model="qwen3-embedding",
        )
        vector = await embedder.embed("hello world")
    """

    def __init__(self, base_url: str, model: str, timeout: int = 120) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    async def embed(self, text: str) -> list[float]:
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embed_sync, text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single HTTP call."""
        if not texts:
            return []
        if len(texts) == 1:
            return [await self.embed(texts[0])]
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embed_batch_sync, texts)

    def _embed_sync(self, text: str) -> list[float]:
        url = f"{self._base_url}/api/embed"
        payload = json.dumps({"model": self._model, "input": text}).encode()
        headers = {"Content-Type": "application/json"}
        req = Request(url, data=payload, headers=headers, method="POST")
        with urlopen(req, timeout=self._timeout) as resp:
            data = json.loads(resp.read())
        return data["embeddings"][0]

    def _embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        url = f"{self._base_url}/api/embed"
        payload = json.dumps({"model": self._model, "input": texts}).encode()
        headers = {"Content-Type": "application/json"}
        req = Request(url, data=payload, headers=headers, method="POST")
        with urlopen(req, timeout=self._timeout) as resp:
            data = json.loads(resp.read())
        return data["embeddings"]


class OpenAIClient:
    """Minimal async chat client that speaks two dialects.

    Two backends are supported:

    * ``api_style="openai"`` (default) — targets the OpenAI-compatible
      ``/v1/chat/completions`` endpoint. Works with OpenAI proper, LM
      Studio, vLLM, Groq, and Ollama's /v1 surface. Reads the answer
      from ``choices[0].message.content``.

    * ``api_style="ollama"`` — targets Ollama's native ``/api/chat``
      endpoint. Lets us pass the top-level ``think: false`` flag that
      /v1 silently ignores on qwen3 / qwq thinking models. Translates
      ``temperature`` and ``max_tokens`` into Ollama's ``options``
      block, reads the answer from ``message.content``. This is the
      ONLY way to keep qwen3.5 from spending its entire token budget
      on reasoning when the prompt is non-trivial — verified against
      Ollama 0.x where /v1 + chat_template_kwargs still emits thinking
      into ``message.reasoning`` and leaves ``content`` empty.

    Usage (openai default)::

        client = OpenAIClient(
            base_url="http://localhost:1234/v1",
            model="qwen2.5:7b",
            api_key="ollama",
        )

    Usage (Ollama native with thinking off)::

        client = OpenAIClient(
            base_url="http://localhost:11434",
            model="qwen3.5:9b",
            api_style="ollama",
            think=False,
        )
        # base_url may end in /v1 — the ``ollama`` style strips it so
        # you can point both styles at the same Ollama daemon without
        # fussing with two URLs.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: int = 120,
        extra_body: dict | None = None,
        api_style: str = "openai",
        think: bool | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        if api_style not in ("openai", "ollama"):
            raise ValueError(
                f"api_style must be 'openai' or 'ollama', got {api_style!r}"
            )
        self._api_style = api_style
        # Only meaningful for ``api_style="ollama"``. None means "don't send the
        # flag at all"; False / True explicitly set it.
        self._think = think
        # Extra fields merged into every chat completion request body.
        # Use this to pass model/runtime-specific knobs that aren't in the
        # OpenAI spec — e.g. ``chat_template_kwargs={"enable_thinking": False}``
        # on vLLM. For Ollama use ``api_style="ollama"`` + ``think=False``
        # instead; Ollama's /v1 surface ignores chat_template_kwargs.
        self._extra_body = dict(extra_body or {})

    # ── Public API (dialect-agnostic) ────────────────────────────────────

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 500,
        json_mode: bool = False,
    ) -> str:
        """Send chat completion request. Returns response content text."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._chat_sync, messages, temperature, max_tokens, json_mode,
        )

    def _chat_sync(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool = False,
    ) -> str:
        if self._api_style == "ollama":
            return self._chat_sync_ollama(
                messages, temperature, max_tokens, json_mode,
            )
        return self._chat_sync_openai(
            messages, temperature, max_tokens, json_mode,
        )

    # ── OpenAI-compatible dialect (/v1/chat/completions) ─────────────────

    def _chat_sync_openai(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        url = f"{self._base_url}/chat/completions"
        body: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        # Merge runtime extras (e.g. chat_template_kwargs for qwen3 thinking toggle).
        for k, v in self._extra_body.items():
            if k not in ("model", "messages", "temperature", "max_tokens"):
                body[k] = v
        payload = json.dumps(body).encode()

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        req = Request(url, data=payload, headers=headers, method="POST")
        with urlopen(req, timeout=self._timeout) as resp:
            data = json.loads(resp.read())

        return data["choices"][0]["message"]["content"]

    # ── Ollama native dialect (/api/chat) ────────────────────────────────

    def _ollama_base(self) -> str:
        """Return the Ollama daemon root (strip trailing /v1 if present)."""
        base = self._base_url
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        return base

    def _chat_sync_ollama(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        url = f"{self._ollama_base()}/api/chat"
        options: dict = {
            "temperature": temperature,
            # Ollama uses ``num_predict`` for the OpenAI ``max_tokens`` concept.
            "num_predict": max_tokens,
        }
        body: dict = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if self._think is not None:
            body["think"] = bool(self._think)
        if json_mode:
            # Ollama honours ``format: "json"`` for structured output at the
            # body top level on /api/chat.
            body["format"] = "json"
        # Caller-supplied extras get merged at top level, last-wins, but can't
        # clobber reserved fields.
        for k, v in self._extra_body.items():
            if k in ("model", "messages", "stream", "options"):
                continue
            body[k] = v

        payload = json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        req = Request(url, data=payload, headers=headers, method="POST")
        with urlopen(req, timeout=self._timeout) as resp:
            data = json.loads(resp.read())

        msg = (data or {}).get("message") or {}
        content = msg.get("content", "")
        # Belt and suspenders: if thinking somehow still slipped through and
        # went to a separate field, surface it to the caller via log so we
        # notice. Don't concatenate — our parsers expect JSON in ``content``.
        reasoning = msg.get("reasoning") or msg.get("thinking")
        if reasoning and not content:
            log.warning(
                "Ollama returned empty content but has reasoning (%d chars). "
                "think flag may not have been honoured. First 200 chars: %.200s",
                len(reasoning), reasoning,
            )
        return content or ""
