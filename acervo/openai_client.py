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
    """Minimal async-compatible OpenAI chat client using stdlib only.

    Usage:
        client = OpenAIClient(
            base_url="http://localhost:1234/v1",
            model="qwen3.5-9b",
            api_key="lm-studio",
        )
        response = await client.chat([{"role": "user", "content": "hi"}])
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: int = 120,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> str:
        """Send chat completion request. Returns response content text."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._chat_sync, messages, temperature, max_tokens,
        )

    def _chat_sync(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        url = f"{self._base_url}/chat/completions"
        payload = json.dumps({
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode()

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        req = Request(url, data=payload, headers=headers, method="POST")
        with urlopen(req, timeout=self._timeout) as resp:
            data = json.loads(resp.read())

        return data["choices"][0]["message"]["content"]
