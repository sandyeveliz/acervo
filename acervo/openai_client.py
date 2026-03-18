"""Built-in OpenAI-compatible LLM client for standalone mode.

Works with any OpenAI-compatible API: LM Studio, Ollama, OpenAI, etc.
No external dependencies — uses stdlib urllib only.
"""

from __future__ import annotations

import json
import logging
from urllib.request import Request, urlopen

log = logging.getLogger(__name__)


class OpenAIClient:
    """Minimal async-compatible OpenAI chat client using stdlib only.

    Usage:
        client = OpenAIClient(
            base_url="http://localhost:1234/v1",
            model="qwen2.5-3b-instruct",
            api_key="lm-studio",
        )
        response = await client.chat([{"role": "user", "content": "hi"}])
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key

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
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        return data["choices"][0]["message"]["content"]
