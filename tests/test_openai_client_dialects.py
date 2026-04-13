"""Unit tests for the two OpenAIClient dialects.

Mocks ``urllib.request.urlopen`` so these run offline. The goal is to
pin the exact wire format for both ``/v1/chat/completions`` (OpenAI
compat) and ``/api/chat`` (Ollama native) — catching regressions in the
body serialization, URL construction and response parsing.

This is the bug-prevention layer for the Ollama ``think: false`` fix:
if someone later tweaks ``_chat_sync_ollama`` and accidentally sends
``max_tokens`` at the top level instead of inside ``options``, these
tests will fail instantly.
"""

from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import patch

import pytest

from acervo.openai_client import OpenAIClient


class _FakeResponse:
    """Context-manager stub that ``urllib.request.urlopen`` can return."""

    def __init__(self, payload: dict):
        self._bytes = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self._bytes


class _UrlopenRecorder:
    """Callable recorder that stores the Request it was called with."""

    def __init__(self, response: dict):
        self._response = response
        self.last_url: str | None = None
        self.last_body: dict | None = None
        self.last_headers: dict | None = None

    def __call__(self, request, timeout=None):
        self.last_url = request.full_url
        self.last_headers = dict(request.headers)
        self.last_body = json.loads(request.data.decode()) if request.data else None
        return _FakeResponse(self._response)


# ── api_style validation ────────────────────────────────────────────────


def test_api_style_must_be_openai_or_ollama():
    with pytest.raises(ValueError, match="api_style"):
        OpenAIClient(base_url="http://x", model="m", api_style="anthropic")


def test_default_api_style_is_openai():
    c = OpenAIClient(base_url="http://x", model="m")
    assert c._api_style == "openai"


# ── OpenAI dialect wire format ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_openai_dialect_hits_v1_chat_completions():
    response = {
        "choices": [{"message": {"role": "assistant", "content": "hello"}}]
    }
    recorder = _UrlopenRecorder(response)

    client = OpenAIClient(
        base_url="http://localhost:1234/v1",
        model="qwen2.5:7b",
        api_key="test-key",
    )

    with patch("acervo.openai_client.urlopen", recorder):
        result = await client.chat(
            [{"role": "user", "content": "hi"}],
            temperature=0.7,
            max_tokens=100,
        )

    assert result == "hello"
    assert recorder.last_url == "http://localhost:1234/v1/chat/completions"
    assert recorder.last_body == {
        "model": "qwen2.5:7b",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.7,
        "max_tokens": 100,
    }
    assert recorder.last_headers.get("Authorization") == "Bearer test-key"


@pytest.mark.asyncio
async def test_openai_dialect_json_mode_adds_response_format():
    recorder = _UrlopenRecorder(
        {"choices": [{"message": {"content": '{"ok": true}'}}]}
    )
    client = OpenAIClient(base_url="http://x/v1", model="m")

    with patch("acervo.openai_client.urlopen", recorder):
        await client.chat([{"role": "user", "content": "x"}], json_mode=True)

    assert recorder.last_body["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_openai_dialect_extra_body_merged_into_request():
    recorder = _UrlopenRecorder(
        {"choices": [{"message": {"content": "ok"}}]}
    )
    client = OpenAIClient(
        base_url="http://x/v1",
        model="m",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    with patch("acervo.openai_client.urlopen", recorder):
        await client.chat([{"role": "user", "content": "x"}])

    assert recorder.last_body["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    # Extras never clobber core fields.
    assert recorder.last_body["model"] == "m"
    assert recorder.last_body["messages"] == [{"role": "user", "content": "x"}]


# ── Ollama dialect wire format ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_ollama_dialect_hits_api_chat_without_v1_suffix():
    """When base_url ends in /v1 we still hit /api/chat on the Ollama root."""
    recorder = _UrlopenRecorder({"message": {"content": "hi"}})
    client = OpenAIClient(
        base_url="http://localhost:11434/v1",
        model="qwen3.5:9b",
        api_style="ollama",
        think=False,
    )

    with patch("acervo.openai_client.urlopen", recorder):
        result = await client.chat(
            [{"role": "user", "content": "x"}],
            temperature=0.0,
            max_tokens=256,
        )

    assert result == "hi"
    assert recorder.last_url == "http://localhost:11434/api/chat"


@pytest.mark.asyncio
async def test_ollama_dialect_puts_max_tokens_in_options_not_top_level():
    """Ollama calls max_tokens 'num_predict' and nests it under 'options'."""
    recorder = _UrlopenRecorder({"message": {"content": "hi"}})
    client = OpenAIClient(
        base_url="http://localhost:11434",
        model="qwen3.5:9b",
        api_style="ollama",
        think=False,
    )

    with patch("acervo.openai_client.urlopen", recorder):
        await client.chat(
            [{"role": "user", "content": "x"}],
            temperature=0.3,
            max_tokens=512,
        )

    body = recorder.last_body
    assert body["model"] == "qwen3.5:9b"
    assert body["stream"] is False
    assert body["think"] is False
    assert body["options"] == {"temperature": 0.3, "num_predict": 512}
    # These fields MUST NOT appear at the top level in Ollama mode:
    assert "max_tokens" not in body
    assert "temperature" not in body


@pytest.mark.asyncio
async def test_ollama_dialect_omits_think_when_none():
    recorder = _UrlopenRecorder({"message": {"content": "ok"}})
    client = OpenAIClient(
        base_url="http://localhost:11434",
        model="qwen3.5:9b",
        api_style="ollama",
        # think left as default None
    )

    with patch("acervo.openai_client.urlopen", recorder):
        await client.chat([{"role": "user", "content": "x"}])

    assert "think" not in recorder.last_body


@pytest.mark.asyncio
async def test_ollama_dialect_think_true_is_serialized():
    recorder = _UrlopenRecorder({"message": {"content": "ok"}})
    client = OpenAIClient(
        base_url="http://localhost:11434",
        model="qwen3.5:9b",
        api_style="ollama",
        think=True,
    )

    with patch("acervo.openai_client.urlopen", recorder):
        await client.chat([{"role": "user", "content": "x"}])

    assert recorder.last_body["think"] is True


@pytest.mark.asyncio
async def test_ollama_dialect_json_mode_uses_format_field():
    """In Ollama mode, json_mode translates to top-level ``format: 'json'``."""
    recorder = _UrlopenRecorder({"message": {"content": '{"ok": true}'}})
    client = OpenAIClient(
        base_url="http://localhost:11434",
        model="qwen3.5:9b",
        api_style="ollama",
        think=False,
    )

    with patch("acervo.openai_client.urlopen", recorder):
        await client.chat([{"role": "user", "content": "x"}], json_mode=True)

    assert recorder.last_body.get("format") == "json"


@pytest.mark.asyncio
async def test_ollama_dialect_reads_message_content_not_choices():
    recorder = _UrlopenRecorder(
        {"message": {"role": "assistant", "content": "the answer"}}
    )
    client = OpenAIClient(
        base_url="http://localhost:11434",
        model="qwen3.5:9b",
        api_style="ollama",
        think=False,
    )

    with patch("acervo.openai_client.urlopen", recorder):
        result = await client.chat([{"role": "user", "content": "x"}])

    assert result == "the answer"


@pytest.mark.asyncio
async def test_ollama_dialect_empty_content_with_reasoning_logs_warning(caplog):
    """When Ollama puts everything in 'reasoning' and leaves content empty,
    we surface a warning so the operator can notice the think flag was ignored."""
    import logging

    recorder = _UrlopenRecorder(
        {
            "message": {
                "role": "assistant",
                "content": "",
                "reasoning": "Let me think step by step about this..." * 10,
            }
        }
    )
    client = OpenAIClient(
        base_url="http://localhost:11434",
        model="qwen3.5:9b",
        api_style="ollama",
        think=False,
    )

    with caplog.at_level(logging.WARNING, logger="acervo.openai_client"):
        with patch("acervo.openai_client.urlopen", recorder):
            result = await client.chat([{"role": "user", "content": "x"}])

    assert result == ""
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("reasoning" in r.getMessage() for r in warnings)


# ── _ollama_dialect_kwargs helper (facade-level auto-detection) ──────


def test_ollama_detection_picks_native_dialect_for_qwen3_on_11434():
    from acervo.facade import _ollama_dialect_kwargs

    kw = _ollama_dialect_kwargs("http://localhost:11434/v1", "qwen3.5:9b")
    assert kw == {"api_style": "ollama", "think": False}


def test_ollama_detection_skips_non_thinking_model():
    from acervo.facade import _ollama_dialect_kwargs

    kw = _ollama_dialect_kwargs("http://localhost:11434/v1", "qwen2.5:7b")
    assert kw == {}


def test_ollama_detection_skips_non_ollama_endpoint():
    from acervo.facade import _ollama_dialect_kwargs

    kw = _ollama_dialect_kwargs("https://api.openai.com/v1", "qwen3.5:9b")
    assert kw == {}


def test_ollama_detection_matches_qwq_and_deepseek_r1():
    from acervo.facade import _ollama_dialect_kwargs

    assert _ollama_dialect_kwargs("http://localhost:11434", "qwq:32b") == {
        "api_style": "ollama", "think": False,
    }
    assert _ollama_dialect_kwargs("http://localhost:11434", "deepseek-r1:7b") == {
        "api_style": "ollama", "think": False,
    }
