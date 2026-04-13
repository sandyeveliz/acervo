"""Minimal end-to-end parse tests.

The goal is to isolate *just* the prompt → Ollama → JSON parse layer,
with zero Acervo pipeline in between. When these fail we know the issue
is in one of three places:

    1. The prompt is too long / confusing and the model rambles or
       enters an endless <think> block.
    2. ``max_tokens`` is too low and the model's output is truncated
       mid-think-block, which then gets wiped by ``strip_think_blocks``.
    3. The model wraps output in markdown fences or adds preamble text
       that our JSON locator can't handle.

These tests print the raw response from Ollama so we can SEE what the
model is actually producing. No guessing, no bisecting, just evidence.

Run with:
    pytest tests/integration/test_llm_parse.py -v -s
"""

from __future__ import annotations

import json
import os
import time

import pytest

from acervo._text import strip_think_blocks
from acervo.extractor import _clean_response, _parse_first_json
from acervo.openai_client import OpenAIClient
from acervo.prompts import load_prompt


_LLM_BASE_URL = os.environ.get("ACERVO_LIGHT_MODEL_URL", "http://localhost:11434/v1")
_LLM_MODEL = os.environ.get("ACERVO_LIGHT_MODEL", "qwen3.5:9b")
_LLM_KEY = os.environ.get("ACERVO_LIGHT_API_KEY", "ollama")


@pytest.fixture(scope="module")
def llm() -> OpenAIClient:
    """Client configured the way the facade configures it for Ollama.

    Uses the native ``/api/chat`` endpoint with ``think: false`` because
    qwen3.5 on /v1 puts all its reasoning into ``message.reasoning`` and
    leaves ``content`` empty, exhausting the token budget in the process.
    See ``test_raw_ollama_response_shape`` for the evidence.
    """
    return OpenAIClient(
        base_url=_LLM_BASE_URL,
        model=_LLM_MODEL,
        api_key=_LLM_KEY,
        timeout=180,
        api_style="ollama",
        think=False,
    )


@pytest.fixture(scope="module")
def llm_thinking_on() -> OpenAIClient:
    """Same client but with thinking ON — only used by the diagnostic test."""
    return OpenAIClient(
        base_url=_LLM_BASE_URL,
        model=_LLM_MODEL,
        api_key=_LLM_KEY,
        timeout=180,
        api_style="ollama",
        think=True,
    )


def _diagnose_response(label: str, raw: str) -> dict:
    """Dump everything we can about a raw LLM response."""
    print(f"\n{'=' * 70}")
    print(f"DIAGNOSIS: {label}")
    print(f"{'=' * 70}")
    print(f"Length: {len(raw)} chars")
    print(f"Has <think>: {'<think>' in raw}")
    print(f"Has </think>: {'</think>' in raw}")
    print(f"Has ``` (markdown fence): {'```' in raw}")
    print(f"Starts with '{{': {raw.lstrip().startswith('{')}")
    print(f"Ends with '}}': {raw.rstrip().endswith('}')}")
    print(f"\n--- RAW (first 500 chars) ---")
    print(raw[:500])
    print(f"\n--- RAW (last 500 chars) ---")
    print(raw[-500:])

    # Pipeline: clean → strip_think → parse
    cleaned = _clean_response(raw)
    stripped = strip_think_blocks(cleaned)
    print(f"\n--- AFTER strip_think_blocks ({len(stripped)} chars) ---")
    print(stripped[:500] if stripped else "(EMPTY)")

    parsed = _parse_first_json(stripped, "object")
    print(f"\n--- PARSE RESULT ---")
    print(f"Parsed to dict: {isinstance(parsed, dict)}")
    if isinstance(parsed, dict):
        print(f"Keys: {list(parsed.keys())}")

    return {
        "raw_len": len(raw),
        "has_think_open": "<think>" in raw,
        "has_think_close": "</think>" in raw,
        "after_strip_len": len(stripped),
        "parsed": isinstance(parsed, dict),
    }


# ── Test 1: S1 prompt returns parseable JSON ────────────────────────────


@pytest.mark.asyncio
async def test_s1_returns_parseable_json(llm: OpenAIClient):
    """Send the real S1 system prompt + a minimal user message. Assert parse."""
    system_prompt = load_prompt("s1_unified")
    user_content = (
        "EXISTING NODES:\n[]\n\n"
        "TOPIC HINT: unresolved — classify the topic yourself\n"
        "CURRENT TOPIC: null\n\n"
        "PREVIOUS ASSISTANT: null\n"
        "USER: Sandy Veliz trabaja en el proyecto Butaco."
    )

    start = time.perf_counter()
    raw = await llm.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
        max_tokens=2048,
    )
    elapsed = time.perf_counter() - start
    print(f"\nLLM elapsed: {elapsed:.1f}s")

    diag = _diagnose_response("S1 with real prompt + simple message", raw)

    # Fail loudly with a useful message if parse failed.
    if not diag["parsed"]:
        if diag["has_think_open"] and not diag["has_think_close"]:
            pytest.fail(
                f"S1 parse failed because the model emitted an OPEN <think> block "
                f"that was truncated at max_tokens=2048. The entire output got "
                f"wiped by strip_think_blocks (open regex matches to end). "
                f"Fix options: (a) add 'Do not use <think> tags' to the prompt, "
                f"(b) bump max_tokens to 4096, or (c) use a /no_think model flag."
            )
        if "```" in raw:
            pytest.fail(
                "S1 output is wrapped in markdown fences — _parse_first_json "
                "may need to handle this. Raw starts with: "
                f"{raw[:200]!r}"
            )
        pytest.fail(f"S1 JSON parse failed for reasons unknown. See diagnosis above.")

    assert elapsed < 30, f"S1 took {elapsed:.1f}s — too slow for iteration"


# ── Test 2: S1.5 prompt returns parseable JSON ──────────────────────────


@pytest.mark.asyncio
async def test_s1_5_returns_parseable_json(llm: OpenAIClient):
    """Same for S1.5. Uses the new Phase 3 prompt with JSON examples."""
    template = load_prompt("s1_5_graph_update")

    # Simulate what S1_5GraphUpdate.run would do.
    prompt = (
        template
        .replace("{new_entities}", '[{"name": "Sandy Veliz", "type": "person"}]')
        .replace("{existing_nodes}", '[{"id": "sandy_veliz", "label": "Sandy Veliz", "type": "person"}]')
        .replace("{current_assistant_msg}", "Entendido.")
    )

    start = time.perf_counter()
    raw = await llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=2048,
    )
    elapsed = time.perf_counter() - start
    print(f"\nLLM elapsed: {elapsed:.1f}s")

    diag = _diagnose_response("S1.5 with real prompt + simple input", raw)

    if not diag["parsed"]:
        if diag["has_think_open"] and not diag["has_think_close"]:
            pytest.fail(
                "S1.5 parse failed — open <think> block truncated at max_tokens. "
                "Same root cause as S1. Fix the underlying LLM configuration."
            )
        pytest.fail("S1.5 JSON parse failed. See diagnosis.")

    assert elapsed < 30, f"S1.5 took {elapsed:.1f}s — too slow"


# ── Test 3: Bare JSON capability check (smallest possible prompt) ───────


@pytest.mark.asyncio
async def test_bare_json_capability(llm: OpenAIClient):
    """Sanity check: can the model return a trivial JSON at all?

    If this fails, the problem is NOT our prompts — it's Ollama config,
    the model itself, or the client. If this passes but Test 1/2 fail,
    the issue is definitely in the S1 / S1.5 prompt.
    """
    start = time.perf_counter()
    raw = await llm.chat(
        [
            {"role": "system", "content": "You return ONLY a JSON object. No markdown, no explanation."},
            {"role": "user", "content": 'Return this exactly: {"ok": true, "n": 42}'},
        ],
        temperature=0.0,
        max_tokens=256,
    )
    elapsed = time.perf_counter() - start
    print(f"\nBare JSON elapsed: {elapsed:.1f}s")

    diag = _diagnose_response("Bare JSON", raw)

    assert diag["parsed"], (
        f"Model cannot return even a trivial JSON. This is a model/Ollama "
        f"issue, not a prompt issue. Raw response: {raw[:300]!r}"
    )
    assert elapsed < 15, f"Even the trivial call took {elapsed:.1f}s — Ollama is slow"


# ── Test -1: Hit Ollama's native /api/chat with think=false ────────────


@pytest.mark.asyncio
async def test_ollama_native_api_with_think_false():
    """Bypass the OpenAI-compat layer entirely.

    Ollama exposes two endpoints:
        - /v1/chat/completions  (OpenAI-compat — what our client uses)
        - /api/chat             (Ollama native — accepts ``think: false``)

    Recent Ollama versions honour ``think: false`` on /api/chat even when
    /v1/chat/completions doesn't accept ``chat_template_kwargs``. If this
    test passes with a non-empty content while the /v1 test fails, we
    know the fix: port OpenAIClient to speak Ollama's native dialect for
    qwen3.x models, or provide a parallel OllamaClient.
    """
    import json as _json
    from urllib.request import Request, urlopen

    # /api/chat is at base_url stripped of the /v1 suffix.
    base = _LLM_BASE_URL.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    url = f"{base}/api/chat"

    body = {
        "model": _LLM_MODEL,
        "messages": [
            {"role": "system", "content": "Return ONLY a JSON object, nothing else."},
            {"role": "user", "content": 'Return: {"ok": true}'},
        ],
        "stream": False,
        "think": False,  # ← the thing we want to validate
        "options": {"temperature": 0.0, "num_predict": 256},
    }

    print(f"\nPOST {url}")
    print(f"Body: {_json.dumps(body, ensure_ascii=False)[:500]}")

    start = time.perf_counter()
    req = Request(
        url,
        data=_json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=120) as resp:
        raw_bytes = resp.read()
    elapsed = time.perf_counter() - start

    data = _json.loads(raw_bytes)
    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"Response size: {len(raw_bytes)} bytes")
    print(f"Top-level keys: {list(data.keys())}")

    if "message" in data:
        msg = data["message"]
        print(f"\nmessage keys: {list(msg.keys())}")
        for k, v in msg.items():
            preview = repr(v)[:300]
            print(f"  {k!r}: {preview}")

    print(f"\ndone={data.get('done')}  done_reason={data.get('done_reason')}")

    print(f"\n--- FULL RESPONSE JSON ---")
    print(_json.dumps(data, indent=2, ensure_ascii=False)[:2500])

    content = (data.get("message") or {}).get("content", "")
    assert content, (
        f"Even /api/chat with think=false returned empty content. "
        f"This suggests the model itself is failing, not the endpoint. "
        f"Full response: {data}"
    )
    assert "<think>" not in content, (
        f"think=false was NOT honoured by /api/chat. Content still has "
        f"<think> tags: {content[:300]!r}"
    )


# ── Test 0: Dump the FULL raw HTTP response from Ollama ─────────────────


@pytest.mark.asyncio
async def test_raw_ollama_response_shape():
    """Bypass OpenAIClient and dump the full JSON Ollama actually returns.

    We've been reading ``choices[0].message.content`` but that's 0 chars
    in the bare JSON test. We need to know WHERE the model's output is
    actually going — maybe Ollama's recent versions put thinking into
    ``message.thinking`` or ``message.reasoning_content`` or similar.

    This test uses urllib directly so there's no layer between us and
    the HTTP response. It dumps every top-level key and every key in
    ``choices[0].message`` so we can see the truth.
    """
    import json as _json
    from urllib.request import Request, urlopen

    url = f"{_LLM_BASE_URL}/chat/completions"
    body = {
        "model": _LLM_MODEL,
        "messages": [
            {"role": "system", "content": "Return ONLY a JSON object, nothing else."},
            {"role": "user", "content": 'Return: {"ok": true}'},
        ],
        "temperature": 0.0,
        "max_tokens": 256,
    }
    payload = _json.dumps(body).encode()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {_LLM_KEY}"}

    print(f"\nPOST {url}")
    print(f"Body keys: {list(body.keys())}")

    start = time.perf_counter()
    req = Request(url, data=payload, headers=headers, method="POST")
    with urlopen(req, timeout=120) as resp:
        raw_bytes = resp.read()
    elapsed = time.perf_counter() - start

    data = _json.loads(raw_bytes)
    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"Response size: {len(raw_bytes)} bytes")
    print(f"Top-level keys: {list(data.keys())}")

    if "choices" in data and data["choices"]:
        choice = data["choices"][0]
        print(f"\nchoice[0] keys: {list(choice.keys())}")
        if "message" in choice:
            msg = choice["message"]
            print(f"\nchoice[0].message keys: {list(msg.keys())}")
            for k, v in msg.items():
                preview = repr(v)[:300] if not isinstance(v, (dict, list)) else str(v)[:300]
                print(f"  {k!r}: {preview}")
        print(f"\nfinish_reason: {choice.get('finish_reason')}")

    # Also dump the full raw JSON (pretty) so we miss nothing.
    print(f"\n--- FULL RESPONSE JSON ---")
    print(_json.dumps(data, indent=2, ensure_ascii=False)[:2000])

    # No assertion — this is pure observation. We want to SEE where the
    # output is going. The following tests will act on what we learn.


# ── Test 4: <think> tag behaviour with/without enable_thinking flag ────


@pytest.mark.asyncio
async def test_think_tags_off_with_flag(llm: OpenAIClient):
    """With ``enable_thinking=false`` the model MUST NOT emit <think> tags.

    This is the whole point of plumbing extra_body through to the client.
    If this test fails, the flag isn't being honoured by the server
    (maybe Ollama doesn't support chat_template_kwargs yet, or the
    Modelfile doesn't use a Jinja template that checks it).
    """
    raw = await llm.chat(
        [{"role": "user", "content": "Answer with ONE word: what color is the sky?"}],
        temperature=0.0,
        max_tokens=256,
    )
    _diagnose_response("enable_thinking=false probe", raw)
    assert "<think>" not in raw, (
        f"Model still emits <think> despite chat_template_kwargs.enable_thinking=false. "
        f"Raw: {raw[:300]!r}. The server may not honour the flag — check Ollama "
        f"version or switch to vLLM."
    )


@pytest.mark.asyncio
async def test_think_tags_on_by_default(llm_thinking_on: OpenAIClient):
    """Diagnostic: confirms that WITHOUT the flag, the model DOES emit <think>.

    No assertion — just prints evidence so we can see the before/after
    clearly in the same test run.
    """
    raw = await llm_thinking_on.chat(
        [{"role": "user", "content": "Answer with ONE word: what color is the sky?"}],
        temperature=0.0,
        max_tokens=512,
    )
    _diagnose_response("enable_thinking NOT set (default probe)", raw)
    has_think = "<think>" in raw
    print(f"\n>>> Model emits <think> when flag is absent: {has_think}")
