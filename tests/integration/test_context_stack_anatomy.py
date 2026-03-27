"""Context stack anatomy — inspects the structure of what Acervo sends to the LLM.

Runs a few turns to build graph knowledge, then examines prep.context_stack
to verify the message structure, warm context markers, hot layer limits, and
total_tokens consistency.

Requires a running LLM server. Run with:
    pytest tests/integration/test_context_stack_anatomy.py -m integration -v -s
"""

from __future__ import annotations

import logging

import pytest

from acervo.token_counter import count_tokens

log = logging.getLogger(__name__)


# Short conversation turns to build some graph knowledge
_SETUP_TURNS = [
    ("Me llamo Sandy y soy programador, vivo en Cipolletti",
     "Hola Sandy! Cipolletti en la Patagonia, lindo lugar."),
    ("Trabajo en Alto Valle Studio, hacemos software",
     "Genial, que tipo de proyectos hacen?"),
    ("Estamos haciendo un SaaS que se llama Chequear, verificacion con NFC",
     "Interesante, NFC tiene mucho potencial."),
    ("El stack es React con Vite y Supabase como backend",
     "React + Supabase es un buen stack moderno."),
    ("Tambien tengo un proyecto personal con Python que se llama Acervo",
     "Un proyecto de AI, interesante."),
]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_context_stack_structure(e2e_memory):
    """Verify the context stack has the expected message structure."""
    memory = e2e_memory
    history = [{"role": "system", "content": "You are a helpful assistant."}]

    # Build graph knowledge over 5 turns
    for user_msg, asst_msg in _SETUP_TURNS:
        await memory.prepare(user_msg, history)
        await memory.process(user_msg, asst_msg)
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": asst_msg})

    # Turn 6: inspect the context stack
    query = "Contame sobre Sandy y Chequear"
    prep = await memory.prepare(query, history)
    stack = prep.context_stack

    print(f"\n=== CONTEXT STACK ANATOMY (Turn 6) ===")
    print(f"Total messages in stack: {len(stack)}")
    print(f"total_tokens: {prep.total_tokens}")
    print(f"warm_tokens: {prep.warm_tokens}")
    print(f"hot_tokens: {prep.hot_tokens}")
    print(f"has_context: {prep.has_context}")
    print(f"warm_content length: {len(prep.warm_content or '')}")
    print()

    for i, msg in enumerate(stack):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        tk = count_tokens(content)
        preview = content[:80].replace("\n", " ")
        print(f"  [{i}] {role:>10} ({tk:>4}tk): {preview}...")
    print("=" * 60)

    # Verify: first message is system
    assert stack[0]["role"] == "system", (
        f"First message should be system, got {stack[0]['role']}"
    )

    # Verify: if warm_tokens > 0, warm context markers should be present
    if prep.warm_tokens > 0:
        # Find the warm context message (should be right after system)
        warm_msg = stack[1] if len(stack) > 1 else None
        assert warm_msg is not None, "Expected warm context message after system"
        assert warm_msg["role"] == "user", (
            f"Warm context should be user role, got {warm_msg['role']}"
        )
        content = warm_msg["content"]
        assert "[VERIFIED CONTEXT]" in content or "[CONVERSATION CONTEXT" in content or "[WEB SEARCH" in content, (
            f"Warm context message missing markers: {content[:100]}"
        )
        # "Understood." acknowledgment should follow
        ack_msg = stack[2] if len(stack) > 2 else None
        assert ack_msg is not None and ack_msg["role"] == "assistant", (
            "Expected 'Understood.' acknowledgment after warm context"
        )

    # Verify: last message is the current user query
    assert stack[-1]["role"] == "user", (
        f"Last message should be user query, got {stack[-1]['role']}"
    )
    assert query in stack[-1]["content"], (
        f"Last message should contain the query"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hot_layer_respects_limit(e2e_memory):
    """Verify: hot layer has at most 2 message pairs (4 messages)."""
    memory = e2e_memory
    history = [{"role": "system", "content": "You are a helpful assistant."}]

    # Build 5 turns of history
    for user_msg, asst_msg in _SETUP_TURNS:
        await memory.prepare(user_msg, history)
        await memory.process(user_msg, asst_msg)
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": asst_msg})

    # Turn 6
    prep = await memory.prepare("Que sabes de mi?", history)
    stack = prep.context_stack

    # Count hot layer messages: everything between warm ack and last user msg
    # Stack structure: [system, warm_ctx?, ack?, hot_pair1?, hot_pair2?, user]
    hot_start = 1  # after system
    if prep.warm_tokens > 0:
        hot_start = 3  # after system + warm_ctx + ack
    hot_end = len(stack) - 1  # before last user msg
    hot_messages = stack[hot_start:hot_end]
    hot_pairs = len(hot_messages) // 2

    print(f"\nHot layer: {len(hot_messages)} messages ({hot_pairs} pairs)")
    for msg in hot_messages:
        print(f"  {msg['role']}: {msg['content'][:60]}...")

    # Default hot_layer_max_messages = 2 pairs
    assert hot_pairs <= 2, (
        f"Hot layer has {hot_pairs} pairs, expected <= 2"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_total_tokens_matches_stack(e2e_memory):
    """Verify: total_tokens approximately matches re-counting stack contents."""
    memory = e2e_memory
    history = [{"role": "system", "content": "You are a helpful assistant."}]

    # Build 5 turns
    for user_msg, asst_msg in _SETUP_TURNS:
        await memory.prepare(user_msg, history)
        await memory.process(user_msg, asst_msg)
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": asst_msg})

    # Turn 6
    prep = await memory.prepare("Contame sobre Sandy y Chequear", history)

    # Re-count all messages in the stack
    recounted = sum(count_tokens(msg.get("content", "")) for msg in prep.context_stack)

    print(f"\nToken comparison:")
    print(f"  prep.total_tokens: {prep.total_tokens}")
    print(f"  recounted from stack: {recounted}")
    print(f"  difference: {abs(prep.total_tokens - recounted)}")

    # Allow small difference for rounding/overhead calculation differences
    diff = abs(prep.total_tokens - recounted)
    assert diff <= 20, (
        f"total_tokens ({prep.total_tokens}) differs from stack recount ({recounted}) by {diff}"
    )
