"""Regression tests for S1.5 prompt placeholder injection.

The Phase 3 prompt rewrite added several JSON few-shot examples to
``acervo/prompts/s1_5_graph_update.txt``. Those curly braces used to
crash the whole S1.5 pipeline because ``S1_5GraphUpdate.run`` injected
inputs via ``str.format()``, which tried to interpret every ``{`` in the
prompt body as a placeholder and raised ``KeyError('\\n  "merges"')``
on turn 1 of every case scenario (49/49 fails on ``test_casa``).

These tests pin the fix: the injection must tolerate arbitrary literal
``{``/``}`` in the prompt template and still substitute the three
documented placeholders.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from acervo.s1_5_graph_update import S1_5GraphUpdate


class _FakeLLM:
    """Captures the prompt that was sent and returns a valid empty S1.5 JSON."""

    def __init__(self):
        self.calls: list[list[dict]] = []

    async def chat(self, messages: list[dict], **_: Any) -> str:
        self.calls.append(messages)
        return json.dumps({
            "merges": [],
            "new_relations": [],
            "type_corrections": [],
            "discards": [],
            "assistant_entities": [],
            "assistant_facts": [],
            "assistant_relations": [],
        })


@pytest.mark.asyncio
async def test_run_does_not_crash_on_prompt_with_literal_curly_braces():
    """Template with JSON examples in the body must not trip str.format."""
    prompt_with_json_examples = (
        "You are a curator. Return JSON.\n"
        "\n"
        "### Example\n"
        'Output: {"merges": [], "new_relations": []}\n'
        "\n"
        "### Example with nested braces\n"
        'Output: {"entities": [{"name": "Alice", "type": "person"}]}\n'
        "\n"
        "New entities: {new_entities}\n"
        "Existing nodes: {existing_nodes}\n"
        "Assistant response: {current_assistant_msg}\n"
    )
    llm = _FakeLLM()
    s15 = S1_5GraphUpdate(llm, prompt_template=prompt_with_json_examples)

    # Before the fix, this raised KeyError('\n  "merges"') on the first call.
    result = await s15.run(
        new_entities_json='[{"name": "Carlos"}]',
        existing_nodes_json='[{"id": "carlos"}]',
        current_assistant_msg="ok",
    )

    # LLM was called exactly once and the prompt substitutions happened.
    assert len(llm.calls) == 1
    sent_prompt = llm.calls[0][0]["content"]
    assert '[{"name": "Carlos"}]' in sent_prompt
    assert '[{"id": "carlos"}]' in sent_prompt
    assert "ok" in sent_prompt

    # The literal JSON examples in the template survived unchanged.
    assert '{"merges": []' in sent_prompt
    assert '{"entities": [{"name": "Alice"' in sent_prompt

    # And the result parses cleanly to an empty S1_5Result.
    assert result.merges == []
    assert result.new_relations == []


@pytest.mark.asyncio
async def test_run_substitutes_all_three_placeholders_exactly_once():
    """Each placeholder gets replaced with its input, not left verbatim."""
    template = (
        "new={new_entities}|existing={existing_nodes}|asst={current_assistant_msg}"
    )
    llm = _FakeLLM()
    s15 = S1_5GraphUpdate(llm, prompt_template=template)

    await s15.run(
        new_entities_json="NEW",
        existing_nodes_json="EXIST",
        current_assistant_msg="ASST",
    )

    sent = llm.calls[0][0]["content"]
    assert sent == "new=NEW|existing=EXIST|asst=ASST"
    # Placeholders themselves must not appear anymore.
    assert "{new_entities}" not in sent
    assert "{existing_nodes}" not in sent
    assert "{current_assistant_msg}" not in sent


@pytest.mark.asyncio
async def test_run_with_default_prompt_does_not_crash():
    """Smoke test against the real production prompt file.

    This is the exact regression: loading the real Phase 3 prompt and
    calling run() used to crash on the first turn because the prompt has
    JSON examples with literal curly braces.
    """
    from acervo.prompts import load_prompt
    real_prompt = load_prompt("s1_5_graph_update")

    llm = _FakeLLM()
    s15 = S1_5GraphUpdate(llm, prompt_template=real_prompt)

    result = await s15.run(
        new_entities_json='[{"name": "Sandy", "type": "person"}]',
        existing_nodes_json="[]",
        current_assistant_msg="Entendido.",
    )

    assert len(llm.calls) == 1
    # All three placeholders were substituted.
    sent = llm.calls[0][0]["content"]
    assert '[{"name": "Sandy"' in sent
    assert "Entendido." in sent
    # And the few-shot JSON examples survived intact.
    assert '"merges":' in sent
    # Empty parse → empty result (no crash).
    assert result.merges == []
