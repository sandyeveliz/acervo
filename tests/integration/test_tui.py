"""Integration tests for AVS-Agents TUI — require LM Studio + Ollama running.

Run with: pytest tests/integration/test_tui.py -m integration -v

These tests use Textual's Pilot API to simulate user interaction.
The TUI app must be importable from the AVS-Agents project (add it to sys.path).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add AVS-Agents to path so we can import the TUI app
_AVS_AGENTS_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "AVS-Agents"
if str(_AVS_AGENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_AVS_AGENTS_ROOT))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tui_sends_message_and_receives_response():
    """TUI shows a response after sending a message."""
    from tui.app import AVSAgentsApp
    from tui.widgets.chat_panel import ChatPanel

    app = AVSAgentsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        # Type a message and send
        await pilot.click("#input")
        await pilot.press(*list("Hola"))
        await pilot.press("enter")

        # Wait for LLM to respond
        await pilot.pause(15.0)

        # Verify the chat panel has children (messages + steps)
        chat = app.query_one("#chat-panel", ChatPanel)
        children = list(chat.children)
        # At minimum: system message + user message + some response
        assert len(children) > 2, \
            f"Expected at least 3 children in chat panel, got {len(children)}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tui_agent_remembers_name_between_sessions():
    """Agent remembers the user's name via the graph across sessions."""
    from tui.app import AVSAgentsApp
    from tui.widgets.chat_panel import ChatPanel

    # Session 1 — tell the agent our name
    app1 = AVSAgentsApp()
    async with app1.run_test(size=(120, 40)) as pilot:
        await pilot.click("#input")
        await pilot.press(*list("Me llamo Sandy y vivo en Cipolletti"))
        await pilot.press("enter")
        await pilot.pause(15.0)

    # Verify the graph has the node after session 1
    assert app1._memory.graph.get_node("sandy") is not None, \
        "Sandy node should exist after session 1"
    assert app1._memory.graph.get_node("cipolletti") is not None, \
        "Cipolletti node should exist after session 1"

    # Session 2 — ask the agent our name
    app2 = AVSAgentsApp()
    async with app2.run_test(size=(120, 40)) as pilot:
        await pilot.click("#input")
        await pilot.press(*list("Como me llamo?"))
        await pilot.press("enter")
        await pilot.pause(15.0)

        # Check if Sandy appears in any message bubble
        chat = app2.query_one("#chat-panel", ChatPanel)
        all_text = " ".join(
            child.renderable if hasattr(child, "renderable") else str(child)
            for child in chat.children
        )
        assert "Sandy" in all_text, \
            f"Agent should remember 'Sandy', got: {all_text[:300]}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tui_graph_builds_correctly_after_conversation():
    """After a multi-turn conversation, the graph has the expected structure."""
    from tui.app import AVSAgentsApp
    from tui.widgets.chat_panel import ChatPanel

    app = AVSAgentsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        messages = [
            "Me llamo Sandy y vivo en Cipolletti",
            "Trabajo en Altovallestudio, es mi empresa de software",
            "Tenemos un proyecto llamado Butaco hecho con Angular",
        ]
        for msg in messages:
            await pilot.click("#input")
            await pilot.press(*list(msg))
            await pilot.press("enter")
            await pilot.pause(15.0)

    # Verify graph structure via the app's memory instance
    graph = app._memory.graph

    # Expected nodes (checking by ID)
    assert graph.get_node("sandy") is not None, "Sandy should exist"
    assert graph.get_node("cipolletti") is not None, "Cipolletti should exist"
    assert graph.get_node("altovallestudio") is not None, "Altovallestudio should exist"
    assert graph.get_node("butaco") is not None, "Butaco should exist"
    assert graph.get_node("angular") is not None, "Angular should exist"

    # Verify layers
    angular = graph.get_node("angular")
    assert angular["layer"] == "UNIVERSAL", "Angular should be universal knowledge"

    avs = graph.get_node("altovallestudio")
    assert avs["layer"] == "PERSONAL", "Altovallestudio should be personal"
    assert avs["owner"] == "Sandy", "Altovallestudio should be owned by Sandy"
