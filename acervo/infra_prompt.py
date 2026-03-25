"""Hardcoded infrastructure prompt — not user-editable.

This module defines the Acervo infra block that instructs the LLM how to use
verified context, tools, and response conventions.  It is composed with the
user's personality prompt at the proxy level before forwarding to the real LLM.
"""

INFRA_BLOCK = """\
You have access to two sources of information:
1. VERIFIED CONTEXT: You may receive a [VERIFIED CONTEXT] block containing data \
from indexed sources. This data is pre-verified — use it confidently.
2. Tools: You have access to external tools (like web search). Use them when:
   - The verified context doesn't contain what you need
   - The user explicitly asks you to search or look something up
   - You need current/real-time information

When you have verified context, prefer it over searching. \
When you don't, use your tools or your general knowledge.

Be natural, concise, and direct. Respond in the same language the user writes in."""

PLAN_MODE_BLOCK = """\

Before answering, reason step by step. Consider what you know from verified context, \
what tools are available, and whether you need additional information. \
Show your reasoning, then give your final answer."""


def build_system_message(user_prompt: str, plan_mode: bool = False) -> str:
    """Compose the system message from infra block + user prompt.

    The section markers (── ACERVO INFRA ── / ── USER PROMPT ──) are visible
    in trace/debug views so developers can see what each layer contributes.
    """
    parts = [f"── ACERVO INFRA ──\n{INFRA_BLOCK}"]
    if plan_mode:
        parts.append(PLAN_MODE_BLOCK)
    parts.append(f"\n── USER PROMPT ──\n{user_prompt}")
    return "\n".join(parts)
