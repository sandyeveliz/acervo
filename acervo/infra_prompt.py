"""Infrastructure prompt — loaded from acervo/prompts/.

Defines the Acervo infra block that instructs the LLM how to use
verified context, tools, and response conventions. Composed with the
user's personality prompt at the proxy level before forwarding to the real LLM.
"""

from acervo.prompts import load_prompt

INFRA_BLOCK = load_prompt("infra")
PLAN_MODE_BLOCK = load_prompt("plan_mode")


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
