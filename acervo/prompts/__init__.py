"""Centralized prompt loading for all pipeline stages.

Prompts live as .txt files in this directory. Each module calls
load_prompt("filename") to get the content, with the hardcoded
default as fallback if the file is missing.

User-level overrides: .acervo/prompts/*.txt (per-project) take
precedence over these defaults when loaded via PromptsConfig.
"""

from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """Load a prompt file by name (without .txt extension).

    Returns the file content, or raises FileNotFoundError if missing.
    """
    path = _PROMPTS_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8").strip()
