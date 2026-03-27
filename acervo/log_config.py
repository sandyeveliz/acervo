"""Logging configuration — configurable levels and optional colors.

Usage:
    from acervo.log_config import setup_logging
    setup_logging(level="info", color=True)

Levels:
    info  — one line per turn (default)
    debug — entities, topic decisions, timing
    trace — full prompts, responses, embeddings
"""

from __future__ import annotations

import logging
import os
import sys

# Custom TRACE level (below DEBUG=10)
TRACE = 5
logging.addLevelName(TRACE, "TRACE")


def _trace(self: logging.Logger, msg: str, *args: object, **kwargs: object) -> None:
    if self.isEnabledFor(TRACE):
        self._log(TRACE, msg, args, **kwargs)


logging.Logger.trace = _trace  # type: ignore[attr-defined]

# ANSI color codes
_COLORS = {
    "TRACE": "\033[36m",     # cyan
    "DEBUG": "\033[34m",     # blue
    "INFO": "\033[32m",      # green
    "WARNING": "\033[33m",   # yellow
    "ERROR": "\033[31m",     # red
    "CRITICAL": "\033[35m",  # magenta
    "RESET": "\033[0m",
}


class ColorFormatter(logging.Formatter):
    """Log formatter with optional ANSI colors."""

    def __init__(self, fmt: str, use_color: bool = True) -> None:
        super().__init__(fmt)
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        if self._use_color:
            color = _COLORS.get(record.levelname, "")
            reset = _COLORS["RESET"]
            record.levelname = f"{color}{record.levelname}{reset}"
            record.name = f"\033[90m{record.name}\033[0m"  # dim gray
        return super().format(record)


_LEVEL_MAP = {
    "trace": TRACE,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def setup_logging(level: str = "info", color: bool | None = None) -> None:
    """Configure logging for Acervo CLI and proxy.

    Args:
        level: One of "trace", "debug", "info", "warning", "error".
        color: Enable ANSI colors. None = auto-detect (True if TTY).
    """
    log_level = _LEVEL_MAP.get(level.lower(), logging.INFO)

    if color is None:
        color = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
        # Windows: enable ANSI if running in a modern terminal
        if color and sys.platform == "win32":
            try:
                os.system("")  # enables ANSI on Windows 10+
            except Exception:
                color = False

    # Format based on level
    if log_level <= TRACE:
        fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    elif log_level <= logging.DEBUG:
        fmt = "%(levelname)-7s %(name)s: %(message)s"
    else:
        fmt = "%(levelname)s: %(message)s"

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(ColorFormatter(fmt, use_color=color))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)

    # Quiet noisy third-party loggers unless TRACE
    if log_level > TRACE:
        for name in ("httpcore", "httpx", "uvicorn.access", "chromadb"):
            logging.getLogger(name).setLevel(max(log_level, logging.WARNING))
