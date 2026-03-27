"""Service utilities for 'acervo up'.

Two modes:
  acervo up        -- proxy only (foreground), with dependency health check
  acervo up --dev  -- all services with multiplexed tagged logs (like docker-compose)
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import platform
import shutil
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from acervo.config import AcervoConfig

log = logging.getLogger(__name__)

_VERSION = "0.2.2"

# -- ANSI colors --

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_MAGENTA = "\033[35m"
_BLUE = "\033[34m"
_RED = "\033[31m"
_WHITE = "\033[97m"

# Tag colors per service
_TAG_COLORS: dict[str, str] = {
    "ollama": _GREEN,
    "lmstudio": _YELLOW,
    "proxy": _CYAN,
    "studio": _MAGENTA,
    "web": _BLUE,
}

_BANNER = rf"""
{_CYAN}{_BOLD}    ___   ____________ _____    ______
   /   | / ____/ ____/ __ \ \  / / __ \
  / /| |/ /   / __/ / /_/ /\ \/ / / / /
 / ___ / /___/ /___/ _, _/  \  / /_/ /
/_/  |_\____/_____/_/ |_|    \/_____/{_RESET}
{_DIM}  v{_VERSION}{_RESET}
"""

_BANNER_DEV = rf"""
{_MAGENTA}{_BOLD}    ___   ____________ _____    ______
   /   | / ____/ ____/ __ \ \  / / __ \
  / /| |/ /   / __/ / /_/ /\ \/ / / / /
 / ___ / /___/ /___/ _, _/  \  / /_/ /
/_/  |_\____/_____/_/ |_|    \/_____/{_RESET}
{_DIM}  v{_VERSION} -- dev mode{_RESET}
"""


def _supports_color() -> bool:
    """Check if the terminal supports ANSI colors."""
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    if platform.system() == "Windows":
        # Windows 10+ supports ANSI via virtual terminal processing
        import os
        os.system("")  # enable ANSI escape processing
        return True
    return True


def _c(color: str, text: str) -> str:
    """Colorize text if terminal supports it."""
    if not _supports_color():
        return text
    return f"{color}{text}{_RESET}"


def _banner(dev: bool = False) -> str:
    """Return the banner string."""
    if not _supports_color():
        lines = (_BANNER_DEV if dev else _BANNER)
        # Strip ANSI codes for non-color terminals
        import re
        return re.sub(r"\033\[[0-9;]*m", "", lines)
    return _BANNER_DEV if dev else _BANNER


# -- Health checks --


def check_health(url: str, timeout: float = 2.0) -> tuple[bool, str]:
    """HTTP GET health check. Returns (ok, response_body)."""
    try:
        req = Request(url, method="GET")
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read(4096).decode("utf-8", errors="replace")
            return True, body
    except (URLError, OSError, TimeoutError):
        return False, ""


def detect_binary(name: str) -> Path | None:
    """Find a binary on PATH."""
    result = shutil.which(name)
    return Path(result) if result else None


def detect_studio_path(config: AcervoConfig) -> Path | None:
    """Try to find the Acervo Studio project directory."""
    # 1. Explicit config path
    if config.services.studio_path:
        p = Path(config.services.studio_path)
        if (p / "server.py").exists():
            return p

    # 2. Check if api module is importable (acervo-studio installed)
    spec = importlib.util.find_spec("api")
    if spec and spec.origin:
        candidate = Path(spec.origin).parent.parent
        if (candidate / "server.py").exists():
            return candidate

    # 3. Check sibling directories (dev layout)
    acervo_spec = importlib.util.find_spec("acervo")
    if acervo_spec and acervo_spec.origin:
        dev_root = Path(acervo_spec.origin).parent.parent.parent
        for dirname in ("AVS-Agents", "acervo-studio"):
            candidate = dev_root / dirname
            if (candidate / "server.py").exists() and (candidate / "api").is_dir():
                return candidate

    return None


# -- Dependency check (for acervo up) --


@dataclass
class DepStatus:
    """Status of a dependency service."""
    name: str
    port: int
    ok: bool
    detail: str = ""


def check_dependencies(config: AcervoConfig) -> list[DepStatus]:
    """Check Ollama and LM Studio health."""
    results: list[DepStatus] = []

    # Ollama
    port = config.services.ollama_port
    ok, body = check_health(f"http://localhost:{port}/")
    results.append(DepStatus(
        name="Ollama (embeddings)",
        port=port,
        ok=ok,
        detail="ok" if ok else "not running",
    ))

    # LM Studio
    port = config.services.lmstudio_port
    ok, body = check_health(f"http://localhost:{port}/v1/models")
    detail = "ok"
    if ok:
        try:
            data = json.loads(body)
            models = data.get("data", [])
            if models:
                names = [m.get("id", "?") for m in models[:3]]
                detail = ", ".join(names)
        except (json.JSONDecodeError, KeyError):
            pass
    else:
        detail = "not running -- start manually"
    results.append(DepStatus(name="LM Studio (chat)", port=port, ok=ok, detail=detail))

    return results


def format_dep_check(results: list[DepStatus]) -> str:
    """Format dependency check results for display."""
    lines = [f"  {_c(_DIM, 'Dependencies:')}"]
    for dep in results:
        if dep.ok:
            icon = _c(_GREEN, "*")
            status = _c(_GREEN, dep.detail)
        else:
            icon = _c(_YELLOW, "o")
            status = _c(_YELLOW, dep.detail)
        lines.append(f"    {icon} {dep.name:<24} :{dep.port:<6} {status}")
    return "\n".join(lines)


# -- DevRunner (for acervo up --dev) --


def _get_npm_cmd() -> list[str]:
    """Get the npm command, handling Windows .cmd extension."""
    npm = detect_binary("npm")
    if not npm:
        return []
    # On Windows, use just "npm" for shell execution (avoids quoting issues
    # with "C:\Program Files\..." paths). shell=True resolves it via PATH.
    if platform.system() == "Windows":
        return ["npm"]
    return [str(npm)]


class DevRunner:
    """Run multiple services with tagged log output. Ctrl+C stops all."""

    def __init__(self, config: AcervoConfig, acervo_dir: Path) -> None:
        self._config = config
        self._acervo_dir = acervo_dir
        self._procs: list[tuple[str, asyncio.subprocess.Process]] = []

    async def run(self) -> None:
        """Start all services and multiplex their output."""
        svc = self._config.services

        print(_banner(dev=True))

        # 1. Check Ollama
        ok, _ = check_health(f"http://localhost:{svc.ollama_port}/")
        if ok:
            self._log("ollama", f"already running on :{svc.ollama_port}")
        else:
            ollama_bin = detect_binary("ollama")
            if ollama_bin:
                self._log("ollama", f"starting on :{svc.ollama_port}...")
                await self._start("ollama", [str(ollama_bin), "serve"])
            else:
                self._log("ollama", "not found -- install from https://ollama.com")

        # 2. Check LM Studio (detect only)
        ok, _ = check_health(f"http://localhost:{svc.lmstudio_port}/v1/models")
        if ok:
            self._log("lmstudio", f"running on :{svc.lmstudio_port}")
        else:
            self._log("lmstudio", f"not running on :{svc.lmstudio_port} -- start manually")

        # 3. Start proxy
        self._log("proxy", f"starting on :{self._config.proxy.port}...")
        await self._start("proxy", [
            sys.executable, "-m", "acervo", "serve",
            "--port", str(self._config.proxy.port),
        ])

        # 4. Start Studio backend (if available)
        studio_path = detect_studio_path(self._config)
        if studio_path:
            self._log("studio", f"starting on :{svc.studio_port}...")
            await self._start("studio", [
                sys.executable, str(studio_path / "server.py"),
            ], cwd=studio_path)

            # 5. Start frontend (if web/ exists)
            web_dir = studio_path / "web"
            if (web_dir / "package.json").exists():
                npm = _get_npm_cmd()
                if npm:
                    self._log("web", f"starting on :{svc.frontend_port}...")
                    use_shell = platform.system() == "Windows"
                    await self._start("web", npm + ["run", "dev"], cwd=web_dir, shell=use_shell)
                else:
                    self._log("web", "npm not found -- skipping frontend")
            else:
                self._log("web", "web/package.json not found -- skipping")
        else:
            self._log("studio", "acervo-studio not found -- skipping")
            self._log("web", "skipping (no studio)")

        print()

        # Wait for all processes, piping output
        if not self._procs:
            print("  No services started.")
            return

        try:
            await asyncio.gather(
                *(self._pipe_output(tag, proc) for tag, proc in self._procs)
            )
        except asyncio.CancelledError:
            pass
        finally:
            await self._shutdown()

    async def _start(
        self,
        tag: str,
        cmd: list[str],
        cwd: Path | None = None,
        shell: bool = False,
    ) -> None:
        """Start a subprocess and register it."""
        try:
            if shell:
                # Quote each arg to handle paths with spaces (e.g. "C:\Program Files\...")
                shell_cmd = " ".join(f'"{c}"' if " " in c else c for c in cmd)
                proc = await asyncio.create_subprocess_shell(
                    shell_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(cwd) if cwd else None,
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(cwd) if cwd else None,
                )
            self._procs.append((tag, proc))
        except (OSError, FileNotFoundError) as e:
            self._log(tag, f"failed to start: {e}")

    async def _pipe_output(self, tag: str, proc: asyncio.subprocess.Process) -> None:
        """Read stdout and stderr and print with tag prefix."""
        async def _read_stream(stream: asyncio.StreamReader | None, is_err: bool = False) -> None:
            if not stream:
                return
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    self._log(tag, text)

        await asyncio.gather(
            _read_stream(proc.stdout),
            _read_stream(proc.stderr, is_err=True),
        )

    async def _shutdown(self) -> None:
        """Terminate all managed processes."""
        print("\n  Shutting down...")
        for tag, proc in reversed(self._procs):
            if proc.returncode is None:
                try:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        proc.kill()
                    self._log(tag, "stopped")
                except (OSError, ProcessLookupError):
                    pass

    def _log(self, tag: str, message: str) -> None:
        """Print a tagged log line with colored tag."""
        color = _TAG_COLORS.get(tag, _WHITE)
        colored_tag = _c(f"{color}{_BOLD}", f"{tag:<8}")
        print(f"  [{colored_tag}] {message}")
