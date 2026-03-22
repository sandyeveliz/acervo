"""Typed configuration for Acervo.

Loaded from .acervo/config.toml. This is the single source of truth for all
Acervo settings — model, indexing, context, proxy.

Usage:
    config = AcervoConfig.load(Path(".acervo/config.toml"))
    config.model.url  # "http://localhost:1234/v1"

    # Or discover from cwd:
    path = AcervoConfig.find_config()
    if path:
        config = AcervoConfig.load(path)
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for the utility model used by Acervo's pipeline."""

    name: str = "qwen2.5-3b-instruct"
    url: str = "http://localhost:1234/v1"
    api_key: str = ""

    def resolve(self) -> ModelConfig:
        """Return a copy with env var overrides applied."""
        return ModelConfig(
            name=os.environ.get("ACERVO_LIGHT_MODEL", self.name),
            url=os.environ.get("ACERVO_LIGHT_MODEL_URL", self.url),
            api_key=os.environ.get("ACERVO_LIGHT_API_KEY", self.api_key),
        )


@dataclass
class ChangelogConfig:
    """Which tool-use names indicate file modifications."""

    write_tools: list[str] = field(default_factory=lambda: [
        "write_file", "edit_file", "create_file", "str_replace_editor",
        "write_to_file", "insert_code_block", "replace_in_file", "str_replace",
    ])
    delete_tools: list[str] = field(default_factory=lambda: [
        "delete_file", "remove_file",
    ])


@dataclass
class EmbeddingsConfig:
    """Configuration for the embeddings model (optional, used for topic detection L2)."""

    url: str = ""
    model: str = ""
    api_key: str = ""

    def resolve(self) -> EmbeddingsConfig:
        """Return a copy with env var overrides applied."""
        return EmbeddingsConfig(
            url=os.environ.get("ACERVO_EMBED_URL", self.url),
            model=os.environ.get("ACERVO_EMBED_MODEL", self.model),
            api_key=os.environ.get("ACERVO_EMBED_API_KEY", self.api_key),
        )


@dataclass
class ProxyConfig:
    """Configuration for 'acervo serve' proxy mode."""

    port: int = 9470
    target: str = ""


@dataclass
class IndexingConfig:
    """What files to index."""

    extensions: list[str] = field(
        default_factory=lambda: [".py", ".ts", ".js", ".tsx", ".jsx", ".md", ".html", ".css"]
    )
    ignore: list[str] = field(
        default_factory=lambda: [
            "node_modules", ".git", "dist", "__pycache__", ".venv", "venv",
            "build", ".next", ".acervo",
        ]
    )


@dataclass
class ContextConfig:
    """Context building settings."""

    max_tokens: int = 32000
    injection: str = "system"  # "system" | "first_user"
    plan_mode: bool = False


@dataclass
class AcervoConfig:
    """Complete Acervo configuration loaded from .acervo/config.toml."""

    workspace: str = "."
    data_dir: str = ".acervo/data"
    owner: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    changelog: ChangelogConfig = field(default_factory=ChangelogConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    context: ContextConfig = field(default_factory=ContextConfig)

    @staticmethod
    def find_config(start_path: Path | None = None) -> Path | None:
        """Walk up from start_path looking for .acervo/config.toml.

        Returns the config file path, or None if not found.
        """
        current = (start_path or Path.cwd()).resolve()
        while True:
            config_path = current / ".acervo" / "config.toml"
            if config_path.exists():
                return config_path
            parent = current.parent
            if parent == current:
                return None
            current = parent

    @staticmethod
    def load(config_path: Path) -> AcervoConfig:
        """Load config from a .toml file. Missing keys use defaults."""
        raw = _load_toml(config_path)
        if not raw:
            return AcervoConfig()

        acervo = raw.get("acervo", {})
        config = AcervoConfig(
            workspace=acervo.get("workspace", "."),
            data_dir=acervo.get("data_dir", ".acervo/data"),
            owner=acervo.get("owner", ""),
        )

        model_raw = acervo.get("model", {})
        config.model = ModelConfig(
            name=model_raw.get("name", config.model.name),
            url=model_raw.get("url", config.model.url),
            api_key=model_raw.get("api_key", config.model.api_key),
        )

        embed_raw = acervo.get("embeddings", {})
        config.embeddings = EmbeddingsConfig(
            url=embed_raw.get("url", config.embeddings.url),
            model=embed_raw.get("model", config.embeddings.model),
            api_key=embed_raw.get("api_key", config.embeddings.api_key),
        )

        proxy_raw = acervo.get("proxy", {})
        config.proxy = ProxyConfig(
            port=proxy_raw.get("port", config.proxy.port),
            target=proxy_raw.get("target", config.proxy.target),
        )

        indexing_raw = acervo.get("indexing", {})
        if "extensions" in indexing_raw:
            config.indexing = IndexingConfig(
                extensions=indexing_raw["extensions"],
                ignore=indexing_raw.get("ignore", config.indexing.ignore),
            )
        if "ignore" in indexing_raw:
            config.indexing.ignore = indexing_raw["ignore"]

        changelog_raw = acervo.get("changelog", {})
        if changelog_raw:
            config.changelog = ChangelogConfig(
                write_tools=changelog_raw.get("write_tools", config.changelog.write_tools),
                delete_tools=changelog_raw.get("delete_tools", config.changelog.delete_tools),
            )

        context_raw = acervo.get("context", {})
        config.context = ContextConfig(
            max_tokens=context_raw.get("max_tokens", config.context.max_tokens),
            injection=context_raw.get("injection", config.context.injection),
            plan_mode=context_raw.get("plan_mode", config.context.plan_mode),
        )

        # Legacy format support: [project] + [llm] sections
        if "project" in raw and "acervo" not in raw:
            proj = raw["project"]
            config.workspace = proj.get("workspace_root", config.workspace)
            config.owner = proj.get("owner", config.owner)
            if "extensions" in proj:
                config.indexing.extensions = proj["extensions"]
            llm = raw.get("llm", {})
            config.model = ModelConfig(
                name=llm.get("model", config.model.name),
                url=llm.get("base_url", config.model.url),
                api_key=llm.get("api_key", config.model.api_key),
            )

        return config

    def save(self, config_path: Path) -> None:
        """Save config to .toml file."""
        exts = ", ".join(f'"{e}"' for e in self.indexing.extensions)
        ignore = ", ".join(f'"{p}"' for p in self.indexing.ignore)

        write_tools = ", ".join(f'"{t}"' for t in self.changelog.write_tools)
        delete_tools = ", ".join(f'"{t}"' for t in self.changelog.delete_tools)

        content = f"""\
# Acervo configuration

[acervo]
workspace = "{self.workspace}"
data_dir = "{self.data_dir}"
owner = "{self.owner}"

[acervo.model]
# Utility model for Acervo's internal pipeline (extraction, planning, etc.)
name = "{self.model.name}"
url = "{self.model.url}"
# api_key = "{self.model.api_key}"

[acervo.embeddings]
# Embeddings model for topic detection L2 (optional — leave empty to skip)
url = "{self.embeddings.url}"
model = "{self.embeddings.model}"
api_key = "{self.embeddings.api_key}"

[acervo.proxy]
port = {self.proxy.port}
# target is optional — clients can send X-Forward-To header instead
# target = "{self.proxy.target}"

[acervo.changelog]
# Tool names that indicate file modifications (for changelog tracking in proxy mode)
write_tools = [{write_tools}]
delete_tools = [{delete_tools}]

[acervo.indexing]
extensions = [{exts}]
ignore = [{ignore}]

[acervo.context]
max_tokens = {self.context.max_tokens}
injection = "{self.context.injection}"
plan_mode = {"true" if self.context.plan_mode else "false"}
"""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(content, encoding="utf-8")

    def resolve_model(self) -> ModelConfig:
        """Return model config with env var overrides applied."""
        return self.model.resolve()

    def resolve_workspace(self, config_path: Path) -> Path:
        """Resolve workspace to absolute path relative to config location."""
        project_root = config_path.parent.parent  # .acervo/config.toml → project root
        return (project_root / self.workspace).resolve()

    def resolve_data_dir(self, config_path: Path) -> Path:
        """Resolve data_dir to absolute path relative to project root."""
        project_root = config_path.parent.parent
        return (project_root / self.data_dir).resolve()

    def get_value(self, key: str) -> str:
        """Get a config value by dotted key (e.g. 'model.url')."""
        parts = key.split(".", 1)
        section = parts[0]
        attr = parts[1] if len(parts) > 1 else None

        if section == "workspace" and not attr:
            return self.workspace
        if section == "data_dir" and not attr:
            return self.data_dir
        if section == "owner" and not attr:
            return self.owner

        obj = getattr(self, section, None)
        if obj is None:
            raise KeyError(f"Unknown config section: {section}")
        if attr is None:
            return str(obj)

        val = getattr(obj, attr, None)
        if val is None:
            raise KeyError(f"Unknown config key: {key}")
        return str(val)

    def set_value(self, key: str, value: str) -> None:
        """Set a config value by dotted key (e.g. 'model.url')."""
        parts = key.split(".", 1)
        section = parts[0]
        attr = parts[1] if len(parts) > 1 else None

        if section == "workspace" and not attr:
            self.workspace = value
            return
        if section == "data_dir" and not attr:
            self.data_dir = value
            return
        if section == "owner" and not attr:
            self.owner = value
            return

        obj = getattr(self, section, None)
        if obj is None:
            raise KeyError(f"Unknown config section: {section}")
        if attr is None:
            raise KeyError(f"Cannot set entire section: {section}")
        if not hasattr(obj, attr):
            raise KeyError(f"Unknown config key: {key}")

        # Coerce type
        current = getattr(obj, attr)
        if isinstance(current, int):
            setattr(obj, attr, int(value))
        elif isinstance(current, float):
            setattr(obj, attr, float(value))
        elif isinstance(current, list):
            setattr(obj, attr, [v.strip() for v in value.split(",")])
        else:
            setattr(obj, attr, value)


def _load_toml(path: Path) -> dict:
    """Parse TOML file using tomllib (Python 3.11+) or tomli fallback."""
    if not path.exists():
        return {}

    if sys.version_info >= (3, 11):
        import tomllib
    else:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            log.warning("tomllib not available (Python <3.11 without tomli). Using empty config.")
            return {}

    with open(path, "rb") as f:
        return tomllib.load(f)
