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

    name: str = "qwen2.5:7b"
    url: str = "http://localhost:11434/v1"
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

    url: str = "http://localhost:11434"
    model: str = "qwen3-embedding"
    api_key: str = "ollama-key"

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
    provider_name: str = ""  # human-readable name, e.g. "LM Studio"


@dataclass
class IndexingConfig:
    """What files to index."""

    extensions: list[str] = field(
        default_factory=lambda: [".py", ".ts", ".js", ".tsx", ".jsx", ".md", ".html", ".css", ".epub"]
    )
    ignore: list[str] = field(
        default_factory=lambda: [
            "node_modules", ".git", "dist", "__pycache__", ".venv", "venv",
            "build", ".next", ".acervo",
        ]
    )
    content_type: str = "auto"  # "auto" | "code" | "prose"


@dataclass
class TimeoutsConfig:
    """Timeout settings per pipeline phase (seconds)."""

    llm_chat: int = 120       # LLM chat/extraction calls
    embedding: int = 30       # Embedding calls
    s1_unified: int = 60      # S1 Unified (topic + extraction)
    s1_5_update: int = 60     # S1.5 Graph Update
    vector_search: int = 10   # Vector store search


@dataclass
class ContextConfig:
    """Context building settings."""

    max_tokens: int = 32000
    injection: str = "system"  # "system" | "first_user"
    plan_mode: bool = False
    history_window: int = 2  # messages to keep when graph has context (0 = no windowing)


@dataclass
class PromptsConfig:
    """Prompt templates for Acervo's pipeline.

    Prompts live in .acervo/prompts/ as plain .txt files.
    The config only stores the directory path.
    """

    prompts_dir: str = ".acervo/prompts"  # relative to project root

    # Known prompt file names → facade key mapping
    _FILE_MAP: dict[str, str] = field(default=False, repr=False, init=False)

    def __post_init__(self):
        # Only Acervo pipeline prompts — agent.txt is for AVS, not loaded here
        self._FILE_MAP = {
            "s1_unified.txt": "s1_unified",
            "s1_5_graph_update.txt": "s1_5_graph_update",
            "extractor_search.txt": "extractor_search",
            "summarizer.txt": "summarizer",
        }

    def load_prompts(self, project_root: Path) -> dict[str, str]:
        """Load all .txt files from prompts_dir. Returns non-empty prompts as dict."""
        prompts_path = (project_root / self.prompts_dir).resolve()
        if not prompts_path.is_dir():
            return {}
        result: dict[str, str] = {}
        for filename, key in self._FILE_MAP.items():
            filepath = prompts_path / filename
            if filepath.is_file():
                content = filepath.read_text(encoding="utf-8").strip()
                if content:
                    result[key] = content
        return result


@dataclass
class ModelsConfig:
    """Per-role model assignments.

    Each role can override the default [acervo.model].
    Empty name = use default model.
    """

    extractor: ModelConfig = field(default_factory=ModelConfig)    # S1 Unified (extractor)
    summarizer: ModelConfig = field(default_factory=ModelConfig)   # hot layer compaction

    def resolve_for_role(self, role: str, default: ModelConfig) -> ModelConfig:
        """Return the model config for a role, falling back to default if not set."""
        role_config = getattr(self, role, None)
        if role_config and role_config.name:
            return role_config
        return default


@dataclass
class ServicesConfig:
    """Configuration for 'acervo up' service ports and paths."""

    ollama_port: int = 11434
    lmstudio_port: int = 1234
    studio_path: str = ""       # path to acervo-studio for --dev mode
    studio_port: int = 8000
    frontend_port: int = 5173


@dataclass
class AcervoConfig:
    """Complete Acervo configuration loaded from .acervo/config.toml."""

    workspace: str = "."
    data_dir: str = ".acervo/data"
    # Default backend as of v0.6.0 is LadybugDB (KuzuDB fork) — it's the
    # embedded Cypher store that supports the Phase 2 bi-temporal Fact
    # schema and native vector/fulltext search. ``json`` (TopicGraph) is
    # kept as an explicit fallback for environments where the driver
    # isn't installed or for lightweight tooling.
    graph_backend: str = "ladybug"  # "ladybug" | "json"
    owner: str = ""
    description: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    changelog: ChangelogConfig = field(default_factory=ChangelogConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)
    timeouts: TimeoutsConfig = field(default_factory=TimeoutsConfig)
    services: ServicesConfig = field(default_factory=ServicesConfig)
    _has_services: bool = field(default=False, repr=False)

    def has_services_config(self) -> bool:
        """Whether [acervo.services] was present in the TOML file."""
        return self._has_services

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
            graph_backend=acervo.get("graph_backend", "ladybug"),
            owner=acervo.get("owner", ""),
            description=acervo.get("description", ""),
        )

        model_raw = acervo.get("model", {})
        config.model = ModelConfig(
            name=model_raw.get("name", config.model.name),
            url=model_raw.get("url", config.model.url),
            api_key=model_raw.get("api_key", config.model.api_key),
        )

        # Per-role model overrides (empty name = use default [acervo.model])
        models_raw = acervo.get("models", {})
        if models_raw:
            def _parse_role_model(role_raw: dict) -> ModelConfig:
                return ModelConfig(
                    name=role_raw.get("name", ""),
                    url=role_raw.get("url", config.model.url),
                    api_key=role_raw.get("api_key", config.model.api_key),
                )
            config.models = ModelsConfig(
                extractor=_parse_role_model(models_raw.get("extractor", {})),
                summarizer=_parse_role_model(models_raw.get("summarizer", {})),
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
            provider_name=proxy_raw.get("provider_name", config.proxy.provider_name),
        )

        indexing_raw = acervo.get("indexing", {})
        if "extensions" in indexing_raw:
            config.indexing = IndexingConfig(
                extensions=indexing_raw["extensions"],
                ignore=indexing_raw.get("ignore", config.indexing.ignore),
            )
        if "ignore" in indexing_raw:
            config.indexing.ignore = indexing_raw["ignore"]
        if "content_type" in indexing_raw:
            config.indexing.content_type = indexing_raw["content_type"]

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
            history_window=context_raw.get("history_window", config.context.history_window),
        )

        prompts_raw = acervo.get("prompts", {})
        if prompts_raw:
            config.prompts = PromptsConfig(
                prompts_dir=prompts_raw.get("dir", config.prompts.prompts_dir),
            )

        services_raw = acervo.get("services", {})
        if services_raw:
            config._has_services = True
            defaults = ServicesConfig()
            config.services = ServicesConfig(
                ollama_port=services_raw.get("ollama_port", defaults.ollama_port),
                lmstudio_port=services_raw.get("lmstudio_port", defaults.lmstudio_port),
                studio_path=services_raw.get("studio_path", defaults.studio_path),
                studio_port=services_raw.get("studio_port", defaults.studio_port),
                frontend_port=services_raw.get("frontend_port", defaults.frontend_port),
            )

        timeouts_raw = acervo.get("timeouts", {})
        if timeouts_raw:
            defaults_t = TimeoutsConfig()
            config.timeouts = TimeoutsConfig(
                llm_chat=timeouts_raw.get("llm_chat", defaults_t.llm_chat),
                embedding=timeouts_raw.get("embedding", defaults_t.embedding),
                s1_unified=timeouts_raw.get("s1_unified", defaults_t.s1_unified),
                s1_5_update=timeouts_raw.get("s1_5_update", defaults_t.s1_5_update),
                vector_search=timeouts_raw.get("vector_search", defaults_t.vector_search),
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
graph_backend = "{self.graph_backend}"
owner = "{self.owner}"
description = "{self.description}"

[acervo.model]
# Single model for everything — qwen2.5:3b handles chat + extraction.
# Behavior is determined by the system prompt, not the model.
name = "{self.model.name}"
url = "{self.model.url}"
# api_key = "{self.model.api_key}"

[acervo.models]
# Per-role model overrides. Empty name = use [acervo.model] as fallback.
[acervo.models.extractor]
# Fine-tuned extractor for S1 Unified (topic + entity extraction)
name = "{self.models.extractor.name}"
# url = "{self.models.extractor.url}"
[acervo.models.summarizer]
name = "{self.models.summarizer.name}"

[acervo.embeddings]
# Embeddings model for topic detection L2 (optional — leave empty to skip)
url = "{self.embeddings.url}"
model = "{self.embeddings.model}"
api_key = "{self.embeddings.api_key}"

[acervo.proxy]
port = {self.proxy.port}
# target is optional — clients can send X-Forward-To header instead
# target = "{self.proxy.target}"
provider_name = "{self.proxy.provider_name}"

[acervo.changelog]
# Tool names that indicate file modifications (for changelog tracking in proxy mode)
write_tools = [{write_tools}]
delete_tools = [{delete_tools}]

[acervo.indexing]
extensions = [{exts}]
ignore = [{ignore}]
content_type = "{self.indexing.content_type}"

[acervo.context]
max_tokens = {self.context.max_tokens}
injection = "{self.context.injection}"
plan_mode = {"true" if self.context.plan_mode else "false"}
history_window = {self.context.history_window}


[acervo.prompts]
# Prompts live as .txt files in this directory (relative to project root).
# Each file overrides the built-in default. Delete a file to use the default.
dir = "{self.prompts.prompts_dir}"
"""
        if self._has_services:
            svc = self.services
            content += f"""
[acervo.services]
# Ports and paths for 'acervo up --dev'
ollama_port = {svc.ollama_port}
lmstudio_port = {svc.lmstudio_port}
studio_path = "{svc.studio_path}"
studio_port = {svc.studio_port}
frontend_port = {svc.frontend_port}
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
        if section == "description" and not attr:
            return self.description

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
        if section == "description" and not attr:
            self.description = value
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
