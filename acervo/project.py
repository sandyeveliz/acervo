"""Project-level .acervo/ directory management.

Handles discovery, initialization, and config loading for the .acervo/
project directory — Acervo's self-contained storage (like .git/).

Usage:
    # Auto-discover from cwd
    project = find_project()

    # Init a new project
    project = init_project(Path("/path/to/my-project"))

    # Load from explicit path
    project = load_project(Path("/path/to/my-project/.acervo"))

Directory structure:
    .acervo/
    ├── config.toml      # workspace root, LLM config, extensions
    ├── data/
    │   ├── graph/
    │   │   ├── nodes.json
    │   │   └── edges.json
    │   └── vectordb/    # ChromaDB persistent storage
    └── .gitignore       # ignores data/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from acervo.config import AcervoConfig

log = logging.getLogger(__name__)

_ACERVO_DIR = ".acervo"
_CONFIG_FILE = "config.toml"
_DATA_DIR = "data"
_GRAPH_DIR = "graph"
_VECTORDB_DIR = "vectordb"


@dataclass
class AcervoProject:
    """Resolved project configuration from a .acervo/ directory."""

    acervo_dir: Path
    workspace_root: Path
    graph_path: Path
    vectordb_path: Path
    config: AcervoConfig

    @property
    def config_path(self) -> Path:
        return self.acervo_dir / _CONFIG_FILE

    @property
    def extensions(self) -> set[str]:
        """File extensions to index."""
        return set(self.config.indexing.extensions)

    @property
    def owner(self) -> str:
        return self.config.owner

    def llm_config(self) -> dict[str, str]:
        """Resolve LLM config: env vars > config.toml > defaults."""
        resolved = self.config.resolve_model()
        return {
            "base_url": resolved.url,
            "model": resolved.name,
            "api_key": resolved.api_key,
        }


def find_project(start: Path | None = None) -> AcervoProject | None:
    """Walk up from start (default: cwd) looking for .acervo/ directory.

    Returns None if no .acervo/ is found up to the filesystem root.
    """
    config_path = AcervoConfig.find_config(start)
    if config_path is None:
        return None
    return load_project(config_path.parent)


def init_project(root: Path, config_overrides: dict | None = None) -> AcervoProject:
    """Create .acervo/ directory with default config.toml.

    Args:
        root: Project root directory (where .acervo/ will be created).
        config_overrides: Optional dict to merge into default config.

    Returns:
        AcervoProject for the newly created directory.
    """
    root = root.resolve()
    acervo_dir = root / _ACERVO_DIR

    if acervo_dir.exists() and (acervo_dir / _CONFIG_FILE).exists():
        log.info("Found existing .acervo/ at %s", acervo_dir)
        return load_project(acervo_dir)

    # Create directory structure
    acervo_dir.mkdir(parents=True, exist_ok=True)
    data_dir = acervo_dir / _DATA_DIR
    data_dir.mkdir(exist_ok=True)
    (data_dir / _GRAPH_DIR).mkdir(exist_ok=True)
    (data_dir / _VECTORDB_DIR).mkdir(exist_ok=True)

    # Write default config
    config = AcervoConfig(data_dir=f".acervo/{_DATA_DIR}")
    config.save(acervo_dir / _CONFIG_FILE)

    # Create .gitignore for machine-specific data
    gitignore_path = acervo_dir / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.write_text(
            "# Machine-specific data (graph, embeddings)\ndata/\n# Service PID files\nrun/\n",
            encoding="utf-8",
        )

    log.info("Initialized .acervo/ at %s", acervo_dir)
    return load_project(acervo_dir)


def load_project(acervo_dir: Path) -> AcervoProject:
    """Load project from an explicit .acervo/ path.

    Args:
        acervo_dir: Path to the .acervo/ directory.

    Returns:
        AcervoProject with resolved paths and parsed config.
    """
    acervo_dir = acervo_dir.resolve()
    config_path = acervo_dir / _CONFIG_FILE
    config = AcervoConfig.load(config_path)

    # Resolve workspace root relative to project root
    project_root = acervo_dir.parent
    workspace_root = (project_root / config.workspace).resolve()

    # Resolve data paths
    data_dir = (project_root / config.data_dir).resolve()
    graph_path = data_dir / _GRAPH_DIR
    vectordb_path = data_dir / _VECTORDB_DIR

    # Ensure dirs exist
    graph_path.mkdir(parents=True, exist_ok=True)
    vectordb_path.mkdir(parents=True, exist_ok=True)

    return AcervoProject(
        acervo_dir=acervo_dir,
        workspace_root=workspace_root,
        graph_path=graph_path,
        vectordb_path=vectordb_path,
        config=config,
    )
