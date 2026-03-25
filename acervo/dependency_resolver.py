"""Dependency resolver — resolves imports to file paths and builds dependency edges.

After all files are structurally parsed, this module resolves relative and
alias imports to actual file paths, producing a list of DependencyEdge objects
for the graph.

Usage:
    resolver = DependencyResolver(workspace_root, file_structures)
    edges = resolver.resolve()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from acervo.structural_parser import FileStructure, Import

log = logging.getLogger(__name__)


@dataclass
class DependencyEdge:
    """A resolved import relationship between two files."""

    source_file: str      # file that imports (relative path)
    target_file: str      # file being imported (relative path)
    imported_names: list[str]
    edge_type: str        # "imports", "extends", "implements"


class DependencyResolver:
    """Resolves import statements to file paths and builds dependency edges."""

    def __init__(
        self,
        workspace_root: Path,
        file_structures: list[FileStructure],
    ) -> None:
        self._workspace_root = workspace_root
        self._structures = {fs.file_path: fs for fs in file_structures}
        # Build lookup: all known file paths (forward slash, relative)
        self._known_files: set[str] = set(self._structures.keys())
        # Load tsconfig/jsconfig path aliases if present
        self._aliases: dict[str, str] = self._load_path_aliases()

    def resolve(self) -> list[DependencyEdge]:
        """Resolve all imports across all files and return dependency edges."""
        edges: list[DependencyEdge] = []

        for file_path, structure in self._structures.items():
            for imp in structure.imports:
                resolved = self._resolve_import(file_path, imp)
                if resolved:
                    edges.append(DependencyEdge(
                        source_file=file_path,
                        target_file=resolved,
                        imported_names=imp.names,
                        edge_type="imports",
                    ))

        return edges

    def _resolve_import(self, importer: str, imp: Import) -> str | None:
        """Resolve a single import to a file path, or None if external/unresolved."""
        source = imp.source

        if imp.is_local:
            return self._resolve_relative(importer, source)

        # Check for path aliases (@/utils, @components/...)
        alias_resolved = self._resolve_alias(source)
        if alias_resolved:
            return alias_resolved

        # Python absolute imports (from acervo.graph import ...)
        language = self._structures.get(importer, None)
        if language and language.language == "python":
            return self._resolve_python_absolute(source)

        # External package — not resolvable to a local file
        return None

    def _resolve_relative(self, importer: str, source: str) -> str | None:
        """Resolve a relative import (./foo, ../bar) to a file path."""
        importer_dir = str(PurePosixPath(importer).parent)

        # Normalize the relative path
        resolved = str(PurePosixPath(importer_dir) / source)
        # Normalize .. segments
        parts = []
        for p in resolved.split("/"):
            if p == "..":
                if parts:
                    parts.pop()
            elif p != ".":
                parts.append(p)
        resolved = "/".join(parts)

        return self._find_file(resolved)

    def _resolve_alias(self, source: str) -> str | None:
        """Resolve a path alias like @/utils or @components/foo."""
        for alias, target_dir in self._aliases.items():
            if source == alias.rstrip("/*"):
                return self._find_file(target_dir.rstrip("/*"))
            if source.startswith(alias.replace("/*", "/")):
                remainder = source[len(alias.replace("/*", "/")):]
                candidate = f"{target_dir.rstrip('/*')}/{remainder}"
                found = self._find_file(candidate)
                if found:
                    return found
        return None

    def _resolve_python_absolute(self, source: str) -> str | None:
        """Resolve a Python absolute import (dotted path) to a file."""
        # Convert dots to path separators: acervo.graph -> acervo/graph
        path = source.replace(".", "/")
        return self._find_file(path)

    def _find_file(self, base_path: str) -> str | None:
        """Try to find a file matching the base path with various extensions."""
        # Direct match
        if base_path in self._known_files:
            return base_path

        # Try with common extensions
        extensions = [".ts", ".tsx", ".js", ".jsx", ".py", ".d.ts"]
        for ext in extensions:
            candidate = base_path + ext
            if candidate in self._known_files:
                return candidate

        # Try index files (JS/TS convention)
        for ext in [".ts", ".tsx", ".js", ".jsx"]:
            candidate = f"{base_path}/index{ext}"
            if candidate in self._known_files:
                return candidate

        # Try __init__.py (Python convention)
        candidate = f"{base_path}/__init__.py"
        if candidate in self._known_files:
            return candidate

        return None

    def _load_path_aliases(self) -> dict[str, str]:
        """Load path aliases from tsconfig.json or jsconfig.json."""
        aliases: dict[str, str] = {}

        for config_name in ("tsconfig.json", "jsconfig.json"):
            config_path = self._workspace_root / config_name
            if not config_path.exists():
                continue

            try:
                raw = json.loads(config_path.read_text(encoding="utf-8"))
                paths = raw.get("compilerOptions", {}).get("paths", {})
                base_url = raw.get("compilerOptions", {}).get("baseUrl", ".")

                for alias, targets in paths.items():
                    if targets and isinstance(targets, list):
                        # Take the first target, resolve relative to baseUrl
                        target = targets[0]
                        resolved_target = str(PurePosixPath(base_url) / target)
                        aliases[alias] = resolved_target

                log.info("Loaded %d path aliases from %s", len(aliases), config_name)
                break  # Use the first config found
            except (json.JSONDecodeError, KeyError) as e:
                log.warning("Failed to parse %s: %s", config_name, e)

        return aliases
