"""Structural file parser — analyzes code and markdown into semantic units.

Parses files into functions, classes, methods, and sections with exact line
ranges. Uses tree-sitter when available, falls back to regex for code files.
Markdown parsing is pure Python (no external deps).

Also extracts imports and exports for dependency graph construction.

Usage:
    parser = StructuralParser()
    structure = parser.parse(Path("src/users.py"), workspace_root)
    for unit in structure.units:
        print(f"{unit.name} ({unit.unit_type}) lines {unit.start_line}-{unit.end_line}")
    for imp in structure.imports:
        print(f"import {imp.names} from {imp.source}")
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ── Data model ──


@dataclass
class StructuralUnit:
    """A semantic unit within a file (function, class, method, section, etc.)."""

    name: str
    unit_type: str       # "function" | "class" | "method" | "interface" | "type_alias" | "section"
    start_line: int      # 1-indexed inclusive
    end_line: int        # 1-indexed inclusive
    parent: str | None = None
    language: str = ""
    signature: str = ""  # first line of declaration or heading text


@dataclass
class Import:
    """An import statement extracted from a source file."""

    source: str              # "@supabase/client", "./todo.service", "../utils"
    names: list[str]         # ["createClient", "SupabaseClient"]
    is_local: bool = False   # True if relative import (starts with . or ..)
    line: int = 0            # 1-indexed line number


@dataclass
class Export:
    """An export extracted from a source file."""

    name: str
    kind: str                # "function", "class", "default", "type", "variable"
    entity_ref: str = ""     # reference to the Entity name (same as name if direct)
    line: int = 0            # 1-indexed line number


@dataclass
class FileStructure:
    """Complete structural analysis of a file."""

    file_path: str       # relative path (forward slashes)
    language: str
    content_hash: str    # SHA-256
    units: list[StructuralUnit] = field(default_factory=list)
    imports: list[Import] = field(default_factory=list)
    exports: list[Export] = field(default_factory=list)
    total_lines: int = 0
    full_text: str = ""  # For binary formats (epub): flattened plaintext


# ── Language detection ──

_LANG_MAP: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".md": "markdown",
    ".html": "html",
    ".css": "css",
    ".epub": "epub",
    ".txt": "plaintext",
    ".pdf": "pdf",
}


def _detect_language(file_path: Path) -> str:
    return _LANG_MAP.get(file_path.suffix.lower(), "unknown")


# ── EPUB helpers ──

def _epub_html_to_text(soup: Any) -> str:
    """Convert parsed HTML from an EPUB document item to markdown-style text.

    Preserves heading hierarchy as markdown markers (# / ## / ###) so the
    downstream section detector can find chapter/sub-section boundaries.
    """
    # Replace heading tags with markdown markers before extracting text
    for level in range(1, 7):
        for tag in soup.find_all(f"h{level}"):
            prefix = "#" * level
            tag.replace_with(f"\n{prefix} {tag.get_text(strip=True)}\n")

    # Replace paragraph and div breaks with newlines
    for tag in soup.find_all(["p", "div", "br"]):
        tag.append("\n")

    text = soup.get_text()

    # Collapse multiple blank lines into at most two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


# ── tree-sitter availability ──

def _check_tree_sitter() -> bool:
    try:
        import tree_sitter  # noqa: F401
        return True
    except ImportError:
        return False


# ── Parser ──


class StructuralParser:
    """Parses code and markdown files into semantic structural units."""

    def __init__(self) -> None:
        self._ts_available = _check_tree_sitter()
        self._ts_parsers: dict[str, Any] = {}

    @property
    def tree_sitter_available(self) -> bool:
        return self._ts_available

    def parse(self, file_path: Path, workspace_root: Path) -> FileStructure:
        """Analyze a file and return its structural breakdown."""
        relative = str(file_path.relative_to(workspace_root)).replace("\\", "/")
        language = _detect_language(file_path)

        # Binary formats get their own parse path
        if language == "epub":
            return self._parse_epub_file(file_path, relative)
        if language == "pdf":
            return self._parse_pdf_file(file_path, relative)

        content = file_path.read_text(encoding="utf-8")
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        total_lines = content.count("\n") + 1

        units: list[StructuralUnit] = []
        imports: list[Import] = []
        exports: list[Export] = []

        if language == "markdown":
            units = self._parse_markdown(content)
        elif language in ("python", "typescript", "javascript"):
            if self._ts_available:
                units = self._parse_with_tree_sitter(content, language)
                imports = self._extract_imports(content, language)
                exports = self._extract_exports(content, language, units)
            else:
                units = self._parse_with_regex(content, language)
                imports = self._extract_imports_regex(content, language)
                exports = self._extract_exports_regex(content, language)
        elif language == "html":
            units = self._parse_html(content)
        elif language == "css":
            units = self._parse_css(content)
        elif language == "plaintext":
            units = self._parse_plaintext(content)

        for u in units:
            u.language = language

        return FileStructure(
            file_path=relative,
            language=language,
            content_hash=content_hash,
            units=units,
            imports=imports,
            exports=exports,
            total_lines=total_lines,
        )

    # ── Markdown parsing (pure Python) ──

    def _parse_markdown(self, content: str) -> list[StructuralUnit]:
        """Split markdown by headings into sections with parent-child hierarchy."""
        lines = content.split("\n")
        heading_re = re.compile(r"^(#{1,6})\s+(.+)")

        # Collect headings with their positions
        headings: list[tuple[int, int, str]] = []  # (line_num, level, title)
        for i, line in enumerate(lines):
            m = heading_re.match(line)
            if m:
                headings.append((i + 1, len(m.group(1)), m.group(2).strip()))

        if not headings:
            return []

        units: list[StructuralUnit] = []
        # Stack tracks (level, name) for parent resolution
        parent_stack: list[tuple[int, str]] = []

        for idx, (line_num, level, title) in enumerate(headings):
            # End line: next heading start - 1, or EOF
            if idx + 1 < len(headings):
                end_line = headings[idx + 1][0] - 1
            else:
                end_line = len(lines)

            # Strip trailing blank lines from section
            while end_line > line_num and not lines[end_line - 1].strip():
                end_line -= 1

            # Update parent stack: pop anything at same or deeper level
            while parent_stack and parent_stack[-1][0] >= level:
                parent_stack.pop()

            parent = parent_stack[-1][1] if parent_stack else None
            parent_stack.append((level, title))

            units.append(StructuralUnit(
                name=title,
                unit_type="section",
                start_line=line_num,
                end_line=end_line,
                parent=parent,
                signature=f"{'#' * level} {title}",
            ))

        return units

    # ── HTML parsing (regex — lightweight structural extraction) ──

    def _parse_html(self, content: str) -> list[StructuralUnit]:
        """Extract template structure from HTML files."""
        lines = content.split("\n")
        units: list[StructuralUnit] = []

        # Extract component references (custom elements, Angular/React-style)
        component_re = re.compile(r"<(app-[\w-]+|[A-Z][\w]+)[\s/>]")
        seen_components: set[str] = set()

        for i, line in enumerate(lines):
            for m in component_re.finditer(line):
                name = m.group(1)
                if name not in seen_components:
                    seen_components.add(name)
                    units.append(StructuralUnit(
                        name=name,
                        unit_type="component_ref",
                        start_line=i + 1,
                        end_line=i + 1,
                        signature=f"<{name}>",
                    ))

        # Extract id and class attributes as structural markers
        id_re = re.compile(r'id=["\']([^"\']+)["\']')
        for i, line in enumerate(lines):
            for m in id_re.finditer(line):
                units.append(StructuralUnit(
                    name=m.group(1),
                    unit_type="html_id",
                    start_line=i + 1,
                    end_line=i + 1,
                    signature=f'id="{m.group(1)}"',
                ))

        return units

    # ── CSS parsing (regex — selectors and custom properties) ──

    def _parse_css(self, content: str) -> list[StructuralUnit]:
        """Extract selectors and custom properties from CSS files."""
        lines = content.split("\n")
        units: list[StructuralUnit] = []

        # CSS custom properties (--variable-name)
        custom_prop_re = re.compile(r"(--[\w-]+)\s*:")
        for i, line in enumerate(lines):
            for m in custom_prop_re.finditer(line):
                units.append(StructuralUnit(
                    name=m.group(1),
                    unit_type="css_variable",
                    start_line=i + 1,
                    end_line=i + 1,
                    signature=m.group(0).strip(),
                ))

        # Top-level selectors (class names, IDs)
        selector_re = re.compile(r"^([.#][\w-]+(?:\s*[,>+~]\s*[.#]?[\w-]+)*)\s*\{")
        brace_depth = 0
        current_selector: tuple[str, int] | None = None

        for i, line in enumerate(lines):
            stripped = line.strip()
            if brace_depth == 0:
                m = selector_re.match(stripped)
                if m:
                    current_selector = (m.group(1).strip(), i + 1)

            brace_depth += stripped.count("{") - stripped.count("}")

            if current_selector and brace_depth == 0:
                units.append(StructuralUnit(
                    name=current_selector[0],
                    unit_type="css_selector",
                    start_line=current_selector[1],
                    end_line=i + 1,
                    signature=f"{current_selector[0]} {{ ... }}",
                ))
                current_selector = None

        return units

    # ── EPUB parsing ──

    def _parse_epub_file(self, file_path: Path, relative: str) -> FileStructure:
        """Parse an EPUB file into structural units (chapters/sections).

        Returns a complete FileStructure with full_text populated for
        downstream enrichment (since epub is binary, not UTF-8 text).
        """
        try:
            import warnings
            from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
            warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
            from ebooklib import epub, ITEM_DOCUMENT
        except ImportError:
            raise ImportError(
                "EPUB support requires additional packages: "
                "pip install acervo[epub]"
            )

        # Hash raw bytes for change detection
        raw_bytes = file_path.read_bytes()
        content_hash = hashlib.sha256(raw_bytes).hexdigest()

        book = epub.read_epub(str(file_path))

        # Extract text from spine-order documents
        all_text_parts: list[str] = []
        units: list[StructuralUnit] = []
        cumulative_line = 1

        # Get spine order (list of item IDs)
        spine_ids = [item_id for item_id, _ in book.spine]

        for item in book.get_items_of_type(ITEM_DOCUMENT):
            if item.get_id() not in spine_ids:
                continue

            html_content = item.get_content()
            soup = BeautifulSoup(html_content, "lxml")

            # Extract chapter title from first heading or item title
            chapter_title = None
            first_heading = soup.find(re.compile(r"^h[1-3]$"))
            if first_heading:
                chapter_title = first_heading.get_text(strip=True)
            if not chapter_title:
                title_tag = soup.find("title")
                if title_tag and title_tag.get_text(strip=True):
                    chapter_title = title_tag.get_text(strip=True)
            if not chapter_title:
                # Use filename without extension
                chapter_title = item.get_name().rsplit("/", 1)[-1].rsplit(".", 1)[0]

            # Skip navigation/toc items with no real content
            text = _epub_html_to_text(soup)
            if len(text.strip()) < 50:
                continue

            text_lines = text.split("\n")
            start_line = cumulative_line
            end_line = cumulative_line + len(text_lines) - 1

            # Create chapter-level section
            units.append(StructuralUnit(
                name=chapter_title,
                unit_type="section",
                start_line=start_line,
                end_line=end_line,
                parent=None,
                language="markdown",
                signature=f"# {chapter_title}",
            ))

            # Detect sub-sections from h2/h3 headings within this chapter
            current_line = start_line
            for line in text_lines:
                heading_match = re.match(r"^(#{2,3})\s+(.+)", line)
                if heading_match:
                    level = len(heading_match.group(1))
                    sub_title = heading_match.group(2).strip()
                    # Find the end of this sub-section (next heading or chapter end)
                    sub_start = current_line
                    sub_end = end_line  # default to chapter end
                    for future_line_idx, future_line in enumerate(
                        text_lines[current_line - start_line + 1:], start=current_line + 1
                    ):
                        if re.match(r"^#{1,3}\s+", future_line):
                            sub_end = future_line_idx - 1
                            break
                    if sub_end >= sub_start:
                        units.append(StructuralUnit(
                            name=sub_title,
                            unit_type="section",
                            start_line=sub_start,
                            end_line=sub_end,
                            parent=chapter_title,
                            language="markdown",
                            signature=f"{'#' * level} {sub_title}",
                        ))
                current_line += 1

            all_text_parts.append(text)
            cumulative_line = end_line + 1

        full_text = "\n".join(all_text_parts)
        total_lines = full_text.count("\n") + 1 if full_text else 0

        return FileStructure(
            file_path=relative,
            language="epub",
            content_hash=content_hash,
            units=units,
            total_lines=total_lines,
            full_text=full_text,
        )

    # ── Plain text parsing ──

    def _parse_plaintext(self, content: str) -> list[StructuralUnit]:
        """Split plain text into sections by blank-line-separated paragraphs.

        Groups consecutive paragraphs into ~50-line sections to avoid
        one-unit-per-paragraph granularity.
        """
        lines = content.split("\n")
        sections: list[StructuralUnit] = []
        section_start = 0
        section_lines: list[str] = []
        section_idx = 1

        for i, line in enumerate(lines):
            stripped = line.strip()
            section_lines.append(line)

            # Split on double blank lines or every ~50 lines
            is_double_blank = (
                not stripped
                and i > 0
                and not lines[i - 1].strip()
            )
            is_long = len(section_lines) >= 50

            if (is_double_blank or is_long) and len(section_lines) > 2:
                # Use first non-empty line as title
                title = next(
                    (l.strip()[:80] for l in section_lines if l.strip()),
                    f"Section {section_idx}",
                )
                sections.append(StructuralUnit(
                    name=title,
                    unit_type="section",
                    start_line=section_start + 1,
                    end_line=i + 1,
                    language="plaintext",
                    signature=title,
                ))
                section_idx += 1
                section_start = i + 1
                section_lines = []

        # Remaining lines
        if section_lines and any(l.strip() for l in section_lines):
            title = next(
                (l.strip()[:80] for l in section_lines if l.strip()),
                f"Section {section_idx}",
            )
            sections.append(StructuralUnit(
                name=title,
                unit_type="section",
                start_line=section_start + 1,
                end_line=len(lines),
                language="plaintext",
                signature=title,
            ))

        return sections

    # ── PDF parsing ──

    def _parse_pdf_file(self, file_path: Path, relative: str) -> FileStructure:
        """Extract text and structure from a PDF using PyMuPDF (fitz).

        Falls back to page-based sectioning if no headings are detected.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PDF parsing. Install with: "
                "pip install PyMuPDF  (or: pip install acervo[pdf])"
            )

        doc = fitz.open(str(file_path))
        content_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()

        all_text_parts: list[str] = []
        units: list[StructuralUnit] = []
        cumulative_line = 1

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if not text or not text.strip():
                continue

            lines = text.split("\n")
            line_count = len(lines)
            start_line = cumulative_line
            end_line = cumulative_line + line_count - 1

            # Use first non-empty line as section title
            title = next(
                (l.strip()[:80] for l in lines if l.strip()),
                f"Page {page_num + 1}",
            )

            units.append(StructuralUnit(
                name=title,
                unit_type="section",
                start_line=start_line,
                end_line=end_line,
                language="pdf",
                signature=f"# Page {page_num + 1}: {title}",
            ))

            all_text_parts.append(text)
            cumulative_line = end_line + 1

        doc.close()

        full_text = "\n".join(all_text_parts)
        total_lines = full_text.count("\n") + 1 if full_text else 0

        return FileStructure(
            file_path=relative,
            language="pdf",
            content_hash=content_hash,
            units=units,
            total_lines=total_lines,
            full_text=full_text,
        )

    # ── Import extraction ──

    def _extract_imports(self, content: str, language: str) -> list[Import]:
        """Extract imports using tree-sitter AST."""
        try:
            parser = self._get_ts_parser(language)
            if not parser:
                return self._extract_imports_regex(content, language)
        except Exception:
            return self._extract_imports_regex(content, language)

        tree = parser.parse(content.encode("utf-8"))

        if language == "python":
            return self._extract_python_imports(tree.root_node)
        return self._extract_js_ts_imports(tree.root_node)

    def _extract_python_imports(self, root: Any) -> list[Import]:
        """Extract Python import statements from AST."""
        imports: list[Import] = []

        for child in root.children:
            if child.type == "import_statement":
                # import foo, import foo.bar
                names = []
                for sub in child.children:
                    if sub.type == "dotted_name":
                        names.append(self._node_text(sub))
                if names:
                    source = names[0]
                    imports.append(Import(
                        source=source,
                        names=names,
                        is_local=False,
                        line=child.start_point[0] + 1,
                    ))

            elif child.type == "import_from_statement":
                # from foo import bar, baz
                source = ""
                names = []
                for sub in child.children:
                    if sub.type == "dotted_name":
                        if not source:
                            source = self._node_text(sub)
                        else:
                            names.append(self._node_text(sub))
                    elif sub.type == "relative_import":
                        source = self._node_text(sub)
                    elif sub.type == "import_prefix":
                        source = self._node_text(sub)
                    elif sub.type in ("identifier",):
                        names.append(self._node_text(sub))
                    elif sub.type == "aliased_import":
                        name_node = self._find_child(sub, "identifier") or self._find_child(sub, "dotted_name")
                        if name_node:
                            names.append(self._node_text(name_node))

                if source:
                    is_local = source.startswith(".")
                    imports.append(Import(
                        source=source,
                        names=names if names else [source.rsplit(".", 1)[-1]],
                        is_local=is_local,
                        line=child.start_point[0] + 1,
                    ))

        return imports

    def _extract_js_ts_imports(self, root: Any) -> list[Import]:
        """Extract JS/TS import statements from AST."""
        imports: list[Import] = []

        for child in root.children:
            if child.type != "import_statement":
                continue

            source = ""
            names: list[str] = []

            for sub in child.children:
                if sub.type == "string":
                    source = self._node_text(sub).strip("'\"")
                elif sub.type == "import_clause":
                    names.extend(self._collect_import_names(sub))

            if source:
                is_local = source.startswith(".")
                imports.append(Import(
                    source=source,
                    names=names if names else ["*"],
                    is_local=is_local,
                    line=child.start_point[0] + 1,
                ))

        return imports

    def _collect_import_names(self, node: Any) -> list[str]:
        """Recursively collect imported names from an import clause."""
        names: list[str] = []
        for child in node.children:
            if child.type == "identifier":
                names.append(self._node_text(child))
            elif child.type == "named_imports":
                for sub in child.children:
                    if sub.type == "import_specifier":
                        name_node = self._find_child(sub, "identifier")
                        if name_node:
                            names.append(self._node_text(name_node))
            elif child.type == "namespace_import":
                name_node = self._find_child(child, "identifier")
                if name_node:
                    names.append(f"* as {self._node_text(name_node)}")
        return names

    # ── Export extraction ──

    def _extract_exports(
        self, content: str, language: str, units: list[StructuralUnit],
    ) -> list[Export]:
        """Extract exports using tree-sitter AST."""
        if language == "python":
            # Python doesn't have explicit exports; __all__ or top-level names
            return self._extract_python_exports(content, units)

        try:
            parser = self._get_ts_parser(language)
            if not parser:
                return self._extract_exports_regex(content, language)
        except Exception:
            return self._extract_exports_regex(content, language)

        tree = parser.parse(content.encode("utf-8"))
        return self._extract_js_ts_exports(tree.root_node)

    def _extract_python_exports(
        self, content: str, units: list[StructuralUnit],
    ) -> list[Export]:
        """Python 'exports' = top-level functions/classes (no parent)."""
        exports: list[Export] = []
        for unit in units:
            if unit.parent is None and unit.unit_type in ("function", "class"):
                exports.append(Export(
                    name=unit.name,
                    kind=unit.unit_type,
                    entity_ref=unit.name,
                    line=unit.start_line,
                ))
        return exports

    def _extract_js_ts_exports(self, root: Any) -> list[Export]:
        """Extract JS/TS export statements from AST."""
        exports: list[Export] = []

        for child in root.children:
            if child.type != "export_statement":
                continue

            line = child.start_point[0] + 1

            for sub in child.children:
                if sub.type == "function_declaration":
                    name = self._get_child_text(sub, "identifier") or ""
                    exports.append(Export(name=name, kind="function", entity_ref=name, line=line))
                elif sub.type == "class_declaration":
                    name = (
                        self._get_child_text(sub, "type_identifier")
                        or self._get_child_text(sub, "identifier")
                        or ""
                    )
                    exports.append(Export(name=name, kind="class", entity_ref=name, line=line))
                elif sub.type == "interface_declaration":
                    name = (
                        self._get_child_text(sub, "type_identifier")
                        or self._get_child_text(sub, "identifier")
                        or ""
                    )
                    exports.append(Export(name=name, kind="type", entity_ref=name, line=line))
                elif sub.type == "type_alias_declaration":
                    name = (
                        self._get_child_text(sub, "type_identifier")
                        or self._get_child_text(sub, "identifier")
                        or ""
                    )
                    exports.append(Export(name=name, kind="type", entity_ref=name, line=line))
                elif sub.type == "lexical_declaration":
                    for decl in sub.children:
                        if decl.type == "variable_declarator":
                            name = self._get_child_text(decl, "identifier") or ""
                            # Check if it's an arrow function
                            has_arrow = any(c.type == "arrow_function" for c in decl.children)
                            kind = "function" if has_arrow else "variable"
                            exports.append(Export(name=name, kind=kind, entity_ref=name, line=line))
                elif sub.type == "export_clause":
                    for spec in sub.children:
                        if spec.type == "export_specifier":
                            name = self._get_child_text(spec, "identifier") or ""
                            exports.append(Export(name=name, kind="variable", entity_ref=name, line=line))
                elif sub.type == "identifier":
                    # export default SomeIdentifier
                    text = self._node_text(sub)
                    exports.append(Export(name=text, kind="default", entity_ref=text, line=line))

            # Check for "export default"
            child_text = self._node_text(child)
            if "export default" in child_text and not exports:
                exports.append(Export(name="default", kind="default", entity_ref="default", line=line))

        return exports

    # ── Regex fallback for imports ──

    def _extract_imports_regex(self, content: str, language: str) -> list[Import]:
        """Regex-based import extraction when tree-sitter is unavailable."""
        if language == "python":
            return self._regex_python_imports(content)
        return self._regex_js_ts_imports(content)

    def _regex_python_imports(self, content: str) -> list[Import]:
        lines = content.split("\n")
        imports: list[Import] = []

        import_re = re.compile(r"^(?:from\s+([\w.]+)\s+)?import\s+(.+)")

        for i, line in enumerate(lines):
            m = import_re.match(line.strip())
            if not m:
                continue
            module = m.group(1) or ""
            names_str = m.group(2).strip().rstrip("\\")
            names = [n.strip().split(" as ")[0].strip() for n in names_str.split(",")]
            source = module if module else names[0]
            imports.append(Import(
                source=source,
                names=[n for n in names if n and n != "("],
                is_local=source.startswith("."),
                line=i + 1,
            ))

        return imports

    def _regex_js_ts_imports(self, content: str) -> list[Import]:
        lines = content.split("\n")
        imports: list[Import] = []

        import_re = re.compile(
            r"""import\s+(?:\{([^}]+)\}|(\w+)|\*\s+as\s+(\w+))\s+from\s+['"]([^'"]+)['"]"""
        )

        for i, line in enumerate(lines):
            m = import_re.search(line)
            if not m:
                continue
            named = m.group(1)
            default = m.group(2)
            namespace = m.group(3)
            source = m.group(4)

            names = []
            if named:
                names = [n.strip().split(" as ")[0].strip() for n in named.split(",")]
            elif default:
                names = [default]
            elif namespace:
                names = [f"* as {namespace}"]

            imports.append(Import(
                source=source,
                names=names,
                is_local=source.startswith("."),
                line=i + 1,
            ))

        return imports

    # ── Regex fallback for exports ──

    def _extract_exports_regex(self, content: str, language: str) -> list[Export]:
        """Regex-based export extraction."""
        if language == "python":
            return []  # Python exports are derived from units

        lines = content.split("\n")
        exports: list[Export] = []

        patterns = [
            (re.compile(r"^export\s+(?:async\s+)?function\s+(\w+)"), "function"),
            (re.compile(r"^export\s+class\s+(\w+)"), "class"),
            (re.compile(r"^export\s+interface\s+(\w+)"), "type"),
            (re.compile(r"^export\s+type\s+(\w+)"), "type"),
            (re.compile(r"^export\s+const\s+(\w+)"), "variable"),
            (re.compile(r"^export\s+default\s+(\w+)"), "default"),
        ]

        for i, line in enumerate(lines):
            stripped = line.strip()
            for pattern, kind in patterns:
                m = pattern.match(stripped)
                if m:
                    name = m.group(1)
                    exports.append(Export(name=name, kind=kind, entity_ref=name, line=i + 1))
                    break

        return exports

    # ── tree-sitter parsing ──

    def _get_ts_parser(self, language: str) -> Any:
        """Get or create a tree-sitter parser for the given language."""
        if language in self._ts_parsers:
            return self._ts_parsers[language]

        import tree_sitter as ts

        parser = ts.Parser()

        if language == "python":
            import tree_sitter_python
            lang = ts.Language(tree_sitter_python.language())
        elif language == "typescript":
            import tree_sitter_typescript
            lang = ts.Language(tree_sitter_typescript.language_typescript())
        elif language == "javascript":
            import tree_sitter_javascript
            lang = ts.Language(tree_sitter_javascript.language())
        elif language == "html":
            try:
                import tree_sitter_html
                lang = ts.Language(tree_sitter_html.language())
            except ImportError:
                return None
        elif language == "css":
            try:
                import tree_sitter_css
                lang = ts.Language(tree_sitter_css.language())
            except ImportError:
                return None
        else:
            return None

        parser.language = lang
        self._ts_parsers[language] = parser
        return parser

    def _parse_with_tree_sitter(self, content: str, language: str) -> list[StructuralUnit]:
        """Parse code using tree-sitter for accurate structural analysis."""
        try:
            parser = self._get_ts_parser(language)
            if not parser:
                return self._parse_with_regex(content, language)
        except Exception as e:
            log.warning("tree-sitter init failed for %s, falling back to regex: %s", language, e)
            return self._parse_with_regex(content, language)

        tree = parser.parse(content.encode("utf-8"))
        lines = content.split("\n")
        units: list[StructuralUnit] = []

        if language == "python":
            self._walk_python(tree.root_node, lines, units, parent=None)
        else:
            self._walk_js_ts(tree.root_node, lines, units, parent=None)

        return units

    def _walk_python(
        self,
        node: Any,
        lines: list[str],
        units: list[StructuralUnit],
        parent: str | None,
    ) -> None:
        """Walk Python AST for function_definition and class_definition."""
        for child in node.children:
            if child.type == "decorated_definition":
                # The actual def/class is inside the decorated_definition
                for sub in child.children:
                    if sub.type in ("function_definition", "class_definition"):
                        self._process_python_node(sub, lines, units, parent, decorated_start=child.start_point[0])
                        break
            elif child.type == "function_definition":
                self._process_python_node(child, lines, units, parent)
            elif child.type == "class_definition":
                self._process_python_node(child, lines, units, parent)

    def _process_python_node(
        self,
        node: Any,
        lines: list[str],
        units: list[StructuralUnit],
        parent: str | None,
        decorated_start: int | None = None,
    ) -> None:
        """Process a single Python function or class node."""
        name = self._get_child_text(node, "identifier")
        if not name:
            return

        start = (decorated_start if decorated_start is not None else node.start_point[0]) + 1
        end = node.end_point[0] + 1

        if node.type == "class_definition":
            sig = lines[node.start_point[0]].strip() if node.start_point[0] < len(lines) else ""
            units.append(StructuralUnit(
                name=name, unit_type="class",
                start_line=start, end_line=end,
                parent=parent, signature=sig[:120],
            ))
            # Recurse into class body for methods
            body = self._find_child(node, "block")
            if body:
                self._walk_python(body, lines, units, parent=name)
        else:
            unit_type = "method" if parent else "function"
            sig = lines[node.start_point[0]].strip() if node.start_point[0] < len(lines) else ""
            units.append(StructuralUnit(
                name=name, unit_type=unit_type,
                start_line=start, end_line=end,
                parent=parent, signature=sig[:120],
            ))

    def _walk_js_ts(
        self,
        node: Any,
        lines: list[str],
        units: list[StructuralUnit],
        parent: str | None,
    ) -> None:
        """Walk JS/TS AST for declarations."""
        for child in node.children:
            if child.type == "export_statement":
                # Unwrap export: the declaration is inside
                for sub in child.children:
                    self._process_js_ts_node(sub, lines, units, parent, export_start=child.start_point[0])
                continue

            self._process_js_ts_node(child, lines, units, parent)

    def _process_js_ts_node(
        self,
        node: Any,
        lines: list[str],
        units: list[StructuralUnit],
        parent: str | None,
        export_start: int | None = None,
    ) -> None:
        """Process a single JS/TS node."""
        start_row = export_start if export_start is not None else node.start_point[0]

        if node.type == "function_declaration":
            name = self._get_child_text(node, "identifier")
            if name:
                units.append(StructuralUnit(
                    name=name, unit_type="function",
                    start_line=start_row + 1, end_line=node.end_point[0] + 1,
                    parent=parent,
                    signature=lines[node.start_point[0]].strip()[:120],
                ))

        elif node.type == "class_declaration":
            name = self._get_child_text(node, "type_identifier") or self._get_child_text(node, "identifier")
            if name:
                units.append(StructuralUnit(
                    name=name, unit_type="class",
                    start_line=start_row + 1, end_line=node.end_point[0] + 1,
                    parent=parent,
                    signature=lines[node.start_point[0]].strip()[:120],
                ))
                # Recurse into class body for methods
                body = self._find_child(node, "class_body")
                if body:
                    self._walk_js_ts_methods(body, lines, units, parent=name)

        elif node.type == "interface_declaration":
            name = self._get_child_text(node, "type_identifier") or self._get_child_text(node, "identifier")
            if name:
                units.append(StructuralUnit(
                    name=name, unit_type="interface",
                    start_line=start_row + 1, end_line=node.end_point[0] + 1,
                    parent=parent,
                    signature=lines[node.start_point[0]].strip()[:120],
                ))

        elif node.type == "type_alias_declaration":
            name = self._get_child_text(node, "type_identifier") or self._get_child_text(node, "identifier")
            if name:
                units.append(StructuralUnit(
                    name=name, unit_type="type_alias",
                    start_line=start_row + 1, end_line=node.end_point[0] + 1,
                    parent=parent,
                    signature=lines[node.start_point[0]].strip()[:120],
                ))

        elif node.type == "lexical_declaration":
            # Exported const arrow functions: export const foo = (...) => { ... }
            self._extract_arrow_functions(node, lines, units, parent, start_row)

    def _extract_arrow_functions(
        self,
        node: Any,
        lines: list[str],
        units: list[StructuralUnit],
        parent: str | None,
        start_row: int,
    ) -> None:
        """Extract arrow function declarations from const/let declarations."""
        for decl in node.children:
            if decl.type != "variable_declarator":
                continue
            name = self._get_child_text(decl, "identifier")
            if not name:
                continue
            # Check if the value is an arrow function
            for child in decl.children:
                if child.type == "arrow_function":
                    units.append(StructuralUnit(
                        name=name, unit_type="function",
                        start_line=start_row + 1, end_line=node.end_point[0] + 1,
                        parent=parent,
                        signature=lines[node.start_point[0]].strip()[:120],
                    ))
                    break

    def _walk_js_ts_methods(
        self,
        body_node: Any,
        lines: list[str],
        units: list[StructuralUnit],
        parent: str,
    ) -> None:
        """Walk class body for method definitions."""
        for child in body_node.children:
            if child.type == "method_definition":
                name = self._get_child_text(child, "property_identifier")
                if name:
                    units.append(StructuralUnit(
                        name=name, unit_type="method",
                        start_line=child.start_point[0] + 1,
                        end_line=child.end_point[0] + 1,
                        parent=parent,
                        signature=lines[child.start_point[0]].strip()[:120],
                    ))

    @staticmethod
    def _node_text(node: Any) -> str:
        """Get the text content of a node."""
        text = node.text
        return text.decode("utf-8") if isinstance(text, bytes) else text

    @staticmethod
    def _get_child_text(node: Any, child_type: str) -> str | None:
        """Get the text of the first child of a given type."""
        for child in node.children:
            if child.type == child_type:
                return child.text.decode("utf-8") if isinstance(child.text, bytes) else child.text
        return None

    @staticmethod
    def _find_child(node: Any, child_type: str) -> Any | None:
        """Find the first child node of a given type."""
        for child in node.children:
            if child.type == child_type:
                return child
        return None

    # ── Regex fallback ──

    def _parse_with_regex(self, content: str, language: str) -> list[StructuralUnit]:
        """Best-effort parsing using regex when tree-sitter is unavailable."""
        if language == "python":
            return self._regex_python(content)
        return self._regex_js_ts(content)

    def _regex_python(self, content: str) -> list[StructuralUnit]:
        """Regex-based Python parsing with indentation tracking."""
        lines = content.split("\n")
        units: list[StructuralUnit] = []
        # Stack of (indent_level, name, type, start_line)
        open_defs: list[tuple[int, str, str, int]] = []

        class_re = re.compile(r"^(\s*)class\s+(\w+)")
        func_re = re.compile(r"^(\s*)(async\s+)?def\s+(\w+)")
        decorator_re = re.compile(r"^(\s*)@")

        decorator_start: int | None = None

        for i, line in enumerate(lines):
            line_num = i + 1
            stripped = line.rstrip()
            if not stripped:
                continue

            # Track decorators
            if decorator_re.match(stripped):
                if decorator_start is None:
                    decorator_start = line_num
                continue

            indent = len(line) - len(line.lstrip())

            # Close any open defs at same or deeper indentation
            while open_defs and open_defs[-1][0] >= indent:
                prev_indent, prev_name, prev_type, prev_start = open_defs.pop()
                # Find the end: last non-empty line before this one
                end = i  # 0-indexed, so line i (1-indexed) minus 1
                while end > prev_start and not lines[end - 1].strip():
                    end -= 1
                parent = open_defs[-1][1] if open_defs else None
                units.append(StructuralUnit(
                    name=prev_name,
                    unit_type=prev_type,
                    start_line=prev_start,
                    end_line=end,
                    parent=parent,
                    signature=lines[prev_start - 1].strip()[:120],
                ))

            cm = class_re.match(line)
            if cm:
                start = decorator_start if decorator_start is not None else line_num
                open_defs.append((indent, cm.group(2), "class", start))
                decorator_start = None
                continue

            fm = func_re.match(line)
            if fm:
                name = fm.group(3)
                start = decorator_start if decorator_start is not None else line_num
                # If inside a class, it's a method
                unit_type = "method" if open_defs and open_defs[-1][2] == "class" else "function"
                open_defs.append((indent, name, unit_type, start))
                decorator_start = None
                continue

            decorator_start = None

        # Close remaining open defs
        total = len(lines)
        while open_defs:
            prev_indent, prev_name, prev_type, prev_start = open_defs.pop()
            end = total
            while end > prev_start and not lines[end - 1].strip():
                end -= 1
            parent = open_defs[-1][1] if open_defs else None
            units.append(StructuralUnit(
                name=prev_name,
                unit_type=prev_type,
                start_line=prev_start,
                end_line=end,
                parent=parent,
                signature=lines[prev_start - 1].strip()[:120],
            ))

        # Sort by start line
        units.sort(key=lambda u: u.start_line)
        return units

    def _regex_js_ts(self, content: str) -> list[StructuralUnit]:
        """Regex-based JS/TS parsing (top-level declarations only)."""
        lines = content.split("\n")
        units: list[StructuralUnit] = []

        patterns = [
            (re.compile(r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)"), "function"),
            (re.compile(r"^(?:export\s+)?class\s+(\w+)"), "class"),
            (re.compile(r"^(?:export\s+)?interface\s+(\w+)"), "interface"),
            (re.compile(r"^(?:export\s+)?type\s+(\w+)"), "type_alias"),
            (re.compile(r"^(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?\("), "function"),
        ]

        brace_depth = 0
        current: tuple[str, str, int] | None = None  # (name, type, start_line)

        for i, line in enumerate(lines):
            line_num = i + 1
            stripped = line.strip()

            if current is None:
                for pattern, unit_type in patterns:
                    m = pattern.match(stripped)
                    if m:
                        current = (m.group(1), unit_type, line_num)
                        brace_depth = 0
                        break

            if current:
                brace_depth += stripped.count("{") - stripped.count("}")
                if brace_depth <= 0 and i > current[2] - 1:
                    units.append(StructuralUnit(
                        name=current[0],
                        unit_type=current[1],
                        start_line=current[2],
                        end_line=line_num,
                        parent=None,
                        signature=lines[current[2] - 1].strip()[:120],
                    ))
                    current = None
                    brace_depth = 0

        # Close any unclosed unit
        if current:
            units.append(StructuralUnit(
                name=current[0],
                unit_type=current[1],
                start_line=current[2],
                end_line=len(lines),
                parent=None,
                signature=lines[current[2] - 1].strip()[:120],
            ))

        return units
