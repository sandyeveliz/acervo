"""Tests for acervo.structural_parser — code and markdown parsing."""

from pathlib import Path

from acervo.structural_parser import StructuralParser, FileStructure


class TestMarkdownParsing:
    def _parse_md(self, tmp_path: Path, content: str) -> FileStructure:
        f = tmp_path / "test.md"
        f.write_text(content, encoding="utf-8")
        return StructuralParser().parse(f, tmp_path)

    def test_heading_sections(self, tmp_path):
        """Parse markdown with h1/h2/h3 — correct sections, parent-child, line ranges."""
        content = """\
# Overview

Some intro text.

## Installation

Steps to install.

### Prerequisites

Need Python 3.11+.

## Usage

How to use it.
"""
        structure = self._parse_md(tmp_path, content)
        assert structure.language == "markdown"
        assert len(structure.units) == 4

        overview, install, prereqs, usage = structure.units

        assert overview.name == "Overview"
        assert overview.parent is None
        assert overview.start_line == 1

        assert install.name == "Installation"
        assert install.parent == "Overview"
        assert install.start_line == 5

        assert prereqs.name == "Prerequisites"
        assert prereqs.parent == "Installation"
        assert prereqs.start_line == 9

        assert usage.name == "Usage"
        assert usage.parent == "Overview"
        assert usage.start_line == 13

    def test_empty_markdown(self, tmp_path):
        """Empty markdown → no units."""
        structure = self._parse_md(tmp_path, "Just plain text, no headings.\n")
        assert structure.units == []

    def test_single_heading(self, tmp_path):
        """One h1 only → one section spanning to EOF."""
        content = "# Title\n\nSome content here.\nMore content.\n"
        structure = self._parse_md(tmp_path, content)
        assert len(structure.units) == 1
        assert structure.units[0].name == "Title"
        assert structure.units[0].start_line == 1
        assert structure.units[0].end_line == 4

    def test_section_signatures(self, tmp_path):
        """Signatures should include the heading markers."""
        content = "# Main\n\n## Sub\n\nText.\n"
        structure = self._parse_md(tmp_path, content)
        assert structure.units[0].signature == "# Main"
        assert structure.units[1].signature == "## Sub"

    def test_all_units_are_sections(self, tmp_path):
        """All markdown units should have unit_type='section'."""
        content = "# A\n## B\n### C\n"
        structure = self._parse_md(tmp_path, content)
        for unit in structure.units:
            assert unit.unit_type == "section"
            assert unit.language == "markdown"


class TestPythonParsing:
    def _parse_py(self, tmp_path: Path, content: str) -> FileStructure:
        f = tmp_path / "test.py"
        f.write_text(content, encoding="utf-8")
        return StructuralParser().parse(f, tmp_path)

    def test_functions(self, tmp_path):
        """Parse .py with def → correct function units."""
        content = """\
def hello():
    return "world"

def goodbye():
    return "bye"
"""
        structure = self._parse_py(tmp_path, content)
        assert structure.language == "python"
        names = [u.name for u in structure.units]
        assert "hello" in names
        assert "goodbye" in names
        for unit in structure.units:
            assert unit.unit_type == "function"
            assert unit.parent is None

    def test_classes_and_methods(self, tmp_path):
        """Parse .py with class + methods → class + nested methods with parent."""
        content = """\
class Dog:
    def __init__(self, name):
        self.name = name

    def bark(self):
        return "woof"

def standalone():
    pass
"""
        structure = self._parse_py(tmp_path, content)
        names = {u.name: u for u in structure.units}

        assert "Dog" in names
        assert names["Dog"].unit_type == "class"
        assert names["Dog"].parent is None

        assert "__init__" in names
        assert names["__init__"].unit_type == "method"
        assert names["__init__"].parent == "Dog"

        assert "bark" in names
        assert names["bark"].unit_type == "method"
        assert names["bark"].parent == "Dog"

        assert "standalone" in names
        assert names["standalone"].unit_type == "function"
        assert names["standalone"].parent is None

    def test_async_functions(self, tmp_path):
        """async def should be detected."""
        content = """\
async def fetch_data():
    return await get()
"""
        structure = self._parse_py(tmp_path, content)
        assert len(structure.units) >= 1
        assert structure.units[0].name == "fetch_data"

    def test_content_hash_deterministic(self, tmp_path):
        """Same content → same hash."""
        content = "def foo():\n    pass\n"
        s1 = self._parse_py(tmp_path, content)
        s2 = self._parse_py(tmp_path, content)
        assert s1.content_hash == s2.content_hash

    def test_content_hash_changes(self, tmp_path):
        """Different content → different hash."""
        f = tmp_path / "test.py"
        f.write_text("def foo():\n    pass\n", encoding="utf-8")
        s1 = StructuralParser().parse(f, tmp_path)

        f.write_text("def bar():\n    pass\n", encoding="utf-8")
        s2 = StructuralParser().parse(f, tmp_path)

        assert s1.content_hash != s2.content_hash


class TestJavaScriptParsing:
    def _parse_js(self, tmp_path: Path, content: str, ext: str = ".js") -> FileStructure:
        f = tmp_path / f"test{ext}"
        f.write_text(content, encoding="utf-8")
        return StructuralParser().parse(f, tmp_path)

    def test_function_declarations(self, tmp_path):
        """function foo() → detected."""
        content = """\
function hello() {
    return "world";
}

function goodbye() {
    return "bye";
}
"""
        structure = self._parse_js(tmp_path, content)
        names = [u.name for u in structure.units]
        assert "hello" in names
        assert "goodbye" in names

    def test_arrow_functions(self, tmp_path):
        """export const foo = () => → detected."""
        content = """\
export const fetchData = async (url) => {
    const res = await fetch(url);
    return res.json();
};
"""
        structure = self._parse_js(tmp_path, content)
        names = [u.name for u in structure.units]
        assert "fetchData" in names

    def test_classes_with_methods(self, tmp_path):
        """class Foo with methods → class + method units with parent."""
        content = """\
class UserService {
    constructor(db) {
        this.db = db;
    }

    async getUser(id) {
        return this.db.find(id);
    }
}
"""
        structure = self._parse_js(tmp_path, content)
        names = {u.name: u for u in structure.units}

        assert "UserService" in names
        assert names["UserService"].unit_type == "class"

        assert "constructor" in names
        assert names["constructor"].parent == "UserService"

        assert "getUser" in names
        assert names["getUser"].parent == "UserService"

    def test_typescript_interface(self, tmp_path):
        """TypeScript interface → detected."""
        content = """\
export interface User {
    id: number;
    name: string;
}
"""
        structure = self._parse_js(tmp_path, content, ext=".ts")
        assert len(structure.units) >= 1
        assert structure.units[0].name == "User"
        assert structure.units[0].unit_type == "interface"

    def test_typescript_type_alias(self, tmp_path):
        """TypeScript type alias → detected."""
        content = """\
export type UserID = string | number;
"""
        structure = self._parse_js(tmp_path, content, ext=".ts")
        assert len(structure.units) >= 1
        assert structure.units[0].name == "UserID"
        assert structure.units[0].unit_type == "type_alias"


class TestFileStructure:
    def test_relative_path_forward_slashes(self, tmp_path):
        """file_path uses forward slashes even on Windows."""
        sub = tmp_path / "src"
        sub.mkdir()
        f = sub / "app.py"
        f.write_text("x = 1\n", encoding="utf-8")
        structure = StructuralParser().parse(f, tmp_path)
        assert "\\" not in structure.file_path
        assert structure.file_path == "src/app.py"

    def test_total_lines(self, tmp_path):
        """total_lines matches actual line count."""
        content = "line1\nline2\nline3\n"
        f = tmp_path / "test.py"
        f.write_text(content, encoding="utf-8")
        structure = StructuralParser().parse(f, tmp_path)
        assert structure.total_lines == 4  # 3 lines + trailing newline counts as 4

    def test_unknown_extension(self, tmp_path):
        """Unknown file extension → no units, language='unknown'."""
        f = tmp_path / "data.csv"
        f.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
        structure = StructuralParser().parse(f, tmp_path)
        assert structure.language == "unknown"
        assert structure.units == []
