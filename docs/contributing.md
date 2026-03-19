# Contributing to Acervo

Thanks for your interest in contributing to Acervo! This document explains how to get started.

## Development setup

1. Clone the repo:

```bash
git clone https://github.com/sandyeveliz/acervo.git
cd acervo
```

2. Create a virtual environment and install dev dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

3. Run the tests:

```bash
pytest tests/
```

## How to contribute

### Reporting bugs

Open a [bug report](https://github.com/sandyeveliz/acervo/issues/new?template=bug_report.md) using the issue template. Include steps to reproduce, expected behavior, and your Python/Acervo versions.

### Proposing new entity types

Acervo's ontology is extensible. To propose a new built-in entity type:

1. Open an issue describing the entity type, its attributes, and which layer it belongs to by default.
2. If accepted, submit a PR that:
   - Adds the type definition to `acervo/ontology.py`
   - Includes tests covering creation and serialization
   - Updates the README's ontology section

### Creating community knowledge packs (Layer 1)

!!! info "Planned"
    Community knowledge packs are a planned feature. The format described below
    is provisional and may change. See [Roadmap](roadmap.md) for status.

Community knowledge packs will provide pre-built universal knowledge for Layer 1. The planned workflow:

1. Create a directory under `packs/` with a descriptive name (e.g., `packs/python/`).
2. Include a `manifest.json` with metadata: name, version, description, and author.
3. Include the knowledge data as JSON files following the node/edge schema.
4. Submit a PR with the pack and a brief description of what it covers.

### Submitting a pull request

1. Fork the repo and create a branch from `main`.
2. Make your changes.
3. Add or update tests as needed.
4. Update `CHANGELOG.md` under `[Unreleased]` with a summary of your changes.
5. Run `pytest tests/` and make sure all tests pass.
6. Open a PR using the pull request template.

### Code style

- Keep code simple and readable.
- Follow existing patterns in the codebase.
- Write docstrings for public functions and classes.

## Building the docs

```bash
pip install -e ".[docs]"
mkdocs serve  # Local preview at http://localhost:8000
mkdocs build  # Build static site to site/
```

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
