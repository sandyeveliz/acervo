# Test Fixtures

Source projects used by Acervo's integration tests.
Each directory represents a different domain for testing the
index -> curate -> synthesize -> conversation pipeline.

## Projects

### P1 — TODO App (code)
A small TypeScript/React TODO application with Express backend.
Tests: code parsing, symbol extraction, import resolution, tech stack detection.

### P2 — Literature (prose)
"The Adventures of Sherlock Holmes" by Arthur Conan Doyle (1892, public domain).
Downloaded from Project Gutenberg. Single epub file with 12 stories.
Tests: character extraction, location detection, cross-story queries,
narrative structure, epub parsing.

### P3 — Project Management Docs (markdown)
Project management documents: roadmaps, sprint plans, issue trackers.
Tests: section extraction, entity connections between docs, deadline detection.

## Usage

These directories contain only source files — NO `.acervo/` generated data.
The integration tests index these projects into temporary directories
or reference existing indexed state from external paths.

To run the pipeline diagnostic tests:
```bash
pytest tests/integration/test_pipeline_validation.py -v
```

Note: Tests will skip if the project hasn't been indexed yet (no `.acervo/` dir).
To index, run `acervo init . && acervo index .` inside each fixture directory.
