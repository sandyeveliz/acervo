# Testing

Requiere: Ollama corriendo con el modelo extractor (`qwen2.5:3b` o `acervo-extractor-v3`).

## Limpiar datos previos

```bash
rm -rf tests/fixtures/p1-todo-app/.acervo/
rm -rf tests/fixtures/p2-literature/.acervo/
rm -rf tests/fixtures/p3-project-docs/.acervo/
```

## Setup (una vez por sesion)

```powershell
# PowerShell
$env:ACERVO_TEST_BACKEND="ladybug"
```

```bash
# Bash / Git Bash
export ACERVO_TEST_BACKEND=ladybug
```

## Proyectos de prueba (p1/p2/p3)

Indexa los 3 fixtures (p1-todo-app, p2-literature, p3-project-docs) y corre las 4 capas:
pipeline validation, graph quality, 5-category benchmark, conversation scenarios.

```bash
# Los 3 proyectos completos (default)
pytest tests/integration/ -v -s

# Solo un proyecto
pytest tests/integration/ -k "p1" -v -s
```

## Conversaciones simuladas (8 dominios JSONL)

Simula conversaciones reales (casa, finanzas, fitness, libro, proyecto, salud, trabajo, viajes)
y mide extraction accuracy por turno.

```bash
# Todos los escenarios (default)
pytest tests/integration/test_case_scenarios.py -k "test_all" -v -s

# Powershell
$env:ACERVO_TEST_BACKEND="ladybug"; pytest tests/integration/test_case_scenarios.py -k "test_all" -v -s

# Un escenario individual
pytest tests/integration/test_case_scenarios.py -k "casa" -v -s
```

## Unit tests (sin LLM)

```bash
pytest tests/ -m "not integration" -v
```

## Reportes

Los reportes se generan en `tests/integration/reports/v0.6.0-ladybug/`.

```bash
python tests/integration/generate_report.py v0.6.0-ladybug
```
