.PHONY: selftest demo test cov clean lint install help

PYTHON ?= python
UV ?= uv

# Target por defecto
help:
	@echo "Uso: make [target]"
	@echo ""
	@echo "Targets disponibles:"
	@echo "  install   - Instalar dependencias de desarrollo con uv"
	@echo "  selftest  - Ejecutar auto-tests del monolito"
	@echo "  demo      - Ejecutar demo end-to-end"
	@echo "  test      - Ejecutar tests rápidos (sin cobertura)"
	@echo "  cov       - Ejecutar tests con informe de cobertura"
	@echo "  lint      - Ejecutar ruff (lint + format check)"
	@echo "  clean     - Limpiar artefactos generados"

install:
	$(UV) pip install --system -e ".[dev]"

selftest:
	$(PYTHON) agent-eval.py --selftest

demo:
	$(PYTHON) agent-eval.py --demo

test:
	$(PYTHON) -m pytest -q --strict-markers

cov:
	$(PYTHON) -m pytest --cov=. --cov-report=term-missing --cov-report=html

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
