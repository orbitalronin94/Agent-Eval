.PHONY: help selftest demo test cov clean

help:
	@echo "Targets disponibles:"
	@echo "  make selftest  - Self-tests embebidos del monolito"
	@echo "  make demo      - Ejecuta la demo end-to-end"
	@echo "  make test      - Tests con pytest"
	@echo "  make cov       - Tests con cobertura"
	@echo "  make clean     - Borra artefactos generados"

selftest:
	python agent-eval.py --selftest

demo:
	python agent-eval.py --demo

test:
	pytest

cov:
	pytest --cov=. --cov-report=term-missing

clean:
	rm -f eval-report.md eval-results.db coverage.xml .coverage
	rm -rf .pytest_cache htmlcov __pycache__ tests/__pycache__ .coverage.*
