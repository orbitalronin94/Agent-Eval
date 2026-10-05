.PHONY: selftest demo test cov clean

PYTHON ?= python

selftest:
$(PYTHON) agent-eval.py --selftest

demo:
$(PYTHON) agent-eval.py --demo

test:
$(PYTHON) -m pytest -q

cov:
$(PYTHON) -m pytest --cov=. --cov-report=term-missing

clean:
rm -rf **pycache** .pytest_cache .coverage htmlcov
find . -type f \(-name "*.pyc" -o -name "*.pyo"\) -delete
