.PHONY: install lint typecheck test check data run gate fetch gate-real leakage-demo coverage mutation all clean

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests

typecheck:
	mypy

test:
	pytest

coverage:
	coverage run -m pytest
	coverage report

# Plants 17 small bugs in metrics/stats one at a time; every one must fail a test.
mutation:
	python scripts/mutation_check.py

check: lint typecheck test

data:
	evalkit data -o data/demand.csv

run:
	evalkit run --config configs/synthetic.yaml --data data/demand.csv -o out

# Dogfood: the backtest on the committed data must match the committed baseline.
gate: run
	evalkit gate out/metrics.csv --baseline reports/baseline_metrics.csv

REAL := hospital tourism_monthly

fetch:
	for d in $(REAL); do evalkit fetch $$d; done

# Real public data (needs `make fetch` once; archives are SHA-256 verified).
gate-real:
	for d in $(REAL); do \
	  evalkit run --config configs/$$d.yaml --data data/$$d.csv -o out/$$d && \
	  evalkit gate out/$$d/metrics.csv --baseline reports/baseline_$$d.csv || exit 1; \
	done

# Planted leaks must win on WAPE *and* fail the gate; the target inverts the exit code.
leakage-demo:
	python examples/leakage_demo.py
	evalkit run --config configs/leakage_demo.yaml --data data/demand.csv -o out/leak
	! evalkit gate out/leak/metrics.csv --baseline out/leak/metrics.csv

all: check coverage gate leakage-demo mutation

clean:
	rm -rf out .coverage .mypy_cache .ruff_cache .pytest_cache
