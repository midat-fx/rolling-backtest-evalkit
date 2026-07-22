.PHONY: install lint typecheck test check data run gate all clean

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests

typecheck:
	mypy

test:
	pytest

check: lint typecheck test

data:
	evalkit data -o data/demand.csv

run:
	evalkit run --config configs/synthetic.yaml --data data/demand.csv -o out

# Dogfood: the backtest on the committed data must match the committed baseline.
gate: run
	evalkit gate out/metrics.csv --baseline reports/baseline_metrics.csv

all: check gate

clean:
	rm -rf out .coverage .mypy_cache .ruff_cache .pytest_cache
