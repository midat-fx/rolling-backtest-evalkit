# Contributing

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
make install      # pip install -e ".[dev]"
make check        # ruff + mypy --strict + pytest
make all          # check + backtest/baseline gate
```

## Rules of the house

- **Reproducibility is the product.** Metrics must be stable run to run. If you
  change the model set, features, or data, regenerate the baseline (below) in the
  same PR and explain the movement.
- **No leakage.** Features must pass the `future_dependence` detector; add a test
  for any new feature. The backtest computes features inside each fold only.
- **Typed and linted.** `mypy --strict` (run on 3.12), `ruff`, `ruff format`.
- **Tested.** Coverage is gated at 90% in CI.

## Regenerating data and baseline

```bash
evalkit data -o data/demand.csv                                   # only if datagen changed
evalkit run --config configs/synthetic.yaml --data data/demand.csv -o reports
cp reports/metrics.csv reports/baseline_metrics.csv               # only on an intended change
```
