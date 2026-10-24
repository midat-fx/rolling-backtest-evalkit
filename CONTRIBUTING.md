# Contributing

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
make install      # pip install -e ".[dev]"
make check        # ruff + mypy --strict + pytest
make all          # check + coverage + gates + leakage demo + mutation check
```

## Rules of the house

- **Reproducibility is the product.** Metrics must be stable run to run. If you
  change the model set, features, or data, regenerate the baseline (below) in the
  same PR and explain the movement.
- **No leakage.** Features must pass the `future_dependence` detector and every
  model must pass `model_future_access` (the run audit and a property test check
  all registered models). The backtest computes features inside each fold only.
- **Typed and linted.** `mypy --strict` (run on 3.12), `ruff`, `ruff format`.
- **Tested.** Coverage of `src/` is gated at 95% in CI, and
  `scripts/mutation_check.py` must kill every mutant.

## Regenerating data and baseline

```bash
evalkit data -o data/demand.csv                                   # only if datagen changed
evalkit run --config configs/synthetic.yaml --data data/demand.csv -o reports
cp reports/metrics.csv reports/baseline_metrics.csv               # only on an intended change

# real data: same, per dataset
make fetch
evalkit run --config configs/hospital.yaml --data data/hospital.csv -o out/hospital
cp out/hospital/metrics.csv reports/baseline_hospital.csv
```
