# evalkit — reproducible forecasting evaluation with leakage detection

[![ci](https://github.com/midat-fx/rolling-backtest-evalkit/actions/workflows/ci.yml/badge.svg)](https://github.com/midat-fx/rolling-backtest-evalkit/actions/workflows/ci.yml)
[![python: 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![types: mypy strict](https://img.shields.io/badge/mypy-strict-blue.svg)](pyproject.toml)
[![coverage: 97%](https://img.shields.io/badge/coverage-97%25-brightgreen.svg)](#develop)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A small framework for evaluating demand-forecasting models **honestly** and
**reproducibly**: rolling-origin backtests, leakage-safe features, a
leakage-detection layer that is tested against real leaks, and a metrics gate
that fails CI when results drift from a committed baseline.

```console
$ evalkit run --config configs/synthetic.yaml --data data/demand.csv -o out
evalkit run: run_id=b2dde5f4a5a0 leakage_ok=True
$ evalkit gate out/metrics.csv --baseline reports/baseline_metrics.csv
OK: metrics match baseline within tolerance
```

## Why

Most backtests lie in one of two ways: a feature quietly peeks at the future, or
the numbers move run-to-run and nobody notices a regression. evalkit attacks
both. Features are computed **inside each fold** and checked by a
future-dependence detector; metrics are written deterministically and gated
against a baseline within a tight tolerance. If a change makes the models look
better because it leaked, or worse because it regressed, the build says so.

## Install

```bash
pip install -e ".[dev]"
```

## Quickstart

```bash
evalkit data    -o data/demand.csv                                   # seeded synthetic panel
evalkit quality data/demand.csv                                      # dup/negative/gap checks
evalkit run     --config configs/synthetic.yaml --data data/demand.csv -o reports
evalkit report  reports/metrics.csv                                  # median metrics by model
evalkit gate    reports/metrics.csv --baseline reports/baseline_metrics.csv
```

## What it does

- **Rolling-origin backtest** — expanding window, configurable `min_train`,
  `horizon`, `step`. One row per (fold, model).
- **Models** — `naive`, `seasonal_naive`, `moving_average`, `drift`, and a numpy
  ridge on lagged features (lags constrained to ≥ horizon for honest direct
  multi-step). No sklearn; the linear algebra is a few lines of numpy.
- **Metrics** — WAPE, sMAPE, MASE, RMSSE, bias, each with an explicit in-sample
  naive scale. See [docs/methodology.md](docs/methodology.md).
- **Leakage detection** — three detectors, run against three planted leaks in the
  test suite. See below.
- **Reproducibility gate** — `evalkit gate` fails on metric drift beyond
  tolerance or if no trained model beats the naive baseline.
- **Provenance** — every run writes `manifest.json` with a `run_id`, a data
  checksum, `leakage_ok`, and the numpy/pandas versions.

## Example report

`evalkit report reports/metrics.csv` on the bundled data:

| model | wape | smape | mase | rmsse | bias |
|---|---|---|---|---|---|
| moving_average | 0.2064 | 0.1984 | 0.9200 | 0.7150 | 0.0160 |
| ridge_lags | 0.2303 | 0.2145 | 1.0597 | 0.8188 | 0.0365 |
| naive | 0.2444 | 0.2338 | 1.1140 | 0.8044 | -0.0143 |
| drift | 0.2758 | 0.2657 | 1.1363 | 0.8443 | -0.0559 |
| seasonal_naive | 0.7787 | 0.5280 | 2.4936 | 1.8283 | 0.4236 |

Honest result: on this short, trending panel a simple moving average wins and
seasonal-naive is *worse* than naive — exactly the kind of thing a real
evaluation surfaces rather than hides.

## Leakage detection (the core idea)

A feature that peeks at the future changes when you remove the future. That is
directly testable:

```python
from evalkit.leakage import recompute_stable
import numpy as np

def lag1(y):                  # safe: only looks back
    out = np.full_like(y, np.nan, float); out[1:] = y[:-1]; return out

def global_z(y):              # LEAK: normalizes using the whole series
    return (y - y.mean()) / (y.std() + 1e-9)

recompute_stable(lag1, y, split=20)      # True  -> no leakage
recompute_stable(global_z, y, split=20)  # False -> leaks the future
```

The suite plants a global normalization, a forward shift, and an
identity-to-target feature and asserts each is caught, while a plain lag feature
passes. The backtest runs this check on its own features and records
`leakage_ok` in the manifest. Full write-up in [docs/leakage.md](docs/leakage.md).

## Use it as a CI gate

The repo dogfoods its own gate — see `.github/workflows/ci.yml`:

```yaml
- run: evalkit quality data/demand.csv
- run: |
    evalkit run --config configs/synthetic.yaml --data data/demand.csv -o out
    evalkit gate out/metrics.csv --baseline reports/baseline_metrics.csv
```

## Develop

```bash
make check     # ruff + mypy --strict + pytest
make all       # check + the backtest/baseline gate
```

Quality bar: `ruff` clean, `mypy --strict` clean, ~97% coverage, tests matrixed
on Python 3.11–3.13.

## Docs

- [docs/methodology.md](docs/methodology.md) — backtest, direct multi-step, metrics, the gate.
- [docs/leakage.md](docs/leakage.md) — the three detectors and the planted-leak tests.
- [docs/determinism.md](docs/determinism.md) — how reproducibility is enforced.

## Roadmap

GitHub issues: real public datasets (Monash) behind a checksum, ETS and Theta
models, Diebold-Mariano significance tests, and an HTML dashboard.

## License

MIT © Midat Faizov. A public, clean-room implementation demonstrating
leakage-aware, reproducible model evaluation.
