# Determinism & reproducibility

- **Seeded data** — the generator uses `numpy.random.default_rng(seed)`; the
  committed `data/demand.csv` is the fixed source of truth for the backtest.
- **Stable metrics output** — `metrics.csv` is sorted by (fold, model) and
  written with fixed `%.6f` formatting, so it is byte-identical run to run.
- **Provenance** — every run writes `manifest.json` with a `run_id` derived from
  the config and a checksum of the data, plus the numpy/pandas versions used.
- **Tolerance gate** — cross-machine floating-point and BLAS differences are
  absorbed by comparing metrics within `atol=rtol=1e-4`; anything larger is a
  real regression and fails the build.

The repo dogfoods all of this: `make gate` (and the CI `gate` job) run the
backtest on the committed data and compare against the committed baseline.
