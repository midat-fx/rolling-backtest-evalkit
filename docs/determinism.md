# Determinism & reproducibility

- **Seeded data** — the generator uses `numpy.random.default_rng(seed)`; the
  committed `data/demand.csv` is the fixed source of truth for the backtest.
- **Stable metrics output** — `metrics.csv` is sorted by (fold, model) and
  written with fixed `%.6f` formatting, so it is byte-identical run to run.
- **Provenance** — every run writes `manifest.json` with a `run_id` derived from
  the config and a checksum of the data, plus the numpy/pandas versions used.
- **Worker-count invariance** — `evalkit run -j N` forecasts series in N
  processes but reassembles them in series order, so `metrics.csv`,
  `report.md`, `report.html`, `stats.json` and `manifest.json` are
  byte-identical for any N (`tests/test_determinism.py`).
- **Seeded statistics** — the bootstrap uses a fixed seed; the planted
  `leak_target_feature` noise is seeded too.
- **run_id** — `sha256(config JSON with sorted keys + data checksum)[:12]`:
  the same config and data give the same id on any machine and any worker
  count; changing either changes it (tested).
- **Pinned external data** — public datasets are accepted only if their
  SHA-256 matches the value pinned in the code (see [data.md](data.md)).
- **Tolerance gate** — cross-machine floating-point and BLAS differences are
  absorbed by comparing metrics within `atol=rtol=1e-4`; anything larger is a
  real regression and fails the build.

The repo dogfoods all of this: `make gate` (and the CI `gate` job) run the
backtest on the committed data and compare against the committed baseline;
`make gate-real` and the pages workflow do the same for the public datasets.
