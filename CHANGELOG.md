# Changelog

Follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-10-23

Real data, more models, significance tests, model-level leakage detection and
an HTML report. All existing commands and the synthetic baseline are unchanged.

### Added
- `evalkit fetch`: Monash `hospital` and `tourism_monthly` (CC BY 4.0) with a
  pinned SHA-256, a cache (`$EVALKIT_CACHE`) and an offline mode; configs and
  committed baselines for both; `make fetch`, `make gate-real`.
- Model registry (`models: [...]` in configs) and six models: `seasonal_drift`,
  `ses`, `holt`, `holt_damped`, `holt_winters` (vectorised grid-search ETS) and
  `theta`.
- Significance: Diebold-Mariano with the HLN correction and Holm adjustment,
  bootstrap CIs over series, Friedman/Nemenyi ranks; `stats.json` and a
  significance section in `report.md`.
- Leakage: `model_future_access` (black-box audit of every model in every run)
  and `temporal_order` detectors; three planted leaky models and a shuffled
  K-fold evaluation in `examples/leakage_demo.py`; `leaky_models` in the
  manifest; `evalkit gate --manifest` (and the sibling `manifest.json`) fails on
  leakage; a CI step asserts the leaky run fails.
- `report.html`: self-contained (inline CSS/SVG, no JS/CDN) report; GitHub
  Pages workflow.
- `evalkit run -j N`: parallel series with byte-identical outputs;
  `scripts/benchmark.py` and `docs/performance.md`.
- Property-based tests (hypothesis, dev extra), determinism tests,
  `scripts/mutation_check.py` (17/17 mutants killed) and a CI job for it.
- `scale_season` config key (MASE/RMSSE scaled by the seasonal-naive error).
- Docs: results, data and licences, performance, notes on known gaps.

### Changed
- The backtest is series-major and keeps raw forecasts; metrics are computed
  on stacked arrays (3.2x faster on the synthetic backtest). Panels with
  unequal series lengths are aligned on each series' end.
- Coverage measures `src/` only; CI floor raised from 90% to 95%.

### Fixed
- Theta's seasonality test produced a NaN autocorrelation on constant series
  (found by a property-based test).

## [1.0.0] - 2026-03-15

First stable release.

### Added
- Deterministic synthetic demand generator (seeded panel of SKU x store series).
- Data-quality checks: duplicate keys, negatives, calendar gaps (fatal vs warn).
- Rolling-origin backtest with in-fold, leakage-safe lag features.
- Models: naive, seasonal_naive, moving_average, drift, and a numpy ridge on
  lagged features (lags constrained to >= horizon for honest multi-step).
- Metrics: WAPE, sMAPE, MASE, RMSSE, bias, with explicit in-sample naive scales.
- A leakage-detection framework (future-dependence, target-correlation,
  split-contamination) with tests that catch three planted leaks.
- Provenance manifest (run_id, data checksum, versions, leakage_ok).
- A baseline gate: metrics must match the committed baseline within tolerance,
  and the best trained model must beat the naive baseline.
- mypy --strict, ~97% coverage, Python 3.11-3.13.

## Roadmap

Gradient boosting on lags as an optional extra; probabilistic forecasts and
interval scores (CRPS, pinball); M5 once it can be fetched reproducibly. See
docs/notes.md.
