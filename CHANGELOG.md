# Changelog

Follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

Tracked as GitHub issues: real public datasets (Monash) behind a checksum, ETS
and Theta models, Diebold-Mariano significance tests, and an HTML dashboard.
