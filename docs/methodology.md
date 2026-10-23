# Methodology

## Rolling-origin backtest

For a series of length `n`, an origin advances from `min_train` in steps of
`step`; at each origin the training data is `y[:origin]` and the test window is
`y[origin:origin+horizon]`. This mimics forecasting in production: you only ever
know the past. Folds accumulate more history as the origin advances (expanding
window).

Panels with unequal series lengths (tourism: 91–333 months) are aligned on each
series' **end**: origins are computed on the shortest length `n` and shifted by
`len(y) - n` for longer series. Fold k therefore tests the same position from
the end for every series, and longer series train on more history.

## Direct multi-step, leakage-safe features

The ridge model uses lagged features, and every lag is constrained to be at least
the forecast `horizon`. Predicting test week `origin+i` (for `0 <= i < horizon`)
with a lag `L >= horizon` reads `y[origin+i-L] <= y[origin-1]` — strictly in the
training region. So a direct multi-step forecast never needs a value it would not
have at the forecast origin, and no recursion or future value is required.

## Models

Registered by name (`models.REGISTRY`), selected per config with
`models: [...]`. The default is the original five, so older configs and
baselines are unchanged.

| name | what it does |
|---|---|
| `naive`, `seasonal_naive`, `drift`, `moving_average` | classical benchmarks |
| `seasonal_drift` | seasonal naive plus the mean year-over-year change |
| `ridge_lags` | numpy ridge on lags ≥ horizon (+ promo, intercept) |
| `ses`, `holt`, `holt_damped`, `holt_winters` | additive ETS; `holt_damped` and `holt_winters` damp the trend with phi = 0.9 |
| `theta` | Theta method in its SES-with-drift form (Hyndman & Billah 2003), with classical multiplicative deseasonalisation when the lag-m autocorrelation passes the 90% test (as in the M3 implementation) |

ETS smoothing parameters are chosen by grid search on in-sample one-step SSE
(alpha ∈ {0.05, …, 0.95}, beta and gamma ∈ {0.01, 0.05, 0.1, 0.2, 0.3}); the
whole grid is evaluated in one vectorised pass (see
[performance.md](performance.md)). Initial states come from the first season
and are not optimised. This is a deliberately small, transparent ETS: it is not
a replacement for `statsmodels` or `fable`, and its numbers will differ from
theirs.

## Metrics

All metrics are computed per series and then averaged over series for each
(fold, model). Scale-dependent metrics use an in-sample naive scale from the
training window:

- **WAPE** = sum|y - yhat| / sum|y|
- **sMAPE** = mean( 2|y - yhat| / (|y| + |yhat|) )
- **MASE** = mean|y - yhat| / mean|y_t - y_{t-m}|(train)
- **RMSSE** = sqrt( mean((y - yhat)^2) / mean((y_t - y_{t-m})^2)(train) )

with `m = scale_season` from the config (default 1, the original definition;
the monthly datasets use 12, the seasonal-naive scaling used by M4 and Monash).
- **bias** = sum(yhat - y) / sum|y|

## Significance

A lower mean is not a better model until the difference is larger than the
noise. Each run writes `stats.json` and a significance section in the report,
computed on one loss per series (its MASE averaged over folds); series are the
independent units, which also absorbs the serial correlation inside a series.

- **Bootstrap CI**: percentile bootstrap of the mean over series (2000
  resamples, seed 0, so the interval is reproducible).
- **Diebold-Mariano** for every model pair, with the Harvey-Leybourne-Newbold
  small-sample correction and a t(n-1) reference; p-values are Holm-adjusted
  over all pairs. `stats.diebold_mariano` also supports h-step loss series
  (autocovariances to lag h-1) for single-series use.
- **Friedman / Nemenyi**: average ranks across series, the Iman-Davenport
  Friedman p-value, and the Nemenyi critical difference (Demšar, 2006). Two
  models whose average ranks differ by less than the CD are not distinguishable
  by rank.

The t, F and studentised-range distributions are implemented in `stats.py`
(regularised incomplete beta, numerical quadrature) and tested against published
tables, so the core stays numpy + pandas.

## The gate

`evalkit gate` fails if any (fold, model) metric moves from the committed
baseline beyond `atol`/`rtol` (default 1e-4), if no trained model beats the
naive baseline on median WAPE, or if the run manifest (`--manifest`, or the
`manifest.json` next to the metrics) records leakage. The tolerance absorbs last-bit differences between
BLAS backends while still catching real regressions.
