# Methodology

## Rolling-origin backtest

For a series of length `n`, an origin advances from `min_train` in steps of
`step`; at each origin the training data is `y[:origin]` and the test window is
`y[origin:origin+horizon]`. This mimics forecasting in production: you only ever
know the past. Folds accumulate more history as the origin advances (expanding
window).

## Direct multi-step, leakage-safe features

The ridge model uses lagged features, and every lag is constrained to be at least
the forecast `horizon`. Predicting test week `origin+i` (for `0 <= i < horizon`)
with a lag `L >= horizon` reads `y[origin+i-L] <= y[origin-1]` — strictly in the
training region. So a direct multi-step forecast never needs a value it would not
have at the forecast origin, and no recursion or future value is required.

## Metrics

All metrics are computed per series and then averaged over series for each
(fold, model). Scale-dependent metrics use an in-sample naive scale from the
training window:

- **WAPE** = sum|y - yhat| / sum|y|
- **sMAPE** = mean( 2|y - yhat| / (|y| + |yhat|) )
- **MASE** = mean|y - yhat| / mean|y_t - y_{t-1}|(train)
- **RMSSE** = sqrt( mean((y - yhat)^2) / mean((y_t - y_{t-1})^2)(train) )
- **bias** = sum(yhat - y) / sum|y|

## The gate

`evalkit gate` fails if any (fold, model) metric moves from the committed
baseline beyond `atol`/`rtol` (default 1e-4), or if no trained model beats the
naive baseline on median WAPE. The tolerance absorbs last-bit differences between
BLAS backends while still catching real regressions.
