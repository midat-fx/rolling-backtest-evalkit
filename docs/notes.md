# Notes: what was not done, and why

Kept honest on purpose; each item is a known gap, not a hidden one.

- **M5 is not included.** The M5 (Walmart) data needs a Kaggle account and
  acceptance of the competition rules before download, so it cannot be fetched
  by `evalkit fetch` with a pinned checksum, and its licence terms are the
  competition's, not an open licence. The two Monash datasets (CC BY 4.0) cover
  the same ground: real demand-type series at scale.
- **No gradient boosting.** Optional by the plan; skipped to keep the core at
  numpy + pandas with no heavy extra. The model registry takes any
  `(y, promo, origin, horizon) -> yhat` callable, so a LightGBM model on lags is
  a factory plus one `register(...)` call.
- **ETS is minimal.** Additive error and season only, grid-searched smoothing
  parameters, fixed initial states, fixed damping (phi = 0.9). Good enough to be
  a strong, transparent baseline; not a substitute for `statsmodels`/`fable`.
- **The DM test runs on per-series losses.** Series are treated as independent
  and each contributes its fold-averaged MASE. Series in a real panel can be
  correlated (common shocks), which this does not model; the bootstrap CIs and
  Nemenyi ranks carry the same assumption.
- **The model audit is a black-box probe**, run on three series at three
  origins per model. A model that only leaks on rare inputs could pass it; the
  property test in `tests/test_properties.py` widens this for the shipped models
  to arbitrary generated series.
- **GitHub Pages is configured, not deployed.** `.github/workflows/pages.yml`
  publishes on push to `main`; nothing was pushed from this branch. Pages also
  has to be enabled once in the repository settings (Source: GitHub Actions).
- **Timings are from a busy laptop** (load average ≈ 5 during the benchmark);
  ratios are stable, absolute times are not.
