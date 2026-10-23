# Leakage detection

Data leakage — letting a model see information it would not have at prediction
time — is the most common way a backtest lies. evalkit ships five detectors. On
every run it checks its own lag features (detector 1) and every configured model
(detector 4); either failing sets `leakage_ok: false` in `manifest.json`, and
`evalkit gate` fails on that even when the metrics match the baseline.

## 1. Future dependence (`recompute_stable`)

A correct feature computed on data truncated at a split point must reproduce the
values it had on the full data, for every position before the split. If removing
the future changes a past feature value, the feature depended on the future.

```python
from evalkit.leakage import recompute_stable
import numpy as np

def lag1(y):                 # safe: only looks back
    out = np.full_like(y, np.nan, float); out[1:] = y[:-1]; return out

def global_z(y):             # LEAK: normalizes by the whole series
    return (y - y.mean()) / (y.std() + 1e-9)

recompute_stable(lag1, y, split=20)     # True  -> safe
recompute_stable(global_z, y, split=20) # False -> leaks
```

This catches global normalization, centered/forward rolling windows, and any
target-encoding computed over the full series.

## 2. Target correlation

A feature whose absolute Pearson correlation with the target exceeds a threshold
(default 0.98) is almost certainly a leaked copy of it.

## 3. Split contamination

The same key appearing in both train and test — the most basic leak — is caught
by an intersection check.

## 4. Model future access (`detect_model_future_access`)

A black-box check on a whole model rather than a feature: forecast at an origin,
then overwrite the test window `y[origin:]` with different values (reversed,
scaled, shifted) and forecast again. A model that only reads the past returns a
bit-identical forecast. Any change means the forecast depended on data after
the origin, whatever the mechanism. The run audit probes each model on three
series at the first, middle and last origin, because a model can hide a leak
behind a short-history fallback at early origins (the ridge falls back to naive
until it has `max(lags)` points).

## 5. Temporal order (`detect_temporal_order`)

Every training index must precede every test index. Shuffled K-fold over time
points fails this by construction.

## Demo: the leak improves the metric, the detector catches it

`evalkit/leaky.py` registers three realistic bugs as models:

- `leak_global_norm` — a scaler fitted on the whole series before the split.
- `leak_centered_window` — `rolling(window, center=True)` used as smoothed demand.
- `leak_target_feature` — a same-period aggregate that contains `y[t]`
  (simulated as `y[t]` plus noise at 10% of the series' scale).

`examples/leakage_demo.py` runs them next to honest models, scores the same
ridge on shuffled 5-fold vs rolling origins, and runs the gate. Output, pasted
verbatim from `python examples/leakage_demo.py --data data/hospital.csv --config
configs/hospital.yaml` (real Monash data, see [data.md](data.md)):

### 1. Rolling origin: honest vs leaky (data/hospital.csv, 767 series)

| model | median WAPE | median MASE | model_future_access |
|---|---|---|---|
| naive | 0.2188 | 1.0056 | clean |
| ridge_lags | 0.2080 | 0.8966 | clean |
| theta | 0.1755 | 0.7978 | clean |
| leak_global_norm | 0.2037 | 0.8690 | **LEAK** |
| leak_centered_window | 0.1110 | 0.5026 | **LEAK** |
| leak_target_feature | 0.0223 | 0.0933 | **LEAK** |

### 2. Same ridge, two evaluation schemes (mean per-series WAPE)

| scheme | WAPE | temporal_order |
|---|---|---|
| rolling origin | 0.2215 | clean |
| shuffled 5-fold | 0.1861 | **LEAK** (39 training index(es) at or after the first test index) |

### 3. The gate

leakage_ok=False leaky_models=['leak_centered_window', 'leak_global_norm', 'leak_target_feature']
FAIL: leakage: leakage_ok=false in manifest.json (leaky models: ['leak_centered_window', 'leak_global_norm', 'leak_target_feature'])

And on the bundled synthetic panel (`python examples/leakage_demo.py`):

### 1. Rolling origin: honest vs leaky (data/demand.csv, 40 series)

| model | median WAPE | median MASE | model_future_access |
|---|---|---|---|
| naive | 0.2444 | 1.1140 | clean |
| ridge_lags | 0.2303 | 1.0597 | clean |
| theta | 0.2181 | 0.9625 | clean |
| leak_global_norm | 0.2132 | 1.0006 | **LEAK** |
| leak_centered_window | 0.1418 | 0.6494 | **LEAK** |
| leak_target_feature | 0.0551 | 0.1686 | **LEAK** |

### 2. Same ridge, two evaluation schemes (mean per-series WAPE)

| scheme | WAPE | temporal_order |
|---|---|---|
| rolling origin | 0.2358 | clean |
| shuffled 5-fold | 0.2234 | **LEAK** (38 training index(es) at or after the first test index) |

### 3. The gate

leakage_ok=False leaky_models=['leak_centered_window', 'leak_global_norm', 'leak_target_feature']
FAIL: leakage: leakage_ok=false in manifest.json (leaky models: ['leak_centered_window', 'leak_global_norm', 'leak_target_feature'])

What to take from it:

- Every leak *improves* the headline metric, from slightly (global
  normalisation: WAPE 0.2080 → 0.2037 on hospital) to absurdly (target feature:
  0.2080 → 0.0223). On the synthetic panel the global-normalisation leak is small
  but enough to flip model selection: it edges out theta, the best honest model.
  Small leaks are the dangerous ones; nobody believes a WAPE of 0.02.
- Shuffled K-fold flatters the same ridge by 16% relative WAPE on hospital
  (0.2215 → 0.1861) without any change to the model.
- None of this reaches a green build: `model_future_access` flags all three
  models, the manifest records them, and the gate fails even when compared to
  its own metrics. CI asserts that failure (`make leakage-demo`, and the
  "Planted leaks must fail the gate" step).

## Tested against planted leaks

`tests/test_leakage.py` plants a global normalisation, a forward shift and an
identity-to-target feature at the feature level, the three `leak_*` models at
the model level, and a shuffled split, and asserts each is flagged. It also
asserts that every honest registered model (all eleven) passes the model audit,
so the detector has no false positives on the shipped models.
