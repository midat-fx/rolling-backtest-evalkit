# Leakage detection

Data leakage — letting a model see information it would not have at prediction
time — is the most common way a backtest lies. evalkit ships three detectors and
runs the first one on its own features as a self-check (`leakage_ok` in the run
manifest).

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

## Tested against real leaks

`tests/test_leakage.py` plants three leaks (global normalization, a forward
shift, and an identity-to-target feature) and asserts each is flagged, while a
plain lag feature is not.
