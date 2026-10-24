"""A small leakage-detection framework.

Five independent detectors catch the common ways a forecasting feature, model
or split peeks at information it should not have:

1. **future_dependence** — recomputing a feature on data truncated at a split
   point must reproduce the pre-split values. A feature that uses future rows
   (a forward shift, a global normalization) changes when the future is removed.
2. **target_correlation** — a feature whose absolute correlation with the target
   is implausibly high is likely a leaked copy of it.
3. **split_contamination** — the same key appearing in both train and test.
4. **model_future_access** — a black-box check on a whole model: overwrite the
   test window ``y[origin:]`` with different values and forecast again. A model
   that only reads the past returns the identical forecast; any change means it
   read the future (a centred window, a global statistic, a leaked target).
5. **temporal_order** — every training index of a split must precede every
   test index. Shuffled K-fold over time points fails this.

The backtest runs detector 1 on its lag features and detector 4 on every model
it evaluates; both feed ``leakage_ok`` in the manifest, which the gate checks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]
FeatureFn = Callable[[Array], Array]
Model = Callable[[Array, Array, int, int], Array]


@dataclass
class LeakageFinding:
    detector: str
    leaked: bool
    detail: str


def recompute_stable(feature_fn: FeatureFn, y: Array, split: int, atol: float = 1e-9) -> bool:
    """True if features on y[:split] reproduce full-data features on [0, split).

    Warm-up positions that are NaN in the full computation are ignored; a
    position that is real in full but NaN or different in the truncated
    computation signals future dependence.
    """
    full = np.asarray(feature_fn(np.asarray(y, dtype=np.float64)), dtype=np.float64)
    trunc = np.asarray(feature_fn(np.asarray(y[:split], dtype=np.float64)), dtype=np.float64)
    a = full[:split]
    b = trunc[:split]
    consider = ~np.isnan(a)
    ok = (~np.isnan(b)) & np.isclose(a, b, atol=atol, equal_nan=False)
    return bool(np.all(~consider | ok))


def pearson(a: Array, b: Array) -> float:
    x = np.asarray(a, dtype=np.float64)
    z = np.asarray(b, dtype=np.float64)
    mask = ~(np.isnan(x) | np.isnan(z))
    x, z = x[mask] - np.mean(x[mask]), z[mask] - np.mean(z[mask])
    denom = float(np.sqrt(np.sum(x * x) * np.sum(z * z)))
    return float(np.sum(x * z) / denom) if denom > 0 else 0.0


def detect_future_dependence(feature_fn: FeatureFn, y: Array, split: int) -> LeakageFinding:
    stable = recompute_stable(feature_fn, y, split)
    detail = (
        "features reproduce under truncation"
        if stable
        else "features change when future is removed"
    )
    return LeakageFinding("future_dependence", not stable, detail)


def detect_target_leakage(feature: Array, target: Array, threshold: float = 0.98) -> LeakageFinding:
    corr = pearson(feature, target)
    return LeakageFinding("target_correlation", abs(corr) > threshold, f"|corr|={abs(corr):.3f}")


def detect_split_contamination(
    train_keys: Iterable[str], test_keys: Iterable[str]
) -> LeakageFinding:
    overlap = set(train_keys) & set(test_keys)
    return LeakageFinding(
        "split_contamination", len(overlap) > 0, f"{len(overlap)} overlapping key(s)"
    )


def audit_feature(
    feature_fn: FeatureFn, y: Array, split: int, target: Array | None = None
) -> list[LeakageFinding]:
    findings = [detect_future_dependence(feature_fn, y, split)]
    if target is not None:
        feats = np.asarray(feature_fn(np.asarray(y, dtype=np.float64)), dtype=np.float64)
        findings.append(detect_target_leakage(feats, target))
    return findings


def poison_future(y: Array, origin: int) -> Array:
    """Copy of y whose test window is replaced by different, plausible values."""
    out = np.array(y, dtype=np.float64)
    tail = out[origin:]
    out[origin:] = tail[::-1] * 1.7 + float(np.std(out[:origin])) + 3.0
    return out


def detect_model_future_access(
    model: Model, y: Array, promo: Array, origin: int, horizon: int
) -> LeakageFinding:
    clean = model(y, promo, origin, horizon)
    poisoned = model(poison_future(y, origin), promo, origin, horizon)
    changed = not np.array_equal(clean, poisoned)
    detail = (
        "forecast changes when y[origin:] is altered"
        if changed
        else "forecast independent of y[origin:]"
    )
    return LeakageFinding("model_future_access", changed, detail)


def detect_temporal_order(train_idx: Sequence[int], test_idx: Sequence[int]) -> LeakageFinding:
    late = sum(1 for t in train_idx if test_idx and t >= min(test_idx))
    return LeakageFinding(
        "temporal_order", late > 0, f"{late} training index(es) at or after the first test index"
    )
