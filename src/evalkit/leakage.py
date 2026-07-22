"""A small leakage-detection framework.

Three independent detectors catch the common ways a forecasting feature or split
peeks at information it should not have:

1. **future_dependence** — recomputing a feature on data truncated at a split
   point must reproduce the pre-split values. A feature that uses future rows
   (a forward shift, a global normalization) changes when the future is removed.
2. **target_correlation** — a feature whose absolute correlation with the target
   is implausibly high is likely a leaked copy of it.
3. **split_contamination** — the same key appearing in both train and test.

The backtest runs detector 1 on its lag features by default.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]
FeatureFn = Callable[[Array], Array]


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
