from __future__ import annotations

import numpy as np

from evalkit.leakage import (
    audit_feature,
    detect_future_dependence,
    detect_split_contamination,
    detect_target_leakage,
    recompute_stable,
)


def lag1(y: np.ndarray) -> np.ndarray:
    out = np.full_like(y, np.nan, dtype=float)
    out[1:] = y[:-1]
    return out


def global_zscore(y: np.ndarray) -> np.ndarray:
    return (y - np.mean(y)) / (np.std(y) + 1e-9)  # uses the WHOLE series -> leak


def forward_shift(y: np.ndarray) -> np.ndarray:
    out = np.full_like(y, np.nan, dtype=float)
    out[:-1] = y[1:]  # uses the future -> leak
    return out


Y = np.arange(1, 41, dtype=float) + np.sin(np.arange(40))


def test_safe_lag_feature_is_stable() -> None:
    assert recompute_stable(lag1, Y, split=20) is True
    assert detect_future_dependence(lag1, Y, 20).leaked is False


def test_global_normalization_is_flagged() -> None:
    assert recompute_stable(global_zscore, Y, split=20) is False
    assert detect_future_dependence(global_zscore, Y, 20).leaked is True


def test_forward_shift_is_flagged() -> None:
    assert recompute_stable(forward_shift, Y, split=20) is False
    assert detect_future_dependence(forward_shift, Y, 20).leaked is True


def test_target_leakage_identity() -> None:
    assert detect_target_leakage(Y, Y).leaked is True  # feature == target


def test_safe_feature_is_not_target_leakage() -> None:
    rng = np.random.default_rng(0)
    noise = rng.normal(size=200)
    feat = lag1(noise)
    mask = ~np.isnan(feat)
    assert detect_target_leakage(feat[mask], noise[mask]).leaked is False


def test_split_contamination() -> None:
    assert detect_split_contamination(["a", "b"], ["b", "c"]).leaked is True
    assert detect_split_contamination(["a", "b"], ["c", "d"]).leaked is False


def test_audit_bundles_findings() -> None:
    findings = audit_feature(global_zscore, Y, 20, target=Y)
    assert any(f.leaked for f in findings)
    assert {f.detector for f in findings} == {"future_dependence", "target_correlation"}
