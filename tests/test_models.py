from __future__ import annotations

import numpy as np

from evalkit.models import build_models, drift, make_moving_average, make_seasonal_naive, naive


def test_naive_is_constant_last_value() -> None:
    y = np.array([1, 2, 3, 9], dtype=float)
    out = naive(y, np.zeros(4), origin=4, horizon=3)
    assert list(out) == [9.0, 9.0, 9.0]


def test_seasonal_naive_uses_season_lag() -> None:
    y = np.arange(60, dtype=float)
    sn = make_seasonal_naive(season=52)
    out = sn(y, np.zeros(60), origin=52, horizon=2)
    assert list(out) == [0.0, 1.0]  # y[52-52], y[53-52]


def test_moving_average() -> None:
    y = np.array([2, 4, 6, 8], dtype=float)
    ma = make_moving_average(window=2)
    out = ma(y, np.zeros(4), origin=4, horizon=2)
    assert list(out) == [7.0, 7.0]  # mean(6, 8)


def test_drift_extrapolates_slope() -> None:
    y = np.array([0, 1, 2, 3], dtype=float)
    out = drift(y, np.zeros(4), origin=4, horizon=2)
    assert out[0] == 4.0 and out[1] == 5.0  # slope 1


def test_registry_has_all_models() -> None:
    models = build_models({"horizon": 4, "season": 52})
    assert set(models) == {"naive", "seasonal_naive", "moving_average", "drift", "ridge_lags"}
    for fn in models.values():
        out = fn(np.arange(100, dtype=float), np.zeros(100), origin=60, horizon=4)
        assert len(out) == 4 and np.all(np.isfinite(out))
