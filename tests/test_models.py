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


def test_vectorised_ets_grid_matches_scalar_recursion() -> None:
    from evalkit.models import ALPHAS, BETAS, GAMMAS, ets_fit_forecast, ets_sse_reference

    rng = np.random.default_rng(1)
    t = np.arange(60, dtype=float)
    y = 50 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 2, 60)
    _, sse = ets_fit_forecast(y, 6, trend=True, season=12, phi=0.9)
    grid = [(a, b, g) for a in ALPHAS for b in BETAS for g in GAMMAS]
    assert len(grid) == sse.size
    for i in (0, 7, 123, len(grid) - 1):
        a, b, g = grid[i]
        assert np.isclose(sse[i], ets_sse_reference(y, a, b, g, 12, 0.9))
    _, sse_holt = ets_fit_forecast(y, 6, trend=True)
    assert np.isclose(sse_holt[3], ets_sse_reference(y, ALPHAS[0], BETAS[3], 0.0))


def test_ets_variants_on_clean_signals() -> None:
    from evalkit.models import make_ets

    flat = np.full(40, 7.0)
    assert np.allclose(make_ets(trend=False)(flat, flat, 40, 3), 7.0)
    line = 10.0 + 2.0 * np.arange(40, dtype=float)
    assert np.allclose(make_ets(trend=True)(line, line, 30, 3), line[30:33])
    t = np.arange(72, dtype=float)
    seasonal = 100 + t + 20 * np.sin(2 * np.pi * t / 12)
    hw = make_ets(trend=True, season=12)(seasonal, seasonal, 60, 12)
    assert np.max(np.abs(hw - seasonal[60:72])) < 2.0
    assert len(make_ets(trend=True)(line, line, 2, 4)) == 4  # too short -> naive


def test_theta_and_seasonal_indices() -> None:
    from evalkit.models import make_theta, seasonal_indices

    t = np.arange(96, dtype=float)
    y = (50 + 0.5 * t) * (1 + 0.3 * np.sin(2 * np.pi * t / 12))
    idx = seasonal_indices(y, 12)
    assert idx is not None and np.isclose(idx.mean(), 1.0)
    assert seasonal_indices(np.arange(1, 97, dtype=float), 12) is None  # trend only
    assert seasonal_indices(y - 1000, 12) is None  # non-positive: no ratio split
    fc = make_theta(12)(y, y, 84, 12)
    assert np.mean(np.abs(fc - y[84:]) / y[84:]) < 0.1
    assert len(make_theta(12)(y, y, 2, 3)) == 3


def test_seasonal_drift_adds_yearly_change() -> None:
    from evalkit.models import make_seasonal_drift

    y = np.tile([1.0, 2.0, 3.0, 4.0], 4) + np.repeat([0.0, 10.0, 20.0, 30.0], 4)
    out = make_seasonal_drift(4)(y, y, 12, 6)
    assert np.allclose(out, [31, 32, 33, 34, 41, 42])
    assert list(make_seasonal_drift(4)(y, y, 3, 2)) == [3.0, 3.0]  # < 1 season -> naive


def test_registry_selection_and_errors() -> None:
    import pytest

    from evalkit.models import REGISTRY, register

    models = build_models({"horizon": 2, "season": 4, "models": ["theta", "naive"]})
    assert list(models) == ["theta", "naive"]
    with pytest.raises(KeyError):
        build_models({"models": ["nope"]})
    register("_test_const", lambda c: lambda y, p, o, h: np.zeros(h))
    try:
        assert list(build_models({"models": ["_test_const"]})["_test_const"](Y100, Y100, 5, 2)) == [
            0.0,
            0.0,
        ]
    finally:
        del REGISTRY["_test_const"]


Y100 = np.arange(100, dtype=float)
