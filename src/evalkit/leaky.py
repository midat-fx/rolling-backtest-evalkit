"""Deliberately leaky models, registered for the leakage demo and its tests.

Each is a realistic bug, not a cartoon. They are registered as ``leak_*`` so a
config can put them next to honest models and watch two things happen: the
metric improves, and ``model_future_access`` flags the model, which makes
``leakage_ok`` false and fails ``evalkit gate``. Never use them for forecasting.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .models import Array, Model, make_ridge_lags, register


def make_global_norm(config: dict[str, Any]) -> Model:
    """Ridge on a series z-scored with the mean/std of the *whole* series.

    The bug: a scaler fitted once on all data, before the split.
    """
    horizon = int(config.get("horizon", 4))
    season = int(config.get("season", 52))
    ridge = make_ridge_lags(sorted({horizon, horizon + 1, season}), 1.0)

    def leak_global_norm(y: Array, promo: Array, origin: int, horizon: int) -> Array:
        mu, sd = float(np.mean(y)), float(np.std(y)) + 1e-9
        z = (y - mu) / sd
        pred = ridge(z - z.min(), promo, origin, horizon) + z.min()
        return np.asarray(np.clip(pred * sd + mu, 0.0, None), dtype=np.float64)

    return leak_global_norm


def make_centered_window(config: dict[str, Any]) -> Model:
    """Forecast = a centred moving average around each target period.

    The bug: ``rolling(window, center=True)`` used as a smoothed-demand feature.
    """
    half = max(1, int(config.get("ma_window", 4)) // 2)

    def leak_centered_window(y: Array, promo: Array, origin: int, horizon: int) -> Array:
        out = np.empty(horizon, dtype=np.float64)
        for i in range(horizon):
            t = origin + i
            out[i] = float(np.mean(y[max(0, t - half) : t + half + 1]))
        return out

    return leak_centered_window


def make_target_feature(config: dict[str, Any]) -> Model:
    """A ridge whose feature set includes a same-period aggregate containing y[t].

    The bug: a "share of category sales" style feature computed from the same
    week's sales. Simulated as y[t] plus 10% noise of its own scale.
    """
    horizon = int(config.get("horizon", 4))
    lag = max(horizon, 1)

    def leak_target_feature(y: Array, promo: Array, origin: int, horizon: int) -> Array:
        rng = np.random.default_rng(0)
        agg = y + rng.normal(0.0, 0.1 * (float(np.std(y[:origin])) + 1e-9), len(y))
        t_train = np.arange(lag, origin)
        x = np.column_stack([agg[t_train], y[t_train - lag], np.ones(len(t_train))])
        coef = np.linalg.solve(x.T @ x + np.eye(3), x.T @ y[t_train])
        t_test = np.arange(origin, origin + horizon)
        x_test = np.column_stack([agg[t_test], y[t_test - lag], np.ones(horizon)])
        return np.asarray(np.clip(x_test @ coef, 0.0, None), dtype=np.float64)

    return leak_target_feature


register("leak_global_norm", make_global_norm)
register("leak_centered_window", make_centered_window)
register("leak_target_feature", make_target_feature)


def shuffled_kfold_split(n: int, k: int, fold: int, seed: int = 0) -> tuple[list[int], list[int]]:
    """Random K-fold over time points: the textbook split that is wrong for time series."""
    perm = np.random.default_rng(seed).permutation(n)
    test = sorted(int(t) for t in perm[fold::k])
    held = set(test)
    return [t for t in range(n) if t not in held], test


def shuffled_kfold_wape(y: Array, lags: list[int], k: int = 5, seed: int = 0) -> float:
    """WAPE of a lag ridge evaluated on shuffled K-fold instead of rolling origins.

    Test points sit between training points, so the lags of a test point and
    its neighbours on both sides are in training: interpolation, not forecasting.
    """
    mlag = max(lags)
    t_all = np.arange(mlag, len(y))
    x = np.column_stack([y[t_all - lag] for lag in lags] + [np.ones(len(t_all))])
    err = tot = 0.0
    for fold in range(k):
        tr, te = shuffled_kfold_split(len(t_all), k, fold, seed)
        coef = np.linalg.solve(x[tr].T @ x[tr] + np.eye(x.shape[1]), x[tr].T @ y[t_all[tr]])
        pred = np.clip(x[te] @ coef, 0.0, None)
        err += float(np.sum(np.abs(y[t_all[te]] - pred)))
        tot += float(np.sum(np.abs(y[t_all[te]])))
    return err / max(tot, 1e-9)
