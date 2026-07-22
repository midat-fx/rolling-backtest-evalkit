"""Forecasting models used by the backtest.

Each model is a callable ``(y, promo, origin, horizon) -> yhat`` where ``y`` is
the full series, ``origin`` is the first test index (so training data is
``y[:origin]``), and the returned array has length ``horizon``. Models never read
``y[origin:]``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from .features import lag_features

Array = npt.NDArray[np.float64]
Model = Callable[[Array, Array, int, int], Array]


def naive(y: Array, promo: Array, origin: int, horizon: int) -> Array:
    return np.full(horizon, y[origin - 1], dtype=np.float64)


def make_seasonal_naive(season: int = 52) -> Model:
    def seasonal_naive(y: Array, promo: Array, origin: int, horizon: int) -> Array:
        out = np.empty(horizon, dtype=np.float64)
        for i in range(horizon):
            src = origin + i - season
            out[i] = y[src] if src >= 0 else y[origin - 1]
        return out

    return seasonal_naive


def make_moving_average(window: int = 4) -> Model:
    def moving_average(y: Array, promo: Array, origin: int, horizon: int) -> Array:
        start = max(0, origin - window)
        return np.full(horizon, float(np.mean(y[start:origin])), dtype=np.float64)

    return moving_average


def drift(y: Array, promo: Array, origin: int, horizon: int) -> Array:
    last = float(y[origin - 1])
    slope = (last - float(y[0])) / max(origin - 1, 1)
    return last + slope * np.arange(1, horizon + 1, dtype=np.float64)


def make_ridge_lags(lags: list[int], alpha: float = 1.0) -> Model:
    def ridge_lags(y: Array, promo: Array, origin: int, horizon: int) -> Array:
        mlag = max(lags)
        train_t = list(range(mlag, origin))
        if len(train_t) < len(lags) + 2:
            return np.full(horizon, y[origin - 1], dtype=np.float64)  # too little data
        x = lag_features(y, promo, lags, train_t)
        target = y[train_t]
        n_features = x.shape[1]
        gram = x.T @ x + alpha * np.eye(n_features)
        coef = np.linalg.solve(gram, x.T @ target)
        test_t = list(range(origin, origin + horizon))
        x_test = lag_features(y, promo, lags, test_t)
        pred = x_test @ coef
        return np.clip(pred, 0.0, None)

    return ridge_lags


def build_models(config: dict[str, Any]) -> dict[str, Model]:
    """Instantiate the model registry from a config dict."""
    season = int(config.get("season", 52))
    ma_window = int(config.get("ma_window", 4))
    horizon = int(config.get("horizon", 4))
    # Lags must be >= horizon so a direct multi-step forecast never needs an
    # unknown recent value; keep a seasonal lag too.
    lags = [lag for lag in config.get("ridge_lags", [horizon, horizon + 1, horizon + 3, season])]
    lags = sorted({max(lag, horizon) for lag in lags})
    alpha = float(config.get("ridge_alpha", 1.0))
    return {
        "naive": naive,
        "seasonal_naive": make_seasonal_naive(season),
        "moving_average": make_moving_average(ma_window),
        "drift": drift,
        "ridge_lags": make_ridge_lags(lags, alpha),
    }
