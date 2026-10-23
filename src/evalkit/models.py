"""Forecasting models used by the backtest.

Each model is a callable ``(y, promo, origin, horizon) -> yhat`` where ``y`` is
the full series, ``origin`` is the first test index (so training data is
``y[:origin]``), and the returned array has length ``horizon``. Models never read
``y[origin:]``.

Models are registered by name in ``REGISTRY`` as factories ``config -> Model``;
a config selects them with ``models: [name, ...]`` (default: the five original
baselines, so existing configs and baselines are unchanged).
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


def make_seasonal_drift(season: int) -> Model:
    """Seasonal naive plus the mean year-over-year change per elapsed season."""

    def seasonal_drift(y: Array, promo: Array, origin: int, horizon: int) -> Array:
        if origin <= season:
            return naive(y, promo, origin, horizon)
        step = float(np.mean(y[season:origin] - y[: origin - season]))
        k = np.arange(horizon) // season + 1  # seasons between source and target
        src = origin + np.arange(horizon) - season * k
        return np.asarray(y[src] + k * step, dtype=np.float64)

    return seasonal_drift


# Smoothing-parameter grids. Fitting evaluates every combination at once: the
# recursion runs over time with the grid as a vector, so one pass over the
# series scores all (alpha, beta, gamma) and the argmin of in-sample SSE wins.
ALPHAS = np.round(np.arange(0.05, 1.0, 0.1), 2)
BETAS = np.array([0.01, 0.05, 0.1, 0.2, 0.3])
GAMMAS = np.array([0.01, 0.05, 0.1, 0.2, 0.3])


def ets_fit_forecast(
    y: Array, horizon: int, trend: bool, season: int = 0, phi: float = 1.0
) -> tuple[Array, Array]:
    """Additive ETS (SES / Holt / Holt-Winters) by grid search on in-sample SSE.

    Returns (forecast, sse_per_grid_point). ``season=0`` disables seasonality;
    ``phi < 1`` damps the trend. Initial states come from the first season (or
    the first two points), are not optimised, and the recursion starts after
    the points they were estimated from.
    """
    n = len(y)
    m = season if season and n >= 2 * season else 0
    grids = np.meshgrid(ALPHAS, BETAS if trend else [0.0], GAMMAS if m else [0.0], indexing="ij")
    a, b, g = (x.ravel() for x in grids)
    k = a.size
    if m:
        l0 = float(np.mean(y[:m]))
        b0 = (float(np.mean(y[m : 2 * m])) - l0) / m if trend else 0.0
        seas = np.tile(y[:m] - l0, (k, 1))
    else:
        l0 = float(y[0])
        b0 = float(y[1] - y[0]) if trend and n > 1 else 0.0
        seas = np.zeros((k, 1))
    level = np.full(k, l0)
    slope = np.full(k, b0)
    sse = np.zeros(k)
    width = seas.shape[1]
    for t in range(m or 1, n):
        s_t = seas[:, t % width]
        err = y[t] - (level + phi * slope + s_t)
        sse += err * err
        new_level = a * (y[t] - s_t) + (1.0 - a) * (level + phi * slope)
        slope = b * (new_level - level) + (1.0 - b) * phi * slope
        if m:
            seas[:, t % width] = g * (y[t] - new_level) + (1.0 - g) * s_t
        level = new_level
    best = int(np.argmin(sse))
    h = np.arange(1, horizon + 1)
    damp = np.cumsum(phi**h) if trend else np.zeros(horizon)
    fc = level[best] + damp * slope[best] + seas[best, (n + h - 1) % width]
    return np.asarray(fc, dtype=np.float64), sse


def ets_sse_reference(
    y: Array, alpha: float, beta: float, gamma: float, season: int = 0, phi: float = 1.0
) -> float:
    """Scalar textbook recursion for one parameter set (Holt-Winters additive;
    ``season=0`` gives Holt). Slow; the vectorised fit is tested against it."""
    m = season if season and len(y) >= 2 * season else 0
    if m:
        level = float(np.mean(y[:m]))
        slope = (float(np.mean(y[m : 2 * m])) - level) / m
        seas = [float(v) - level for v in y[:m]]
    else:
        level, slope, seas = float(y[0]), float(y[1] - y[0]), [0.0]
    width = len(seas)
    sse = 0.0
    for t in range(m or 1, len(y)):
        s = seas[t % width]
        sse += (float(y[t]) - (level + phi * slope + s)) ** 2
        new_level = alpha * (float(y[t]) - s) + (1 - alpha) * (level + phi * slope)
        slope = beta * (new_level - level) + (1 - beta) * phi * slope
        if m:
            seas[t % width] = gamma * (float(y[t]) - new_level) + (1 - gamma) * s
        level = new_level
    return sse


def make_ets(trend: bool, season: int = 0, phi: float = 1.0) -> Model:
    def ets(y: Array, promo: Array, origin: int, horizon: int) -> Array:
        if origin < 3:
            return naive(y, promo, origin, horizon)
        fc, _ = ets_fit_forecast(
            np.asarray(y[:origin], dtype=np.float64), horizon, trend, season, phi
        )
        return np.clip(fc, 0.0, None)

    return ets


def seasonal_indices(y: Array, season: int) -> Array | None:
    """Classical multiplicative seasonal indices, or None if not seasonal.

    Seasonality is tested as in the M3 Theta method: the lag-``season``
    autocorrelation must exceed its 90% bound. Non-positive data stays
    unadjusted (a ratio decomposition is undefined there).
    """
    n = len(y)
    if season < 2 or n < 3 * season or np.any(y <= 0):
        return None
    d = y - y.mean()
    acf = np.array([np.sum(d[k:] * d[:-k]) for k in range(1, season + 1)]) / np.sum(d * d)
    limit = 1.645 * np.sqrt((1.0 + 2.0 * np.sum(acf[:-1] ** 2)) / n)
    if abs(acf[-1]) <= limit:
        return None
    weights = np.ones(season + 1) if season % 2 else np.r_[0.5, np.ones(season - 1), 0.5]
    cma = np.convolve(y, weights / season, mode="valid")
    off = season // 2
    ratio = y[off : off + len(cma)] / cma
    pos = (np.arange(len(cma)) + off) % season
    idx = np.array([ratio[pos == j].mean() for j in range(season)])
    return np.asarray(idx / idx.mean(), dtype=np.float64)


def make_theta(season: int) -> Model:
    """Theta method (Assimakopoulos & Nikolopoulos 2000) in its SES-with-drift
    form (Hyndman & Billah 2003): SES on the deseasonalised series plus half
    the slope of a linear trend, then reseasonalised."""

    def theta(y: Array, promo: Array, origin: int, horizon: int) -> Array:
        train = np.asarray(y[:origin], dtype=np.float64)
        n = len(train)
        if n < 3:
            return naive(y, promo, origin, horizon)
        idx = seasonal_indices(train, season)
        x = train / idx[np.arange(n) % season] if idx is not None else train
        _, sse = ets_fit_forecast(x, 1, trend=False)
        alpha = float(ALPHAS[int(np.argmin(sse))])
        level = x[0]
        for v in x[1:]:
            level = alpha * v + (1.0 - alpha) * level
        slope = float(np.polyfit(np.arange(n, dtype=np.float64), x, 1)[0])
        h = np.arange(1, horizon + 1, dtype=np.float64)
        fc = level + 0.5 * slope * (h - 1 + 1 / alpha - (1 - alpha) ** n / alpha)
        if idx is not None:
            fc = fc * idx[(n + np.arange(horizon)) % season]
        return np.asarray(np.clip(fc, 0.0, None), dtype=np.float64)

    return theta


def _ridge(config: dict[str, Any]) -> Model:
    horizon = int(config.get("horizon", 4))
    season = int(config.get("season", 52))
    # Lags must be >= horizon so a direct multi-step forecast never needs an
    # unknown recent value; keep a seasonal lag too.
    lags = [lag for lag in config.get("ridge_lags", [horizon, horizon + 1, horizon + 3, season])]
    lags = sorted({max(lag, horizon) for lag in lags})
    return make_ridge_lags(lags, float(config.get("ridge_alpha", 1.0)))


def _season(config: dict[str, Any]) -> int:
    return int(config.get("season", 52))


Factory = Callable[[dict[str, Any]], Model]
REGISTRY: dict[str, Factory] = {
    "naive": lambda c: naive,
    "seasonal_naive": lambda c: make_seasonal_naive(_season(c)),
    "moving_average": lambda c: make_moving_average(int(c.get("ma_window", 4))),
    "drift": lambda c: drift,
    "ridge_lags": _ridge,
    "seasonal_drift": lambda c: make_seasonal_drift(_season(c)),
    "ses": lambda c: make_ets(trend=False),
    "holt": lambda c: make_ets(trend=True),
    "holt_damped": lambda c: make_ets(trend=True, phi=0.9),
    "holt_winters": lambda c: make_ets(trend=True, season=_season(c), phi=0.9),
    "theta": lambda c: make_theta(_season(c)),
}
DEFAULT_MODELS = ["naive", "seasonal_naive", "moving_average", "drift", "ridge_lags"]


def register(name: str, factory: Factory) -> None:
    """Add a model to the registry (used by the leakage demo's planted models)."""
    REGISTRY[name] = factory


def build_models(config: dict[str, Any]) -> dict[str, Model]:
    """Instantiate the models a config names (``models:``), in that order."""
    names = list(config.get("models", DEFAULT_MODELS))
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise KeyError(f"unknown model(s) {unknown}; registered: {sorted(REGISTRY)}")
    return {name: REGISTRY[name](config) for name in names}
