"""Forecast accuracy metrics.

Pure functions over numpy arrays. Scale-dependent metrics (MASE, RMSSE) take a
pre-computed in-sample naive scale so the caller controls what "naive" means for
each series. All functions guard against division by zero with a small epsilon.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]
_EPS = 1e-9


def _f(a: npt.ArrayLike) -> Array:
    return np.asarray(a, dtype=np.float64)


def wape(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    y, p = _f(y_true), _f(y_pred)
    return float(np.sum(np.abs(y - p)) / max(float(np.sum(np.abs(y))), _EPS))


def smape(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    y, p = _f(y_true), _f(y_pred)
    denom = np.abs(y) + np.abs(p) + _EPS
    return float(np.mean(2.0 * np.abs(y - p) / denom))


def bias(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    y, p = _f(y_true), _f(y_pred)
    return float(np.sum(p - y) / max(float(np.sum(np.abs(y))), _EPS))


def naive_mae_scale(train: npt.ArrayLike, season: int = 1) -> float:
    """In-sample MAE of a (seasonal) naive forecast on the training series."""
    t = _f(train)
    if t.size <= season:
        return _EPS
    return max(float(np.mean(np.abs(t[season:] - t[:-season]))), _EPS)


def naive_mse_scale(train: npt.ArrayLike, season: int = 1) -> float:
    """In-sample MSE of a (seasonal) naive forecast on the training series."""
    t = _f(train)
    if t.size <= season:
        return _EPS
    return max(float(np.mean((t[season:] - t[:-season]) ** 2)), _EPS)


def mase(y_true: npt.ArrayLike, y_pred: npt.ArrayLike, scale: float) -> float:
    y, p = _f(y_true), _f(y_pred)
    return float(np.mean(np.abs(y - p)) / max(scale, _EPS))


def rmsse(y_true: npt.ArrayLike, y_pred: npt.ArrayLike, scale_mse: float) -> float:
    y, p = _f(y_true), _f(y_pred)
    return float(np.sqrt(np.mean((y - p) ** 2) / max(scale_mse, _EPS)))
