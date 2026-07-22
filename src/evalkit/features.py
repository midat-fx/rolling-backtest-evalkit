"""Leakage-safe feature construction.

The lag feature matrix for a set of time indices only ever references past
observations (``y[t - lag]`` with ``lag >= 1``), so it cannot see the value it
is used to predict. In the backtest, lags are constrained to be at least the
forecast horizon, which keeps direct multi-step forecasts honest.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def lag_features(y: Array, promo: Array, lags: Sequence[int], t_index: Sequence[int]) -> Array:
    """Build a feature matrix: one row per t in t_index.

    Row = [y[t-lag] for lag in lags] + [promo[t], 1.0(intercept)].
    Every t in t_index must satisfy t - max(lags) >= 0.
    """
    rows: list[list[float]] = []
    for t in t_index:
        row = [float(y[t - lag]) for lag in lags]
        row.append(float(promo[t]))
        row.append(1.0)
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)
