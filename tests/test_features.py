from __future__ import annotations

import numpy as np

from evalkit.features import lag_features


def test_shape_and_values() -> None:
    y = np.arange(10, dtype=float)
    promo = np.zeros(10)
    lags = [1, 2]
    x = lag_features(y, promo, lags, t_index=[5, 6])
    # columns: y[t-1], y[t-2], promo[t], intercept
    assert x.shape == (2, 4)
    assert list(x[0]) == [4.0, 3.0, 0.0, 1.0]
    assert list(x[1]) == [5.0, 4.0, 0.0, 1.0]


def test_only_uses_past() -> None:
    # Feature rows must not depend on y at or after t.
    y = np.arange(20, dtype=float)
    promo = np.zeros(20)
    x = lag_features(y, promo, [4], t_index=[10])
    assert x[0, 0] == 6.0  # y[10-4], strictly in the past
