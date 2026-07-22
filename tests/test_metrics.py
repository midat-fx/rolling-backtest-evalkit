from __future__ import annotations

import numpy as np
import pytest

from evalkit import metrics as M


def test_perfect_forecast_is_zero_error() -> None:
    y = [10, 20, 30]
    assert M.wape(y, y) == pytest.approx(0.0)
    assert M.smape(y, y) == pytest.approx(0.0, abs=1e-6)
    assert M.bias(y, y) == pytest.approx(0.0)


def test_wape_hand_computed() -> None:
    # errors |{-2, +2}| = 4 over sum|y| = 30
    assert M.wape([10, 20], [12, 18]) == pytest.approx(4 / 30)


def test_bias_sign() -> None:
    assert M.bias([10, 10], [12, 13]) > 0  # over-forecast
    assert M.bias([10, 10], [8, 7]) < 0  # under-forecast


def test_naive_scales() -> None:
    assert M.naive_mae_scale([10, 20, 30], 1) == pytest.approx(10.0)
    assert M.naive_mse_scale([10, 20, 30], 1) == pytest.approx(100.0)


def test_mase_and_rmsse() -> None:
    assert M.mase([10, 20], [12, 18], scale=5.0) == pytest.approx(2 / 5)
    assert M.rmsse([10, 20], [13, 16], scale_mse=25.0) == pytest.approx(np.sqrt(12.5 / 25.0))


def test_zero_division_guarded() -> None:
    assert M.wape([0, 0], [1, 1]) == pytest.approx(2 / 1e-9)  # eps denominator, finite
    assert np.isfinite(M.smape([0], [0]))
