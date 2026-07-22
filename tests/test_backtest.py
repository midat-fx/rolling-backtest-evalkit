from __future__ import annotations

from evalkit.backtest import METRIC_NAMES, rolling_origins, run_backtest
from evalkit.datagen import generate


def test_rolling_origins() -> None:
    # n=60, min_train=52, horizon=4, step=4 -> origins 52, 56 (56+4<=60)
    assert rolling_origins(60, 52, 4, 4) == [52, 56]
    assert rolling_origins(52, 52, 4, 4) == []  # no room for a test window


def test_backtest_shape_and_determinism() -> None:
    df = generate(seed=42, n_sku=4, n_store=1, n_weeks=104)
    cfg = {"min_train": 52, "horizon": 4, "step": 4, "season": 52, "ridge_lags": [4, 52]}
    a = run_backtest(df, cfg)
    b = run_backtest(df, cfg)
    assert list(a.columns) == ["fold", "model"] + METRIC_NAMES
    assert a["model"].nunique() == 5
    assert a.equals(b)  # deterministic
