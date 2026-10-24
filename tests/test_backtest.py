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


def test_scale_season_sets_the_mase_denominator() -> None:
    import numpy as np

    from evalkit.backtest import backtest_detail

    df = generate(seed=3, n_sku=2, n_store=1, n_weeks=60)
    cfg = {"min_train": 40, "horizon": 4, "step": 8, "season": 52, "models": ["naive"]}
    for season in (1, 4):
        d = backtest_detail(df, {**cfg, "scale_season": season})
        y = df[df["unique_id"] == d.uids[0]]["y"].to_numpy(dtype=float)[:40]
        assert d.scale[0, 0] == np.mean(np.abs(y[season:] - y[:-season]))
