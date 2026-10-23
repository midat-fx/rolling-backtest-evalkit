"""Rolling-origin backtest over a panel of series.

For each origin, training data is ``y[:origin]`` and the test window is
``y[origin:origin+horizon]``. Features are recomputed inside each fold from the
training window only; no global statistics leak across the split. Metrics are
computed per series and averaged, giving one row per (fold, model).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from .metrics import bias, mase, naive_mae_scale, naive_mse_scale, rmsse, smape, wape
from .models import build_models

Array = npt.NDArray[np.float64]
METRIC_NAMES = ["wape", "smape", "mase", "rmsse", "bias"]


def to_panel(df: pd.DataFrame) -> dict[str, tuple[Array, Array]]:
    panel: dict[str, tuple[Array, Array]] = {}
    ordered = df.sort_values(["unique_id", "ds"])
    for uid, g in ordered.groupby("unique_id", sort=True):
        y = g["y"].to_numpy(dtype=np.float64)
        promo = g["promo"].to_numpy(dtype=np.float64)
        panel[str(uid)] = (y, promo)
    return panel


def rolling_origins(n: int, min_train: int, horizon: int, step: int) -> list[int]:
    origins: list[int] = []
    o = min_train
    while o + horizon <= n:
        origins.append(o)
        o += step
    return origins


def run_backtest(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    panel = to_panel(df)
    # Series may differ in length; folds are aligned on each series' end, so
    # fold k tests the same position from the end and longer series simply
    # train on more history.
    n = min(len(y) for y, _ in panel.values())
    origins = rolling_origins(
        n, int(config["min_train"]), int(config["horizon"]), int(config["step"])
    )
    models = build_models(config)
    horizon = int(config["horizon"])
    # MASE/RMSSE scale: in-sample naive with this lag (1 = naive, m = seasonal naive).
    scale_season = int(config.get("scale_season", 1))

    records: list[dict[str, Any]] = []
    for fold, origin in enumerate(origins):
        for mname, mfn in models.items():
            acc: dict[str, list[float]] = {k: [] for k in METRIC_NAMES}
            for uid in sorted(panel):
                y, promo = panel[uid]
                o = origin + len(y) - n
                y_true = y[o : o + horizon]
                y_hat = mfn(y, promo, o, horizon)
                train = y[:o]
                scale = naive_mae_scale(train, scale_season)
                scale_mse = naive_mse_scale(train, scale_season)
                acc["wape"].append(wape(y_true, y_hat))
                acc["smape"].append(smape(y_true, y_hat))
                acc["mase"].append(mase(y_true, y_hat, scale))
                acc["rmsse"].append(rmsse(y_true, y_hat, scale_mse))
                acc["bias"].append(bias(y_true, y_hat))
            record: dict[str, Any] = {"fold": fold, "model": mname}
            for k in METRIC_NAMES:
                record[k] = float(np.mean(acc[k]))
            records.append(record)
    return pd.DataFrame.from_records(records)
