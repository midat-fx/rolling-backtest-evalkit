"""Rolling-origin backtest over a panel of series.

For each origin, training data is ``y[:origin]`` and the test window is
``y[origin:origin+horizon]``. Features are recomputed inside each fold from the
training window only; no global statistics leak across the split. Metrics are
computed per series and averaged, giving one row per (fold, model).

The work is series-major: ``backtest_detail`` forecasts every (fold, model) for
one series at a time and keeps the raw forecasts, which the statistics and the
HTML report need; ``run_backtest`` aggregates them to the per-fold table.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import leaky  # noqa: F401  (worker processes must see the leak_* models too)
from .metrics import naive_mae_scale, naive_mse_scale
from .models import build_models

Array = npt.NDArray[np.float64]
METRIC_NAMES = ["wape", "smape", "mase", "rmsse", "bias"]
_EPS = 1e-9


@dataclass
class Detail:
    """Raw backtest output. Shapes: S series, F folds, M models, H horizon."""

    uids: list[str]
    models: list[str]
    y_true: Array  # (S, F, H)
    y_hat: Array  # (S, F, M, H)
    scale: Array  # (S, F) in-sample naive MAE
    scale_mse: Array  # (S, F) in-sample naive MSE

    def metric(self, name: str) -> Array:
        """Per-(series, fold, model) metric, shape (S, F, M)."""
        y = self.y_true[:, :, None, :]
        e = self.y_hat - y
        denom = np.maximum(np.sum(np.abs(y), axis=-1), _EPS)
        if name == "wape":
            return np.asarray(np.sum(np.abs(e), axis=-1) / denom)
        if name == "bias":
            return np.asarray(np.sum(e, axis=-1) / denom)
        if name == "smape":
            return np.asarray(
                np.mean(2.0 * np.abs(e) / (np.abs(y) + np.abs(self.y_hat) + _EPS), -1)
            )
        if name == "mase":
            return np.asarray(np.mean(np.abs(e), axis=-1) / self.scale[:, :, None])
        if name == "rmsse":
            return np.asarray(np.sqrt(np.mean(e * e, axis=-1) / self.scale_mse[:, :, None]))
        raise KeyError(name)

    def scaled_abs_error(self) -> Array:
        """|error| / naive MAE scale, shape (S, F, M, H): the per-step MASE."""
        return np.asarray(
            np.abs(self.y_hat - self.y_true[:, :, None, :]) / self.scale[:, :, None, None]
        )


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


def forecast_series(
    y: Array, promo: Array, origins: list[int], config: dict[str, Any]
) -> tuple[Array, Array, Array, Array]:
    """All folds x models for one series. ``origins`` are already tail-aligned."""
    models = build_models(config)
    horizon = int(config["horizon"])
    # MASE/RMSSE scale: in-sample naive with this lag (1 = naive, m = seasonal naive).
    scale_season = int(config.get("scale_season", 1))
    f = len(origins)
    y_true = np.empty((f, horizon))
    y_hat = np.empty((f, len(models), horizon))
    scale = np.empty(f)
    scale_mse = np.empty(f)
    for i, o in enumerate(origins):
        y_true[i] = y[o : o + horizon]
        train = y[:o]
        scale[i] = naive_mae_scale(train, scale_season)
        scale_mse[i] = naive_mse_scale(train, scale_season)
        for j, mfn in enumerate(models.values()):
            y_hat[i, j] = mfn(y, promo, o, horizon)
    return y_true, y_hat, scale, scale_mse


def _forecast_task(
    task: tuple[Array, Array, list[int], dict[str, Any]],
) -> tuple[Array, Array, Array, Array]:
    return forecast_series(*task)


def backtest_detail(df: pd.DataFrame, config: dict[str, Any], workers: int = 1) -> Detail:
    """Forecast every series; ``workers > 1`` fans series out to processes.

    Results come back in series order whatever the worker count, so the output
    is identical for any ``workers``. Models used with ``workers > 1`` must be
    registered at import time (as all built-in and ``leak_*`` models are).
    """
    panel = to_panel(df)
    # Series may differ in length; folds are aligned on each series' end, so
    # fold k tests the same position from the end and longer series simply
    # train on more history.
    n = min(len(y) for y, _ in panel.values())
    origins = rolling_origins(
        n, int(config["min_train"]), int(config["horizon"]), int(config["step"])
    )
    uids = sorted(panel)
    tasks = [
        (panel[u][0], panel[u][1], [o + len(panel[u][0]) - n for o in origins], config)
        for u in uids
    ]
    if workers > 1:
        chunk = max(1, len(tasks) // (workers * 4))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_forecast_task, tasks, chunksize=chunk))
    else:
        parts = [_forecast_task(t) for t in tasks]
    return Detail(
        uids=uids,
        models=list(build_models(config)),
        y_true=np.stack([p[0] for p in parts]),
        y_hat=np.stack([p[1] for p in parts]),
        scale=np.stack([p[2] for p in parts]),
        scale_mse=np.stack([p[3] for p in parts]),
    )


def aggregate(detail: Detail) -> pd.DataFrame:
    """One row per (fold, model): each metric averaged over series."""
    means = {k: detail.metric(k).mean(axis=0) for k in METRIC_NAMES}  # (F, M)
    n_folds = detail.y_true.shape[1]
    records: list[dict[str, Any]] = []
    for fold in range(n_folds):
        for j, mname in enumerate(detail.models):
            record: dict[str, Any] = {"fold": fold, "model": mname}
            for k in METRIC_NAMES:
                record[k] = float(means[k][fold, j])
            records.append(record)
    return pd.DataFrame.from_records(records)


def run_backtest(df: pd.DataFrame, config: dict[str, Any], workers: int = 1) -> pd.DataFrame:
    return aggregate(backtest_detail(df, config, workers))
