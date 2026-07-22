"""Orchestration: config, provenance, the leakage self-check, and the gate."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml

from .backtest import METRIC_NAMES, run_backtest
from .leakage import recompute_stable
from .report import per_model_summary, render_report, write_metrics

Array = npt.NDArray[np.float64]


def load_config(path: str) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config {path} must be a mapping")
    return data


def data_checksum(df: pd.DataFrame) -> str:
    ordered = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return hashlib.sha256(ordered.to_csv(index=False).encode("utf-8")).hexdigest()


def run_id(config: dict[str, Any], checksum: str) -> str:
    payload = json.dumps(config, sort_keys=True) + checksum
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _lag_feature_fn(lags: list[int]) -> Any:
    lag0, mlag = lags[0], max(lags)

    def fn(y: Array) -> Array:
        out = np.full(len(y), np.nan, dtype=np.float64)
        for t in range(mlag, len(y)):
            out[t] = y[t - lag0]
        return out

    return fn


def leakage_self_check(df: pd.DataFrame, config: dict[str, Any]) -> bool:
    """Confirm the backtest's lag features do not depend on the future."""
    horizon = int(config.get("horizon", 4))
    season = int(config.get("season", 52))
    lags = sorted({max(lag, horizon) for lag in config.get("ridge_lags", [horizon, season])})
    series = df.sort_values(["unique_id", "ds"]).groupby("unique_id")["y"].first().index[0]
    y = df[df["unique_id"] == series]["y"].to_numpy(dtype=np.float64)
    split = max(len(y) // 2, max(lags) + 1)
    return recompute_stable(_lag_feature_fn(lags), y, split)


def run_evaluation(config_path: str, data_path: str, outdir: str) -> dict[str, Any]:
    config = load_config(config_path)
    df = pd.read_csv(data_path)
    bt = run_backtest(df, config)

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    write_metrics(bt, str(out / "metrics.csv"))
    (out / "report.md").write_text(render_report(bt), encoding="utf-8")

    checksum = data_checksum(df)
    manifest: dict[str, Any] = {
        "run_id": run_id(config, checksum),
        "config": config,
        "data_checksum": checksum,
        "leakage_ok": leakage_self_check(df, config),
        "folds": int(bt["fold"].nunique()),
        "models": sorted(bt["model"].unique().tolist()),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest


def sanity_best_beats_naive(metrics_csv: str) -> tuple[bool, str]:
    """A backtest is only credible if at least one trained model beats the
    naive baseline on median WAPE. If nothing beats naive, either the models or
    the evaluation is broken."""
    bt = pd.read_csv(metrics_csv)
    summary = per_model_summary(bt).set_index("model")
    if "naive" not in summary.index:
        return True, "no naive baseline to compare against"
    naive_wape = float(summary.loc["naive", "wape"])
    others = summary.drop(index="naive")["wape"]
    if others.empty:
        return True, "only a naive baseline present"
    best_model = str(others.idxmin())
    best_wape = float(others.min())
    return (
        best_wape <= naive_wape,
        f"best={best_model} WAPE {best_wape:.4f} vs naive {naive_wape:.4f}",
    )


def gate_metrics(
    current_csv: str, baseline_csv: str, atol: float = 1e-4, rtol: float = 1e-4
) -> tuple[bool, list[str]]:
    cur = pd.read_csv(current_csv)
    base = pd.read_csv(baseline_csv)
    problems: list[str] = []
    merged = cur.merge(base, on=["fold", "model"], suffixes=("_cur", "_base"))
    if len(merged) != len(base) or len(merged) != len(cur):
        problems.append(
            f"row set differs: current={len(cur)}, baseline={len(base)}, matched={len(merged)}"
        )
    for m in METRIC_NAMES:
        a = merged[f"{m}_cur"].to_numpy(dtype=np.float64)
        b = merged[f"{m}_base"].to_numpy(dtype=np.float64)
        bad = ~np.isclose(a, b, atol=atol, rtol=rtol)
        for i in np.where(bad)[0]:
            fold = int(merged["fold"].iloc[i])
            model = str(merged["model"].iloc[i])
            problems.append(f"fold {fold} {model} {m}: {b[i]:.6f} -> {a[i]:.6f}")
    ok_h, detail = sanity_best_beats_naive(current_csv)
    if not ok_h:
        problems.append(f"sanity: no trained model beats naive ({detail})")
    return len(problems) == 0, problems
