from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evalkit import leaky
from evalkit.cli import main
from evalkit.datagen import generate
from evalkit.leakage import (
    audit_feature,
    detect_future_dependence,
    detect_model_future_access,
    detect_split_contamination,
    detect_target_leakage,
    detect_temporal_order,
    poison_future,
    recompute_stable,
)
from evalkit.models import REGISTRY, build_models
from evalkit.run import gate_metrics, model_leakage_audit, run_evaluation


def lag1(y: np.ndarray) -> np.ndarray:
    out = np.full_like(y, np.nan, dtype=float)
    out[1:] = y[:-1]
    return out


def global_zscore(y: np.ndarray) -> np.ndarray:
    return (y - np.mean(y)) / (np.std(y) + 1e-9)  # uses the WHOLE series -> leak


def forward_shift(y: np.ndarray) -> np.ndarray:
    out = np.full_like(y, np.nan, dtype=float)
    out[:-1] = y[1:]  # uses the future -> leak
    return out


Y = np.arange(1, 41, dtype=float) + np.sin(np.arange(40))


def test_safe_lag_feature_is_stable() -> None:
    assert recompute_stable(lag1, Y, split=20) is True
    assert detect_future_dependence(lag1, Y, 20).leaked is False


def test_global_normalization_is_flagged() -> None:
    assert recompute_stable(global_zscore, Y, split=20) is False
    assert detect_future_dependence(global_zscore, Y, 20).leaked is True


def test_forward_shift_is_flagged() -> None:
    assert recompute_stable(forward_shift, Y, split=20) is False
    assert detect_future_dependence(forward_shift, Y, 20).leaked is True


def test_target_leakage_identity() -> None:
    assert detect_target_leakage(Y, Y).leaked is True  # feature == target


def test_safe_feature_is_not_target_leakage() -> None:
    rng = np.random.default_rng(0)
    noise = rng.normal(size=200)
    feat = lag1(noise)
    mask = ~np.isnan(feat)
    assert detect_target_leakage(feat[mask], noise[mask]).leaked is False


def test_split_contamination() -> None:
    assert detect_split_contamination(["a", "b"], ["b", "c"]).leaked is True
    assert detect_split_contamination(["a", "b"], ["c", "d"]).leaked is False


def test_audit_bundles_findings() -> None:
    findings = audit_feature(global_zscore, Y, 20, target=Y)
    assert any(f.leaked for f in findings)
    assert {f.detector for f in findings} == {"future_dependence", "target_correlation"}


# --- model-level and split-level detectors, against the planted leak_* models ---

CFG = {"min_train": 52, "horizon": 4, "step": 4, "season": 52, "ridge_lags": [4, 5, 7, 52]}
PANEL = generate(seed=7, n_sku=3, n_store=1, n_weeks=104)
LEAKS = ["leak_global_norm", "leak_centered_window", "leak_target_feature"]


def test_poison_future_only_touches_the_test_window() -> None:
    y = np.arange(20, dtype=float)
    p = poison_future(y, 12)
    assert np.array_equal(p[:12], y[:12]) and not np.allclose(p[12:], y[12:])


def test_every_honest_model_passes_future_access() -> None:
    honest = [m for m in REGISTRY if not m.startswith("leak_") and not m.startswith("_")]
    audit = model_leakage_audit(PANEL, {**CFG, "models": honest})
    assert audit == {m: False for m in honest}


@pytest.mark.parametrize("name", LEAKS)
def test_each_planted_leak_is_flagged(name: str) -> None:
    assert model_leakage_audit(PANEL, {**CFG, "models": [name]}) == {name: True}


def test_future_access_finding_detail() -> None:
    y = PANEL[PANEL["unique_id"] == PANEL["unique_id"].iloc[0]]["y"].to_numpy(dtype=float)
    models = build_models({**CFG, "models": ["naive", "leak_centered_window"]})
    clean = detect_model_future_access(models["naive"], y, y * 0, 80, 4)
    leak = detect_model_future_access(models["leak_centered_window"], y, y * 0, 80, 4)
    assert not clean.leaked and "independent" in clean.detail
    assert leak.leaked and "changes" in leak.detail


def test_temporal_order_flags_shuffled_kfold_not_rolling() -> None:
    assert not detect_temporal_order(range(50), range(50, 54)).leaked
    train, test = leaky.shuffled_kfold_split(100, 5, 0)
    assert sorted(train + test) == list(range(100)) and not set(train) & set(test)
    assert detect_temporal_order(train, test).leaked
    y = 10 + np.sin(np.arange(80))
    assert 0.0 <= leaky.shuffled_kfold_wape(y, [4, 5]) < 1.0


def test_leaky_run_fails_the_gate(tmp_path: Path) -> None:
    data = tmp_path / "d.csv"
    PANEL.to_csv(data, index=False)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "min_train: 52\nhorizon: 4\nstep: 4\nseason: 52\nridge_lags: [4, 52]\n"
        "models: [naive, ridge_lags, leak_target_feature]\n"
    )
    manifest = run_evaluation(str(cfg), str(data), str(tmp_path / "out"))
    assert manifest["leakage_ok"] is False
    assert manifest["leaky_models"] == ["leak_target_feature"]
    metrics = str(tmp_path / "out" / "metrics.csv")
    ok, problems = gate_metrics(metrics, metrics)  # identical to baseline, still fails
    assert not ok and any("leakage" in p for p in problems)
    manifest_path = str(tmp_path / "out" / "manifest.json")
    assert main(["gate", metrics, "--baseline", metrics, "--manifest", manifest_path]) == 1
    honest = pd.read_csv(metrics)
    assert set(honest["model"]) == {"naive", "ridge_lags", "leak_target_feature"}
