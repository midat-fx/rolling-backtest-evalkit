from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from evalkit.cli import main
from evalkit.run import gate_metrics, run_evaluation, sanity_best_beats_naive

ROOT = Path(__file__).parent.parent
CONFIG = str(ROOT / "configs" / "synthetic.yaml")
DATA = str(ROOT / "data" / "demand.csv")
BASELINE = str(ROOT / "reports" / "baseline_metrics.csv")


def test_run_produces_manifest_and_leakage_ok(tmp_path: Path) -> None:
    manifest = run_evaluation(CONFIG, DATA, str(tmp_path))
    assert manifest["leakage_ok"] is True
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "report.md").exists()
    assert set(manifest["models"]) == {
        "naive",
        "seasonal_naive",
        "moving_average",
        "drift",
        "ridge_lags",
    }


def test_metrics_match_committed_baseline(tmp_path: Path) -> None:
    run_evaluation(CONFIG, DATA, str(tmp_path))
    ok, problems = gate_metrics(str(tmp_path / "metrics.csv"), BASELINE)
    assert ok, problems


def test_run_is_deterministic(tmp_path: Path) -> None:
    run_evaluation(CONFIG, DATA, str(tmp_path / "a"))
    run_evaluation(CONFIG, DATA, str(tmp_path / "b"))
    a = (tmp_path / "a" / "metrics.csv").read_bytes()
    b = (tmp_path / "b" / "metrics.csv").read_bytes()
    assert a == b


def test_gate_detects_regression(tmp_path: Path) -> None:
    df = pd.read_csv(BASELINE)
    df.loc[0, "wape"] += 0.5
    bad = tmp_path / "bad.csv"
    df.to_csv(bad, index=False, float_format="%.6f")
    ok, problems = gate_metrics(str(bad), BASELINE)
    assert not ok and any("wape" in p for p in problems)


def test_sanity_best_beats_naive() -> None:
    ok, detail = sanity_best_beats_naive(BASELINE)
    assert ok, detail


def test_cli_gate_exit_codes() -> None:
    assert main(["gate", BASELINE, "--baseline", BASELINE]) == 0


def test_cli_data_and_quality(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "d.csv"
    assert main(["data", "-o", str(out), "--seed", "42"]) == 0
    assert main(["quality", str(out)]) == 0
