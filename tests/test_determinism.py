from __future__ import annotations

import filecmp
import json
from pathlib import Path

from evalkit.run import run_evaluation

ROOT = Path(__file__).parent.parent
CONFIG = str(ROOT / "configs" / "synthetic.yaml")
DATA = str(ROOT / "data" / "demand.csv")
OUTPUTS = ["metrics.csv", "report.md", "report.html", "stats.json", "manifest.json"]


def test_outputs_identical_across_worker_counts(tmp_path: Path) -> None:
    a = run_evaluation(CONFIG, DATA, str(tmp_path / "j1"), workers=1)
    b = run_evaluation(CONFIG, DATA, str(tmp_path / "j3"), workers=3)
    assert a["run_id"] == b["run_id"]
    for f in OUTPUTS:
        assert filecmp.cmp(tmp_path / "j1" / f, tmp_path / "j3" / f, shallow=False), f


def test_run_id_tracks_config_and_data(tmp_path: Path) -> None:
    base = run_evaluation(CONFIG, DATA, str(tmp_path / "a"))["run_id"]
    cfg = tmp_path / "c.yaml"
    cfg.write_text(Path(CONFIG).read_text() + "ridge_alpha: 2.0\n")
    assert run_evaluation(str(cfg), DATA, str(tmp_path / "b"))["run_id"] != base
    data = tmp_path / "d.csv"
    lines = Path(DATA).read_text().splitlines()
    lines[1] = lines[1].rsplit(",", 2)[0] + ",999," + lines[1].rsplit(",", 1)[1]
    data.write_text("\n".join(lines) + "\n")
    assert run_evaluation(CONFIG, str(data), str(tmp_path / "c"))["run_id"] != base
    manifest = json.loads((tmp_path / "a" / "manifest.json").read_text())
    assert manifest["run_id"] == base
