from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

from evalkit.run import run_evaluation

ROOT = Path(__file__).parent.parent


def _run(tmp: Path, config: str) -> str:
    run_evaluation(str(ROOT / "configs" / config), str(ROOT / "data" / "demand.csv"), str(tmp))
    return (tmp / "report.html").read_text(encoding="utf-8")


def test_report_is_self_contained_and_complete(tmp_path: Path) -> None:
    page = _run(tmp_path, "synthetic.yaml")
    for section in [
        "Models",
        "WAPE by fold",
        "Error by horizon step",
        "Average rank",
        "Pairwise Diebold-Mariano",
        "Leakage audit",
        "Manifest",
    ]:
        assert f">{section}" in page
    assert "<script" not in page and not re.search(r'(src|href)="https?://', page)
    assert page.count("<svg") == 3 and "leakage_ok = true" in page
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["run_id"] in page


def test_report_is_deterministic_and_shows_leaks(tmp_path: Path) -> None:
    assert _run(tmp_path / "a", "synthetic.yaml") == _run(tmp_path / "b", "synthetic.yaml")
    leaky = _run(tmp_path / "c", "leakage_demo.yaml")
    assert "leakage_ok = false" in leaky and leaky.count(">LEAK<") == 3


def test_build_site(tmp_path: Path) -> None:
    _run(tmp_path / "run1", "synthetic.yaml")
    spec = importlib.util.spec_from_file_location("build_site", ROOT / "scripts" / "build_site.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(str(tmp_path / "site"), [str(tmp_path / "run1")])
    index = (tmp_path / "site" / "index.html").read_text()
    assert 'href="run1.html"' in index and (tmp_path / "site" / "run1.html").exists()
