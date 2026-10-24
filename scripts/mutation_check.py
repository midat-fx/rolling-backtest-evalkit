"""Mutation check for the metric and statistics code.

    python scripts/mutation_check.py

Each mutant is a small, plausible bug (a dropped abs, a wrong denominator, an
off-by-one in degrees of freedom). For each one the repo is copied to a temp
dir, the bug is applied there, and the relevant tests run; a mutant the tests
do not fail on "survives" and means a test is missing. Exit code 1 if any
mutant survives. The working tree is never modified.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = [
    "tests/test_metrics.py",
    "tests/test_properties.py",
    "tests/test_stats.py",
    "tests/test_backtest.py",
    "tests/test_cli.py",
]

# (file, original snippet, mutated snippet). Each snippet must occur exactly once.
MUTANTS = [
    ("metrics.py", "np.sum(np.abs(y - p)) / max", "np.sum(y - p) / max"),
    (
        "metrics.py",
        "np.sum(np.abs(y - p)) / max(float(np.sum(np.abs(y)))",
        "np.sum(np.abs(y - p)) / max(float(np.sum(np.abs(p)))",
    ),
    ("metrics.py", "np.mean(2.0 * np.abs(y - p) / denom)", "np.mean(np.abs(y - p) / denom)"),
    ("metrics.py", "denom = np.abs(y) + np.abs(p) + _EPS", "denom = np.abs(y) + _EPS"),
    ("metrics.py", "np.sum(p - y) / max", "np.sum(y - p) / max"),
    ("metrics.py", "np.mean(np.abs(t[season:] - t[:-season]))", "np.mean(np.abs(t[1:] - t[:-1]))"),
    (
        "metrics.py",
        "np.mean((t[season:] - t[:-season]) ** 2)",
        "np.mean(np.abs(t[season:] - t[:-season]))",
    ),
    (
        "metrics.py",
        "float(np.mean(np.abs(y - p)) / max(scale",
        "float(np.sum(np.abs(y - p)) / max(scale",
    ),
    (
        "metrics.py",
        "np.sqrt(np.mean((y - p) ** 2) / max(scale_mse",
        "(np.mean((y - p) ** 2) / max(scale_mse",
    ),
    (
        "backtest.py",
        "np.mean(np.abs(e), axis=-1) / self.scale[:, :, None]",
        "np.mean(np.abs(e), axis=-1) / self.scale_mse[:, :, None]",
    ),
    (
        "backtest.py",
        "scale[i] = naive_mae_scale(train, scale_season)",
        "scale[i] = naive_mae_scale(train, 1)",
    ),
    ("stats.py", "(n + 1 - 2 * h + h * (h - 1) / n) / n", "(n + 1 - h + h * (h - 1) / n) / n"),
    ("stats.py", "2.0 * t_sf(abs(hln), n - 1)", "2.0 * t_sf(abs(hln), n)"),
    ("stats.py", "(len(pvalues) - rank) * pvalues[i]", "len(pvalues) * pvalues[i]"),
    ("stats.py", "[alpha / 2, 1 - alpha / 2]", "[alpha, 1 - alpha]"),
    ("stats.py", "ranks = 1.0 + less + (equal - 1) / 2.0", "ranks = 1.0 + less + (equal - 1)"),
    ("stats.py", "math.sqrt(k * (k + 1) / (6.0 * n))", "math.sqrt(k * (k + 1) / (12.0 * n))"),
]


def main() -> int:
    survivors = 0
    for i, (fname, old, new) in enumerate(MUTANTS, 1):
        with tempfile.TemporaryDirectory() as tmp:
            for d in ["src", "tests", "configs", "data", "reports", "scripts"]:
                shutil.copytree(ROOT / d, Path(tmp) / d)
            shutil.copy(ROOT / "pyproject.toml", tmp)
            target = Path(tmp) / "src" / "evalkit" / fname
            text = target.read_text()
            assert text.count(old) == 1, f"mutant {i}: snippet not found once in {fname}"
            target.write_text(text.replace(old, new))
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *TESTS],
                cwd=tmp,
                env={"PYTHONPATH": str(Path(tmp) / "src"), "PATH": "/usr/bin:/bin"},
                capture_output=True,
                text=True,
            )
        killed = proc.returncode != 0
        survivors += not killed
        print(f"{'killed  ' if killed else 'SURVIVED'} {fname}: {old!r} -> {new!r}")
    print(f"\n{len(MUTANTS) - survivors}/{len(MUTANTS)} mutants killed")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
