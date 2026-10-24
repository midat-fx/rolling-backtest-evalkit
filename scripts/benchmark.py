"""Timing for docs/performance.md.

    python scripts/benchmark.py            # needs data/hospital.csv, data/tourism_monthly.csv

1. ETS fit: the scalar recursion looped over the parameter grid vs the
   vectorised grid (same SSEs), on one hospital series.
2. Full `run_evaluation` (backtest + stats + audit + report) by worker count,
   checking that every output file is byte-identical across worker counts.
"""

from __future__ import annotations

import filecmp
import os
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evalkit.models import ALPHAS, BETAS, GAMMAS, ets_fit_forecast, ets_sse_reference
from evalkit.run import run_evaluation

OUTPUTS = ["metrics.csv", "report.md", "report.html", "stats.json", "manifest.json"]


def best_of(fn: object, repeat: int = 3) -> float:
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()  # type: ignore[operator]
        times.append(time.perf_counter() - t)
    return min(times)


def main() -> None:
    print(
        f"python {platform.python_version()}, numpy {np.__version__}, {os.cpu_count()} CPUs, "
        f"{platform.machine()} {platform.system()}\n"
    )
    df = pd.read_csv("data/hospital.csv")
    y = df[df["unique_id"] == "T1"]["y"].to_numpy(dtype=np.float64)[:72]
    grid = [(a, b, g) for a in ALPHAS for b in BETAS for g in GAMMAS]
    scalar = best_of(lambda: [ets_sse_reference(y, a, b, g, 12, 0.9) for a, b, g in grid])
    vector = best_of(lambda: ets_fit_forecast(y, 12, trend=True, season=12, phi=0.9))
    print("## ETS (Holt-Winters) fit, one series of 72 months, 250-point grid\n")
    print("| implementation | time per fit | speed-up |\n|---|---|---|")
    print(f"| scalar loop over grid | {scalar * 1e3:.1f} ms | 1.0x |")
    print(f"| vectorised grid | {vector * 1e3:.2f} ms | {scalar / vector:.0f}x |\n")

    print("## Full run (`evalkit run -j N`), 11 models\n")
    print("| dataset | series | workers | wall time | speed-up | outputs identical |")
    print("|---|---|---|---|---|---|")
    for name in ["hospital", "tourism_monthly"]:
        data = f"data/{name}.csv"
        n_series = pd.read_csv(data)["unique_id"].nunique()
        with tempfile.TemporaryDirectory() as tmp:
            base = None
            for workers in [1, 2, 4, 8]:
                out = Path(tmp) / f"j{workers}"
                t = time.perf_counter()
                run_evaluation(f"configs/{name}.yaml", data, str(out), workers=workers)
                dt = time.perf_counter() - t
                base = base or dt
                same = all(
                    filecmp.cmp(Path(tmp) / "j1" / f, out / f, shallow=False) for f in OUTPUTS
                )
                print(
                    f"| {name} | {n_series} | {workers} | {dt:.2f} s | {base / dt:.1f}x | {same} |"
                )


if __name__ == "__main__":
    main()
