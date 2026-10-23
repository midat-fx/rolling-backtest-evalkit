"""Honest models vs planted leaks: the metric improves, the detectors fire.

    python examples/leakage_demo.py                         # synthetic panel
    python examples/leakage_demo.py --data data/hospital.csv --config configs/hospital.yaml

Prints (1) rolling-origin WAPE/MASE and the model_future_access verdict for
honest and leaky models, (2) the same ridge scored on shuffled K-fold vs rolling
origins with the temporal_order verdict, and (3) what `evalkit gate` says.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from evalkit.backtest import aggregate, backtest_detail, to_panel
from evalkit.leakage import detect_temporal_order
from evalkit.leaky import shuffled_kfold_split, shuffled_kfold_wape
from evalkit.report import per_model_summary
from evalkit.run import gate_metrics, load_config, model_leakage_audit, run_evaluation

HONEST = ["naive", "ridge_lags", "theta"]
LEAKS = ["leak_global_norm", "leak_centered_window", "leak_target_feature"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/demand.csv")
    p.add_argument("--config", default="configs/synthetic.yaml")
    args = p.parse_args()
    config = {**load_config(args.config), "models": HONEST + LEAKS}
    df = pd.read_csv(args.data)

    detail = backtest_detail(df, config)
    summary = per_model_summary(aggregate(detail)).set_index("model")
    audit = model_leakage_audit(df, config)
    print(f"## 1. Rolling origin: honest vs leaky ({args.data}, {len(detail.uids)} series)\n")
    print("| model | median WAPE | median MASE | model_future_access |")
    print("|---|---|---|---|")
    for m in HONEST + LEAKS:
        verdict = "**LEAK**" if audit[m] else "clean"
        print(f"| {m} | {summary.loc[m, 'wape']:.4f} | {summary.loc[m, 'mase']:.4f} | {verdict} |")

    panel = to_panel(df)
    h = int(config["horizon"])
    lags = sorted({max(lag, h) for lag in config["ridge_lags"]})
    shuffled = float(np.mean([shuffled_kfold_wape(y, lags) for y, _ in panel.values()]))
    rolling = float(detail.metric("wape")[:, :, detail.models.index("ridge_lags")].mean())
    n0 = len(next(iter(panel.values()))[0]) - max(lags)
    o = int(config["min_train"])
    honest_order = detect_temporal_order(list(range(o)), list(range(o, o + h)))
    order = detect_temporal_order(*shuffled_kfold_split(n0, 5, 0))
    print("\n## 2. Same ridge, two evaluation schemes (mean per-series WAPE)\n")
    print("| scheme | WAPE | temporal_order |")
    print("|---|---|---|")
    schemes = [("rolling origin", rolling, honest_order), ("shuffled 5-fold", shuffled, order)]
    for scheme, wape, f in schemes:
        verdict = f"**LEAK** ({f.detail})" if f.leaked else "clean"
        print(f"| {scheme} | {wape:.4f} | {verdict} |")

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        manifest = run_evaluation(str(cfg_path), args.data, tmp)
        metrics = str(Path(tmp) / "metrics.csv")
        ok, problems = gate_metrics(metrics, metrics)  # even against itself
        problems = [p.replace(tmp + "/", "") for p in problems]
    print("\n## 3. The gate\n")
    print(f"leakage_ok={manifest['leakage_ok']} leaky_models={manifest['leaky_models']}")
    print(("OK" if ok else "FAIL") + ": " + "; ".join(problems))


if __name__ == "__main__":
    main()
