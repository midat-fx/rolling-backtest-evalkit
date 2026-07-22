"""Command-line interface for evalkit."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import __version__
from .datagen import write_csv
from .quality import check_quality
from .report import per_model_summary
from .run import gate_metrics, run_evaluation


def cmd_data(args: argparse.Namespace) -> int:
    df = write_csv(args.output, seed=args.seed)
    print(f"evalkit data: wrote {args.output} ({len(df)} rows, {df['unique_id'].nunique()} series)")
    return 0


def cmd_quality(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.data)
    issues, summary = check_quality(df)
    for i in issues:
        print(f"  {i}")
    print(f"evalkit quality: {summary}")
    return 1 if summary.get("fatal") else 0


def cmd_run(args: argparse.Namespace) -> int:
    manifest = run_evaluation(args.config, args.data, args.output)
    print(f"evalkit run: run_id={manifest['run_id']} leakage_ok={manifest['leakage_ok']}")
    print(f"evalkit run: wrote {args.output}/metrics.csv, report.md, manifest.json")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    bt = pd.read_csv(args.metrics)
    print(per_model_summary(bt).to_string(index=False))
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    ok, problems = gate_metrics(args.metrics, args.baseline, atol=args.atol, rtol=args.rtol)
    for p in problems:
        print(f"  {p}")
    print(
        "OK: metrics match baseline within tolerance" if ok else f"FAIL: {len(problems)} problem(s)"
    )
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evalkit",
        description="Reproducible rolling-origin forecasting evaluation with leakage detection.",
    )
    p.add_argument("--version", action="version", version=f"evalkit {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("data", help="generate the synthetic demand dataset")
    sp.add_argument("-o", "--output", default="data/demand.csv")
    sp.add_argument("--seed", type=int, default=42)

    sp = sub.add_parser("quality", help="run data-quality checks")
    sp.add_argument("data")

    sp = sub.add_parser("run", help="run the backtest and write metrics + report + manifest")
    sp.add_argument("--config", required=True)
    sp.add_argument("--data", required=True)
    sp.add_argument("-o", "--output", default="reports")

    sp = sub.add_parser("report", help="print median metrics by model")
    sp.add_argument("metrics", help="a metrics.csv")

    sp = sub.add_parser("gate", help="compare metrics.csv to a baseline within tolerance")
    sp.add_argument("metrics")
    sp.add_argument("--baseline", required=True)
    sp.add_argument("--atol", type=float, default=1e-4)
    sp.add_argument("--rtol", type=float, default=1e-4)

    return p


_HANDLERS = {
    "data": cmd_data,
    "quality": cmd_quality,
    "run": cmd_run,
    "report": cmd_report,
    "gate": cmd_gate,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _HANDLERS[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
