"""Write and render backtest metrics deterministically."""

from __future__ import annotations

import pandas as pd

from .backtest import METRIC_NAMES


def write_metrics(bt: pd.DataFrame, path: str) -> None:
    ordered = bt.sort_values(["fold", "model"]).reset_index(drop=True)
    ordered.to_csv(path, index=False, float_format="%.6f")


def per_model_summary(bt: pd.DataFrame) -> pd.DataFrame:
    summary = bt.groupby("model")[METRIC_NAMES].median().reset_index()
    return summary.sort_values("wape").reset_index(drop=True)


def render_report(bt: pd.DataFrame) -> str:
    summary = per_model_summary(bt)
    lines: list[str] = ["# Backtest report", "", "## Median metrics by model", ""]
    lines.append("| model | " + " | ".join(METRIC_NAMES) + " |")
    lines.append("|---|" + "|".join(["---"] * len(METRIC_NAMES)) + "|")
    for _, row in summary.iterrows():
        cells = " | ".join(f"{row[m]:.4f}" for m in METRIC_NAMES)
        lines.append(f"| {row['model']} | {cells} |")
    lines.append("")
    winner = str(summary.iloc[0]["model"])
    lines.append(f"**Winner by median WAPE:** {winner}")
    lines.append("")
    return "\n".join(lines) + "\n"
