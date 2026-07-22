"""Data-quality checks run before a backtest."""

from __future__ import annotations

from typing import Any

import pandas as pd

REQUIRED = ["unique_id", "ds", "y", "promo"]


def check_quality(df: pd.DataFrame) -> tuple[list[str], dict[str, Any]]:
    """Return (issues, summary). Fatal issues are duplicates, negatives, or
    missing columns; calendar gaps are reported but non-fatal."""
    issues: list[str] = []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        issues.append(f"missing required columns: {missing}")
        return issues, {"fatal": True, "missing_columns": missing}

    duplicates = int(df.duplicated(subset=["unique_id", "ds"]).sum())
    if duplicates:
        issues.append(f"{duplicates} duplicate (unique_id, ds) row(s)")
    negatives = int((df["y"] < 0).sum())
    if negatives:
        issues.append(f"{negatives} negative y value(s)")

    gaps = 0
    for _uid, g in df.groupby("unique_id"):
        ds = g["ds"].to_list()
        expected = set(range(int(min(ds)), int(max(ds)) + 1))
        gaps += len(expected - set(int(d) for d in ds))
    if gaps:
        issues.append(f"{gaps} missing calendar slot(s) across series")

    summary: dict[str, Any] = {
        "rows": int(len(df)),
        "series": int(df["unique_id"].nunique()),
        "duplicates": duplicates,
        "negatives": negatives,
        "calendar_gaps": gaps,
        "fatal": bool(duplicates or negatives),
    }
    return issues, summary
