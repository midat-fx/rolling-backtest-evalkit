"""Write and render backtest metrics deterministically."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .backtest import METRIC_NAMES, Detail
from .stats import bootstrap_ci, dm_matrix, friedman_nemenyi, holm


def write_metrics(bt: pd.DataFrame, path: str) -> None:
    ordered = bt.sort_values(["fold", "model"]).reset_index(drop=True)
    ordered.to_csv(path, index=False, float_format="%.6f")


def per_model_summary(bt: pd.DataFrame) -> pd.DataFrame:
    summary = bt.groupby("model")[METRIC_NAMES].median().reset_index()
    return summary.sort_values("wape").reset_index(drop=True)


def significance(detail: Detail, metric: str = "mase", alpha: float = 0.05) -> dict[str, Any]:
    """Bootstrap CIs, ranks and pairwise DM tests on per-series ``metric``.

    Each series contributes one loss per model: its metric averaged over folds.
    DM p-values are Holm-adjusted across all model pairs.
    """
    loss = detail.metric(metric).mean(axis=1)  # (S, M)
    names = detail.models
    ci = {m: bootstrap_ci(loss[:, j]) for j, m in enumerate(names)}
    ranks = friedman_nemenyi(loss, names, alpha)
    dm = dm_matrix(loss, names)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
    adj = dict(zip(pairs, holm([dm[a][b].pvalue for a, b in pairs]), strict=True))
    best = min(names, key=lambda m: ci[m][0])
    return {
        "metric": metric,
        "n_series": int(loss.shape[0]),
        "alpha": alpha,
        "best": best,
        "mean_ci": {m: {"mean": v[0], "lo": v[1], "hi": v[2]} for m, v in ci.items()},
        "avg_rank": ranks.avg_rank,
        "critical_difference": ranks.critical_difference,
        "friedman_pvalue": ranks.friedman_pvalue,
        "dm": {
            a: {
                b: {
                    "stat": r.stat,
                    "pvalue": r.pvalue,
                    "pvalue_holm": adj.get((a, b), adj.get((b, a), 1.0)),
                }
                for b, r in row.items()
            }
            for a, row in dm.items()
        },
    }


def _p(p: float) -> str:
    return "< 1e-16" if p < 1e-16 else f"{p:.2g}"


def render_significance(sig: dict[str, Any]) -> list[str]:
    m, best, alpha = sig["metric"], sig["best"], sig["alpha"]
    lines = [
        f"## Significance (per-series {m.upper()}, {sig['n_series']} series)",
        "",
        f"| model | mean {m} | 95% bootstrap CI | avg rank | DM vs {best} (Holm p) |",
        "|---|---|---|---|---|",
    ]
    for name in sorted(sig["mean_ci"], key=lambda k: sig["mean_ci"][k]["mean"]):
        c = sig["mean_ci"][name]
        if name == best:
            vs = "—"
        else:
            p = sig["dm"][best][name]["pvalue_holm"]
            vs = _p(p) + (" *" if p < alpha else "")
        rank = sig["avg_rank"][name]
        lines.append(
            f"| {name} | {c['mean']:.4f} | [{c['lo']:.4f}, {c['hi']:.4f}] | {rank:.2f} | {vs} |"
        )
    lines += [
        "",
        f"Friedman (Iman-Davenport) p = {_p(sig['friedman_pvalue'])}; Nemenyi critical "
        f"difference = {sig['critical_difference']:.3f} ranks. `*` = DM rejects equal "
        f"accuracy with {best} at {alpha} after Holm adjustment.",
        "",
    ]
    return lines


def render_report(bt: pd.DataFrame, sig: dict[str, Any] | None = None) -> str:
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
    if sig is not None:
        lines += render_significance(sig)
    return "\n".join(lines) + "\n"
