"""Render the README figures (standalone SVG) from real runs.

    make fetch && python scripts/readme_figures.py

Writes docs/img/{leakage_hospital,ranks_hospital,horizon_tourism}.svg. The
charts reuse the HTML report's SVG code; CSS variables are bound inside each
file so they render the same on GitHub as in the report (light theme).
"""

from __future__ import annotations

import pandas as pd

from evalkit.backtest import aggregate, backtest_detail
from evalkit.dashboard import PALETTE, _colors, line_chart, rank_chart
from evalkit.report import per_model_summary, significance
from evalkit.run import load_config, model_leakage_audit

VARS = (
    "<style>svg{--line:#e3e6ec;--muted-line:#c9ced8;--sub:#5b6375;--sig:#dbe9f7;"
    "font-family:system-ui,-apple-system,'Segoe UI',sans-serif}text{fill:#5b6375;"
    "font-size:11px}.t{fill:#1d2230;font-size:13px;font-weight:600}</style>"
)


def standalone(svg: str, title: str, keys: dict[str, str] | None = None) -> str:
    """Bind the CSS variables, add a white background, a title row and an
    optional legend row (name -> colour) under the chart."""
    head, rest = svg.split(">", 1)
    vb = [float(v) for v in head.split('viewBox="')[1].split('"')[0].split()]
    body_h = vb[3]
    extra = ""
    if keys:
        extra = "".join(
            f'<rect x="{12 + 128 * (i % 4)}" y="{body_h + 6 + 16 * (i // 4)}" width="12" '
            f'height="3" fill="{c}"/><text x="{28 + 128 * (i % 4)}" '
            f'y="{body_h + 11 + 16 * (i // 4)}">{m}</text>'
            for i, (m, c) in enumerate(keys.items())
        )
        body_h += 16 * ((len(keys) + 3) // 4) + 4
    w, h = vb[2], body_h + 24
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -24 {w:g} {h:g}" '
        f'width="{w * 1.25:g}" height="{h * 1.25:g}">{VARS}'
        f'<rect x="0" y="-24" width="{w:g}" height="{h:g}" fill="#fff"/>'
        f'<text class="t" x="8" y="-8">{title}</text>{rest.replace("</svg>", extra + "</svg>")}'
    )


def leakage_bars(wape: dict[str, float], leaked: dict[str, bool]) -> str:
    names = list(wape)
    w, row, pl, pr = 560, 24, 150, 130
    h = row * len(names) + 8
    top = max(wape.values())
    parts = [f'<svg viewBox="0 0 {w} {h}">']
    for i, m in enumerate(names):
        y = 4 + i * row
        bw = (w - pl - pr) * wape[m] / top
        color = "#D55E00" if leaked[m] else PALETTE[0]
        tag = "LEAK caught" if leaked[m] else "clean"
        parts.append(
            f'<text x="{pl - 8}" y="{y + 15}" text-anchor="end">{m}</text>'
            f'<rect x="{pl}" y="{y + 3}" width="{bw:.1f}" height="{row - 8}" fill="{color}"/>'
            f'<text x="{pl + bw + 6:.1f}" y="{y + 15}">{wape[m]:.3f} · {tag}</text>'
        )
    return "".join(parts) + "</svg>"


def main() -> None:
    cfg = load_config("configs/hospital.yaml")
    df = pd.read_csv("data/hospital.csv")
    demo = {
        **cfg,
        "models": [
            "naive",
            "ridge_lags",
            "theta",
            "leak_global_norm",
            "leak_centered_window",
            "leak_target_feature",
        ],
    }
    summary = per_model_summary(aggregate(backtest_detail(df, demo, workers=8))).set_index("model")
    audit = model_leakage_audit(df, demo)
    wape = {m: float(summary.loc[m, "wape"]) for m in demo["models"]}
    svg = leakage_bars(wape, audit)
    with open("docs/img/leakage_hospital.svg", "w", encoding="utf-8") as f:
        f.write(standalone(svg, "Hospital (767 series): median WAPE, honest vs planted leaks"))

    detail = backtest_detail(df, cfg, workers=8)
    sig = significance(detail)
    svg = rank_chart(sig["avg_rank"], sig["critical_difference"])
    with open("docs/img/ranks_hospital.svg", "w", encoding="utf-8") as f:
        f.write(standalone(svg, "Hospital: average rank by per-series MASE (Nemenyi CD shaded)"))

    cfg = load_config("configs/tourism_monthly.yaml")
    detail = backtest_detail(pd.read_csv("data/tourism_monthly.csv"), cfg, workers=8)
    sig = significance(detail)
    ranked = sorted(sig["mean_ci"], key=lambda m: sig["mean_ci"][m]["mean"])
    colors = _colors(ranked)
    err = detail.scaled_abs_error().mean(axis=(0, 1))
    series = {m: err[detail.models.index(m)].tolist() for m in ranked}
    svg = line_chart(series, colors, "months ahead (mean |error| / seasonal-naive MAE)")
    keys = {m: c for m, c in colors.items() if c != "var(--muted-line)"}
    keys["naive (dashed)"] = keys.pop("naive")
    keys["other models"] = "#c9ced8"
    with open("docs/img/horizon_tourism.svg", "w", encoding="utf-8") as f:
        f.write(standalone(svg, "Tourism (366 series): error by months ahead", keys))


if __name__ == "__main__":
    main()
