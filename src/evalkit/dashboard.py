"""A self-contained HTML report: inline CSS and SVG, no JavaScript, no CDN.

Sections: headline metrics with bootstrap CIs and ranks, WAPE by fold, error by
horizon step, average ranks with the Nemenyi critical difference, the pairwise
DM matrix, the leakage audit, and the manifest. Output depends only on its
inputs, so the same run renders byte-identical HTML.
"""

from __future__ import annotations

import html
import json
from typing import Any

import numpy as np
import pandas as pd

from .backtest import METRIC_NAMES, Detail
from .report import per_model_summary

# Okabe-Ito: distinguishable for common colour-vision deficiencies.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]
MUTED = "var(--muted-line)"

CSS = """
:root{--bg:#fff;--fg:#1d2230;--sub:#5b6375;--line:#e3e6ec;--muted-line:#c9ced8;
--card:#f7f8fa;--ok:#1a7f37;--bad:#c62828;--sig:#dbe9f7;--sig2:#f9e0dc}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#14171d;--fg:#e6e8ec;
--sub:#9aa2b1;--line:#2b303a;--muted-line:#454c59;--card:#1b1f27;--ok:#4cc26b;--bad:#ff6b6b;
--sig:#1e3550;--sig2:#4a2622}}
:root[data-theme=dark]{--bg:#14171d;--fg:#e6e8ec;--sub:#9aa2b1;--line:#2b303a;
--muted-line:#454c59;--card:#1b1f27;--ok:#4cc26b;--bad:#ff6b6b;--sig:#1e3550;--sig2:#4a2622}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",
sans-serif;margin:0;padding:24px 16px}main{max-width:1040px;margin:0 auto}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 8px}
.sub{color:var(--sub)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
gap:16px}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}
.scroll{overflow-x:auto}table{border-collapse:collapse;font-size:13px;font-variant-numeric:
tabular-nums}th,td{padding:4px 8px;border-bottom:1px solid var(--line);text-align:right;
white-space:nowrap}th:first-child,td:first-child{text-align:left}
.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}
td.s1{background:var(--sig)}td.s2{background:var(--sig2)}
svg{width:100%;height:auto}svg.rank{max-width:620px}svg text{fill:var(--sub);font-size:11px}
.legend span{display:inline-block;margin-right:12px;font-size:12px}
.legend i{display:inline-block;width:12px;height:3px;margin-right:4px;vertical-align:middle}
pre{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;
overflow-x:auto;font-size:12px}
"""


def _esc(x: Any) -> str:
    return html.escape(str(x))


def _colors(ranked: list[str]) -> dict[str, str]:
    """Best models get palette colours; naive is always shown; the rest are muted."""
    colors = {m: MUTED for m in ranked}
    top = [m for m in ranked if m != "naive"][: len(PALETTE) - 1]
    for m, c in zip(top, PALETTE, strict=False):
        colors[m] = c
    if "naive" in colors:
        colors["naive"] = PALETTE[-1]
    return colors


def line_chart(series: dict[str, list[float]], colors: dict[str, str], xlabel: str) -> str:
    """Inline SVG line chart; x = 1..n, one line per entry; muted lines drawn first."""
    w, h, pl, pr, pt, pb = 520, 260, 48, 12, 10, 34
    values = [v for vs in series.values() for v in vs if np.isfinite(v)]
    lo, hi = min(values), max(values)
    pad = (hi - lo) * 0.05 or 1.0
    lo, hi = lo - pad, hi + pad
    n = max(len(v) for v in series.values())

    def sx(i: int) -> float:
        return pl + (w - pl - pr) * (i / max(n - 1, 1))

    def sy(v: float) -> float:
        return pt + (h - pt - pb) * (1 - (v - lo) / (hi - lo))

    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{_esc(xlabel)} chart">']
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        y = sy(v)
        parts.append(
            f'<line x1="{pl}" x2="{w - pr}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--line)"/>'
            f'<text x="{pl - 6}" y="{y + 4:.1f}" text-anchor="end">{v:.3f}</text>'
        )
    for i in range(n):
        parts.append(f'<text x="{sx(i):.1f}" y="{h - pb + 16}" text-anchor="middle">{i + 1}</text>')
    parts.append(f'<text x="{(w + pl) / 2}" y="{h - 2}" text-anchor="middle">{_esc(xlabel)}</text>')
    order = sorted(series, key=lambda m: colors[m] != MUTED)  # muted underneath
    for m in order:
        pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(series[m]))
        width = 1.2 if colors[m] == MUTED else 2.2
        dash = ' stroke-dasharray="5 3"' if m == "naive" else ""
        parts.append(
            f'<polyline fill="none" stroke="{colors[m]}" stroke-width="{width}"{dash} '
            f'points="{pts}">'
            f"<title>{_esc(m)}</title></polyline>"
        )
    parts.append("</svg>")
    return "".join(parts)


def legend(colors: dict[str, str]) -> str:
    items = [
        f'<span><i style="background:{c}"></i>{_esc(m)}{" (dashed)" if m == "naive" else ""}</span>'
        for m, c in colors.items()
        if c != MUTED
    ]
    return '<div class="legend">' + "".join(items) + " <span>grey: other models</span></div>"


def rank_chart(avg_rank: dict[str, float], cd: float) -> str:
    """Average rank per model (1 = best) with the Nemenyi CD as a bar from the best."""
    names = sorted(avg_rank, key=lambda m: avg_rank[m])
    k = len(names)
    w, row, pl, pr = 520, 20, 130, 16
    h = row * k + 40
    lo, hi = 1.0, float(k)

    def sx(r: float) -> float:
        return pl + (w - pl - pr) * (r - lo) / max(hi - lo, 1e-9)

    best = avg_rank[names[0]]
    parts = [f'<svg class="rank" viewBox="0 0 {w} {h}" role="img" aria-label="average ranks">']
    parts.append(
        f'<rect x="{sx(best):.1f}" y="4" width="{sx(best + cd) - sx(best):.1f}" '
        f'height="{row * k}" fill="var(--sig)"><title>within CD of the best</title></rect>'
    )
    for i, m in enumerate(names):
        y = 4 + row * i + row / 2
        r = avg_rank[m]
        parts.append(
            f'<text x="{pl - 8}" y="{y + 4:.1f}" text-anchor="end">{_esc(m)}</text>'
            f'<line x1="{pl}" x2="{w - pr}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--line)"/>'
            f'<circle cx="{sx(r):.1f}" cy="{y:.1f}" r="5" fill="{PALETTE[0]}">'
            f"<title>{_esc(m)}: {r:.2f}</title></circle>"
        )
    for t in range(1, k + 1):
        parts.append(f'<text x="{sx(t):.1f}" y="{row * k + 22}" text-anchor="middle">{t}</text>')
    parts.append(
        f'<text x="{(w + pl) / 2}" y="{h - 2}" text-anchor="middle">average rank '
        f"(shaded: within CD = {cd:.3f} of the best)</text></svg>"
    )
    return "".join(parts)


def _p(p: float) -> str:
    return "&lt;1e-16" if p < 1e-16 else f"{p:.2g}"


def dm_table(sig: dict[str, Any], names: list[str]) -> str:
    """Row model vs column model: Holm p, coloured when the row is significantly better
    (blue) or worse (red)."""
    alpha = sig["alpha"]
    head = "".join(f"<th>{_esc(n)}</th>" for n in names)
    rows = []
    for a in names:
        cells = []
        for b in names:
            if a == b:
                cells.append("<td>—</td>")
                continue
            r = sig["dm"][a][b]
            cls = ""
            if r["pvalue_holm"] < alpha:
                cls = ' class="s1"' if r["stat"] < 0 else ' class="s2"'
            cells.append(f"<td{cls}>{_p(r['pvalue_holm'])}</td>")
        rows.append(f"<tr><td>{_esc(a)}</td>{''.join(cells)}</tr>")
    return f"<table><tr><th>row vs col</th>{head}</tr>{''.join(rows)}</table>"


def render_html(
    bt: pd.DataFrame,
    detail: Detail,
    sig: dict[str, Any],
    manifest: dict[str, Any],
    title: str = "evalkit report",
) -> str:
    summary = per_model_summary(bt).set_index("model")
    ranked = sorted(sig["mean_ci"], key=lambda m: sig["mean_ci"][m]["mean"])
    colors = _colors(ranked)
    metric = sig["metric"]
    best = sig["best"]

    head = "".join(f"<th>median {m}</th>" for m in METRIC_NAMES)
    rows = []
    for m in ranked:
        c = sig["mean_ci"][m]
        dm = "—" if m == best else _p(sig["dm"][best][m]["pvalue_holm"])
        cells = "".join(f"<td>{summary.loc[m, k]:.4f}</td>" for k in METRIC_NAMES)
        rows.append(
            f"<tr><td>{_esc(m)}</td>{cells}<td>{c['mean']:.4f} [{c['lo']:.4f}, {c['hi']:.4f}]"
            f"</td><td>{sig['avg_rank'][m]:.2f}</td><td>{dm}</td></tr>"
        )
    table = (
        f"<table><tr><th>model</th>{head}<th>mean {metric} [95% CI]</th><th>avg rank</th>"
        f"<th>DM vs {_esc(best)} (Holm p)</th></tr>{''.join(rows)}</table>"
    )

    by_fold = {m: bt[bt["model"] == m].sort_values("fold")["wape"].tolist() for m in ranked}
    step_err = detail.scaled_abs_error().mean(axis=(0, 1))  # (M, H)
    by_step = {m: step_err[detail.models.index(m)].tolist() for m in ranked}

    leaky = set(manifest.get("leaky_models", []))
    leak_rows = "".join(
        f'<tr><td>{_esc(m)}</td><td class="{"bad" if m in leaky else "ok"}">'
        f"{'LEAK' if m in leaky else 'clean'}</td></tr>"
        for m in detail.models
    )
    ok = bool(manifest.get("leakage_ok"))
    status = f'<span class="{"ok" if ok else "bad"}">leakage_ok = {str(ok).lower()}</span>'
    n_series, n_folds, horizon = len(detail.uids), detail.y_true.shape[1], detail.y_true.shape[2]

    body = f"""<main>
<h1>{_esc(title)}</h1>
<p class="sub">run_id <code>{_esc(manifest.get("run_id"))}</code> · {n_series} series ·
{n_folds} rolling origins · horizon {horizon} · {status}</p>
<h2>Models</h2>
<p class="sub">Sorted by mean per-series {metric}. CI: percentile bootstrap over series.
DM: Diebold-Mariano with the Harvey-Leybourne-Newbold correction, Holm-adjusted.
Friedman p = {_p(sig["friedman_pvalue"])}.</p>
<div class="card scroll">{table}</div>
<div class="grid">
<div><h2>WAPE by fold</h2><div class="card">{line_chart(by_fold, colors, "fold")}
{legend(colors)}</div></div>
<div><h2>Error by horizon step</h2><div class="card">
{line_chart(by_step, colors, "steps ahead (mean |error| / naive MAE)")}{legend(colors)}</div></div>
</div>
<h2>Average rank across series</h2>
<div class="card">{rank_chart(sig["avg_rank"], sig["critical_difference"])}</div>
<h2>Pairwise Diebold-Mariano (Holm p)</h2>
<p class="sub">Blue: the row model is significantly more accurate than the column model;
red: significantly less accurate (alpha = {sig["alpha"]}).</p>
<div class="card scroll">{dm_table(sig, ranked)}</div>
<h2>Leakage audit</h2>
<p class="sub">model_future_access: forecasts must not change when y[origin:] is altered.</p>
<div class="card scroll"><table><tr><th>model</th><th>verdict</th></tr>{leak_rows}</table></div>
<h2>Manifest</h2>
<pre>{_esc(json.dumps(manifest, sort_keys=True, indent=2))}</pre>
</main>"""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{_esc(title)}</title><style>{CSS}</style></head><body>{body}</body></html>\n"
    )
