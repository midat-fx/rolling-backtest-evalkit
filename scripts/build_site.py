"""Assemble the GitHub Pages site from finished runs.

    python scripts/build_site.py site out/hospital out/tourism_monthly out/synthetic ...

Copies each run's report.html to site/<run>.html and writes an index linking
them with their run_id and leakage status, read from each manifest.json.
"""

from __future__ import annotations

import html
import json
import shutil
import sys
from pathlib import Path


def main(site: str, runs: list[str]) -> None:
    out = Path(site)
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for run in runs:
        name = Path(run).name
        manifest = json.loads((Path(run) / "manifest.json").read_text(encoding="utf-8"))
        shutil.copy(Path(run) / "report.html", out / f"{name}.html")
        status = "leakage_ok" if manifest["leakage_ok"] else "LEAKAGE DETECTED (planted demo)"
        items.append(
            f'<li><a href="{name}.html">{html.escape(name)}</a> — run_id '
            f"<code>{manifest['run_id']}</code>, {len(manifest['models'])} models, "
            f"{manifest['folds']} folds, {status}</li>"
        )
    (out / "index.html").write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>evalkit reports</title><style>body{font:16px/1.6 system-ui,sans-serif;"
        "max-width:760px;margin:40px auto;padding:0 16px;background:#fff;color:#1d2230}"
        "@media (prefers-color-scheme:dark){body{background:#14171d;color:#e6e8ec}"
        "a{color:#7ab8ff}}</style></head><body><h1>evalkit reports</h1>"
        "<p>Rolling-origin backtests built by CI from public data. Source: "
        '<a href="https://github.com/midat-fx/rolling-backtest-evalkit">'
        "midat-fx/rolling-backtest-evalkit</a>.</p><ul>" + "".join(items) + "</ul></body></html>\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
