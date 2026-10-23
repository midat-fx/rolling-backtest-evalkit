"""Public benchmark datasets behind a SHA-256 checksum.

Datasets come from the Monash Time Series Forecasting Repository (Godahewa et
al., 2021), hosted on Zenodo under CC BY 4.0. Each entry pins the archive's
SHA-256: a download that does not match is rejected, never parsed. Archives are
cached under ``$EVALKIT_CACHE`` (default ``~/.cache/evalkit``); with
``offline=True`` nothing is downloaded and a missing archive is an error, which
is how the test suite runs.

Series become the evalkit panel schema ``unique_id, ds, y, promo`` with ``ds``
an integer period index from each series' start and ``promo = 0`` (these
datasets carry no covariates).
"""

from __future__ import annotations

import hashlib
import io
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Dataset:
    name: str
    url: str
    sha256: str
    member: str
    season: int
    description: str


DATASETS: dict[str, Dataset] = {
    d.name: d
    for d in [
        Dataset(
            "hospital",
            "https://zenodo.org/api/records/4656014/files/hospital_dataset.zip/content",
            "8fd4085efb4517508e30fc69247eee9b74b300765bc6e94482c5d250efba66e5",
            "hospital_dataset.tsf",
            12,
            "767 monthly patient-count series for medical products, 2000-2006 (84 months)",
        ),
        Dataset(
            "tourism_monthly",
            "https://zenodo.org/api/records/4656096/files/tourism_monthly_dataset.zip/content",
            "6e434ad8a0ef6acb55ebee0e1d1ffe22707ea4e62c05e091326267f3a5b4a93c",
            "tourism_monthly_dataset.tsf",
            12,
            "366 monthly tourism-demand series from the Kaggle tourism competition (91-333 months)",
        ),
    ]
}


class ChecksumError(ValueError):
    pass


def cache_dir() -> Path:
    return Path(os.environ.get("EVALKIT_CACHE", Path.home() / ".cache" / "evalkit"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, sha256: str) -> Path:
    """Download url to dest, verifying the checksum before the file is kept."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as f:  # noqa: S310
        f.write(resp.read())
    got = sha256_file(tmp)
    if got != sha256:
        tmp.unlink()
        raise ChecksumError(f"{url}: sha256 {got} != expected {sha256}")
    tmp.replace(dest)
    return dest


def parse_tsf(text: str) -> pd.DataFrame:
    """Parse a Monash .tsf file (header lines, then ``name:start:v1,v2,...``)."""
    ids: list[str] = []
    ds: list[int] = []
    ys: list[float] = []
    in_data = False
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not in_data:
            in_data = line.lower() == "@data"
            continue
        name, _, values = line.partition(":")
        values = values.rsplit(":", 1)[-1]  # drop the start timestamp, if present
        series = [float(v) for v in values.split(",")]
        ids.extend([name] * len(series))
        ds.extend(range(len(series)))
        ys.extend(series)
    if not ids:
        raise ValueError("no series found after @data")
    return pd.DataFrame({"unique_id": ids, "ds": ds, "y": ys, "promo": 0})


def load(name: str, offline: bool = False, root: Path | None = None) -> pd.DataFrame:
    """Return the named dataset as a panel, downloading (and verifying) if needed."""
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}")
    d = DATASETS[name]
    archive = (root or cache_dir()) / f"{name}.zip"
    if archive.exists():
        got = sha256_file(archive)
        if got != d.sha256:
            raise ChecksumError(f"cached {archive}: sha256 {got} != expected {d.sha256}")
    elif offline:
        raise FileNotFoundError(f"{archive} not cached and offline=True")
    else:
        download(d.url, archive, d.sha256)
    with zipfile.ZipFile(archive) as zf:
        text = io.TextIOWrapper(zf.open(d.member), encoding="latin-1").read()
    return parse_tsf(text)
