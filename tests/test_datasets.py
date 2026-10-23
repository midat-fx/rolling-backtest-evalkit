from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evalkit import datasets
from evalkit.backtest import run_backtest

FIXTURE = Path(__file__).parent / "fixtures" / "hospital_sample.tsf"


def _zip_fixture(tmp_path: Path, member: str) -> Path:
    archive = tmp_path / "src.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(FIXTURE, member)
    return archive


def test_parse_tsf_fixture() -> None:
    df = datasets.parse_tsf(FIXTURE.read_text(encoding="latin-1"))
    assert list(df.columns) == ["unique_id", "ds", "y", "promo"]
    assert df["unique_id"].nunique() == 8
    assert df.groupby("unique_id").size().eq(84).all()
    first = df[df["unique_id"] == "T1"]["y"].to_numpy()
    assert first[:3].tolist() == [27.0, 16.0, 18.0]  # the published first values


def test_parse_tsf_without_timestamp_and_empty() -> None:
    df = datasets.parse_tsf("@data\na:1,2,3\nb:4,5\n")
    assert df.groupby("unique_id").size().to_dict() == {"a": 3, "b": 2}
    with pytest.raises(ValueError):
        datasets.parse_tsf("# only a header\n@data\n")


def test_download_verifies_checksum(tmp_path: Path) -> None:
    src = _zip_fixture(tmp_path, "x.tsf")
    good = datasets.sha256_file(src)
    dest = datasets.download(src.as_uri(), tmp_path / "cache" / "x.zip", good)
    assert datasets.sha256_file(dest) == good
    with pytest.raises(datasets.ChecksumError):
        datasets.download(src.as_uri(), tmp_path / "cache" / "y.zip", "0" * 64)
    assert not (tmp_path / "cache" / "y.zip").exists()  # rejected files are not kept


def test_load_offline_from_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    member = datasets.DATASETS["hospital"].member
    src = _zip_fixture(tmp_path, member)
    fake = datasets.Dataset("hospital", src.as_uri(), datasets.sha256_file(src), member, 12, "")
    monkeypatch.setitem(datasets.DATASETS, "hospital", fake)
    root = tmp_path / "cache"
    with pytest.raises(FileNotFoundError):
        datasets.load("hospital", offline=True, root=root)
    df = datasets.load("hospital", root=root)  # downloads from the file:// url
    assert df["unique_id"].nunique() == 8
    assert datasets.load("hospital", offline=True, root=root).equals(df)
    (root / "hospital.zip").write_bytes(b"tampered")
    with pytest.raises(datasets.ChecksumError):
        datasets.load("hospital", offline=True, root=root)


def test_load_unknown_and_cache_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        datasets.load("nope")
    monkeypatch.setenv("EVALKIT_CACHE", str(tmp_path))
    assert datasets.cache_dir() == tmp_path


def test_backtest_aligns_unequal_series_on_their_end() -> None:
    long_y = np.arange(30, dtype=float) + 1.0
    df = pd.DataFrame(
        {
            "unique_id": ["long"] * 30 + ["short"] * 20,
            "ds": list(range(30)) + list(range(20)),
            "y": np.concatenate([long_y, long_y[10:]]),  # short == tail of long
            "promo": 0,
        }
    )
    cfg = {"min_train": 10, "horizon": 2, "step": 4, "season": 4, "ridge_lags": [2, 3]}
    bt = run_backtest(df, cfg)
    alone = run_backtest(df[df["unique_id"] == "short"], cfg)
    naive_both = bt[bt["model"] == "naive"]["wape"].to_numpy()
    naive_short = alone[alone["model"] == "naive"]["wape"].to_numpy()
    # Tail-aligned: the long series tests the same values, so naive errors agree.
    assert np.allclose(naive_both, naive_short)


def test_cli_fetch_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evalkit.cli import main

    member = datasets.DATASETS["hospital"].member
    cache = tmp_path / "cache"
    cache.mkdir()
    src = _zip_fixture(cache, member)
    src.rename(cache / "hospital.zip")
    sha = datasets.sha256_file(cache / "hospital.zip")
    monkeypatch.setitem(
        datasets.DATASETS, "hospital", datasets.Dataset("hospital", "", sha, member, 12, "")
    )
    monkeypatch.setenv("EVALKIT_CACHE", str(cache))
    out = tmp_path / "h.csv"
    assert main(["fetch", "hospital", "--offline", "-o", str(out)]) == 0
    assert pd.read_csv(out)["unique_id"].nunique() == 8
