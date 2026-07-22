from __future__ import annotations

import pandas as pd

from evalkit.quality import check_quality


def _df(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["unique_id", "ds", "y", "promo"])


def test_clean_data() -> None:
    issues, summary = check_quality(_df([("a", 0, 1, 0), ("a", 1, 2, 0)]))
    assert issues == [] and summary["fatal"] is False


def test_duplicate_key_is_fatal() -> None:
    issues, summary = check_quality(_df([("a", 0, 1, 0), ("a", 0, 2, 0)]))
    assert summary["duplicates"] == 1 and summary["fatal"] is True


def test_negative_is_fatal() -> None:
    _, summary = check_quality(_df([("a", 0, -1, 0)]))
    assert summary["negatives"] == 1 and summary["fatal"] is True


def test_calendar_gap_is_reported_not_fatal() -> None:
    _, summary = check_quality(_df([("a", 0, 1, 0), ("a", 2, 1, 0)]))
    assert summary["calendar_gaps"] == 1 and summary["fatal"] is False


def test_missing_column() -> None:
    issues, summary = check_quality(pd.DataFrame({"unique_id": ["a"], "ds": [0]}))
    assert summary["fatal"] is True and any("missing" in i for i in issues)
