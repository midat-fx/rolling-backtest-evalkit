from __future__ import annotations

from evalkit.datagen import generate


def test_shape_and_columns() -> None:
    df = generate(seed=42, n_sku=20, n_store=2, n_weeks=104)
    assert list(df.columns) == ["unique_id", "ds", "y", "promo"]
    assert len(df) == 20 * 2 * 104
    assert df["unique_id"].nunique() == 40


def test_deterministic() -> None:
    a = generate(seed=42)
    b = generate(seed=42)
    assert a.equals(b)


def test_no_negatives_and_promo_binary() -> None:
    df = generate(seed=1)
    assert (df["y"] >= 0).all()
    assert set(df["promo"].unique()) <= {0, 1}
