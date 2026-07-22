"""Deterministic synthetic demand data.

A panel of SKU x store series with level, weekly seasonality, a mild trend,
random promotions with an uplift, and multiplicative noise. Seeded with numpy's
default_rng so re-running produces a byte-identical CSV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["unique_id", "ds", "y", "promo"]


def generate(seed: int = 42, n_sku: int = 20, n_store: int = 2, n_weeks: int = 104) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids: list[str] = []
    ds: list[int] = []
    ys: list[int] = []
    promos: list[int] = []
    for sku in range(n_sku):
        level = float(np.exp(rng.normal(3.0, 0.5)))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        trend = float(rng.normal(0.0, 0.02))
        for store in range(n_store):
            store_mult = 1.0 + 0.3 * store
            uid = f"sku{sku:02d}_st{store}"
            for w in range(n_weeks):
                seasonal = 1.0 + 0.3 * np.sin(2.0 * np.pi * w / 52.0 + phase)
                trend_factor = max(0.2, 1.0 + trend * w)
                promo = 1 if rng.random() < 0.05 else 0
                lift = float(rng.uniform(1.5, 2.5)) if promo else 1.0
                mean = max(1.0, level * store_mult * seasonal * trend_factor * lift)
                qty = int(round(max(0.0, rng.normal(mean, mean * 0.15))))
                ids.append(uid)
                ds.append(w)
                ys.append(qty)
                promos.append(promo)
    return pd.DataFrame({"unique_id": ids, "ds": ds, "y": ys, "promo": promos})


def write_csv(path: str, **kwargs: int) -> pd.DataFrame:
    df = generate(**kwargs)
    df.to_csv(path, index=False)
    return df
