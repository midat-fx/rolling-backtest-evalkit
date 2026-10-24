"""Property-based tests: invariants that must hold for any input, not just the
hand-picked ones."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from evalkit import metrics as M
from evalkit.backtest import Detail, rolling_origins
from evalkit.leakage import detect_model_future_access
from evalkit.models import REGISTRY, build_models
from evalkit.stats import bootstrap_ci, diebold_mariano, holm

positive = st.floats(0.1, 1e4, allow_nan=False, allow_infinity=False)


def vec(n: int) -> st.SearchStrategy[np.ndarray]:
    return arrays(np.float64, n, elements=positive)


@given(vec(12), vec(12), st.floats(0.01, 100))
def test_wape_is_scale_invariant_and_nonnegative(y: np.ndarray, p: np.ndarray, c: float) -> None:
    assert M.wape(y, p) >= 0
    assert np.isclose(M.wape(c * y, c * p), M.wape(y, p), rtol=1e-9)
    assert M.wape(y, y) == 0


@given(vec(12), vec(12))
def test_smape_bounded_and_symmetric(y: np.ndarray, p: np.ndarray) -> None:
    s = M.smape(y, p)
    assert 0 <= s <= 2
    assert np.isclose(s, M.smape(p, y))


@given(vec(12), vec(12), positive)
def test_rmsse_dominates_mase_on_same_scale(y: np.ndarray, p: np.ndarray, s: float) -> None:
    # RMSE >= MAE, so with matching scales (MSE scale = MAE scale^2) RMSSE >= MASE.
    assert M.rmsse(y, p, s * s) >= M.mase(y, p, s) - 1e-9
    assert M.mase(y, y, s) == 0


@given(vec(30), st.integers(1, 12))
def test_seasonal_scale_matches_definition(t: np.ndarray, m: int) -> None:
    assert np.isclose(M.naive_mae_scale(t, m), max(np.mean(np.abs(t[m:] - t[:-m])), 1e-9))
    assert np.isclose(M.naive_mse_scale(t, m), max(np.mean((t[m:] - t[:-m]) ** 2), 1e-9))


@given(st.data())
def test_vectorised_metrics_equal_reference_functions(data: st.DataObject) -> None:
    s, f, m, h = 2, 2, 2, 3
    y_true = data.draw(arrays(np.float64, (s, f, h), elements=positive))
    y_hat = data.draw(arrays(np.float64, (s, f, m, h), elements=positive))
    scale = data.draw(arrays(np.float64, (s, f), elements=positive))
    detail = Detail(["a", "b"], ["x", "y"], y_true, y_hat, scale, scale * scale)
    ref = {
        "wape": lambda yt, yh, i, j: M.wape(yt, yh),
        "smape": lambda yt, yh, i, j: M.smape(yt, yh),
        "bias": lambda yt, yh, i, j: M.bias(yt, yh),
        "mase": lambda yt, yh, i, j: M.mase(yt, yh, scale[i, j]),
        "rmsse": lambda yt, yh, i, j: M.rmsse(yt, yh, scale[i, j] ** 2),
    }
    for name, fn in ref.items():
        got = detail.metric(name)
        for i in range(s):
            for j in range(f):
                for k in range(m):
                    want = fn(y_true[i, j], y_hat[i, j, k], i, j)
                    assert np.isclose(got[i, j, k], want, rtol=1e-9), name


@given(st.integers(0, 200), st.integers(1, 100), st.integers(1, 24), st.integers(1, 24))
def test_rolling_origins_invariants(n: int, min_train: int, horizon: int, step: int) -> None:
    origins = rolling_origins(n, min_train, horizon, step)
    expected = max(0, (n - min_train - horizon) // step + 1)
    assert len(origins) == expected
    assert all(o >= min_train and o + horizon <= n for o in origins)
    assert all(b - a == step for a, b in zip(origins, origins[1:], strict=False))


HONEST = sorted(m for m in REGISTRY if not m.startswith(("leak_", "_")))


@settings(max_examples=25, deadline=None)
@given(arrays(np.float64, 60, elements=st.floats(0, 500)), st.integers(30, 55))
def test_no_honest_model_reads_the_future(y: np.ndarray, origin: int) -> None:
    models = build_models({"horizon": 4, "season": 12, "ridge_lags": [4, 12], "models": HONEST})
    promo = np.zeros_like(y)
    for name, model in models.items():
        horizon = min(4, len(y) - origin)
        fc = model(y, promo, origin, horizon)
        assert fc.shape == (horizon,) and np.all(np.isfinite(fc)), name
        assert not detect_model_future_access(model, y, promo, origin, horizon).leaked, name


@given(arrays(np.float64, 20, elements=positive), arrays(np.float64, 20, elements=positive))
def test_dm_antisymmetric_and_valid_pvalue(a: np.ndarray, b: np.ndarray) -> None:
    ab, ba = diebold_mariano(a, b), diebold_mariano(b, a)
    assert 0.0 <= ab.pvalue <= 1.0
    assert np.isclose(ab.pvalue, ba.pvalue) and np.isclose(ab.mean_diff, -ba.mean_diff)


@given(st.lists(st.floats(0, 1), min_size=1, max_size=20))
def test_holm_never_below_raw_and_bounded(p: list[float]) -> None:
    adj = holm(p)
    assert all(r <= a + 1e-12 <= 1 + 1e-12 for r, a in zip(p, adj, strict=True))


@settings(max_examples=25)
@given(arrays(np.float64, st.integers(2, 60), elements=positive))
def test_bootstrap_interval_contains_sample_mean(v: np.ndarray) -> None:
    m, lo, hi = bootstrap_ci(v, n_boot=200)
    assert lo - 1e-9 <= m <= hi + 1e-9
