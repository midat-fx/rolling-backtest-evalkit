from __future__ import annotations

import math

import numpy as np
import pytest

from evalkit.stats import (
    betainc,
    bootstrap_ci,
    diebold_mariano,
    dm_matrix,
    f_sf,
    friedman_nemenyi,
    holm,
    nemenyi_q,
    t_sf,
)


def test_distribution_functions_match_tables() -> None:
    assert math.isclose(t_sf(2.228, 10), 0.025, abs_tol=1e-4)  # t table, 97.5%, df=10
    assert math.isclose(t_sf(-2.228, 10), 0.975, abs_tol=1e-4)
    assert math.isclose(f_sf(3.0, 2, 10), 1.6**-5, rel_tol=1e-9)  # closed form for d1=2
    assert f_sf(0.0, 3, 5) == 1.0
    assert betainc(2, 3, 0.0) == 0.0 and betainc(2, 3, 1.0) == 1.0
    assert math.isclose(betainc(1, 1, 0.3), 0.3, rel_tol=1e-12)  # uniform
    # Demšar (2006), Table 5(a): q_0.05 for k = 2..10.
    table = [1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164]
    for k, q in zip(range(2, 11), table, strict=True):
        assert math.isclose(nemenyi_q(k), q, abs_tol=2e-3)


def test_dm_identical_and_clear_difference() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=200) ** 2
    same = diebold_mariano(a, a)
    assert same.stat == 0.0 and same.pvalue == 1.0
    better = diebold_mariano(a, a + 0.5 + rng.normal(0, 0.1, 200))
    assert better.stat < -10 and better.pvalue < 1e-10 and better.mean_diff < 0
    noise = diebold_mariano(a, rng.permutation(a))
    assert noise.pvalue > 0.01
    with pytest.raises(ValueError):
        diebold_mariano(np.ones(1), np.ones(1))


def test_dm_hln_by_hand() -> None:
    d = np.array([1.0, -0.5, 2.0, 0.5, 1.5, -1.0, 0.0, 1.0])
    n, h = d.size, 2
    dc = d - d.mean()
    g0, g1 = np.dot(dc, dc) / n, np.dot(dc[1:], dc[:-1]) / n
    dm = d.mean() / math.sqrt((g0 + 2 * g1) / n)
    hln = dm * math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    r = diebold_mariano(d, np.zeros(n), h=h)
    assert math.isclose(r.stat, hln, rel_tol=1e-12)
    assert math.isclose(r.pvalue, 2 * t_sf(abs(hln), n - 1), rel_tol=1e-12)


def test_dm_degenerate_constant_difference() -> None:
    r = diebold_mariano(np.ones(5), np.zeros(5))
    assert math.isinf(r.stat) and r.pvalue == 0.0


def test_holm() -> None:
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_bootstrap_ci_is_seeded_and_brackets_mean() -> None:
    v = np.random.default_rng(3).gamma(2.0, size=500)
    m, lo, hi = bootstrap_ci(v)
    assert lo < m < hi and math.isclose(m, v.mean())
    assert bootstrap_ci(v) == (m, lo, hi)
    assert hi - lo < bootstrap_ci(v[:50])[2] - bootstrap_ci(v[:50])[1]  # more data, tighter


def test_friedman_nemenyi_ranks_and_ties() -> None:
    loss = np.array([[1.0, 2.0, 3.0], [1.0, 3.0, 2.0], [2.0, 2.0, 3.0]])
    r = friedman_nemenyi(loss, ["a", "b", "c"])
    assert r.avg_rank == pytest.approx({"a": 7 / 6, "b": 13 / 6, "c": 8 / 3})
    assert sum(r.avg_rank.values()) == pytest.approx(6.0)  # ranks sum to k(k+1)/2
    assert r.critical_difference == pytest.approx(nemenyi_q(3) * math.sqrt(3 * 4 / 18))
    rng = np.random.default_rng(1)
    strong = rng.normal(size=(100, 3)) + np.array([0.0, 1.0, 2.0])
    assert friedman_nemenyi(strong, ["a", "b", "c"]).friedman_pvalue < 1e-10
    perfect = np.tile([1.0, 2.0], (5, 1))
    assert friedman_nemenyi(perfect, ["a", "b"]).friedman_pvalue == 0.0


def test_dm_matrix_is_antisymmetric() -> None:
    loss = np.random.default_rng(2).random((40, 3))
    dm = dm_matrix(loss, ["a", "b", "c"])
    assert dm["a"]["b"].stat == pytest.approx(-dm["b"]["a"].stat)
    assert dm["a"]["b"].pvalue == pytest.approx(dm["b"]["a"].pvalue)
