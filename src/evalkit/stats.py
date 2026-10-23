"""Is the difference between two models signal or noise?

- ``diebold_mariano``: the DM test on a loss-differential series with the
  Harvey, Leybourne & Newbold (1997) small-sample correction and a Student-t
  reference, for h-step forecasts (autocovariances up to lag h-1).
- ``bootstrap_ci``: a percentile bootstrap of a mean, resampling series.
- ``friedman_nemenyi``: average ranks across series, the Iman-Davenport
  Friedman test and the Nemenyi critical difference (Demšar, 2006).

In the panel the unit of observation is the series: each series' loss is first
averaged over its folds and horizon steps, which absorbs the serial correlation
inside a series, and series are treated as independent. The distribution
functions are implemented here (regularised incomplete beta, studentised range)
so the core still needs only numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def _betacf(a: float, b: float, x: float) -> float:
    # Continued fraction for the incomplete beta (Numerical Recipes, betacf).
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf(t: float, df: float) -> float:
    """P(T > t) for Student's t with df degrees of freedom."""
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return tail if t >= 0 else 1.0 - tail


def f_sf(x: float, d1: float, d2: float) -> float:
    """P(F > x) for the F distribution."""
    if x <= 0:
        return 1.0
    return betainc(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * x))


@dataclass
class DMResult:
    stat: float  # HLN-corrected DM statistic; < 0 means the first model is better
    pvalue: float  # two-sided, t(n-1)
    mean_diff: float


def diebold_mariano(loss1: Array, loss2: Array, h: int = 1) -> DMResult:
    """DM test of equal predictive accuracy on paired losses.

    ``loss1``/``loss2`` are per-period losses (e.g. squared or absolute errors)
    of two forecasts of the same targets. For h-step forecasts the variance of
    the mean differential uses autocovariances up to lag h-1.
    """
    d = np.asarray(loss1, dtype=np.float64) - np.asarray(loss2, dtype=np.float64)
    n = d.size
    if n < 2:
        raise ValueError("need at least two loss differentials")
    dbar = float(d.mean())
    dc = d - dbar
    gamma = [float(np.dot(dc[k:], dc[: n - k]) / n) for k in range(h)]
    var = (gamma[0] + 2.0 * sum(gamma[1:])) / n
    if var <= 0:
        return DMResult(0.0 if dbar == 0 else math.copysign(math.inf, dbar), float(dbar == 0), dbar)
    dm = dbar / math.sqrt(var)
    hln = dm * math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return DMResult(hln, min(1.0, 2.0 * t_sf(abs(hln), n - 1)), dbar)


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def dm_matrix(loss: Array, names: list[str]) -> dict[str, dict[str, DMResult]]:
    """Pairwise DM over series. ``loss`` is (S, M): one loss per series and model."""
    out: dict[str, dict[str, DMResult]] = {a: {} for a in names}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i != j:
                out[a][b] = diebold_mariano(loss[:, i], loss[:, j], h=1)
    return out


def bootstrap_ci(
    values: Array, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """(mean, lo, hi): percentile bootstrap CI of the mean, resampling rows."""
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


def _norm_cdf(x: Array) -> Array:
    erf = np.vectorize(math.erf)
    return np.asarray(0.5 * (1.0 + erf(x / math.sqrt(2.0))), dtype=np.float64)


def studentized_range_cdf(q: float, k: int) -> float:
    """P(range of k iid N(0,1) < q), by quadrature (infinite degrees of freedom)."""
    z = np.linspace(-8.0, 8.0, 4001)
    pdf = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    integrand = k * pdf * (_norm_cdf(z + q) - _norm_cdf(z)) ** (k - 1)
    return float(np.sum(integrand) * (z[1] - z[0]))


def nemenyi_q(k: int, alpha: float = 0.05) -> float:
    """Critical value q_alpha of the Nemenyi test (studentised range / sqrt 2)."""
    lo, hi = 0.0, 10.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if studentized_range_cdf(mid, k) < 1.0 - alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi) / math.sqrt(2.0)


@dataclass
class RankResult:
    avg_rank: dict[str, float]
    critical_difference: float
    friedman_pvalue: float  # Iman-Davenport F version


def friedman_nemenyi(loss: Array, names: list[str], alpha: float = 0.05) -> RankResult:
    """Average ranks (1 = best) over series and the Nemenyi critical difference."""
    n, k = loss.shape
    # Average ranks with ties shared: rank = 1 + #smaller + (#equal - 1) / 2.
    less = (loss[:, None, :] < loss[:, :, None]).sum(axis=2)
    equal = (loss[:, None, :] == loss[:, :, None]).sum(axis=2)
    ranks = 1.0 + less + (equal - 1) / 2.0
    avg = ranks.mean(axis=0)
    chi2 = 12.0 * n / (k * (k + 1)) * (float(np.sum(avg**2)) - k * (k + 1) ** 2 / 4.0)
    denom = n * (k - 1) - chi2
    ff = (n - 1) * chi2 / denom if denom > 0 else math.inf
    p = 0.0 if math.isinf(ff) else f_sf(ff, k - 1, (k - 1) * (n - 1))
    cd = nemenyi_q(k, alpha) * math.sqrt(k * (k + 1) / (6.0 * n))
    return RankResult(dict(zip(names, map(float, avg), strict=True)), cd, p)
