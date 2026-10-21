"""Descriptive, time-structure and drought-run statistics."""

import numpy as np
import pandas as pd
import pymannkendall as mk
from scipy import signal, stats
from scipy.optimize import minimize_scalar
from statsmodels.nonparametric.smoothers_lowess import lowess


def describe(values):
    """Summary statistics of a 1-D sample."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    q = np.percentile(x, [5, 25, 50, 75, 95, 99])
    return pd.Series({
        "n": x.size,
        "mean": x.mean(),
        "sd": x.std(ddof=1),
        "median": q[2],
        "mad": stats.median_abs_deviation(x, scale="normal"),
        "skewness": stats.skew(x, bias=False),
        "ex_kurtosis": stats.kurtosis(x, bias=False),
        "min": x.min(),
        "q05": q[0],
        "q25": q[1],
        "q75": q[3],
        "q95": q[4],
        "q99": q[5],
        "max": x.max(),
    })


def tau_ar1(t, x):
    """Persistence time of an unevenly spaced AR(1) process.

    Least-squares estimator of Mudelsee (2002): minimizes
    sum_i [z_i - exp(-dt_i / tau) z_{i-1}]^2 on the standardized series.
    """
    t = np.asarray(t, dtype=float)
    z = np.asarray(x, dtype=float)
    z = (z - z.mean()) / z.std()
    dt = np.diff(t)
    keep = dt > 0
    z0, z1, dt = z[:-1][keep], z[1:][keep], dt[keep]

    def sse(log_tau):
        a = np.exp(-dt / np.exp(log_tau))
        return np.sum((z1 - a * z0) ** 2)

    res = minimize_scalar(sse, bounds=(np.log(0.05), np.log(2000.0)),
                          method="bounded")
    return float(np.exp(res.x))


def ar1_surrogate(t, tau, rng):
    """Unit-variance AR(1) realization on an uneven time axis."""
    t = np.asarray(t, dtype=float)
    out = np.empty(t.size)
    out[0] = rng.standard_normal()
    a = np.exp(-np.diff(t) / tau)
    eps = rng.standard_normal(t.size - 1)
    for i in range(1, t.size):
        out[i] = a[i - 1] * out[i - 1] + np.sqrt(1 - a[i - 1] ** 2) * \
            eps[i - 1]
    return out


def mann_kendall(x):
    """Hamed-Rao modified Mann-Kendall test for autocorrelated data."""
    res = mk.hamed_rao_modification_test(np.asarray(x, dtype=float))
    return {"trend": res.trend, "p": res.p, "z": res.z, "tau": res.Tau,
            "sen_slope_per_step": res.slope}


def pettitt(x):
    """Pettitt (1979) single change-point test."""
    x = np.asarray(x, dtype=float)
    n = x.size
    sgn = np.sign(x[:, None] - x[None, :])
    u = np.array([sgn[:k + 1, k + 1:].sum() for k in range(n - 1)])
    k = int(np.argmax(np.abs(u)))
    big_k = np.abs(u[k])
    p = min(1.0, 2.0 * np.exp(-6.0 * big_k ** 2 / (n ** 3 + n ** 2)))
    return {"index": k + 1, "K": float(big_k), "p": float(p)}


def binary_segmentation(x, min_size=30, penalty=None):
    """Detect mean shifts by binary segmentation with a BIC penalty.

    A split is accepted when it lowers the Gaussian log-likelihood cost by
    more than ``penalty`` (default 2 ln n, one new mean plus one break).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    pen = 2.0 * np.log(n) if penalty is None else penalty
    var = np.var(np.diff(x)) / 2.0

    def cost(seg):
        return np.sum((seg - seg.mean()) ** 2) / var

    breaks = []
    stack = [(0, n)]
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2 * min_size:
            continue
        seg = x[lo:hi]
        full = cost(seg)
        best, best_k = np.inf, None
        for k in range(min_size, hi - lo - min_size + 1):
            c = cost(seg[:k]) + cost(seg[k:])
            if c < best:
                best, best_k = c, k
        if best_k is not None and full - best > pen:
            breaks.append(lo + best_k)
            stack += [(lo, lo + best_k), (lo + best_k, hi)]
    return sorted(breaks)


def lomb_scargle(t, x, periods):
    """Normalized Lomb-Scargle power at the requested periods."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    omega = 2.0 * np.pi / np.asarray(periods, dtype=float)
    return signal.lombscargle(t, x - x.mean(), omega, normalize=True)


def red_noise_levels(t, tau, periods, n_sim, rng, levels=(95, 99)):
    """Percentiles of Lomb-Scargle power for AR(1) surrogates."""
    sims = np.empty((n_sim, len(periods)))
    for i in range(n_sim):
        sims[i] = lomb_scargle(t, ar1_surrogate(t, tau, rng), periods)
    return {lv: np.percentile(sims, lv, axis=0) for lv in levels}


def background(years, values, window_yr, robust=True):
    """Robust LOWESS background with a window of about window_yr."""
    years = np.asarray(years, dtype=float)
    frac = min(1.0, window_yr / (years.max() - years.min()))
    fit = lowess(values, years, frac=frac, it=3 if robust else 0,
                 return_sorted=False)
    return np.asarray(fit)


def drought_runs(years, values, threshold, step, pool_bins=1):
    """Runs of consecutive bins above a threshold (run theory).

    Runs separated by at most ``pool_bins`` sub-threshold bins are pooled.
    Severity is the summed exceedance times the bin width (permil yr).
    """
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    above = values > threshold
    idx = np.flatnonzero(above)
    if idx.size == 0:
        return pd.DataFrame()
    groups = [[idx[0]]]
    for i in idx[1:]:
        if i - groups[-1][-1] - 1 <= pool_bins:
            groups[-1].append(i)
        else:
            groups.append([i])
    rows = []
    for g in groups:
        lo, hi = g[0], g[-1]
        seg = values[lo:hi + 1]
        exc = np.clip(seg - threshold, 0.0, None)
        rows.append({
            "start_ce": years[lo] - step / 2.0,
            "end_ce": years[hi] + step / 2.0,
            "duration_yr": (hi - lo + 1) * step,
            "severity": exc.sum() * step,
            "peak": seg.max(),
            "peak_age_ce": years[lo + int(np.argmax(seg))],
        })
    runs = pd.DataFrame(rows)
    runs["intensity"] = runs["severity"] / runs["duration_yr"]
    return runs


def percentile_rank(sample, value):
    """Fraction of the sample not exceeding value, in percent."""
    sample = np.asarray(sample, dtype=float)
    return 100.0 * np.mean(sample <= value)
