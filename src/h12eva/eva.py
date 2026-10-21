"""Extreme value models built on pyextremes, plus Bayesian refits.

Sign conventions: the shape parameter xi is reported in the Coles (2001)
convention, xi < 0 meaning a bounded (Weibull-type) upper tail. SciPy and
pyextremes use c = xi for genpareto and c = -xi for genextreme.
"""

import warnings

import emcee
import numpy as np
from pyextremes import EVA, get_extremes
from scipy import stats

from h12eva import config as cfg
from h12eva import io

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RP_SIZE = "1D"  # one pseudo-day equals one calendar year


def span_years(years):
    """Record length used for event rates (same rule as pyextremes)."""
    years = np.asarray(years, dtype=float)
    return years.max() - years.min()


# ----------------------------------------------------------------------
# Peaks over threshold (generalized Pareto)
# ----------------------------------------------------------------------
def pot_model(years, values, threshold, decluster_yr):
    """Fit a GPD to declustered threshold excesses with pyextremes."""
    model = EVA(io.as_series(years, values))
    model.get_extremes("POT", threshold=threshold, r=f"{decluster_yr}D")
    model.fit_model("MLE", distribution="genpareto")
    return model


def pot_params(model):
    """GPD parameters (xi, sigma) from a fitted pyextremes model."""
    p = model.distribution.mle_parameters
    return float(p["c"]), float(p["scale"])


def gpd_return_period(level, u, xi, sigma, rate):
    """Return period (years) of exceeding ``level`` under a POT model."""
    sf = stats.genpareto.sf(level, xi, loc=u, scale=sigma)
    sf = np.asarray(sf, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(sf > 0, 1.0 / (rate * sf), np.inf)


def gpd_return_level(periods, u, xi, sigma, rate):
    """Return level for return periods (years) under a POT model."""
    periods = np.asarray(periods, dtype=float)
    p_exc = 1.0 / (rate * periods)
    p_exc = np.where(p_exc < 1.0, p_exc, np.nan)
    return stats.genpareto.isf(p_exc, xi, loc=u, scale=sigma)


def gpd_endpoint(u, xi, sigma):
    """Upper endpoint of the GPD (infinite for xi >= 0)."""
    return u - sigma / xi if xi < 0 else np.inf


def bootstrap_pot(excesses_abs, u, span, level, periods, rng,
                  n_boot=cfg.N_BOOT):
    """Nonparametric bootstrap of a POT fit.

    Resamples the declustered peaks with replacement and the number of
    peaks from a Poisson law, then records the shape, the return period of
    ``level`` and the return levels at ``periods``.
    """
    x = np.asarray(excesses_abs, dtype=float)
    n = x.size
    xis, t_level, rls = [], [], []
    for _ in range(n_boot):
        sample = rng.choice(x, size=n, replace=True)
        n_star = max(rng.poisson(n), 1)
        try:
            xi, _, sig = stats.genpareto.fit(sample, floc=u)
        except Exception:
            continue
        rate = n_star / span
        xis.append(xi)
        t_level.append(float(gpd_return_period(level, u, xi, sig, rate)))
        rls.append(gpd_return_level(periods, u, xi, sig, rate))
    return np.array(xis), np.array(t_level), np.array(rls)


# ----------------------------------------------------------------------
# Block maxima (generalized extreme value)
# ----------------------------------------------------------------------
def bm_model(years, values, block_yr):
    """Fit a GEV to block maxima with pyextremes."""
    model = EVA(io.as_series(years, values))
    model.get_extremes("BM", block_size=f"{block_yr}D")
    model.fit_model("MLE", distribution="genextreme")
    return model


def bm_params(model):
    """GEV parameters (xi, mu, sigma) in the Coles convention."""
    p = model.distribution.mle_parameters
    return -float(p["c"]), float(p["loc"]), float(p["scale"])


def gev_return_period(level, xi, mu, sigma, block_yr):
    """Return period (years) of exceeding ``level`` under a GEV model."""
    sf = np.asarray(stats.genextreme.sf(level, -xi, loc=mu, scale=sigma),
                    dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(sf > 0, block_yr / sf, np.inf)


def gev_return_level(periods, xi, mu, sigma, block_yr):
    """Return level for return periods (years) under a GEV model."""
    p = block_yr / np.asarray(periods, dtype=float)
    return stats.genextreme.isf(p, -xi, loc=mu, scale=sigma)


def bootstrap_bm(maxima, level, periods, block_yr, rng,
                 n_boot=cfg.N_BOOT):
    """Nonparametric bootstrap of a block-maxima GEV fit."""
    x = np.asarray(maxima, dtype=float)
    xis, t_level, rls = [], [], []
    for _ in range(n_boot):
        sample = rng.choice(x, size=x.size, replace=True)
        try:
            c, mu, sig = stats.genextreme.fit(sample)
        except Exception:
            continue
        xis.append(-c)
        t_level.append(float(gev_return_period(level, -c, mu, sig,
                                               block_yr)))
        rls.append(gev_return_level(periods, -c, mu, sig, block_yr))
    return np.array(xis), np.array(t_level), np.array(rls)


# ----------------------------------------------------------------------
# Bayesian fits with the Martins-Stedinger (2000) shape prior
# ----------------------------------------------------------------------
def ms_log_prior(xi):
    """Geophysical shape prior: (xi + 0.5) ~ Beta(9, 6) on (-0.5, 0.5).

    Martins and Stedinger (2000) place Beta(6, 9) on kappa = -xi, which
    favors moderately heavy tails (prior mean xi = +0.1). For a bounded
    proxy tail this pulls xi upward, which shortens return periods and
    makes extremeness claims conservative.
    """
    if not -0.5 < xi < 0.5:
        return -np.inf
    return 8.0 * np.log(0.5 + xi) + 5.0 * np.log(0.5 - xi)


def uniform_log_prior(xi):
    """Neutral prior: xi uniform on (-0.5, 0.5), the regular MLE range."""
    return 0.0 if -0.5 < xi < 0.5 else -np.inf


PRIORS = {"ms": ms_log_prior, "uniform": uniform_log_prior}


def _gpd_log_post(theta, exc, prior="ms"):
    xi, log_sig = theta
    lp = PRIORS[prior](xi)
    if not np.isfinite(lp):
        return -np.inf
    ll = stats.genpareto.logpdf(exc, xi, loc=0.0, scale=np.exp(log_sig))
    ll = np.sum(ll)
    return lp + ll if np.isfinite(ll) else -np.inf


def _gev_log_post(theta, x, prior="ms"):
    xi, mu, log_sig = theta
    lp = PRIORS[prior](xi)
    if not np.isfinite(lp):
        return -np.inf
    ll = np.sum(stats.genextreme.logpdf(x, -xi, loc=mu,
                                        scale=np.exp(log_sig)))
    return lp + ll if np.isfinite(ll) else -np.inf


def _run_emcee(log_post, start, args, rng):
    ndim = len(start)
    p0 = np.asarray(start) + 1e-3 * rng.standard_normal((cfg.N_WALKERS,
                                                        ndim))
    p0[:, 0] = np.clip(p0[:, 0], -0.49, 0.49)
    # emcee copies NumPy's global random state when the sampler is
    # built, so the seed must be set before construction.
    state = np.random.get_state()
    np.random.seed(int(rng.integers(0, 2 ** 31 - 1)))
    sampler = emcee.EnsembleSampler(cfg.N_WALKERS, ndim, log_post,
                                    args=args)
    sampler.run_mcmc(p0, cfg.N_STEPS, progress=False)
    np.random.set_state(state)
    return sampler.get_chain(discard=cfg.N_BURN, thin=5, flat=True)


def bayes_gpd(peaks, u, rng, prior="ms"):
    """Posterior samples (xi, sigma) of a GPD fitted to peak excesses."""
    exc = np.asarray(peaks, dtype=float) - u
    xi0, _, sig0 = stats.genpareto.fit(exc, floc=0.0)
    start = [float(np.clip(xi0, -0.45, 0.45)), np.log(sig0)]
    chain = _run_emcee(_gpd_log_post, start, (exc, prior), rng)
    return chain[:, 0], np.exp(chain[:, 1])


def bayes_gev(maxima, rng, prior="ms"):
    """Posterior samples (xi, mu, sigma) of a GEV fitted to maxima."""
    x = np.asarray(maxima, dtype=float)
    c0, mu0, sig0 = stats.genextreme.fit(x)
    start = [float(np.clip(-c0, -0.45, 0.45)), mu0, np.log(sig0)]
    chain = _run_emcee(_gev_log_post, start, (x, prior), rng)
    return chain[:, 0], chain[:, 1], np.exp(chain[:, 2])


# ----------------------------------------------------------------------
# Non-stationary GEV (location linear in time)
# ----------------------------------------------------------------------
def nonstationary_gev(covariate, maxima):
    """Compare a stationary GEV with mu = mu0 + mu1 * covariate.

    The covariate is centered before fitting. Returns parameter
    estimates and the likelihood-ratio test (1 degree of freedom).
    """
    from scipy.optimize import minimize

    x = np.asarray(maxima, dtype=float)
    cov = np.asarray(covariate, dtype=float)
    s = cov - cov.mean()
    c0, mu0, sig0 = stats.genextreme.fit(x)

    def nll(theta, linear):
        if linear:
            xi, m0, m1, ls = theta
            mu = m0 + m1 * s
        else:
            xi, m0, ls = theta
            mu = m0
        val = -np.sum(stats.genextreme.logpdf(x, -xi, loc=mu,
                                              scale=np.exp(ls)))
        return val if np.isfinite(val) else 1e10

    st = minimize(nll, [-c0, mu0, np.log(sig0)], args=(False,),
                  method="Nelder-Mead", options={"maxiter": 20000})
    ns = minimize(nll, [-c0, mu0, 0.0, np.log(sig0)], args=(True,),
                  method="Nelder-Mead", options={"maxiter": 20000})
    lr = 2.0 * (st.fun - ns.fun)
    return {
        "stationary": st.x, "nonstationary": ns.x,
        "nll_stationary": st.fun, "nll_nonstationary": ns.fun,
        "lr": lr, "p": float(stats.chi2.sf(max(lr, 0.0), df=1)),
        "covariate_mean": float(cov.mean()),
    }


def declustered_peaks(years, values, threshold, decluster_yr):
    """Declustered peaks (years, values) above a threshold."""
    ext = get_extremes(io.as_series(years, values), "POT",
                       threshold=threshold, r=f"{decluster_yr}D")
    return io.from_pseudo_index(ext.index), ext.to_numpy()


def block_maxima(years, values, block_yr):
    """Block maxima (years, values)."""
    ext = get_extremes(io.as_series(years, values), "BM",
                       block_size=f"{block_yr}D")
    return io.from_pseudo_index(ext.index), ext.to_numpy()


def summarize_period(samples):
    """Median and 5-95 % range of return-period samples (years)."""
    t = np.asarray(samples, dtype=float)
    t = t[~np.isnan(t)]
    q = np.quantile(t, [0.05, 0.50, 0.95], method="inverted_cdf")
    return {"median": q[1], "p05": q[0], "p95": q[2],
            "frac_inf": float(np.mean(np.isinf(t)))}


def prob_in_record(period, record_yr):
    """Probability of at least one event of return period T in a record."""
    period = np.asarray(period, dtype=float)
    with np.errstate(divide="ignore"):
        return 1.0 - np.exp(-record_yr / period)
