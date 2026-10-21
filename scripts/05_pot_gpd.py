#!/usr/bin/env python
"""Step 5: peaks over threshold with a generalized Pareto model.

Threshold selection (mean residual life, parameter stability), pyextremes
MLE fit with bootstrap intervals, pyextremes Emcee fit, diagnostics, and
the return period of the sixth-century peak.

Outputs
-------
figures/fig05_pot_threshold.(pdf|png)
figures/fig06_pot_fit.(pdf|png)
reports/05_pot_gpd.txt
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps

from h12eva import config as cfg
from h12eva import eva, io, results, style
from h12eva.report import Report, fmt_years


def stability_scan(yrs, a, rng, n_boot=300):
    """Mean excess and GPD parameters over a range of thresholds."""
    qs = np.arange(cfg.STABILITY_QUANTILES[0],
                   cfg.STABILITY_QUANTILES[1] + 1e-9, 0.01)
    rows = []
    for q in qs:
        u = float(np.quantile(a, q))
        _, peaks = eva.declustered_peaks(yrs, a, u, cfg.DECLUSTER_YR)
        exc = peaks - u
        if exc.size < 6:
            continue
        xi, _, sig = sps.genpareto.fit(peaks, floc=u)
        boot = []
        for _ in range(n_boot):
            s = rng.choice(peaks, peaks.size, replace=True)
            bx, _, bs = sps.genpareto.fit(s, floc=u)
            boot.append((bx, bs - bx * u))
        boot = np.array(boot)
        rows.append({
            "q": q, "u": u, "n": exc.size,
            "me": exc.mean(),
            "me_se": exc.std(ddof=1) / np.sqrt(exc.size),
            "xi": xi, "xi_lo": np.percentile(boot[:, 0], 5),
            "xi_hi": np.percentile(boot[:, 0], 95),
            "sstar": sig - xi * u,
            "sstar_lo": np.percentile(boot[:, 1], 5),
            "sstar_hi": np.percentile(boot[:, 1], 95),
        })
    return rows


def main():
    style.apply()
    rng = np.random.default_rng(cfg.SEED)
    binned = io.load_binned()
    yrs = binned["age_ce"].to_numpy()
    a = binned["anomaly"].to_numpy()
    ev_mask = io.event_mask(yrs)
    level = float(a[ev_mask].max())
    span = eva.span_years(yrs)
    u = float(np.quantile(a, cfg.POT_QUANTILE))

    scan = stability_scan(yrs, a, rng)

    model = eva.pot_model(yrs, a, u, cfg.DECLUSTER_YR)
    xi, sig = eva.pot_params(model)
    peaks = model.extremes.to_numpy()
    peak_yrs = io.from_pseudo_index(model.extremes.index)
    n = peaks.size
    rate = n / span
    t_mle = float(eva.gpd_return_period(level, u, xi, sig, rate))
    endpoint = eva.gpd_endpoint(u, xi, sig)

    rp = np.array(cfg.RETURN_PERIODS, dtype=float)
    np.random.seed(cfg.SEED)  # pyextremes bootstrap uses global RNG
    rl, rl_lo, rl_hi = model.get_return_value(
        return_period=rp, return_period_size=eva.RP_SIZE, alpha=0.90,
        n_samples=cfg.N_BOOT)
    rp_dense = np.geomspace(1.2 / rate, 3.0e4, 60)
    bx, bt, brl = eva.bootstrap_pot(peaks, u, span, level, rp_dense, rng)
    t_boot = eva.summarize_period(bt)
    band = np.nanpercentile(brl, [5, 95], axis=0)

    # pyextremes Bayesian fit (flat priors, emcee sampler)
    bay = eva.EVA(io.as_series(yrs, a))
    bay.get_extremes("POT", threshold=u, r=f"{cfg.DECLUSTER_YR}D")
    np.random.seed(cfg.SEED)
    bay.fit_model("Emcee", distribution="genpareto",
                  n_walkers=cfg.N_WALKERS, n_samples=cfg.N_STEPS,
                  progress=False)
    trace = bay.model.trace[:, cfg.N_BURN:, :].reshape(-1, 2)
    t_bay = eva.gpd_return_period(level, u, trace[:, 0], trace[:, 1],
                                  rate)
    t_bay_sum = eva.summarize_period(t_bay)
    brv, brv_lo, brv_hi = bay.get_return_value(
        return_period=rp, return_period_size=eva.RP_SIZE, alpha=0.90,
        burn_in=cfg.N_BURN)

    # diagnostics
    srt = np.sort(peaks)
    pp = (np.arange(1, n + 1) - 0.44) / (n + 0.12)
    model_q = sps.genpareto.ppf(pp, xi, loc=u, scale=sig)
    emp_t = 1.0 / (rate * (1.0 - pp))
    ks = sps.kstest(peaks, "genpareto", args=(xi, u, sig))

    # ------------------------------------------------ figure: threshold
    q_arr = np.array([r["q"] for r in scan])
    u_arr = np.array([r["u"] for r in scan])
    fig, axs = plt.subplots(2, 2, figsize=(style.WIDTH_FULL, 4.4),
                            gridspec_kw={"hspace": 0.62, "wspace": 0.3})
    ax = axs[0, 0]
    me = np.array([r["me"] for r in scan])
    se = np.array([r["me_se"] for r in scan])
    ax.fill_between(u_arr, me - 1.96 * se, me + 1.96 * se,
                    color=style.LIGHT_GREY, lw=0)
    ax.plot(u_arr, me, color=style.BLACK)
    ax.axvline(u, color=style.VERMILION, lw=0.8, ls="--")
    ax.set_xlabel("Threshold u (‰)")
    ax.set_ylabel("Mean excess (‰)")
    style.panel_label(ax, "a")

    ax = axs[0, 1]
    for key, lab, col in (("xi", r"Shape $\xi$", style.BLACK),):
        lo = np.array([r[key + "_lo"] for r in scan])
        hi = np.array([r[key + "_hi"] for r in scan])
        ax.fill_between(u_arr, lo, hi, color=style.LIGHT_GREY, lw=0)
        ax.plot(u_arr, [r[key] for r in scan], color=col)
        ax.set_ylabel(lab)
    ax.axhline(0, color=style.GREY, lw=0.5)
    ax.axvline(u, color=style.VERMILION, lw=0.8, ls="--")
    ax.set_xlabel("Threshold u (‰)")
    style.panel_label(ax, "b")

    ax = axs[1, 0]
    lo = np.array([r["sstar_lo"] for r in scan])
    hi = np.array([r["sstar_hi"] for r in scan])
    ax.fill_between(u_arr, lo, hi, color=style.LIGHT_GREY, lw=0)
    ax.plot(u_arr, [r["sstar"] for r in scan], color=style.BLACK)
    ax.axvline(u, color=style.VERMILION, lw=0.8, ls="--")
    ax.set_xlabel("Threshold u (‰)")
    ax.set_ylabel(r"Modified scale $\sigma^{*}$ (‰)")
    style.panel_label(ax, "c")
    sec = ax.secondary_xaxis(
        "top", functions=(lambda x: np.interp(x, u_arr, q_arr * 100),
                          lambda p: np.interp(p, q_arr * 100, u_arr)))
    sec.set_xlabel("Percentile", fontsize=7)
    sec.tick_params(labelsize=6.5)

    ax = axs[1, 1]
    ax.plot(yrs, a, color=style.LIGHT_GREY, lw=0.5)
    ax.axhline(u, color=style.VERMILION, lw=0.8, ls="--")
    ax.plot(peak_yrs, peaks, "o", ms=3, mfc="none", mec=style.BLACK,
            mew=0.7)
    j = int(np.argmax(peaks))
    ax.plot(peak_yrs[j], peaks[j], "*", color=style.EVENT, ms=8)
    ax.set_xlim(-650, 2010)
    ax.set_xlabel(style.YEAR_LABEL)
    ax.set_ylabel("Anomaly (‰), " + style.DRY_DOWN)
    style.dry_down(ax)
    ax.text(0.99, 0.97, f"{n} declustered peaks\nr = "
            f"{cfg.DECLUSTER_YR:.0f} yr", transform=ax.transAxes,
            ha="right", va="top", fontsize=7)
    style.panel_label(ax, "d")
    style.save(fig, "fig05_pot_threshold")

    # ------------------------------------------------------ figure: fit
    fig = plt.figure(figsize=(style.WIDTH_FULL, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.7, 1, 1], wspace=0.42)
    ax = fig.add_subplot(gs[0])
    ax.fill_between(rp_dense, band[0], band[1], color=style.SKY,
                    alpha=0.35, lw=0, label="90 % bootstrap")
    ax.plot(rp_dense, eva.gpd_return_level(rp_dense, u, xi, sig, rate),
            color=style.BLUE, lw=1.2, label="GPD (pyextremes MLE)")
    ax.plot(emp_t, srt, "o", ms=3, mfc="none", mec=style.BLACK, mew=0.7,
            label="Declustered peaks")
    ax.plot(t_mle, level, "*", color=style.EVENT, ms=9, zorder=4)
    ax.hlines(level, rp_dense[0], t_mle, color=style.EVENT, lw=0.6,
              ls=":")
    ax.annotate(f"520-525 CE\nT ≈ {t_mle:,.0f} yr", xy=(t_mle, level),
                xytext=(t_mle * 0.06, level - 0.15), fontsize=7,
                color=style.EVENT, ha="center", va="top")
    ax.axvline(span, color=style.GREY, lw=0.6, ls="--")
    ax.text(span * 1.08, ax.get_ylim()[0] + 0.15, "record\nlength",
            fontsize=6.5, color=style.GREY)
    ax.set_xscale("log")
    style.plain_log_axis(ax)
    ax.set_xlabel("Return period (yr)")
    ax.set_ylabel("Return level (‰)")
    ax.legend(loc="upper left")
    style.panel_label(ax, "a", x=-0.12)

    ax = fig.add_subplot(gs[1])
    lim = [u - 0.05, max(srt.max(), model_q.max()) + 0.2]
    ax.plot(lim, lim, color=style.GREY, lw=0.6)
    ax.plot(model_q, srt, "o", ms=3, mfc="none", mec=style.BLACK,
            mew=0.7)
    ax.plot(model_q[-1], srt[-1], "*", color=style.EVENT, ms=8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Model quantile (‰)")
    ax.set_ylabel("Empirical quantile (‰)")
    style.panel_label(ax, "b", x=-0.30)

    ax = fig.add_subplot(gs[2])
    ax.hist(peaks - u, bins=np.arange(0, 3.0, 0.25), density=True,
            color=style.LIGHT_GREY, edgecolor="white", lw=0.5)
    xx = np.linspace(0, 3.0, 200)
    ax.plot(xx, sps.genpareto.pdf(xx, xi, scale=sig), color=style.BLUE,
            lw=1.2)
    ax.set_xlabel("Excess over u (‰)")
    ax.set_ylabel("Density")
    style.panel_label(ax, "c", x=-0.30)
    style.save(fig, "fig06_pot_fit")

    results.update("pot", {
        "threshold": u, "quantile": cfg.POT_QUANTILE, "n_peaks": n,
        "rate_per_yr": rate, "xi": xi, "sigma": sig,
        "endpoint": endpoint, "event_level": level,
        "T_mle": t_mle, "T_boot": t_boot, "T_emcee": t_bay_sum,
        "xi_boot_p05": np.percentile(bx, 5),
        "xi_boot_p95": np.percentile(bx, 95),
        "p_record_mle": float(eva.prob_in_record(t_mle, span)),
        "ks_p": ks.pvalue, "span_yr": span,
        "return_levels": {int(p): [float(v), float(lo_), float(hi_)]
                          for p, v, lo_, hi_ in zip(rp, rl, rl_lo,
                                                    rl_hi)},
    })

    # ------------------------------------------------------------ report
    rep = Report("05_pot_gpd", "Step 5. Peaks over threshold (GPD)")
    rep.section("Set-up")
    rep.kv("Series", f"5-yr bins, {len(a)} values, {span:.0f} yr")
    rep.kv("Threshold u", f"{u:.3f} permil (q{cfg.POT_QUANTILE * 100:.0f})")
    rep.kv("Declustering run length r", f"{cfg.DECLUSTER_YR:.0f} yr")
    rep.kv("Declustered peaks", f"{n} (rate {rate * 100:.2f} per century)")
    rep.para(
        "Time is mapped onto a pseudo-calendar in which one year equals "
        "one day, because pandas cannot represent dates before 1677. "
        "pyextremes then works unchanged, and every return period is "
        "expressed in calendar years.")
    rep.para(
        "The mean residual life plot is roughly linear above about the "
        "85th percentile, and the shape and modified scale are stable "
        "within their uncertainty from the 85th to the 95th percentile. "
        "The 90th percentile balances bias (too low a threshold) "
        "against variance (too few peaks).")
    rep.section("pyextremes MLE fit")
    rep.kv("Shape xi", f"{xi:.3f} (bootstrap 90 %: "
           f"{np.percentile(bx, 5):.2f} to {np.percentile(bx, 95):.2f})")
    rep.kv("Scale sigma", f"{sig:.3f} permil")
    rep.kv("Upper endpoint u - sigma/xi",
           "unbounded" if np.isinf(endpoint) else f"{endpoint:.2f} permil")
    rep.kv("Kolmogorov-Smirnov p (fitted, indicative)", f"{ks.pvalue:.2f}")
    rep.section("Return levels (permil), 90 % bootstrap interval")
    for p, v, lo_, hi_ in zip(rp, rl, rl_lo, rl_hi):
        rep.kv(f"  {p:>6.0f} yr", f"{v:5.2f}  [{lo_:5.2f}, {hi_:5.2f}]")
    rep.section("Return period of the 520-525 CE anomaly "
                f"({level:.2f} permil)")
    rep.kv("MLE point estimate", f"{t_mle:,.0f} yr")
    rep.kv("Bootstrap median [5 %, 95 %]",
           f"{fmt_years(t_boot['median'])} [{fmt_years(t_boot['p05'])}, "
           f"{fmt_years(t_boot['p95'])}] yr")
    rep.kv("Bootstrap fraction beyond endpoint",
           f"{t_boot['frac_inf'] * 100:.1f} %")
    rep.kv("pyextremes Emcee median [5 %, 95 %]",
           f"{fmt_years(t_bay_sum['median'])} "
           f"[{fmt_years(t_bay_sum['p05'])}, "
           f"{fmt_years(t_bay_sum['p95'])}] yr")
    rep.kv("Emcee return level 1000 yr [90 %]",
           f"{brv[2]:.2f} [{brv_lo[2]:.2f}, {brv_hi[2]:.2f}] permil")
    rep.kv("P(at least one such event in record)",
           f"{eva.prob_in_record(t_mle, span) * 100:.0f} %")
    rep.section("Interpretation")
    rep.para(
        f"With the event included, the GPD places the 520-525 CE anomaly "
        f"at a return period of about {t_mle:,.0f} yr, roughly "
        f"{t_mle / span:.0f} times the length of the record. The "
        "bootstrap interval is wide because only "
        f"{n} independent peaks constrain the tail, and its upper end "
        "includes resamples whose bounded tail cannot reach the event at "
        "all. The point estimate is also optimistic for a subtle reason: "
        "the event itself is one of the peaks used for fitting, which "
        "lengthens the fitted tail toward it. Step 7 removes this "
        "circularity.")
    rep.write()
    print(f"step 5 done: u={u:.2f} n={n} xi={xi:.3f} T_mle={t_mle:,.0f} "
          f"boot={t_boot} emcee={t_bay_sum}")


if __name__ == "__main__":
    main()
