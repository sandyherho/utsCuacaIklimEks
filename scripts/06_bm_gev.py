#!/usr/bin/env python
"""Step 6: block maxima with a generalized extreme value model.

Outputs
-------
figures/fig07_bm_gev.(pdf|png)
reports/06_bm_gev.txt
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps

from h12eva import config as cfg
from h12eva import eva, io, results, style
from h12eva.report import Report, fmt_years


def main():
    style.apply()
    rng = np.random.default_rng(cfg.SEED + 1)
    binned = io.load_binned()
    yrs = binned["age_ce"].to_numpy()
    a = binned["anomaly"].to_numpy()
    level = float(a[io.event_mask(yrs)].max())
    span = eva.span_years(yrs)
    rp = np.array(cfg.RETURN_PERIODS, dtype=float)

    fits = {}
    for block in cfg.BLOCK_YR:
        model = eva.bm_model(yrs, a, block)
        xi, mu, sig = eva.bm_params(model)
        maxima = model.extremes.to_numpy()
        np.random.seed(cfg.SEED)  # pyextremes bootstrap uses global RNG
        rl, lo, hi = model.get_return_value(
            return_period=rp, return_period_size=eva.RP_SIZE, alpha=0.90,
            n_samples=cfg.N_BOOT // 2)
        dense = np.geomspace(block * 1.05, 3.0e4, 60)
        bx, bt, brl = eva.bootstrap_bm(maxima, level, dense, block, rng)
        t_mle = float(eva.gev_return_period(level, xi, mu, sig, block))
        # Same fit with the 500-530 CE bins removed, to see how much the
        # event pulls the fitted curve toward itself.
        keep = ~io.event_mask(yrs)
        m_ex = eva.bm_model(yrs[keep], a[keep], block)
        xi_ex, mu_ex, sig_ex = eva.bm_params(m_ex)
        t_ex = float(eva.gev_return_period(level, xi_ex, mu_ex, sig_ex,
                                           block))
        fits[block] = {
            "xi_without": xi_ex, "T_without": t_ex,
            "max_without": float(m_ex.extremes.max()),
            "n_without": int(m_ex.extremes.size),
            "xi": xi, "mu": mu, "sigma": sig, "n": maxima.size,
            "maxima": maxima, "rl": (rl, lo, hi), "dense": dense,
            "band": np.nanpercentile(brl, [5, 95], axis=0),
            "T_mle": t_mle, "T_boot": eva.summarize_period(bt),
            "xi_ci": (np.percentile(bx, 5), np.percentile(bx, 95)),
        }

    # ------------------------------------------------------------ figure
    fig = plt.figure(figsize=(style.WIDTH_FULL, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.25, 1], wspace=0.4)
    for k, (block, letter) in enumerate(zip(cfg.BLOCK_YR, "ab")):
        f = fits[block]
        ax = fig.add_subplot(gs[k])
        srt = np.sort(f["maxima"])
        n = srt.size
        pp = (np.arange(1, n + 1) - 0.44) / (n + 0.12)
        emp_t = block / (1.0 - pp)
        ax.fill_between(f["dense"], f["band"][0], f["band"][1],
                        color=style.SKY, alpha=0.35, lw=0,
                        label="90 % bootstrap")
        ax.plot(f["dense"], eva.gev_return_level(
            f["dense"], f["xi"], f["mu"], f["sigma"], block),
            color=style.BLUE, lw=1.2, label="GEV (pyextremes MLE)")
        ax.plot(emp_t, srt, "o", ms=3, mfc="none", mec=style.BLACK,
                mew=0.7, label=f"{block:.0f}-yr maxima")
        ax.plot(min(f["T_mle"], 3.0e4), level, "*", color=style.EVENT,
                ms=9, zorder=4)
        ax.text(0.97, 0.05, f"T ≈ {fmt_years(f['T_mle'])} yr\n"
                rf"$\xi$ = {f['xi']:.2f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, color=style.EVENT)
        ax.axvline(span, color=style.GREY, lw=0.6, ls="--")
        ax.set_xscale("log")
        style.plain_log_axis(ax)
        ax.set_xlabel("Return period (yr)")
        if k == 0:
            ax.set_ylabel("Return level (‰)")
            ax.legend(loc="upper left")
        ax.set_title(f"{block:.0f}-yr blocks (n = {n})", fontsize=8)
        ax.set_ylim(-0.5, 6.0)
        style.panel_label(ax, letter, x=-0.16 if k == 0 else -0.10)

    ax = fig.add_subplot(gs[2])
    lim = [-1.0, 3.8]
    ax.plot(lim, lim, color=style.GREY, lw=0.6)
    for block, mk, col in zip(cfg.BLOCK_YR, ("o", "s"),
                              (style.BLACK, style.ORANGE)):
        f = fits[block]
        srt = np.sort(f["maxima"])
        n = srt.size
        pp = (np.arange(1, n + 1) - 0.44) / (n + 0.12)
        mq = sps.genextreme.ppf(pp, -f["xi"], loc=f["mu"],
                                scale=f["sigma"])
        ax.plot(mq, srt, mk, ms=3, mfc="none", mec=col, mew=0.7,
                label=f"{block:.0f}-yr")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Model quantile (‰)")
    ax.set_ylabel("Empirical quantile (‰)")
    ax.legend(loc="upper left")
    style.panel_label(ax, "c", x=-0.28)
    style.save(fig, "fig07_bm_gev")

    results.update("bm", {
        str(int(b)): {
            **{k: v for k, v in f.items()
               if k in ("xi", "mu", "sigma", "n", "T_mle", "T_boot",
                        "xi_ci", "xi_without", "T_without", "max_without",
                        "n_without")},
            "return_levels": {
                int(p_): [float(v), float(lo_), float(hi_)]
                for p_, v, lo_, hi_ in zip(rp, *f["rl"])},
            "p_record": float(eva.prob_in_record(f["T_mle"], span)),
        }
        for b, f in fits.items()})

    # ------------------------------------------------------------ report
    rep = Report("06_bm_gev", "Step 6. Block maxima (GEV)")
    rep.para(
        "Block maxima are taken from the 5-yr series and fitted with a "
        "GEV by maximum likelihood in pyextremes. Shape xi follows the "
        "Coles (2001) convention: xi < 0 bounded tail, xi = 0 Gumbel, "
        "xi > 0 heavy tail.")
    for block, f in fits.items():
        rep.section(f"{block:.0f}-yr blocks ({f['n']} maxima)")
        rep.kv("xi [bootstrap 90 %]",
               f"{f['xi']:.3f} [{f['xi_ci'][0]:.2f}, {f['xi_ci'][1]:.2f}]")
        rep.kv("mu, sigma (permil)", f"{f['mu']:.3f}, {f['sigma']:.3f}")
        rl, lo, hi = f["rl"]
        for p, v, l_, h_ in zip(rp, rl, lo, hi):
            rep.kv(f"  return level {p:>6.0f} yr",
                   f"{v:5.2f} [{l_:5.2f}, {h_:5.2f}] permil")
        tb = f["T_boot"]
        rep.kv(f"Return period of {level:.2f} permil (MLE)",
               f"{fmt_years(f['T_mle'])} yr")
        rep.kv("Without 500-530 CE: xi, return period",
               f"{f['xi_without']:.3f}, {fmt_years(f['T_without'])} yr")
        rep.kv("  bootstrap median [5 %, 95 %]",
               f"{fmt_years(tb['median'])} [{fmt_years(tb['p05'])}, "
               f"{fmt_years(tb['p95'])}] yr; beyond endpoint in "
               f"{tb['frac_inf'] * 100:.0f} %")
    rep.section("Interpretation")
    f50, f100 = fits[cfg.BLOCK_YR[0]], fits[cfg.BLOCK_YR[1]]
    rep.para(
        f"Both block sizes give shapes close to zero (Gumbel-like), and "
        f"the threshold model shape is also compatible with zero within "
        f"its interval. The return "
        f"period of the event is {fmt_years(f50['T_mle'])} yr with 50-yr "
        f"blocks and {fmt_years(f100['T_mle'])} yr with 100-yr blocks, "
        "the same order of magnitude as the threshold model. Block "
        "maxima waste information, since many blocks contain no real "
        "drought, so they serve here as an independent cross-check, not "
        "as the main estimate.")
    rep.write()
    print("step 6 done:", {b: (round(f["xi"], 3), round(f["T_mle"]))
                           for b, f in fits.items()})


if __name__ == "__main__":
    main()
