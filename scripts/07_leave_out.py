#!/usr/bin/env python
"""Step 7: circularity test, refitting the tail without 500-530 CE.

Asking whether the largest value is extreme with a model fitted to that
same value is circular. Here the models are refitted after removing the
event window, then asked how surprising the event is.

Outputs
-------
figures/fig08_leave_out.(pdf|png)
reports/07_leave_out.txt
"""

import matplotlib.pyplot as plt
import numpy as np

from h12eva import config as cfg
from h12eva import eva, io, results, style
from h12eva.report import Report, fmt_years


def posterior_gpd(peaks, u, span, level, grid, rng, prior="ms"):
    """Posterior summaries of a GPD fit under a given shape prior."""
    xi, sig = eva.bayes_gpd(peaks, u, rng, prior=prior)
    rate = peaks.size / span
    t = eva.gpd_return_period(level, u, xi, sig, rate)
    rl = np.array([eva.gpd_return_level(grid, u, x, s, rate)
                   for x, s in zip(xi[::10], sig[::10])])
    endpoint = np.where(xi < 0, u - sig / np.where(xi < 0, xi, -1.0),
                        np.inf)
    p_rec = eva.prob_in_record(t, span)
    return {"xi": xi, "t": t, "rl": rl, "endpoint": endpoint,
            "p_record": p_rec}


def main():
    style.apply()
    rng = np.random.default_rng(cfg.SEED + 2)
    binned = io.load_binned()
    yrs = binned["age_ce"].to_numpy()
    a = binned["anomaly"].to_numpy()
    mask = io.event_mask(yrs)
    level = float(a[mask].max())
    span = eva.span_years(yrs)
    u = float(np.quantile(a, cfg.POT_QUANTILE))
    grid = np.geomspace(300.0, 1.0e5, 50)

    # pyextremes MLE without the window
    m_ex = eva.pot_model(yrs[~mask], a[~mask], u, cfg.DECLUSTER_YR)
    xi_ex, sig_ex = eva.pot_params(m_ex)
    pk_ex = m_ex.extremes.to_numpy()
    t_ex_mle = float(eva.gpd_return_period(level, u, xi_ex, sig_ex,
                                           pk_ex.size / span))
    end_ex = eva.gpd_endpoint(u, xi_ex, sig_ex)
    second = float(np.sort(pk_ex)[-1])

    _, pk_in = eva.declustered_peaks(yrs, a, u, cfg.DECLUSTER_YR)

    post_in = posterior_gpd(pk_in, u, span, level, grid, rng)
    post_ex = posterior_gpd(pk_ex, u, span, level, grid, rng)
    uni_in = posterior_gpd(pk_in, u, span, level, grid, rng, "uniform")
    uni_ex = posterior_gpd(pk_ex, u, span, level, grid, rng, "uniform")

    # GEV 50-yr blocks, Bayesian, with and without the window
    block = cfg.BLOCK_YR[0]
    _, bm_in = eva.block_maxima(yrs, a, block)
    _, bm_ex = eva.block_maxima(yrs[~mask], a[~mask], block)
    gev = {}
    for key, bm in (("with", bm_in), ("without", bm_ex)):
        x, mu, s = eva.bayes_gev(bm, rng)
        gev[key] = eva.gev_return_period(level, x, mu, s, block)

    def summ(t):
        return eva.summarize_period(t)

    out = {
        "mle_without": {"xi": xi_ex, "sigma": sig_ex, "endpoint": end_ex,
                        "T": t_ex_mle, "n": pk_ex.size,
                        "largest_remaining_peak": second},
        "bayes_gpd_with": summ(post_in["t"]),
        "bayes_gpd_without": summ(post_ex["t"]),
        "uniform_gpd_with": summ(uni_in["t"]),
        "uniform_gpd_without": summ(uni_ex["t"]),
        "p_record_uniform_with": float(np.mean(uni_in["p_record"])),
        "p_record_uniform_without": float(np.mean(uni_ex["p_record"])),
        "xi_uniform_without_median": float(np.median(uni_ex["xi"])),
        "bayes_gev50_with": summ(gev["with"]),
        "bayes_gev50_without": summ(gev["without"]),
        "p_beyond_endpoint_without": float(np.mean(np.isinf(
            post_ex["t"]))),
        "p_record_with": float(np.mean(post_in["p_record"])),
        "p_record_without": float(np.mean(post_ex["p_record"])),
        "xi_without_median": float(np.median(post_ex["xi"])),
        "xi_with_median": float(np.median(post_in["xi"])),
    }
    results.update("leave_out", out)

    # ------------------------------------------------------------ figure
    fig = plt.figure(figsize=(style.WIDTH_FULL, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1.1, 1.0], wspace=0.42)
    ax = fig.add_subplot(gs[0])
    for post, col, lab in ((post_in, style.BLUE, "With 500-530 CE"),
                           (post_ex, style.GREEN,
                            "Without 500-530 CE")):
        q = np.nanpercentile(post["rl"], [5, 50, 95], axis=0)
        ax.fill_between(grid, q[0], q[2], color=col, alpha=0.2, lw=0)
        ax.plot(grid, q[1], color=col, lw=1.2, label=lab)
    ax.axhline(level, color=style.EVENT, lw=0.7, ls=":")
    ax.text(grid[0] * 1.05, level + 0.06, "520-525 CE anomaly",
            color=style.EVENT, fontsize=7, ha="left", va="bottom")
    ax.text(0.03, 0.03, "MS prior, median and 90 %",
            transform=ax.transAxes, fontsize=6.5, color=style.GREY)
    ax.axvline(span, color=style.GREY, lw=0.6, ls="--")
    ax.set_xscale("log")
    style.plain_log_axis(ax)
    ax.set_xlabel("Return period (yr)")
    ax.set_ylabel("Return level (‰)")
    ax.set_ylim(0.5, 4.6)
    ax.legend(loc="upper left")
    style.panel_label(ax, "a", x=-0.13)

    ax = fig.add_subplot(gs[1])
    bins = np.linspace(2.5, 6.5, 33)
    sets = ((post_in, style.BLUE, "With, MS prior"),
            (post_ex, style.GREEN, "Without, MS prior"),
            (uni_ex, style.PURPLE, "Without, uniform prior"))
    for k, (post, col, lab) in enumerate(sets):
        t = post["t"]
        fin = np.log10(t[np.isfinite(t)])
        w = np.ones_like(fin) / t.size
        ax.hist(fin, bins=bins, weights=w, histtype="step", lw=1.1,
                color=col, label=lab)
        frac = np.mean(np.isinf(t))
        if frac > 0.05:
            ax.text(0.98, 0.55, f"{frac * 100:.0f} % beyond\nendpoint",
                    transform=ax.transAxes, ha="right", fontsize=6.5,
                    color=col)
    ax.axvline(np.log10(span), color=style.GREY, lw=0.6, ls="--")
    ax.set_xlabel(r"log$_{10}$ return period of event (yr)")
    ax.set_ylabel("Posterior probability")
    ax.legend(loc="upper left", fontsize=6.5)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    style.panel_label(ax, "b", x=-0.22)

    ax = fig.add_subplot(gs[2])
    bins = np.linspace(-0.5, 0.5, 41)
    for post, col, lab in sets:
        ax.hist(post["xi"], bins=bins, density=True, histtype="step",
                color=col, lw=1.1)
    xx = np.linspace(-0.499, 0.499, 200)
    prior = np.exp([eva.ms_log_prior(v) for v in xx])
    prior /= np.trapezoid(prior, xx)
    ax.plot(xx, prior, color=style.GREY, lw=0.8, ls="--",
            label="MS prior")
    ax.text(0.97, 0.45, "colors as\nin (b)", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.5, color=style.GREY)
    ax.set_xlabel(r"Shape $\xi$")
    ax.set_ylabel("Posterior density")
    ax.legend(loc="upper left", fontsize=6.5)
    style.panel_label(ax, "c", x=-0.25)
    style.save(fig, "fig08_leave_out")

    # ------------------------------------------------------------ report
    rep = Report("07_leave_out", "Step 7. Circularity test")
    rep.para(
        "The six 5-yr bins between 500 and 530 CE are removed. The "
        "threshold, declustering and block length stay as in steps 5 and "
        "6, so only the information carried by the event changes.")
    rep.section("pyextremes MLE without the window")
    rep.kv("Declustered peaks", f"{pk_ex.size}")
    rep.kv("Largest remaining peak", f"{second:.2f} permil")
    rep.kv("Shape xi", f"{xi_ex:.3f}")
    rep.kv("Upper endpoint", "unbounded" if np.isinf(end_ex)
           else f"{end_ex:.2f} permil")
    rep.kv(f"Return period of {level:.2f} permil", f"{fmt_years(t_ex_mle)}"
           " yr")
    rep.para(
        "Without the event the likelihood pushes the shape strongly "
        "negative, so the fitted tail ends just above the largest "
        "remaining peak. Maximum likelihood is not regular for xi below "
        "-0.5, which is why the Bayesian refit below constrains xi with "
        "a geophysical prior instead of trusting this estimate.")
    rep.section("Bayesian fits with the Martins-Stedinger (2000) prior")
    rep.para(
        "The prior restricts xi to (-0.5, 0.5) and favors mildly heavy "
        "tails (prior mean +0.1). It therefore pulls the tail upward and "
        "shortens return periods, so any extremeness that survives it is "
        "conservative.")
    for key, lab in (("bayes_gpd_with", "GPD, with window"),
                     ("bayes_gpd_without", "GPD, without window"),
                     ("uniform_gpd_with", "GPD, uniform prior, with"),
                     ("uniform_gpd_without", "GPD, uniform prior, without"),
                     ("bayes_gev50_with", "GEV 50-yr, with window"),
                     ("bayes_gev50_without", "GEV 50-yr, without window")):
        s = out[key]
        rep.kv(lab, f"median {fmt_years(s['median'])} yr [5 %: "
               f"{fmt_years(s['p05'])}, 95 %: {fmt_years(s['p95'])}]; "
               f"beyond endpoint {s['frac_inf'] * 100:.0f} %")
    rep.kv("Posterior median xi, with / without",
           f"{out['xi_with_median']:.2f} / {out['xi_without_median']:.2f}")
    rep.kv("P(such an event in a record this long), with",
           f"{out['p_record_with'] * 100:.0f} %")
    rep.kv("P(such an event in a record this long), without",
           f"{out['p_record_without'] * 100:.0f} %")
    rep.kv("  same, uniform prior, with / without",
           f"{out['p_record_uniform_with'] * 100:.0f} % / "
           f"{out['p_record_uniform_without'] * 100:.0f} %")
    rep.kv("Posterior median xi, uniform prior, without",
           f"{out['xi_uniform_without_median']:.2f}")
    rep.para(
        "Under the uniform prior the posterior of xi piles up against "
        "the lower bound of -0.5: without the event, the data favor a "
        "tail that is bounded more sharply than the prior allows.")
    rep.section("Interpretation")
    sw, so = out["bayes_gpd_with"], out["bayes_gpd_without"]
    uo = out["uniform_gpd_without"]
    rep.para(
        f"Three statements follow. First, judged by the data alone, the "
        f"rest of the record has a bounded dry tail ending near "
        f"{end_ex:.1f} permil, and the sixth-century value exceeds every "
        f"other independent dry extreme by {level - second:.1f} permil. "
        "In that sense the event lies outside the behavior of the "
        "stalagmite over the other twenty-five centuries.")
    rep.para(
        f"Second, that conclusion depends on the tail shape, which "
        f"{pk_ex.size} peaks cannot pin down. Under a neutral uniform "
        f"prior on xi, {uo['frac_inf'] * 100:.0f} % of posterior draws "
        "without the window place the event beyond the upper limit of "
        "the tail, and the chance of such an event in a record of this "
        f"length is {out['p_record_uniform_without'] * 100:.0f} %. "
        "Under the heavy-tail Martins-Stedinger prior it is "
        f"{fmt_years(so['median'])} yr (versus {fmt_years(sw['median'])} "
        "yr with the event included), and that chance rises to about "
        f"{out['p_record_without'] * 100:.0f} %. The prior on the tail "
        "shape, not the data, decides between 'outside the normal "
        "range' and 'rare but expected once in a few such records'.")
    rep.para(
        "Third, every variant agrees that removing the event lengthens "
        "its return period, which confirms that fitting a model to the "
        "event itself flatters it. The defensible summary is a "
        "record-breaking, multi-millennial drought whose exact return "
        "period is uncertain by an order of magnitude or more, not a "
        "precise recurrence time.")
    rep.write()
    print("step 7 done:", {k: out[k] for k in ("bayes_gpd_with",
                                               "bayes_gpd_without")},
          "mle_ex", round(xi_ex, 2), fmt_years(t_ex_mle))


if __name__ == "__main__":
    main()
