#!/usr/bin/env python
"""Step 8: extremeness relative to the contemporaneous climate state.

The modern baseline is one reference frame. The mean state of the record
shifts by more than 1 permil between regimes (step 3), so the event is
also measured against a robust 300-yr LOWESS background, and the GEV
location is allowed to follow time or that background.

Outputs
-------
figures/fig09_local_frame.(pdf|png)
reports/08_local_frame.txt
"""

import matplotlib.pyplot as plt
import numpy as np

from h12eva import config as cfg
from h12eva import eva, io, results, style
from h12eva.report import Report, fmt_years


def main():
    style.apply()
    rng = np.random.default_rng(cfg.SEED + 3)
    binned = io.load_binned()
    yrs = binned["age_ce"].to_numpy()
    a = binned["anomaly"].to_numpy()
    bg = binned["background"].to_numpy()
    resid = a - bg
    mask = io.event_mask(yrs)
    span = eva.span_years(yrs)
    level_r = float(resid[mask].max())
    yr_r = float(yrs[mask][np.argmax(resid[mask])])
    bg_event = float(np.interp(yr_r, yrs, bg))

    u_r = float(np.quantile(resid, cfg.POT_QUANTILE))
    model = eva.pot_model(yrs, resid, u_r, cfg.DECLUSTER_YR)
    xi, sig = eva.pot_params(model)
    peaks = model.extremes.to_numpy()
    rate = peaks.size / span
    t_r = float(eva.gpd_return_period(level_r, u_r, xi, sig, rate))
    dense = np.geomspace(1.2 / rate, 3.0e4, 60)
    _, bt, brl = eva.bootstrap_pot(peaks, u_r, span, level_r, dense, rng)
    t_r_boot = eva.summarize_period(bt)
    band = np.nanpercentile(brl, [5, 95], axis=0)
    rank_r = int((resid > level_r).sum() + 1)

    m_ex = eva.pot_model(yrs[~mask], resid[~mask], u_r, cfg.DECLUSTER_YR)
    xi_ex, sig_ex = eva.pot_params(m_ex)
    second_r = float(m_ex.extremes.max())
    t_r_ex = float(eva.gpd_return_period(level_r, u_r, xi_ex, sig_ex,
                                         m_ex.extremes.size / span))

    # non-stationary GEV on 50-yr maxima of the anomaly
    block = cfg.BLOCK_YR[0]
    bm_t, bm_x = eva.block_maxima(yrs, a, block)
    bg_blocks = np.interp(bm_t, yrs, bg)
    ns_time = eva.nonstationary_gev(bm_t / 1000.0, bm_x)
    ns_bg = eva.nonstationary_gev(bg_blocks, bm_x)
    xi_n, m0, m1, ls = ns_bg["nonstationary"]
    mu_event = m0 + m1 * (bg_event - ns_bg["covariate_mean"])
    level = float(a[mask].max())
    t_ns = float(eva.gev_return_period(level, xi_n, mu_event,
                                       np.exp(ls), block))

    out = {"u_resid": u_r, "xi_resid": xi, "sigma_resid": sig,
           "event_resid": level_r, "event_resid_year": yr_r,
           "background_at_event": bg_event, "rank_resid": rank_r,
           "T_resid_mle": t_r, "T_resid_boot": t_r_boot,
           "T_resid_without": t_r_ex, "largest_other_resid": second_r,
           "ns_time_lr": ns_time["lr"], "ns_time_p": ns_time["p"],
           "ns_time_slope_per_kyr": ns_time["nonstationary"][2],
           "ns_bg_lr": ns_bg["lr"], "ns_bg_p": ns_bg["p"],
           "ns_bg_slope": m1, "T_ns_bg": t_ns}
    results.update("local", out)

    # ------------------------------------------------------------ figure
    fig = plt.figure(figsize=(style.WIDTH_FULL, 4.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.55,
                          wspace=0.45)
    ax = fig.add_subplot(gs[0, :])
    ax.plot(yrs, resid, color=style.BLACK, lw=0.6)
    ax.axhline(u_r, color=style.VERMILION, lw=0.7, ls="--")
    pk_t = io.from_pseudo_index(model.extremes.index)
    ax.plot(pk_t, peaks, "o", ms=3, mfc="none", mec=style.BLACK, mew=0.6)
    ax.plot(yr_r, level_r, "*", color=style.EVENT, ms=9)
    ax.axvspan(*cfg.EVENT_WINDOW, color=style.ORANGE, alpha=0.25, lw=0)
    ax.set_xlim(-650, 2010)
    ax.set_xlabel(style.YEAR_LABEL)
    ax.set_ylabel("Anomaly minus\nbackground (‰), " + style.DRY_DOWN)
    style.dry_down(ax)
    ax.text(0.99, 0.95, f"threshold q{cfg.POT_QUANTILE * 100:.0f} = "
            f"{u_r:.2f}‰, {peaks.size} peaks", transform=ax.transAxes,
            ha="right", va="top", fontsize=7)
    style.panel_label(ax, "a", x=-0.07)

    ax = fig.add_subplot(gs[1, 0])
    z = (yrs >= 300) & (yrs <= 750)
    ax.plot(yrs[z], a[z], color=style.BLACK, lw=0.8, label="Anomaly")
    ax.plot(yrs[z], bg[z], color=style.BLUE, lw=1.4, label="Background")
    ax.fill_between(yrs[z], bg[z], a[z], where=a[z] > bg[z],
                    color=style.DRY, alpha=0.3, lw=0)
    ax.axvspan(*cfg.EVENT_WINDOW, color=style.ORANGE, alpha=0.25, lw=0)
    ax.set_xlabel("Year (CE)")
    ax.set_ylabel("Anomaly (‰), " + style.DRY_DOWN)
    style.dry_down(ax)
    ax.legend(loc="lower right", fontsize=6.5)
    style.panel_label(ax, "b", x=-0.30)

    ax = fig.add_subplot(gs[1, 1])
    srt = np.sort(peaks)
    n = srt.size
    pp = (np.arange(1, n + 1) - 0.44) / (n + 0.12)
    ax.fill_between(dense, band[0], band[1], color=style.SKY, alpha=0.35,
                    lw=0)
    ax.plot(dense, eva.gpd_return_level(dense, u_r, xi, sig, rate),
            color=style.BLUE, lw=1.2)
    ax.plot(1.0 / (rate * (1.0 - pp)), srt, "o", ms=3, mfc="none",
            mec=style.BLACK, mew=0.6)
    ax.plot(min(t_r, 3.0e4), level_r, "*", color=style.EVENT, ms=9)
    ax.text(0.04, 0.96, f"T ≈ {fmt_years(t_r)} yr",
            transform=ax.transAxes, va="top", fontsize=7,
            color=style.EVENT)
    ax.set_xscale("log")
    style.plain_log_axis(ax)
    ax.set_xlabel("Return period (yr)")
    ax.set_ylabel("Return level (‰)")
    style.panel_label(ax, "c", x=-0.30)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(bg_blocks, bm_x, "o", ms=3, mfc="none", mec=style.BLACK,
            mew=0.6)
    xx = np.linspace(bg_blocks.min(), bg_blocks.max(), 50)
    ax.plot(xx, m0 + m1 * (xx - ns_bg["covariate_mean"]),
            color=style.BLUE, lw=1.2, label=r"GEV location $\mu$")
    j = int(np.argmax(bm_x))
    ax.plot(bg_blocks[j], bm_x[j], "*", color=style.EVENT, ms=9)
    ax.set_xlabel("Background (‰)")
    ax.set_ylabel("50-yr maximum (‰)")
    ax.legend(loc="upper left")
    p_txt = ("p < 0.001" if ns_bg["p"] < 0.001
             else f"p = {ns_bg['p']:.3f}")
    ax.text(0.97, 0.04, f"LR test {p_txt}", transform=ax.transAxes,
            ha="right", fontsize=7)
    style.panel_label(ax, "d", x=-0.30)
    style.save(fig, "fig09_local_frame")

    # ------------------------------------------------------------ report
    rep = Report("08_local_frame", "Step 8. Local reference frame")
    rep.para(
        "A robust LOWESS curve with a 300-yr window describes the slowly "
        "varying mean state. Robust iterations down-weight outliers, so "
        "the event itself has little influence on its own background.")
    rep.section("Event relative to the background")
    rep.kv("Background at the event (permil)", f"{bg_event:.2f}")
    rep.kv("Event residual (permil)", f"{level_r:.2f} at {yr_r:.1f} CE")
    rep.kv("Rank of residual among all bins", f"{rank_r} of {len(resid)}")
    rep.kv("Largest residual peak outside window",
           f"{second_r:.2f} permil")
    rep.section("Threshold model on residuals (pyextremes MLE)")
    rep.kv("Threshold", f"{u_r:.3f} permil, {peaks.size} peaks")
    rep.kv("xi, sigma", f"{xi:.3f}, {sig:.3f}")
    rep.kv("Return period of event residual",
           f"{fmt_years(t_r)} yr; bootstrap median "
           f"{fmt_years(t_r_boot['median'])} [{fmt_years(t_r_boot['p05'])}"
           f", {fmt_years(t_r_boot['p95'])}]")
    rep.kv("Same, window excluded from the fit", f"{fmt_years(t_r_ex)} yr")
    rep.section("Non-stationary GEV, 50-yr maxima")
    rep.kv("Location linear in time: LR, p",
           f"{ns_time['lr']:.2f}, {ns_time['p']:.3f}")
    rep.kv("Location linear in background: LR, p",
           f"{ns_bg['lr']:.2f}, {ns_bg['p']:.3g}")
    rep.kv("Slope of mu on background", f"{m1:.2f} permil per permil")
    rep.kv("Event return period given its background",
           f"{fmt_years(t_ns)} yr")
    rep.section("Interpretation")
    verdict = ("improves" if ns_bg["p"] < 0.05
               else "does not significantly improve")
    rep.para(
        f"The event occurred while the background itself was "
        f"{'above' if bg_event > 0 else 'below'} the modern baseline "
        f"({bg_event:+.2f} permil), so part of its anomaly reflects a "
        "drier regime. Even after removing the background, the "
        f"sixth-century residual ranks {rank_r} of {len(resid)} and "
        f"exceeds the largest residual peak elsewhere by "
        f"{level_r - second_r:.1f} permil. Allowing the GEV location to "
        f"follow the background {verdict} "
        f"the fit (p = {ns_bg['p']:.3g}); conditional on the regime of "
        f"the time, the return period is about {fmt_years(t_ns)} yr. "
        "That p-value is optimistic, because the background is estimated "
        "from the same series as the maxima, so the covariate is not "
        "independent of the response. "
        "The event is therefore extreme in both frames: against the "
        "present day and against the climate of its own century.")
    rep.write()
    print("step 8 done:", {k: out[k] for k in ("event_resid", "rank_resid",
                                               "T_resid_mle", "ns_bg_p",
                                               "T_ns_bg", "ns_time_p")})


if __name__ == "__main__":
    main()
