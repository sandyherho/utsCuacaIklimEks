#!/usr/bin/env python
"""Step 10: fingerprint of the sixth-century drought and summary report.

Outputs
-------
figures/fig11_fingerprint.(pdf|png)
figures/fig12_context.(pdf|png)
reports/10_fingerprint.txt
reports/00_summary.txt
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from h12eva import config as cfg
from h12eva import io, results, style
from h12eva.report import Report, fmt_years

INF_PLOT = 2.0e6


def _f(v):
    return results.as_float(v)


def forest_rows(res):
    """Rows (label, median, low, high, group) for the forest plot."""
    pot, bm, lo, loc = res["pot"], res["bm"], res["leave_out"], res["local"]
    rows = [
        ("POT-GPD, MLE + bootstrap", _f(pot["T_mle"]),
         _f(pot["T_boot"]["p05"]), _f(pot["T_boot"]["p95"]), "with"),
        ("POT-GPD, pyextremes Emcee", _f(pot["T_emcee"]["median"]),
         _f(pot["T_emcee"]["p05"]), _f(pot["T_emcee"]["p95"]), "with"),
        ("GEV, 50-yr blocks", _f(bm["50"]["T_mle"]),
         _f(bm["50"]["T_boot"]["p05"]), _f(bm["50"]["T_boot"]["p95"]),
         "with"),
        ("GEV, 100-yr blocks", _f(bm["100"]["T_mle"]),
         _f(bm["100"]["T_boot"]["p05"]), _f(bm["100"]["T_boot"]["p95"]),
         "with"),
        ("GPD, MS prior", _f(lo["bayes_gpd_with"]["median"]),
         _f(lo["bayes_gpd_with"]["p05"]), _f(lo["bayes_gpd_with"]["p95"]),
         "with"),
        ("Local frame, POT-GPD", _f(loc["T_resid_mle"]),
         _f(loc["T_resid_boot"]["p05"]), _f(loc["T_resid_boot"]["p95"]),
         "local"),
        ("Local frame, GEV with background", _f(loc["T_ns_bg"]),
         np.nan, np.nan, "local"),
        ("GPD, MS prior", _f(lo["bayes_gpd_without"]["median"]),
         _f(lo["bayes_gpd_without"]["p05"]),
         _f(lo["bayes_gpd_without"]["p95"]), "without"),
        ("GPD, uniform prior", _f(lo["uniform_gpd_without"]["median"]),
         _f(lo["uniform_gpd_without"]["p05"]),
         _f(lo["uniform_gpd_without"]["p95"]), "without"),
        ("GEV 50-yr, MS prior", _f(lo["bayes_gev50_without"]["median"]),
         _f(lo["bayes_gev50_without"]["p05"]),
         _f(lo["bayes_gev50_without"]["p95"]), "without"),
        ("POT-GPD, MLE", _f(lo["mle_without"]["T"]), np.nan, np.nan,
         "without"),
    ]
    return rows


def main():
    style.apply()
    res = results.load()
    binned = io.load_binned()
    native = io.load_native()
    runs = pd.read_csv(cfg.PROC_DIR / "drought_runs.csv")
    yrs = binned["age_ce"].to_numpy()
    a = binned["anomaly"].to_numpy()
    mask = io.event_mask(yrs)
    level = float(a[mask].max())
    ev_run = runs[runs["event"]].sort_values("severity").iloc[-1]
    span = res["pot"]["span_yr"]
    rest = a[~mask]
    z = (level - rest.mean()) / rest.std(ddof=1)
    second = float(res["leave_out"]["mle_without"]["largest_remaining_peak"])

    metrics = [
        ("5-yr anomaly", 1, len(a)),
        ("Anomaly minus background", res["local"]["rank_resid"], len(a)),
        ("Declustered peak", 1, res["pot"]["n_peaks"]),
        ("Spell peak", int(ev_run["rank_peak"]), len(runs)),
        ("Spell severity", int(ev_run["rank_severity"]), len(runs)),
        ("Spell duration", int(ev_run["rank_duration_yr"]), len(runs)),
        ("Spell intensity", int(ev_run["rank_intensity"]), len(runs)),
    ]

    # --------------------------------------------- figure: fingerprint
    rows = forest_rows(res)
    fig = plt.figure(figsize=(style.WIDTH_FULL, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], wspace=0.75)
    ax = fig.add_subplot(gs[0])
    colors = {"with": style.BLUE, "local": style.ORANGE,
              "without": style.GREEN}
    ypos = np.arange(len(rows))[::-1].astype(float)
    groups = [r[4] for r in rows]
    for k in range(1, len(rows)):
        if groups[k] != groups[k - 1]:
            ax.axhline(ypos[k] + 0.5, color=style.LIGHT_GREY, lw=0.6)
    for y, (lab, med, lo_, hi_, grp) in zip(ypos, rows):
        col = colors[grp]
        lo_p = np.nan if np.isnan(lo_) else min(lo_, INF_PLOT)
        hi_p = np.nan if np.isnan(hi_) else min(hi_, INF_PLOT)
        if not np.isnan(lo_p):
            ax.hlines(y, lo_p, hi_p, color=col, lw=1.4)
            if np.isinf(hi_):
                ax.annotate("", xy=(INF_PLOT * 1.6, y), xytext=(hi_p, y),
                            arrowprops={"arrowstyle": "->", "color": col,
                                        "lw": 1.0})
        if np.isinf(med):
            ax.plot(INF_PLOT, y, ">", color=col, ms=5)
            ax.text(INF_PLOT * 0.75, y + 0.2, "∞", fontsize=7,
                    color=col, ha="right")
        else:
            ax.plot(med, y, "o", color=col, ms=4)
    ax.axvline(span, color=style.GREY, lw=0.7, ls="--")
    ax.text(span * 1.1, ypos[0] + 0.55, "record length", fontsize=6.5,
            color=style.GREY)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.8)
    ax.set_xscale("log")
    style.plain_log_axis(ax)
    ax.set_xlim(300, 6.0e6)
    ax.set_ylim(-0.7, len(rows) - 0.1)
    ax.set_xlabel("Return period of 520-525 CE anomaly (yr)")
    for grp, lab in (("with", "Event in fit"), ("local", "Local frame"),
                     ("without", "Event left out")):
        ax.plot([], [], "o-", color=colors[grp], ms=4, label=lab)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              fontsize=6.8)
    style.panel_label(ax, "a", x=-0.62, y=1.06)

    ax = fig.add_subplot(gs[1])
    yy = np.arange(len(metrics))[::-1]
    pct = [100.0 * (n - r + 1) / n for _, r, n in metrics]
    ax.hlines(yy, 0, pct, color=style.LIGHT_GREY, lw=2.5)
    ax.plot(pct, yy, "o", color=style.EVENT, ms=4)
    for y, (_, r, n), p in zip(yy, metrics, pct):
        ax.text(p - 2, y + 0.25, f"{r} of {n}", fontsize=6.5, ha="right")
    ax.set_yticks(yy)
    ax.set_yticklabels([m[0] for m in metrics], fontsize=6.8)
    ax.set_xlim(0, 104)
    ax.set_ylim(-0.6, len(metrics) - 0.2)
    ax.set_xlabel("Percentile of event (%)")
    ax.set_title(f"Margin over next peak: {level - second:.1f}‰; "
                 f"z = {z:.1f}", fontsize=7)
    style.panel_label(ax, "b", x=-0.75, y=1.06)
    style.save(fig, "fig11_fingerprint")

    # ----------------------------------------------- figure: context
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, 2.9))
    z0, z1 = 380.0, 720.0
    sel = (native["age_ce"] >= z0) & (native["age_ce"] <= z1)
    zb = (yrs >= z0) & (yrs <= z1)
    ax.axvspan(ev_run["start_ce"], ev_run["end_ce"], color=style.DRY,
               alpha=0.12, lw=0)
    ax.text(ev_run["start_ce"] + 2, -1.7,
            f"drought spell\n{ev_run['start_ce']:.0f}-"
            f"{ev_run['end_ce']:.0f} CE", ha="left", va="center",
            fontsize=7, color=style.VERMILION)
    ax.axhline(0, color=style.GREY, lw=0.5, ls="--")
    ax.plot(native["age_ce"][sel], native["anomaly"][sel],
            color=style.GREY, lw=0.6, label="Native samples")
    ax.plot(yrs[zb], a[zb], color=style.BLACK, lw=1.0, label="5-yr bins")
    ax.plot(yrs[zb], binned["background"][zb], color=style.BLUE, lw=1.4,
            label="Background")
    pk = native.loc[native["anomaly"].idxmax()]
    ax.errorbar(cfg.EVENT_PEAK_AGE, pk["anomaly"] + 0.45,
                xerr=cfg.EVENT_AGE_ERROR, fmt="^", color=style.EVENT,
                ms=4, capsize=2.5, lw=0.9,
                label="Peak 523 ± 30 CE (paper)")
    # Each event line stops at its own label row, so no label is crossed.
    # The anomaly axis is inverted (drought down), so label rows sit at
    # negative values near the top of the panel.
    for k, (yr, lab) in enumerate(cfg.HISTORICAL_EVENTS):
        y_lab = -3.9 + 0.45 * k
        ax.vlines(yr, y_lab, 4.8, color=style.PURPLE, lw=0.7, ls=":")
        ax.text(yr + 2, y_lab, lab, fontsize=6.8, color=style.PURPLE,
                va="center")
    ax.set_xlim(z0, z1)
    ax.set_ylim(4.8, -4.2)
    ax.set_yticks(np.arange(-2, 5, 1))
    ax.set_xlabel("Year (CE)")
    ax.set_ylabel(style.ANOMALY_LABEL + "\n" + style.DRY_DOWN)
    leg = ax.legend(loc="lower right", ncol=2, fontsize=6.8, frameon=True,
                    framealpha=1.0, edgecolor="none")
    leg.set_zorder(5)
    style.save(fig, "fig12_context")

    # ------------------------------------------------------- reports
    rep = Report("10_fingerprint", "Step 10. Fingerprint of the event")
    rep.section("Magnitude")
    rep.kv("Peak native anomaly",
           f"{pk['anomaly']:.2f} permil at {pk['age_ce']:.1f} CE")
    rep.kv("Peak 5-yr anomaly", f"{level:.2f} permil (rank 1 of {len(a)})")
    rep.kv("z-score against rest of record", f"{z:.1f}")
    rep.kv("Margin over next independent peak",
           f"{level - second:.2f} permil")
    rep.section("Persistence (run theory)")
    rep.kv("Spell", f"{ev_run['start_ce']:.0f} to {ev_run['end_ce']:.0f} CE")
    for lab, r, n in metrics[3:]:
        rep.kv(f"  {lab}", f"rank {r} of {n}")
    rep.section("Return period of the 520-525 CE anomaly")
    for lab, med, lo_, hi_, grp in rows:
        rng_txt = ("" if np.isnan(lo_) else
                   f" [{fmt_years(lo_)}, {fmt_years(hi_)}]")
        rep.kv(f"  [{grp}] {lab}", f"{fmt_years(med)}{rng_txt}")
    rob = res["robustness"]
    rep.kv("  robustness grid, event in fit",
           f"{fmt_years(rob['T_with_min'])} to "
           f"{fmt_years(rob['T_with_max'])} yr over {rob['n_cells']} "
           "choices")
    rep.section("Timing")
    rep.para(
        f"The binned peak falls in the 520-525 CE bin and the native peak "
        f"at {pk['age_ce']:.0f} CE. The published age uncertainty of about "
        f"+/-{cfg.EVENT_AGE_ERROR:.0f} yr spans {cfg.EVENT_PEAK_AGE - 30:.0f}"
        f" to {cfg.EVENT_PEAK_AGE + 30:.0f} CE, which contains the "
        "Aksumite conquest (525 CE) and the last Himyarite inscription "
        "(about 559 CE) but would also contain any other event of the "
        "same decades. Timing is consistent with the historical sequence "
        "without being diagnostic of it.")
    rep.write()

    pot, lo = res["pot"], res["leave_out"]
    s = Report("00_summary", "Summary for presentation")
    s.para(
        "Aim: to quantify how extreme, in statistical terms, the "
        "sixth-century drought recorded by Hoti Cave stalagmite H12 "
        "(Fleitmann et al. 2022) is, and to identify which parts of that "
        "claim are robust.")
    s.section("Answers")
    s.para(
        f"1. Magnitude. The 520-525 CE anomaly of {level:.2f} permil is "
        f"the largest value in {span:.0f} yr, {z:.1f} standard deviations "
        f"above the rest of the record and {level - second:.1f} permil "
        "above the next independent dry peak. A 30-yr window carries "
        "most of the excess kurtosis of the whole record.")
    s.para(
        f"2. Persistence. It sits inside a {ev_run['duration_yr']:.0f}-yr "
        f"spell ({ev_run['start_ce']:.0f} to {ev_run['end_ce']:.0f} CE), "
        "ranked first in peak and second in severity and duration among "
        f"{len(runs)} spells.")
    s.para(
        f"3. Rarity with the event in the fit. Threshold and block models "
        f"agree on a return period of roughly "
        f"{fmt_years(rob['T_with_min'])} to {fmt_years(rob['T_with_max'])}"
        f" yr across {rob['n_cells']} analysis choices (MLE "
        f"{fmt_years(pot['T_mle'])} yr). No choice brings it below the "
        "record length.")
    s.para(
        "4. Rarity judged by the rest of the record. Leaving the window "
        "out, the data alone bound the dry tail near "
        f"{_f(lo['mle_without']['endpoint']):.1f} permil, below the "
        "event. With a neutral prior on the tail shape, "
        f"{_f(lo['uniform_gpd_without']['frac_inf']) * 100:.0f} % of "
        "posterior draws put the event beyond the tail. With a heavy-tail "
        "geophysical prior the median is "
        f"{fmt_years(_f(lo['bayes_gpd_without']['median']))} yr. The tail "
        "shape, which 19 peaks cannot resolve, controls the answer.")
    s.para(
        "5. Local frame. Against the climate state of its own century the "
        f"event still ranks first (residual {res['local']['event_resid']:.2f}"
        " permil), so it is not a by-product of a drier regime.")
    s.para(
        "6. Timing. The +/-30 yr age uncertainty overlaps the Aksumite "
        "conquest and the decline of Himyar but cannot single them out.")
    s.section("Bottom line")
    s.para(
        "The H12 record supports a record-breaking, multi-millennial "
        "drought in the early sixth century CE. The statistical evidence "
        "for its magnitude is strong and robust; its exact return period "
        "is uncertain by an order of magnitude; its link to Himyar rests "
        "on timing that the dating cannot resolve below a few decades.")
    s.section("Caveats")
    s.para(
        "Single stalagmite; d18O only (no d13C); no age-model ensemble; "
        "the transcribed table should be checked against the original "
        "NOAA file; isotopes are a proxy for winter-spring rainfall, not "
        "rainfall itself; 19 to 21 declustered peaks limit every tail "
        "model.")
    s.write()
    print("step 10 done")


if __name__ == "__main__":
    main()
