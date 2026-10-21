#!/usr/bin/env python
"""Step 3: persistence, trend, regime shifts and spectral content.

Outputs
-------
figures/fig03_time_structure.(pdf|png)
reports/03_time_structure.txt
data/processed/time_structure.json
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from h12eva import config as cfg
from h12eva import io, stats, style
from h12eva.report import Report


def main():
    style.apply()
    rng = np.random.default_rng(cfg.SEED)
    native = io.load_native()
    binned = io.load_binned()
    t_nat = native["age_ce"].to_numpy()
    a_nat = native["anomaly"].to_numpy()
    yrs = binned["age_ce"].to_numpy()
    a = binned["anomaly"].to_numpy()

    tau_native = stats.tau_ar1(t_nat, a_nat)
    resid = a - binned["background"].to_numpy()
    tau_binned = stats.tau_ar1(yrs, a)
    tau_resid = stats.tau_ar1(yrs, resid)

    trend = stats.mann_kendall(a)
    pet = stats.pettitt(a)
    breaks = stats.binary_segmentation(a, min_size=30)
    seg_edges = [0] + breaks + [len(a)]
    seg_means = [a[i:j].mean() for i, j in zip(seg_edges[:-1],
                                               seg_edges[1:])]

    periods = np.geomspace(12.0, 1000.0, 600)
    keep = np.r_[True, np.diff(t_nat) > 0]
    t_ls, a_ls = t_nat[keep], a_nat[keep]
    power = stats.lomb_scargle(t_ls, a_ls, periods)
    red = stats.red_noise_levels(t_ls, tau_native, periods,
                                 cfg.N_SURROGATE, rng)
    sig95 = power > red[95]
    peaks = []
    for i in range(1, len(periods) - 1):
        if sig95[i] and power[i] >= power[i - 1] and \
                power[i] >= power[i + 1]:
            peaks.append((periods[i], power[i], power[i] > red[99][i]))

    lags = np.arange(0, 41)
    acf = np.array([1.0] + [np.corrcoef(a[:-k], a[k:])[0, 1]
                            for k in lags[1:]])
    acf_r = np.array([1.0] + [np.corrcoef(resid[:-k], resid[k:])[0, 1]
                              for k in lags[1:]])

    # ------------------------------------------------------------ figure
    fig = plt.figure(figsize=(style.WIDTH_FULL, 4.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.5,
                          wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    ax.plot(yrs, a, color=style.GREY, lw=0.6, label="5-yr anomaly")
    for i, j, m in zip(seg_edges[:-1], seg_edges[1:], seg_means):
        ax.hlines(m, yrs[i] - 2.5, yrs[j - 1] + 2.5, color=style.BLACK,
                  lw=1.6)
    ax.plot([], [], color=style.BLACK, lw=1.6,
            label="Segment means (binary segmentation)")
    for b in breaks:
        ax.axvline(yrs[b] - 2.5, color=style.BLUE, lw=0.6, ls="--")
    ax.axvline(yrs[pet["index"]] - 2.5, color=style.VERMILION, lw=1.2,
               label=f"Pettitt break ({yrs[pet['index']] - 2.5:.0f} CE)")
    ax.axvspan(*cfg.EVENT_WINDOW, color=style.ORANGE, alpha=0.25, lw=0)
    ax.set_xlim(-650, 2010)
    ax.set_xlabel(style.YEAR_LABEL)
    ax.set_ylabel("Anomaly (‰)\n" + style.DRY_DOWN)
    style.dry_down(ax)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.0))
    style.panel_label(ax, "a", x=-0.06, y=1.10)

    ax = fig.add_subplot(gs[1, 0])
    ax.semilogx(periods, power, color=style.BLACK, lw=0.9,
                label="Lomb-Scargle")
    ax.semilogx(periods, red[95], color=style.BLUE, lw=0.8, ls="--",
                label="AR(1) 95 %")
    ax.semilogx(periods, red[99], color=style.BLUE, lw=0.8, ls=":",
                label="AR(1) 99 %")
    for p, w, s99 in peaks:
        if s99:
            ax.annotate(f"{p:.0f} yr", xy=(p, w), xytext=(p, w * 1.06),
                        ha="center", fontsize=7, color=style.VERMILION)
    ax.set_ylim(0, max(power.max(), red[99].max()) * 1.4)
    ax.set_xticks([1000, 500, 200, 100, 50, 20])
    ax.set_xticklabels(["1000", "500", "200", "100", "50", "20"])
    ax.minorticks_off()
    ax.set_xlabel("Period (yr)")
    ax.set_ylabel("Normalized power")
    ax.invert_xaxis()
    ax.legend(loc="upper right")
    style.panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 1])
    lag_yr = lags * cfg.BIN_WIDTH
    ax.plot(lag_yr, acf, "o-", color=style.BLACK, ms=2.5, lw=0.8,
            label="Anomaly")
    ax.plot(lag_yr, acf_r, "s-", color=style.VERMILION, ms=2.5, lw=0.8,
            label="Minus LOWESS background")
    ax.plot(lag_yr, np.exp(-lag_yr / tau_binned), color=style.BLUE,
            lw=0.8, ls="--",
            label=rf"AR(1), $\tau$ = {tau_binned:.0f} yr")
    ax.axhline(0, color=style.GREY, lw=0.5)
    ax.set_xlabel("Lag (yr)")
    ax.set_ylabel("Autocorrelation")
    ax.legend(loc="upper right")
    style.panel_label(ax, "c")
    style.save(fig, "fig03_time_structure")

    out = {"tau_native": tau_native, "tau_binned": tau_binned,
           "tau_residual": tau_resid, "pettitt_year": yrs[pet["index"]]
           - 2.5, "pettitt_p": pet["p"],
           "breaks_ce": [float(yrs[b] - 2.5) for b in breaks],
           "segment_means": seg_means, "mk": trend,
           "spectral_peaks_95": [[float(p), float(w), bool(s)]
                                 for p, w, s in peaks]}
    (cfg.PROC_DIR / "time_structure.json").write_text(
        json.dumps(out, indent=2, default=float))

    # ------------------------------------------------------------ report
    rep = Report("03_time_structure", "Step 3. Time structure")
    rep.section("Persistence (Mudelsee 2002 least-squares estimator)")
    rep.kv("tau, native uneven series", f"{tau_native:.1f} yr")
    rep.kv("tau, 5-yr bins", f"{tau_binned:.1f} yr")
    rep.kv("tau, 5-yr bins minus LOWESS background",
           f"{tau_resid:.1f} yr")
    rep.para(
        f"Excursions persist for roughly a decade. Binning raises tau "
        f"because the 5-yr average itself adds correlation, so the "
        f"native value is the better physical estimate. The declustering "
        f"window for the threshold models is set to {cfg.DECLUSTER_YR:.0f}"
        f" yr, close to tau, so that peaks closer than one persistence "
        "time count as a single event. The residual series still carries "
        "multi-decadal memory, visible as slowly decaying "
        "autocorrelation, so effective sample sizes are much smaller than "
        "the number of bins.")
    rep.section("Monotonic trend (Hamed-Rao modified Mann-Kendall)")
    rep.kv("Kendall tau", f"{trend['tau']:.3f}")
    rep.kv("z, p", f"{trend['z']:.2f}, {trend['p']:.3g}")
    rep.kv("Sen slope", f"{trend['sen_slope_per_step'] * 200:.3f} "
           "permil per kyr")
    rep.section("Regime shifts")
    rep.kv("Pettitt change point",
           f"{yrs[pet['index']] - 2.5:.0f} CE (p = {pet['p']:.2g})")
    rep.kv("Binary segmentation breaks (CE)",
           ", ".join(f"{yrs[b] - 2.5:.0f}" for b in breaks))
    rep.kv("Segment means (permil)",
           ", ".join(f"{m:.2f}" for m in seg_means))
    rep.para(
        "Binary segmentation uses a BIC penalty and a minimum segment of "
        "150 yr. Autocorrelation makes such tests liberal, so breaks are "
        "read as regime descriptions, not as formal detections. The "
        "segment means show that the mean state shifts by more than "
        "1 permil over the record, which is the main reason a single "
        "stationary extreme value model needs the checks in step 8.")
    rep.section("Spectral content (Lomb-Scargle vs AR(1) red noise)")
    rep.para(
        f"Red-noise levels come from {cfg.N_SURROGATE} AR(1) surrogates "
        "generated on the native sampling times with the estimated tau. "
        "Local maxima above the 95 % level:")
    if peaks:
        for p, w, s99 in peaks:
            rep.kv(f"  period {p:7.1f} yr", f"power {w:.3f}"
                   + ("  (also above 99 %)" if s99 else ""))
    else:
        rep.kv("  none", "")
    rep.para(
        "Isolated exceedances of the 95 % level are expected by chance "
        "across many frequencies, so only peaks above the 99 % level "
        "deserve attention. A peak near 409 yr clears the 99 % level, but "
        "the record contains only about six such cycles, so it is read "
        "as suggestive multi-centennial variability, not as a periodic "
        "driver. Most variance sits in red-noise-like power at long "
        "periods, consistent with regime behavior.")
    rep.write()
    print("step 3 done: tau", round(tau_native, 1), round(tau_binned, 1),
          "breaks", out["breaks_ce"], "pettitt", out["pettitt_year"],
          "peaks", [(round(p), s) for p, w, s in peaks])


if __name__ == "__main__":
    main()
