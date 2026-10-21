#!/usr/bin/env python
"""Step 2: descriptive statistics, overall and by hydroclimate epoch.

Outputs
-------
figures/fig02_descriptive.(pdf|png)
reports/02_descriptive.txt
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

from h12eva import config as cfg
from h12eva import io, stats, style
from h12eva.report import Report


def epoch_of(years):
    """Label each year with its epoch name."""
    labels = np.empty(len(years), dtype=object)
    for name, (lo, hi) in cfg.EPOCHS.items():
        labels[(years >= lo) & (years < hi)] = name
    return labels


def main():
    style.apply()
    binned = io.load_binned()
    native = io.load_native()
    yrs = binned["age_ce"].to_numpy()
    a = binned["anomaly"].to_numpy()
    binned["epoch"] = epoch_of(yrs)
    ev = io.event_mask(yrs)
    names = list(cfg.EPOCHS)

    table = pd.DataFrame({"All bins": stats.describe(a)})
    table["Native samples"] = stats.describe(native["anomaly"])
    for name in names:
        table[name] = stats.describe(a[binned["epoch"] == name])
    table["Excluding 500-530 CE"] = stats.describe(a[~ev])
    kw = sps.kruskal(*[a[binned["epoch"] == n] for n in names])
    jb = sps.jarque_bera(a)
    jb_ex = sps.jarque_bera(a[~ev])
    z_event = (a[ev].max() - a.mean()) / a.std(ddof=1)
    z_event_ex = (a[ev].max() - a[~ev].mean()) / a[~ev].std(ddof=1)

    win = int(200 / cfg.BIN_WIDTH)
    run_sd = pd.Series(a).rolling(win, center=True, min_periods=win // 2)
    run_sd = run_sd.std()

    # ------------------------------------------------------------ figure
    fig, axs = plt.subplots(2, 2, figsize=(style.WIDTH_FULL, 4.6),
                            gridspec_kw={"hspace": 0.45, "wspace": 0.28})
    ax = axs[0, 0]
    bins = np.arange(-2.6, 3.8, 0.2)
    for name, col in zip(names, style.EPOCH_COLORS):
        ax.hist(a[binned["epoch"] == name], bins=bins, histtype="step",
                color=col, lw=1.1, density=True, label=name.split(" ")[0])
    ax.axvline(a[ev].max(), color=style.EVENT, lw=0.8, ls=":")
    ax.annotate("520-525 CE", xy=(a[ev].max(), 0.05),
                xytext=(2.0, 0.35), color=style.EVENT, fontsize=7,
                arrowprops={"arrowstyle": "->", "color": style.EVENT,
                            "lw": 0.6})
    ax.set_xlabel(style.ANOMALY_LABEL)
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", bbox_to_anchor=(0.9, 1.0))
    style.panel_label(ax, "a")

    ax = axs[0, 1]
    data = [a[binned["epoch"] == n] for n in names]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=True,
                    flierprops={"marker": "o", "ms": 2,
                                "mec": style.GREY, "mfc": "none"},
                    medianprops={"color": style.BLACK, "lw": 1.0})
    for patch, col in zip(bp["boxes"], style.EPOCH_COLORS):
        patch.set_facecolor(col)
        patch.set_alpha(0.35)
        patch.set_edgecolor(col)
    ax.plot(2, a[ev].max(), "*", color=style.EVENT, ms=8,
            label="520-525 CE bin")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([n.split(" ")[0] for n in names])
    ax.set_ylabel(style.ANOMALY_LABEL)
    ax.legend(loc="upper left")
    style.panel_label(ax, "b")

    ax = axs[1, 0]
    (osm, osr), (slope, icpt, _) = sps.probplot(a, dist="norm")
    ax.plot(osm, slope * osm + icpt, color=style.GREY, lw=0.8)
    ax.plot(osm, osr, "o", ms=2.2, mfc="none", mec=style.BLACK, mew=0.5)
    top = np.argmax(osr)
    ax.plot(osm[top], osr[top], "*", color=style.EVENT, ms=8)
    ax.set_xlabel("Normal theoretical quantile")
    ax.set_ylabel("Sample quantile (‰)")
    style.panel_label(ax, "c")

    ax = axs[1, 1]
    ax.plot(yrs, run_sd, color=style.BLACK)
    ax.axvspan(*cfg.EVENT_WINDOW, color=style.ORANGE, alpha=0.25, lw=0)
    for (lo, hi), col in zip(cfg.EPOCHS.values(), style.EPOCH_COLORS):
        ax.axvspan(max(lo, yrs.min()), min(hi, yrs.max()), ymin=0,
                   ymax=0.04, color=col, lw=0)
        ax.text((max(lo, yrs.min()) + min(hi, yrs.max())) / 2, 0.06,
                [k for k, v in cfg.EPOCHS.items() if v == (lo, hi)][0]
                .split(" ")[0], transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7, color=col)
    ax.set_xlabel(style.YEAR_LABEL)
    ax.set_ylabel("200-yr running SD (‰)")
    ax.set_xlim(-650, 2010)
    style.panel_label(ax, "d")
    style.save(fig, "fig02_descriptive")

    # ------------------------------------------------------------ report
    rep = Report("02_descriptive", "Step 2. Descriptive statistics")
    rep.para(
        "Statistics refer to d18O anomalies (permil VPDB, relative to "
        "-1.8 permil) on the 5-yr grid unless stated. Positive anomaly "
        "means drier than 1978-1997.")
    rep.section("Summary table")
    rep.table(table.T)
    rep.section("Distribution shape")
    rep.kv("Jarque-Bera, all bins", f"JB = {jb.statistic:.1f}, "
           f"p = {jb.pvalue:.2e}")
    rep.kv("Jarque-Bera, excluding 500-530 CE",
           f"JB = {jb_ex.statistic:.1f}, p = {jb_ex.pvalue:.2e}")
    rep.kv("Kruskal-Wallis across epochs",
           f"H = {kw.statistic:.1f}, p = {kw.pvalue:.2e}")
    rep.kv("Event bin z-score, all bins", f"{z_event:.2f} SD")
    rep.kv("Event bin z-score vs rest of record", f"{z_event_ex:.2f} SD")
    rep.section("Interpretation")
    e1, e2, e3 = names
    rep.para(
        f"Median anomalies differ between epochs: "
        f"{table.loc['median', e1]:.2f}, {table.loc['median', e2]:.2f} "
        f"and {table.loc['median', e3]:.2f} permil for {e1}, {e2} and "
        f"{e3}. The middle epoch is the driest, consistent with the "
        "persistently low effective moisture reported for about 480 to "
        "1400 CE. The record is therefore not stationary in its mean, "
        "which matters for the extreme value models in steps 5 to 8.")
    rep.para(
        f"The 520-525 CE bin lies {z_event_ex:.1f} standard deviations "
        "above the mean of the remaining record and is the largest value "
        "on the grid. Positive skewness and excess kurtosis indicate a "
        "heavier dry tail than a Gaussian, and the normal quantile plot "
        "shows the event as the single point furthest from the line.")
    rep.para(
        f"Removing the six bins of 500-530 CE lowers the excess kurtosis "
        f"from {table.loc['ex_kurtosis', 'All bins']:.2f} to "
        f"{table.loc['ex_kurtosis', 'Excluding 500-530 CE']:.2f} and the "
        f"skewness from {table.loc['skewness', 'All bins']:.2f} to "
        f"{table.loc['skewness', 'Excluding 500-530 CE']:.2f}. A 30-yr "
        "window, about 1 % of the record, therefore carries most of the "
        "heavy-tail signature of the full distribution.")
    rep.write()
    print("step 2 done: event z =", round(z_event_ex, 2))


if __name__ == "__main__":
    main()
