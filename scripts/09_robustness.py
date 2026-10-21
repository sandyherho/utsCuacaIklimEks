#!/usr/bin/env python
"""Step 9: sensitivity of the event return period to analysis choices.

Grid: time base (native, 2.5, 5, 10 yr) x threshold quantile x
declustering run length, each fitted with and without the event window.

Outputs
-------
data/processed/robustness_grid.csv
figures/fig10_robustness.(pdf|png)
reports/09_robustness.txt
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from h12eva import config as cfg
from h12eva import eva, io, results, style
from h12eva.report import Report, fmt_years


def series_for(base, native):
    """Return (years, anomaly) for a time base."""
    if base == "native":
        keep = np.r_[True, np.diff(native["age_ce"].to_numpy()) > 0]
        return (native["age_ce"].to_numpy()[keep],
                native["anomaly"].to_numpy()[keep])
    b = io.bin_record(native, width=float(base))
    b = b.dropna(subset=["anomaly"])
    return b["age_ce"].to_numpy(), b["anomaly"].to_numpy()


def main():
    style.apply()
    rng = np.random.default_rng(cfg.SEED + 4)
    native = io.load_native()
    rows = []
    boot = {}
    for base in cfg.ROBUST_BINS:
        yrs, a = series_for(base, native)
        mask = io.event_mask(yrs)
        level = float(a[mask].max())
        span = eva.span_years(yrs)
        for q in cfg.ROBUST_QUANTILES:
            u = float(np.quantile(a, q))
            for r in cfg.ROBUST_DECLUSTER:
                row = {"base": str(base), "q": q, "r_yr": r,
                       "level": level, "u": u}
                for tag, sel in (("with", slice(None)), ("without",
                                                         ~mask)):
                    m = eva.pot_model(yrs[sel], a[sel], u, r)
                    xi, sig = eva.pot_params(m)
                    rate = m.extremes.size / span
                    row[f"n_{tag}"] = m.extremes.size
                    row[f"xi_{tag}"] = xi
                    row[f"T_{tag}"] = float(eva.gpd_return_period(
                        level, u, xi, sig, rate))
                rows.append(row)
        u = float(np.quantile(a, cfg.POT_QUANTILE))
        _, pk = eva.declustered_peaks(yrs, a, u, cfg.DECLUSTER_YR)
        _, bt, _ = eva.bootstrap_pot(pk, u, span, level, [1000.0], rng,
                                     n_boot=1000)
        boot[str(base)] = eva.summarize_period(bt)
    grid = pd.DataFrame(rows)
    grid.to_csv(cfg.PROC_DIR / "robustness_grid.csv", index=False,
                float_format="%.4g")

    fin = grid["T_with"].replace(np.inf, np.nan)
    results.update("robustness", {
        "T_with_min": float(fin.min()), "T_with_max": float(fin.max()),
        "T_with_median": float(fin.median()),
        "frac_with_above_record": float((grid["T_with"] > 2630).mean()),
        "frac_without_infinite": float(np.isinf(grid["T_without"]).mean()),
        "n_cells": len(grid), "boot_by_base": boot,
    })

    # ------------------------------------------------------------ figure
    fig, axs = plt.subplots(1, 2, figsize=(style.WIDTH_FULL, 2.8),
                            gridspec_kw={"wspace": 0.62,
                                         "width_ratios": [1, 1.25]})
    ax = axs[0]
    sub = grid[grid["base"] == "5.0"]
    mat = sub.pivot(index="q", columns="r_yr", values="T_with")
    img = ax.imshow(np.log10(mat.to_numpy()), cmap="Oranges",
                    vmin=np.log10(4000), vmax=np.log10(25000),
                    aspect="auto", origin="lower")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.to_numpy()[i, j]
            ax.text(j, i, f"{v / 1000:.1f}k", ha="center", va="center",
                    fontsize=7, color=style.BLACK)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels([f"{c:.0f}" for c in mat.columns])
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels([f"q{v * 100:.0f}" for v in mat.index])
    ax.set_xlabel("Declustering run length (yr)")
    ax.set_ylabel("Threshold")
    ax.set_title("Return period, 5-yr bins, event included", fontsize=8)
    cb = fig.colorbar(img, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label(r"log$_{10}$ T (yr)")
    cb.outline.set_linewidth(0.5)
    style.panel_label(ax, "a", x=-0.22)

    ax = axs[1]
    bases = list(cfg.ROBUST_BINS)
    xs = np.arange(len(bases))
    for k, base in enumerate(bases):
        s = grid[(grid["base"] == str(base))]
        tw = s["T_with"].to_numpy()
        ax.plot(np.full(tw.size, k - 0.12), tw, "o", ms=3.5, mfc="none",
                mec=style.BLUE, mew=0.7)
        to = s["T_without"].to_numpy()
        n_inf = int(np.isinf(to).sum())
        ax.plot(np.full((~np.isinf(to)).sum(), k + 0.12),
                to[~np.isinf(to)], "s", ms=3.5, mfc="none",
                mec=style.GREEN, mew=0.7)
        ax.text(k + 0.12, 1.5e7, f"∞: {n_inf}/{to.size}", ha="center",
                va="bottom", fontsize=6.5, color=style.GREEN)
    ax.plot([], [], "o", mfc="none", mec=style.BLUE, label="With window")
    ax.plot([], [], "s", mfc="none", mec=style.GREEN,
            label="Without window (∞ = beyond endpoint)")
    ax.axhline(2630, color=style.GREY, lw=0.6, ls="--")
    ax.text(len(bases) - 0.55, 2630 * 1.25, "record length", fontsize=6.5,
            color=style.GREY, ha="right")
    ax.set_yscale("log")
    style.plain_log_axis(ax, axis="y")
    ax.set_ylim(500, 6.0e7)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Native" if b == "native" else f"{b:g}-yr"
                        for b in bases])
    ax.set_xlabel("Time base")
    ax.set_ylabel("Return period of event (yr)")
    ax.legend(loc="lower left", fontsize=6.5)
    style.panel_label(ax, "b", x=-0.20)
    style.save(fig, "fig10_robustness")

    # ------------------------------------------------------------ report
    rep = Report("09_robustness", "Step 9. Robustness of the return period")
    rep.para(
        f"{len(grid)} combinations of time base, threshold quantile and "
        "declustering length are fitted with pyextremes (MLE), with and "
        "without the 500-530 CE window. The event level is the largest "
        "window value on each time base, so it differs between bases.")
    rep.section("Summary over the grid")
    rep.kv("T with window, min / median / max",
           f"{fmt_years(fin.min())} / {fmt_years(fin.median())} / "
           f"{fmt_years(fin.max())} yr")
    rep.kv("Cells with T longer than the record",
           f"{(grid['T_with'] > 2630).sum()} of {len(grid)}")
    rep.kv("Without window, event beyond endpoint",
           f"{int(np.isinf(grid['T_without']).sum())} of {len(grid)}")
    rep.section("Bootstrap at q90, r = 10 yr, by time base (with window)")
    for base, s in boot.items():
        rep.kv(f"  {base}", f"median {fmt_years(s['median'])} yr "
               f"[{fmt_years(s['p05'])}, {fmt_years(s['p95'])}], "
               f"beyond endpoint {s['frac_inf'] * 100:.0f} %")
    rep.section("Full grid")
    show = grid[["base", "q", "r_yr", "level", "n_with", "xi_with",
                 "T_with", "n_without", "xi_without", "T_without"]]
    rep.table(show.set_index(["base", "q", "r_yr"]), floatfmt="{:.3g}")
    rep.section("Interpretation")
    rep.para(
        "No reasonable choice brings the return period below the length "
        "of the record. Including the event, estimates span roughly "
        f"{fmt_years(fin.min())} to {fmt_years(fin.max())} yr, a range "
        "set mostly by the tail shape at each threshold. Excluding it, "
        f"{int(np.isinf(grid['T_without']).sum())} of {len(grid)} "
        "combinations place the event outside the fitted tail, and the "
        "rest give longer return periods than with it included. The "
        "qualitative conclusion, a multi-millennial event, does not "
        "depend on resolution or on the threshold rules.")
    rep.write()
    print("step 9 done: T_with", fin.min(), fin.median(), fin.max(),
          "inf_without", int(np.isinf(grid["T_without"]).sum()), len(grid))


if __name__ == "__main__":
    main()
