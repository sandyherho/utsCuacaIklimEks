#!/usr/bin/env python
"""Step 13: explanatory figure for the block maxima method.

Panel (a) shows the 5-yr record cut into 50-yr blocks with the driest
value of each block marked. Panel (b) compares the 53 block maxima with
the fitted GEV density. Panel (c) shows how the GEV shape parameter
changes the tail.

Outputs
-------
figures/fig14_bm_method.(pdf|png)
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps

from h12eva import config as cfg
from h12eva import eva, io, style


def main():
    style.apply()
    binned = io.load_binned()
    yrs = binned["age_ce"].to_numpy()
    a = binned["anomaly"].to_numpy()
    block = cfg.BLOCK_YR[0]
    model = eva.bm_model(yrs, a, block)
    xi, mu, sig = eva.bm_params(model)
    bm_t = io.from_pseudo_index(model.extremes.index)
    bm_x = model.extremes.to_numpy()
    level = float(a[io.event_mask(yrs)].max())

    fig = plt.figure(figsize=(style.WIDTH_FULL, 4.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.55,
                          wspace=0.32)

    ax = fig.add_subplot(gs[0, :])
    edges = np.arange(yrs.min() - cfg.BIN_WIDTH / 2, yrs.max() + block,
                      block)
    for k, e in enumerate(edges[:-1]):
        if k % 2 == 0:
            ax.axvspan(e, e + block, color=style.LIGHT_GREY, alpha=0.45,
                       lw=0)
    ax.plot(yrs, a, color=style.GREY, lw=0.6, label="5-yr anomaly")
    ax.plot(bm_t, bm_x, "o", ms=3.2, mfc="white", mec=style.BLACK,
            mew=0.8, label=f"Driest value in each {block:.0f}-yr block")
    j = int(np.argmax(bm_x))
    ax.plot(bm_t[j], bm_x[j], "*", color=style.EVENT, ms=9, zorder=4,
            label="520-525 CE")
    ax.set_xlim(yrs.min() - 5, yrs.max() + 5)
    ax.set_xlabel(style.YEAR_LABEL)
    ax.set_ylabel("Anomaly (‰), " + style.DRY_DOWN)
    style.dry_down(ax)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.0))
    style.panel_label(ax, "a", x=-0.07, y=1.10)

    ax = fig.add_subplot(gs[1, 0])
    bins = np.arange(-1.2, 3.8, 0.3)
    ax.hist(bm_x, bins=bins, density=True, color=style.LIGHT_GREY,
            edgecolor="white", lw=0.5, label=f"{bm_x.size} block maxima")
    z = np.linspace(-1.2, 3.8, 300)
    ax.plot(z, sps.genextreme.pdf(z, -xi, loc=mu, scale=sig),
            color=style.BLUE, lw=1.3, label="Fitted GEV")
    ax.axvline(level, color=style.EVENT, lw=0.8, ls=":")
    ax.text(level - 0.05, 0.42, "520-525 CE", rotation=90, ha="right",
            va="top", fontsize=7, color=style.EVENT)
    ax.set_xlabel(f"{block:.0f}-yr maximum anomaly z (‰)")
    ax.set_ylabel("Probability density")
    ax.legend(loc="upper left", fontsize=6.8)
    ax.text(0.80, 0.62, rf"$\mu$ = {mu:.2f}‰" "\n"
            rf"$\sigma$ = {sig:.2f}‰" "\n" rf"$\xi$ = {xi:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7)
    style.panel_label(ax, "b", x=-0.20)

    ax = fig.add_subplot(gs[1, 1])
    zz = np.linspace(-2.0, 6.0, 500)
    for x_, col, lab in ((-0.3, style.GREEN, r"$\xi<0$: has a ceiling"),
                         (0.0, style.BLACK, r"$\xi=0$: Gumbel"),
                         (0.3, style.VERMILION, r"$\xi>0$: no ceiling")):
        sf = sps.genextreme.sf(zz, -x_, loc=0.0, scale=0.7)
        ax.semilogy(zz, np.where(sf > 0, sf, np.nan), color=col, lw=1.3,
                    label=lab)
    ax.set_ylim(1e-4, 1.5)
    ax.set_xlim(-1.0, 6.0)
    ax.set_xlabel("Block maximum z (‰)")
    ax.set_ylabel("P(block maximum > z)")
    ax.legend(loc="lower left", fontsize=6.8)
    ax.text(0.97, 0.97, r"$\mu$ = 0, $\sigma$ = 0.7‰",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            color=style.GREY)
    style.panel_label(ax, "c", x=-0.20)
    style.save(fig, "fig14_bm_method")
    print("step 13 done")


if __name__ == "__main__":
    main()
