#!/usr/bin/env python
"""Step 12: explanatory figure for the peaks-over-threshold method.

Panel (a) shows, on the real record, how independent dry events and their
excesses over the threshold are extracted. Panel (b) shows how the GPD
shape parameter controls the tail.

Outputs
-------
figures/fig13_pot_method.(pdf|png)
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
    u = float(np.quantile(a, cfg.POT_QUANTILE))
    pk_t, pk_x = eva.declustered_peaks(yrs, a, u, cfg.DECLUSTER_YR)

    fig, axs = plt.subplots(1, 2, figsize=(style.WIDTH_FULL, 2.7),
                            gridspec_kw={"width_ratios": [1.7, 1],
                                         "wspace": 0.32})
    ax = axs[0]
    z0, z1 = 380.0, 700.0
    z = (yrs >= z0) & (yrs <= z1)
    ax.fill_between(yrs[z], u, a[z], where=a[z] > u, color=style.DRY,
                    alpha=0.35, lw=0, interpolate=True)
    ax.plot(yrs[z], a[z], color=style.BLACK, lw=0.9)
    ax.axhline(u, color=style.VERMILION, lw=0.9, ls="--")
    ax.text(z1 - 3, u - 0.06, f"u = {u:.2f}‰ (q90)",
            ha="right", va="bottom", fontsize=7, color=style.VERMILION)
    sel = (pk_t >= z0) & (pk_t <= z1)
    ax.plot(pk_t[sel], pk_x[sel], "o", ms=4, mfc="white",
            mec=style.BLACK, mew=0.8, zorder=4,
            label="Independent event peak")
    j = int(np.argmax(pk_x))
    ax.annotate("", xy=(pk_t[j] + 9, pk_x[j]), xytext=(pk_t[j] + 9, u),
                arrowprops={"arrowstyle": "<->", "color": style.BLUE,
                            "lw": 0.9})
    ax.text(pk_t[j] + 13, (u + pk_x[j]) / 2, "excess\ny = x - u",
            fontsize=7, color=style.BLUE, va="center")
    ax.annotate("peaks closer than 10 yr\nmerge into one event",
                xy=(497.5, 1.75), xytext=(395, 2.9), fontsize=7,
                color=style.GREY,
                arrowprops={"arrowstyle": "->", "color": style.GREY,
                            "lw": 0.6})
    ax.set_xlim(z0, z1)
    ax.set_ylim(3.9, -1.6)
    ax.set_xlabel("Year (CE)")
    ax.set_ylabel("Anomaly x (‰), " + style.DRY_DOWN)
    ax.legend(loc="upper right")
    style.panel_label(ax, "a", x=-0.10)

    ax = axs[1]
    y = np.linspace(0.0, 4.0, 400)
    for xi, col, lab in ((-0.3, style.GREEN, r"$\xi<0$: upper limit"),
                         (0.0, style.BLACK, r"$\xi=0$: exponential"),
                         (0.3, style.VERMILION, r"$\xi>0$: heavy tail")):
        sf = sps.genpareto.sf(y, xi, scale=0.8)
        ax.semilogy(y, np.where(sf > 0, sf, np.nan), color=col, lw=1.3,
                    label=lab)
    ax.axvline(0.8 / 0.3, color=style.GREEN, lw=0.6, ls=":")
    ax.text(0.8 / 0.3 + 0.06, 2e-3, "limit\n" + r"$-\sigma/\xi$",
            fontsize=6.5, color=style.GREEN)
    ax.set_ylim(1e-4, 1.2)
    ax.set_xlabel("Excess y (‰)")
    ax.set_ylabel("P(excess > y)")
    ax.legend(loc="lower left", fontsize=6.8)
    ax.text(0.97, 0.97, r"$\sigma$ = 0.8‰", transform=ax.transAxes,
            ha="right", va="top", fontsize=7, color=style.GREY)
    style.panel_label(ax, "b", x=-0.22)
    style.save(fig, "fig13_pot_method")
    print("step 12 done")


if __name__ == "__main__":
    main()
