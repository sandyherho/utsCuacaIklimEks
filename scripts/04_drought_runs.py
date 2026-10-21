#!/usr/bin/env python
"""Step 4: drought spells by run theory (duration, severity, peak).

Outputs
-------
data/processed/drought_runs.csv
figures/fig04_drought_runs.(pdf|png)
reports/04_drought_runs.txt
"""

import matplotlib.pyplot as plt
import numpy as np

from h12eva import config as cfg
from h12eva import io, stats, style
from h12eva.report import Report


def main():
    style.apply()
    binned = io.load_binned()
    yrs = binned["age_ce"].to_numpy()
    a = binned["anomaly"].to_numpy()
    thr = float(np.quantile(a, cfg.RUN_QUANTILE))
    runs = stats.drought_runs(yrs, a, thr, cfg.BIN_WIDTH,
                              cfg.RUN_POOL_BINS)
    lo, hi = cfg.EVENT_WINDOW
    is_event = (runs["start_ce"] <= hi) & (runs["end_ce"] >= lo)
    runs["event"] = is_event
    for col in ("duration_yr", "severity", "peak", "intensity"):
        runs[f"rank_{col}"] = runs[col].rank(ascending=False,
                                             method="min").astype(int)
    runs.to_csv(cfg.PROC_DIR / "drought_runs.csv", index=False,
                float_format="%.3f")
    ev = runs[runs["event"]].sort_values("severity").iloc[-1]

    # ------------------------------------------------------------ figure
    fig, axs = plt.subplots(1, 3, figsize=(style.WIDTH_FULL, 2.6),
                            gridspec_kw={"wspace": 0.42,
                                         "width_ratios": [1.6, 1, 1]})
    ax = axs[0]
    ax.plot(yrs, a, color=style.GREY, lw=0.5)
    ax.axhline(thr, color=style.BLACK, lw=0.6, ls="--")
    ax.text(1150, thr + 0.35, f"q{cfg.RUN_QUANTILE * 100:.0f} = "
            f"{thr:.2f}‰", fontsize=7)
    for _, r in runs.iterrows():
        col = style.EVENT if r["event"] else style.BLUE
        ax.axvspan(r["start_ce"], r["end_ce"], ymin=0.94, ymax=1.0,
                   color=col, lw=0)
    ax.set_xlim(-650, 2010)
    ax.set_xlabel(style.YEAR_LABEL)
    ax.set_ylabel("Anomaly (‰), " + style.DRY_DOWN)
    style.dry_down(ax)
    style.panel_label(ax, "a", x=-0.14)

    ax = axs[1]
    other = runs[~runs["event"]]
    ax.scatter(other["duration_yr"], other["severity"],
               s=6 + 18 * (other["peak"] - thr), c=style.BLUE,
               alpha=0.6, lw=0)
    ax.scatter(ev["duration_yr"], ev["severity"],
               s=6 + 18 * (ev["peak"] - thr), c=style.EVENT,
               marker="*", lw=0, zorder=3)
    ax.annotate(f"{ev['start_ce']:.0f}-{ev['end_ce']:.0f} CE",
                xy=(ev["duration_yr"], ev["severity"]),
                xytext=(0.45, 0.50), textcoords="axes fraction",
                fontsize=7, color=style.EVENT,
                arrowprops={"arrowstyle": "->", "color": style.EVENT,
                            "lw": 0.6})
    longest = runs.sort_values("severity").iloc[-1]
    if not longest["event"]:
        ax.annotate(f"{longest['start_ce']:.0f}-{longest['end_ce']:.0f} CE",
                    xy=(longest["duration_yr"], longest["severity"]),
                    xytext=(-4, -14), textcoords="offset points",
                    ha="right", fontsize=7, color=style.BLUE)
    ax.text(0.97, 0.03, "marker size = peak", transform=ax.transAxes,
            ha="right", fontsize=6.5, color=style.GREY)
    ax.set_xlabel("Duration (yr)")
    ax.set_ylabel("Severity (‰ yr)")
    style.panel_label(ax, "b", x=-0.30)

    ax = axs[2]
    metrics = [("peak", "Peak"), ("severity", "Severity"),
               ("intensity", "Intensity"), ("duration_yr", "Duration")]
    pct = [stats.percentile_rank(runs[m], ev[m]) for m, _ in metrics]
    ranks = [int(ev[f"rank_{m}"]) for m, _ in metrics]
    ypos = np.arange(len(metrics))[::-1]
    ax.hlines(ypos, 0, pct, color=style.LIGHT_GREY, lw=2.5)
    ax.plot(pct, ypos, "o", color=style.EVENT, ms=4)
    for y, p, r in zip(ypos, pct, ranks):
        ax.text(p - 3, y + 0.22, f"rank {r}/{len(runs)}", fontsize=6.5,
                ha="right", color=style.BLACK)
    ax.set_yticks(ypos)
    ax.set_yticklabels([lab for _, lab in metrics])
    ax.set_xlim(0, 104)
    ax.set_ylim(-0.6, len(metrics) - 0.3)
    ax.set_xlabel("Percentile among spells (%)")
    style.panel_label(ax, "c", x=-0.42)
    style.save(fig, "fig04_drought_runs")

    # ------------------------------------------------------------ report
    rep = Report("04_drought_runs", "Step 4. Drought spells (run theory)")
    rep.para(
        f"A drought spell is a run of consecutive 5-yr bins with anomaly "
        f"above the {cfg.RUN_QUANTILE * 100:.0f}th percentile "
        f"({thr:.2f} permil). Runs separated by at most "
        f"{cfg.RUN_POOL_BINS} sub-threshold bin are pooled. Severity is "
        "the summed exceedance multiplied by the bin width (permil yr); "
        "intensity is severity divided by duration.")
    rep.kv("Number of spells", f"{len(runs)}")
    rep.kv("Median duration (yr)", f"{runs.duration_yr.median():.0f}")
    rep.kv("Median severity (permil yr)", f"{runs.severity.median():.2f}")
    rep.section("Spell containing 500-530 CE")
    rep.kv("Interval (CE)", f"{ev.start_ce:.0f} to {ev.end_ce:.0f}")
    for m, lab in metrics:
        pctl = stats.percentile_rank(runs[m], ev[m])
        rep.kv(f"{lab}", f"{ev[m]:.2f}  (rank {int(ev['rank_' + m])} of "
               f"{len(runs)}, percentile {pctl:.0f})")
    rep.section("Ten most severe spells")
    top = runs.sort_values("severity", ascending=False).head(10)
    rep.table(top[["start_ce", "end_ce", "duration_yr", "severity",
                   "peak", "intensity"]].reset_index(drop=True),
              floatfmt="{:.2f}")
    rep.section("Interpretation")
    worst = top.iloc[0]
    rep.para(
        f"The sixth-century spell ({ev.start_ce:.0f} to {ev.end_ce:.0f} "
        f"CE) ranks {int(ev['rank_peak'])} in peak, "
        f"{int(ev['rank_severity'])} in severity, "
        f"{int(ev['rank_intensity'])} in intensity and "
        f"{int(ev['rank_duration_yr'])} in duration among {len(runs)} "
        f"spells. The {worst.start_ce:.0f} to {worst.end_ce:.0f} CE spell "
        f"accumulates more severity ({worst.severity:.0f} permil yr) "
        f"because it lasts {worst.duration_yr:.0f} yr, but its intensity "
        f"({worst.intensity:.2f} permil) is about "
        f"{ev.intensity / worst.intensity:.1f} times lower. The "
        "sixth-century spell is therefore both long and deep, and its "
        "peak is unmatched. Its onset in the 470s CE agrees with the "
        "weakening of Himyar from the 480s CE described by Fleitmann "
        "et al. (2022).")
    rep.write()
    print("step 4 done:", len(runs), "runs; event ranks",
          {m: int(ev['rank_' + m]) for m, _ in metrics},
          f"interval {ev.start_ce:.0f}-{ev.end_ce:.0f}")


if __name__ == "__main__":
    main()
