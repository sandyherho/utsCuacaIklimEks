#!/usr/bin/env python
"""Step 1: load, check and homogenize the H12 d18O record.

Outputs
-------
data/processed/h12_d18o_native.csv
data/processed/h12_d18o_binned_5yr.csv
figures/fig01_record_overview.(pdf|png)
reports/01_data.txt
"""

import matplotlib.pyplot as plt
import numpy as np

from h12eva import config as cfg
from h12eva import io, stats, style
from h12eva.report import Report


def main():
    style.apply()
    df, meta = io.load_record()
    binned = io.bin_record(df)
    binned["background"] = stats.background(
        binned["age_ce"], binned["anomaly"], cfg.LOWESS_WINDOW_YR)

    cfg.PROC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.PROC_DIR / "h12_d18o_native.csv", index=False,
              float_format="%.4f")
    binned.to_csv(cfg.PROC_DIR / "h12_d18o_binned_5yr.csv", index=False,
                  float_format="%.4f")

    lo, hi = cfg.BASELINE_PERIOD
    modern = df[(df.age_ce >= lo) & (df.age_ce <= hi)]["d18o"]
    dt = df["dt_yr"].dropna()
    dt_pos = dt[dt > 0]
    i_max = int(df["anomaly"].idxmax())
    j_max = int(binned["anomaly"].idxmax())

    # ------------------------------------------------------------ figure
    fig, axes = plt.subplots(
        2, 1, figsize=(style.WIDTH_FULL, 4.3),
        gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.35})
    ax = axes[0]
    t, a = df["age_ce"].to_numpy(), df["anomaly"].to_numpy()
    ax.axhline(0.0, color=style.GREY, lw=0.6, ls="--", zorder=1)
    ax.fill_between(t, 0.0, a, where=a > 0, color=style.DRY, alpha=0.35,
                    lw=0, interpolate=True, label="Drier than 1978-1997")
    ax.plot(t, a, color=style.GREY, lw=0.5, label="Native samples")
    ax.plot(binned["age_ce"], binned["anomaly"], color=style.BLACK, lw=0.8,
            label="5-yr bins")
    ax.plot(binned["age_ce"], binned["background"], color=style.BLUE,
            lw=1.4, label=f"Robust LOWESS ({cfg.LOWESS_WINDOW_YR:.0f} yr)")
    ax.axvspan(*cfg.EVENT_WINDOW, color=style.ORANGE, alpha=0.25, lw=0,
               label="500-530 CE")
    ax.set_xlim(-650, 2010)
    ax.set_ylabel(style.ANOMALY_LABEL + "\n" + style.DRY_DOWN)
    style.dry_down(ax)
    ax.legend(ncol=5, loc="upper left", bbox_to_anchor=(0.0, 1.13),
              handlelength=1.6, columnspacing=1.2)
    style.panel_label(ax, "a", x=-0.07, y=1.10)

    ax = axes[1]
    ax.semilogy(df["age_ce"].iloc[1:][dt > 0], dt_pos, ".",
                color=style.BLACK, ms=1.5)
    ax.set_xlim(-650, 2010)
    ax.set_ylabel("Spacing (yr)")
    ax.set_ylim(0.08, 6)
    ax.set_xlabel(style.YEAR_LABEL)
    style.panel_label(ax, "b", x=-0.07, y=1.04)
    style.save(fig, "fig01_record_overview")

    # ------------------------------------------------------------ report
    rep = Report("01_data", "Step 1. Data provenance and homogenization")
    rep.section("Source")
    rep.kv("File read", meta["source_file"])
    rep.kv("Original NOAA file", str(meta["is_original_noaa"]))
    rep.para(
        "The transcribed table reproduces the NOAA NCEI study 38500 file "
        "(doi:10.25921/erfq-6235). Place the original "
        "fleitmann2022-stalagmite12.txt in data/raw to use it instead; "
        "the pipeline then reads it automatically.")
    rep.kv("Column identified as d18O", meta["d18o_column_in_file"])
    for col, mis in meta["baseline_misfit"].items():
        rep.kv(f"|1978-1997 mean - (-1.8)| for {col}", f"{mis:.3f} permil")
    rep.para(
        "The NOAA header labels the two isotope columns in the opposite "
        "order to their values. The column whose 1978-1997 mean matches "
        "the published modern baseline of -1.8 permil is taken as d18O.")

    rep.section("Native record")
    rep.kv("Samples", f"{len(df)}")
    rep.kv("Age range (CE)",
           f"{df.age_ce.min():.1f} to {df.age_ce.max():.1f}")
    rep.kv("Depth range (mm)",
           f"{df.depth_mm.min():.2f} to {df.depth_mm.max():.2f}")
    rep.kv("1978-1997 mean d18O (permil VPDB)", f"{modern.mean():.3f}")
    rep.kv("Samples sharing an identical age", f"{int((dt == 0).sum())}")
    rep.kv("Spacing median / 5th / 95th pct (yr)",
           f"{dt_pos.median():.2f} / {dt_pos.quantile(0.05):.2f} / "
           f"{dt_pos.quantile(0.95):.2f}")
    rep.kv("Maximum anomaly (permil) and age (CE)",
           f"{df.anomaly.iloc[i_max]:.2f} at {df.age_ce.iloc[i_max]:.1f}")

    rep.section("Homogenized record")
    rep.para(
        f"Samples are averaged into {cfg.BIN_WIDTH:.0f}-yr bins. Sample "
        "spacing varies from sub-annual near the top to more than 3 yr "
        "at depth, so each native sample integrates a different time "
        "span. Binning removes most of this resolution contrast, which "
        "would otherwise inflate the apparent variance of the young part "
        "of the record.")
    rep.kv("Bins", f"{len(binned)} (empty: "
           f"{int((binned.n_samples == 0).sum())})")
    rep.kv("Samples per bin, median (range)",
           f"{binned.n_samples.median():.0f} "
           f"({binned.n_samples.min()}-{binned.n_samples.max()})")
    rep.kv("Maximum binned anomaly (permil) and bin",
           f"{binned.anomaly.iloc[j_max]:.2f} at "
           f"{binned.age_ce.iloc[j_max]:.1f} CE")
    rep.para(
        "Anomalies are d18O minus the 1978-1997 baseline. Positive values "
        "indicate less winter-spring rainfall than at present.")
    rep.write()
    print("step 1 done:", len(df), "samples,", len(binned), "bins;",
          f"max {np.nanmax(a):.2f} permil")


if __name__ == "__main__":
    main()
