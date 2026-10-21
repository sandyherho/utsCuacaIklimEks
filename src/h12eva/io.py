"""Reading, cleaning, binning and time mapping of the H12 record."""

import io as _io

import numpy as np
import pandas as pd

from h12eva import config as cfg

# One calendar year is mapped to one pseudo-day so that the full record
# (631 BCE to 1996 CE) fits inside the pandas datetime range.
PSEUDO_ORIGIN_YEAR = -700.0
PSEUDO_ORIGIN = pd.Timestamp("1800-01-01")


def _read_table(path):
    """Read a NOAA-style tab table, skipping '#' comment lines."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        lines = [ln for ln in handle if ln.strip() and not ln.startswith("#")]
    return pd.read_csv(_io.StringIO("".join(lines)), sep="\t",
                       na_values=["NA", "NaN", "nan", "-999", "-999.0"])


def _identify_d18o(table, age_col, iso_cols):
    """Return the isotope column whose modern mean matches the baseline.

    The NOAA header lists d13C before d18O. The 1978-1997 mean of the
    true d18O series equals the published baseline (-1.8 permil), while
    d13C sits near -6 permil, so the match is unambiguous.
    """
    lo, hi = cfg.BASELINE_PERIOD
    modern = table[(table[age_col] >= lo) & (table[age_col] <= hi)]
    misfit = {c: abs(modern[c].mean() - cfg.BASELINE_PERMIL)
              for c in iso_cols}
    best = min(misfit, key=misfit.get)
    return best, misfit


def load_record():
    """Load H12 d18O, sorted by increasing age (CE).

    Returns
    -------
    df : DataFrame with columns age_ce, depth_mm, d18o, anomaly, dt_yr.
    meta : dict describing provenance and column identification.
    """
    path = cfg.NOAA_FILE if cfg.NOAA_FILE.exists() else cfg.TRANSCRIBED_FILE
    table = _read_table(path)
    age_col = [c for c in table.columns if "age" in c.lower()][0]
    depth_col = [c for c in table.columns if "depth" in c.lower()][0]
    iso_cols = [c for c in table.columns if c.lower().startswith("d1")]
    d18o_col, misfit = _identify_d18o(table, age_col, iso_cols)

    df = pd.DataFrame({
        "age_ce": table[age_col].astype(float),
        "depth_mm": table[depth_col].astype(float),
        "d18o": table[d18o_col].astype(float),
    })
    df = df.drop_duplicates().dropna(subset=["d18o"])
    df = df.sort_values(["age_ce", "depth_mm"],
                        ascending=[True, False]).reset_index(drop=True)
    df["anomaly"] = df["d18o"] - cfg.BASELINE_PERMIL
    df["dt_yr"] = np.r_[np.nan, np.diff(df["age_ce"].to_numpy())]

    meta = {
        "source_file": path.name,
        "is_original_noaa": path == cfg.NOAA_FILE,
        "d18o_column_in_file": d18o_col,
        "baseline_misfit": misfit,
    }
    return df, meta


def bin_record(df, width=cfg.BIN_WIDTH, start=cfg.BIN_START,
               end=cfg.BIN_END):
    """Average the anomaly into regular, non-overlapping bins."""
    edges = np.arange(start, end + width, width)
    cut = pd.cut(df["age_ce"], edges, right=False)
    grp = df.groupby(cut, observed=False)["anomaly"]
    out = pd.DataFrame({
        "age_ce": edges[:-1] + width / 2.0,
        "anomaly": grp.mean().to_numpy(),
        "n_samples": grp.size().to_numpy(),
    })
    first = out["n_samples"].to_numpy().nonzero()[0]
    out = out.iloc[first[0]:first[-1] + 1].reset_index(drop=True)
    return out


def to_pseudo_index(years):
    """Map calendar years to a DatetimeIndex (1 year -> 1 day)."""
    days = np.asarray(years, dtype=float) - PSEUDO_ORIGIN_YEAR
    return PSEUDO_ORIGIN + pd.to_timedelta(days, unit="D")


def from_pseudo_index(index):
    """Inverse of to_pseudo_index."""
    days = (pd.DatetimeIndex(index) - PSEUDO_ORIGIN) / pd.Timedelta("1D")
    return np.asarray(days, dtype=float) + PSEUDO_ORIGIN_YEAR


def as_series(years, values):
    """Build a pandas Series on the pseudo time axis for pyextremes."""
    return pd.Series(np.asarray(values, dtype=float),
                     index=to_pseudo_index(years), name="anomaly")


def event_mask(years):
    """Boolean mask of the sixth-century event window."""
    lo, hi = cfg.EVENT_WINDOW
    years = np.asarray(years, dtype=float)
    return (years >= lo) & (years <= hi)


def load_binned():
    """Load the processed 5-yr series written by 01_prepare_data.py."""
    return pd.read_csv(cfg.PROC_DIR / "h12_d18o_binned_5yr.csv")


def load_native():
    """Load the processed native series written by 01_prepare_data.py."""
    return pd.read_csv(cfg.PROC_DIR / "h12_d18o_native.csv")
