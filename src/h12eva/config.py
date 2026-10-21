"""Central configuration: paths and every analysis choice in one place."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "reports"

# Original NOAA file (preferred when present) and transcribed fallback.
NOAA_FILE = RAW_DIR / "fleitmann2022-stalagmite12.txt"
TRANSCRIBED_FILE = RAW_DIR / "h12_transcribed_noaa38500.tsv"
NOAA_URL = ("https://www.ncei.noaa.gov/pub/data/paleo/speleothem/asia/"
            "oman/fleitmann2023/fleitmann2022-stalagmite12.txt")

# Modern reference: 1978-1997 mean d18O used by Fleitmann et al. (2022).
BASELINE_PERMIL = -1.8
BASELINE_PERIOD = (1978.0, 1997.0)

# Homogenized time base.
BIN_WIDTH = 5.0
BIN_START = -635.0
BIN_END = 2000.0

# Sixth-century event window and its reported age uncertainty.
EVENT_WINDOW = (500.0, 530.0)
EVENT_PEAK_AGE = 523.0
EVENT_AGE_ERROR = 30.0

# Epochs following the hydroclimate phases described in the paper.
EPOCHS = {
    "E1 (631 BCE-479 CE)": (-640.0, 480.0),
    "E2 (480-1399 CE)": (480.0, 1400.0),
    "E3 (1400-1996 CE)": (1400.0, 2001.0),
}

# Drought-run definition.
RUN_QUANTILE = 0.75
RUN_POOL_BINS = 1

# Peaks over threshold.
POT_QUANTILE = 0.90
DECLUSTER_YR = 10.0
STABILITY_QUANTILES = (0.70, 0.97)

# Block maxima.
BLOCK_YR = (50.0, 100.0)

# Local (detrended) frame.
LOWESS_WINDOW_YR = 300.0

# Resampling and Monte Carlo.
N_BOOT = 2000
N_SURROGATE = 1000
N_WALKERS = 32
N_STEPS = 4000
N_BURN = 1000
SEED = 20221017

# Robustness grid.
ROBUST_BINS = ("native", 2.5, 5.0, 10.0)
ROBUST_QUANTILES = (0.85, 0.90, 0.95)
ROBUST_DECLUSTER = (5.0, 10.0, 20.0)

# Return periods (years) at which return levels are reported.
RETURN_PERIODS = (200, 500, 1000, 2000, 5000, 10000)

# Historical events (CE) summarized in Fleitmann et al. (2022).
HISTORICAL_EVENTS = (
    (525.0, "Aksumite conquest"),
    (559.0, "Last Himyarite inscription"),
    (570.0, "Persian domination"),
    (580.0, "Marib dam failure"),
)
