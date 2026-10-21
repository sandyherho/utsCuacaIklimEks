#!/usr/bin/env python
"""Run the full analysis pipeline in order."""

import runpy
import time
from pathlib import Path

STEPS = (
    "01_prepare_data.py",
    "02_descriptive.py",
    "03_time_structure.py",
    "04_drought_runs.py",
    "05_pot_gpd.py",
    "06_bm_gev.py",
    "07_leave_out.py",
    "08_local_frame.py",
    "09_robustness.py",
    "10_fingerprint.py",
    "11_slide_numbers.py",
    "12_method_figure.py",
    "13_bm_method_figure.py",
)


def main():
    here = Path(__file__).resolve().parent
    t_start = time.time()
    for step in STEPS:
        t0 = time.time()
        runpy.run_path(str(here / step), run_name="__main__")
        print(f"  {step}: {time.time() - t0:.0f} s")
    print(f"all steps finished in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
