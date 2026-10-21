#!/usr/bin/env python
"""Step 11: write slides/numbers.tex so the slides quote current results.

Every number in the presentation comes from this file, which is rebuilt
from the stored results whenever the pipeline runs.
"""

import json

import numpy as np
import pandas as pd
from scipy import stats as sps

from h12eva import config as cfg
from h12eva import io, results


def yrs(v):
    """Format a return period for LaTeX."""
    v = results.as_float(v)
    if np.isinf(v):
        return r"$\infty$"
    return f"{v:,.0f}".replace(",", "{,}")


def main():
    res = results.load()
    ts = json.loads((cfg.PROC_DIR / "time_structure.json").read_text())
    binned = io.load_binned()
    native = io.load_native()
    runs = pd.read_csv(cfg.PROC_DIR / "drought_runs.csv")
    a = binned["anomaly"].to_numpy()
    mask = io.event_mask(binned["age_ce"].to_numpy())
    ev = runs[runs["event"]].sort_values("severity").iloc[-1]
    worst = runs.sort_values("severity").iloc[-1]
    pot, bm, lo, loc = (res["pot"], res["bm"], res["leave_out"],
                        res["local"])
    rob = res["robustness"]
    second = lo["mle_without"]["largest_remaining_peak"]
    level = pot["event_level"]
    rest = a[~mask]
    epochs = list(cfg.EPOCHS.values())

    def med(lo_, hi_):
        y = binned["age_ce"].to_numpy()
        return np.median(a[(y >= lo_) & (y < hi_)])

    high = native["anomaly"] > 2.5
    age = native["age_ce"]
    in_win = (age >= 515.0) & (age <= 535.0)
    dt = native["dt_yr"].dropna()
    dt = dt[dt > 0]
    lam = {
        "NHighAll": f"{int(high.sum())}",
        "NHighWindow": f"{int((high & in_win).sum())}",
        "MaxSpacing": f"{dt.max():.1f}",
        "NSamples": f"{len(native)}",
        "NBins": f"{len(a)}",
        "Span": f"{pot['span_yr']:,.0f}".replace(",", "{,}"),
        "NativePeak": f"{native['anomaly'].max():.2f}",
        "EventLevel": f"{level:.2f}",
        "SecondPeak": f"{second:.2f}",
        "Margin": f"{level - second:.1f}",
        "Zscore": f"{(level - rest.mean()) / rest.std(ddof=1):.1f}",
        "KurtAll": f"{sps.kurtosis(a, bias=False):.2f}",
        "KurtEx": f"{sps.kurtosis(rest, bias=False):.2f}",
        "MedEOne": f"{med(*epochs[0]):.2f}",
        "MedETwo": f"{med(*epochs[1]):.2f}",
        "MedEThree": f"{med(*epochs[2]):.2f}",
        "TauNative": f"{ts['tau_native']:.1f}",
        "TauBinned": f"{ts['tau_binned']:.1f}",
        "PettittYear": f"{ts['pettitt_year']:.0f}",
        "MKp": f"{ts['mk']['p']:.2f}",
        "NRuns": f"{len(runs)}",
        "SpellStart": f"{ev['start_ce']:.0f}",
        "SpellEnd": f"{ev['end_ce']:.0f}",
        "SpellDur": f"{ev['duration_yr']:.0f}",
        "RankSeverity": f"{int(ev['rank_severity'])}",
        "RankDuration": f"{int(ev['rank_duration_yr'])}",
        "RankIntensity": f"{int(ev['rank_intensity'])}",
        "WorstStart": f"{worst['start_ce']:.0f}",
        "WorstEnd": f"{worst['end_ce']:.0f}",
        "Threshold": f"{pot['threshold']:.2f}",
        "NPeaks": f"{pot['n_peaks']}",
        "RatePerCentury": f"{pot['rate_per_yr'] * 100:.2f}",
        "Xi": f"{pot['xi']:.2f}",
        "XiLo": f"{pot['xi_boot_p05']:.2f}",
        "XiHi": f"{pot['xi_boot_p95']:.2f}",
        "TMLE": yrs(pot["T_mle"]),
        "TBootLo": yrs(pot["T_boot"]["p05"]),
        "TBootInf": f"{pot['T_boot']['frac_inf'] * 100:.0f}",
        "TEmcee": yrs(pot["T_emcee"]["median"]),
        "PRecord": f"{pot['p_record_mle'] * 100:.0f}",
        "RLThousand": f"{pot['return_levels']['1000'][0]:.2f}",
        "XiFifty": f"{bm['50']['xi']:.2f}",
        "XiHundred": f"{bm['100']['xi']:.2f}",
        "TFifty": yrs(bm["50"]["T_mle"]),
        "THundred": yrs(bm["100"]["T_mle"]),
        "XiWithout": f"{lo['mle_without']['xi']:.2f}",
        "EndWithout": f"{results.as_float(lo['mle_without']['endpoint']):.2f}",
        "TMSWith": yrs(lo["bayes_gpd_with"]["median"]),
        "TMSWithout": yrs(lo["bayes_gpd_without"]["median"]),
        "TMSWithoutHi": yrs(lo["bayes_gpd_without"]["p95"]),
        "TUniWith": yrs(lo["uniform_gpd_with"]["median"]),
        "UniInf": f"{lo['uniform_gpd_without']['frac_inf'] * 100:.0f}",
        "PRecMSWith": f"{lo['p_record_with'] * 100:.0f}",
        "PRecMSWithout": f"{lo['p_record_without'] * 100:.0f}",
        "PRecUniWith": f"{lo['p_record_uniform_with'] * 100:.0f}",
        "PRecUniWithout": f"{lo['p_record_uniform_without'] * 100:.0f}",
        "BgEvent": f"{loc['background_at_event']:+.2f}",
        "Resid": f"{loc['event_resid']:.2f}",
        "TResid": yrs(loc["T_resid_mle"]),
        "TNS": yrs(loc["T_ns_bg"]),
        "GridMin": yrs(rob["T_with_min"]),
        "GridMax": yrs(rob["T_with_max"]),
        "GridCells": f"{rob['n_cells']}",
        "GridInf": f"{rob['frac_without_infinite'] * rob['n_cells']:.0f}",
    }
    for blk, tag in (("50", "Fifty"), ("100", "Hundred")):
        b = bm[blk]
        rl = b["return_levels"]
        lam.update({
            f"Mu{tag}": f"{b['mu']:.2f}",
            f"Sigma{tag}": f"{b['sigma']:.2f}",
            f"N{tag}": f"{b['n']}",
            f"XiLo{tag}": f"{b['xi_ci'][0]:.2f}",
            f"XiHi{tag}": f"{b['xi_ci'][1]:.2f}",
            f"TBootLo{tag}": yrs(b["T_boot"]["p05"]),
            f"TBootMed{tag}": yrs(b["T_boot"]["median"]),
            f"TBootInf{tag}": f"{b['T_boot']['frac_inf'] * 100:.0f}",
            f"RLThousand{tag}": f"{rl['1000'][0]:.2f}",
            f"RLThousandLo{tag}": f"{rl['1000'][1]:.2f}",
            f"RLThousandHi{tag}": f"{rl['1000'][2]:.2f}",
            f"RLFiveHundred{tag}": f"{rl['500'][0]:.2f}",
            f"RLTenK{tag}": f"{rl['10000'][0]:.2f}",
            f"XiWithout{tag}": f"{b['xi_without']:.2f}",
            f"TWithout{tag}": yrs(b["T_without"]),
            f"PRec{tag}": f"{b['p_record'] * 100:.0f}",
        })
    for blk, tag in (("50", "Fifty"), ("100", "Hundred")):
        rho = np.exp(-float(blk) / ts["tau_native"])
        lam[f"Rho{tag}"] = (f"{rho:.2f}" if rho >= 0.005
                            else r"$<$\,0.001")
    lines = ["% Generated by scripts/11_slide_numbers.py. Do not edit."]
    lines += [rf"\newcommand{{\n{k}}}{{{v}}}" for k, v in lam.items()]
    out = cfg.ROOT / "slides" / "numbers.tex"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"step 11 done: {len(lam)} macros written to {out.name}")


if __name__ == "__main__":
    main()
