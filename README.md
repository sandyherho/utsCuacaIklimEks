# hoti-h12-eva

Repository: <https://github.com/sandyherho/utsCuacaIklimEks>

Extreme value analysis of the sixth-century drought in the Hoti Cave
stalagmite H12 oxygen isotope record (northern Oman).

Course project, Weather and Climate Extremes.
Sandy Hardian Susanto Herho (NIM 32426003).

## Question

Fleitmann et al. (2022, *Science* 376, 1317-1321,
[doi:10.1126/science.abg4044](https://doi.org/10.1126/science.abg4044))
report unprecedented aridity in southern Arabia between about 500 and 530
CE and link it to the decline of the Himyarite kingdom. This repository
asks how extreme that drought is in statistical terms, using only the H12
d18O series, and which parts of the claim survive changes in method.

## Main results

| Quantity | Value |
|---|---|
| Peak 5-yr anomaly (520-525 CE) | +3.43 permil vs. 1978-1997, rank 1 of 527 bins |
| Distance from rest of record | 5.1 standard deviations; 1.3 permil above the next independent peak |
| Drought spell | 475-545 CE: rank 1 in peak, 2 in severity and duration of 26 spells |
| Return period, event in fit | about 5,600 to 20,800 yr over 36 analysis choices (POT MLE 11,600 yr) |
| Return period, event left out | beyond the fitted tail (data alone); 9,100 yr median with a heavy-tail prior |
| Local frame (vs. 300-yr background) | still rank 1; return period about 13,000 yr |

The drought is record-breaking and multi-millennial in magnitude. Its exact
return period is uncertain by an order of magnitude because 19 independent
peaks cannot constrain the tail shape, and the +/-30 yr age uncertainty
makes the match with the Aksumite conquest of 525 CE consistent but not
diagnostic. Full numbers are in `reports/00_summary.txt`.

## Repository layout

```
hoti-h12-eva/
├── data/
│   ├── raw/                 H12 table (see data/raw/README.md)
│   └── processed/           written by the pipeline
├── figures/                 PDF (vector) and PNG (300 dpi), AGU sizing
├── reports/                 plain-text reports, one per step
├── scripts/                 analysis steps 01-10 and run_all.py
├── slides/                  Beamer presentation (main.tex, refs.bib)
├── src/h12eva/              package: config, io, stats, eva, style
├── CITATION.cff
├── LICENSE                  MIT
├── pyproject.toml
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/sandyherho/utsCuacaIklimEks.git
cd utsCuacaIklimEks
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Python 3.10 or newer is required. Dependencies: numpy, scipy, pandas,
matplotlib, statsmodels, pymannkendall,
[pyextremes](https://github.com/georgebv/pyextremes) and emcee.

## Usage

```bash
python scripts/run_all.py          # full pipeline, about 10 min on a laptop
python scripts/05_pot_gpd.py       # or any single step
```

Every random step is seeded in `src/h12eva/config.py`, so results are
reproducible. All analysis choices (bin width, thresholds, declustering,
block sizes, priors, Monte Carlo sizes) live in that file.

To build the slides after running the pipeline:

```bash
cd slides && latexmk -pdf main.tex
```

## Slides

The presentation in `slides/` reports the block maxima (GEV) analysis
only, with every equation explained, together with the background on the
stalagmite record and the amount effect. The full set of methods
(threshold models, Bayesian refits, local frame, robustness grid) is in
`reports/` and `figures/`.

## Data

The pipeline reads `data/raw/fleitmann2022-stalagmite12.txt` (the original
NOAA NCEI file, study 38500,
[doi:10.25921/erfq-6235](https://doi.org/10.25921/erfq-6235)) when present,
and otherwise a transcription of it. See `data/raw/README.md` for
provenance, the check on the swapped isotope labels in the NOAA header, and
the download command.

## Methods

| Step | Script | Content |
|---|---|---|
| 1 | `01_prepare_data.py` | Column identification, anomalies vs. -1.8 permil, 5-yr bins, LOWESS background |
| 2 | `02_descriptive.py` | Moments, quantiles, epochs, normality, running variability |
| 3 | `03_time_structure.py` | AR(1) persistence for uneven sampling, Mann-Kendall, Pettitt, binary segmentation, Lomb-Scargle vs. red noise |
| 4 | `04_drought_runs.py` | Run theory: duration, severity, intensity, peak |
| 5 | `05_pot_gpd.py` | Threshold selection, GPD by pyextremes MLE and Emcee, bootstrap, diagnostics |
| 6 | `06_bm_gev.py` | GEV on 50-yr and 100-yr block maxima |
| 7 | `07_leave_out.py` | Circularity test: refits without 500-530 CE, Bayesian GPD/GEV with two shape priors |
| 8 | `08_local_frame.py` | Extremes relative to the 300-yr background; non-stationary GEV |
| 9 | `09_robustness.py` | 36-cell grid of time base, threshold and declustering |
| 10 | `10_fingerprint.py` | Return periods across methods, rank metrics, historical context, summary |
| 11 | `11_slide_numbers.py` | Writes `slides/numbers.tex` so the slides quote the stored results |
| 12 | `12_method_figure.py` | Explanatory figure of the peaks-over-threshold method |
| 13 | `13_bm_method_figure.py` | Explanatory figure of the block maxima method |

pandas cannot store dates before 1677, so years are mapped onto a
pseudo-calendar in which one calendar year equals one day. pyextremes then
runs unchanged and all return periods are reported in calendar years.

## Figures

| File | Content |
|---|---|
| `fig01_record_overview` | Native and binned anomaly, background, sample spacing |
| `fig02_descriptive` | Distributions by epoch, box plots, normal QQ plot, running SD |
| `fig03_time_structure` | Regime shifts, Lomb-Scargle spectrum, autocorrelation |
| `fig04_drought_runs` | Drought spells and the rank of the 475-545 CE spell |
| `fig05_pot_threshold` | Mean residual life, parameter stability, declustered peaks |
| `fig06_pot_fit` | GPD return levels, QQ plot, excess density |
| `fig07_bm_gev` | GEV return levels for 50-yr and 100-yr blocks |
| `fig08_leave_out` | Return levels and posteriors with and without the event |
| `fig09_local_frame` | Residual extremes and background-dependent GEV |
| `fig10_robustness` | Return period across 36 analysis choices |
| `fig11_fingerprint` | Return periods across all methods; rank metrics |
| `fig12_context` | 380-720 CE with age uncertainty and historical events |
| `fig13_pot_method` | How threshold excesses are extracted; effect of the GPD shape on the tail |
| `fig14_bm_method` | 50-yr blocks and their maxima, fitted GEV, effect of the GEV shape |

In every time-series panel the anomaly axis is inverted, so drought (higher
d18O) plots downward, following the usual convention for speleothem records.

## Limitations

The analysis uses one stalagmite and d18O only, carries no age-model
ensemble, and treats d18O as a proxy for winter-spring rainfall rather than
rainfall itself. Tail estimates rest on 19 to 21 declustered peaks.

## License

MIT. See `LICENSE`.

## Citation

Cite the original study and data (Fleitmann et al., 2022; NOAA NCEI study
38500) and this repository (`CITATION.cff`).
