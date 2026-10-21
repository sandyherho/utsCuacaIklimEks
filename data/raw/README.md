# Raw data

`h12_transcribed_noaa38500.tsv` reproduces the H12 data table of NOAA NCEI
paleoclimatology study 38500 (Fleitmann et al., 2022; dataset
doi:[10.25921/erfq-6235](https://doi.org/10.25921/erfq-6235)). Columns are
copied in the order of the NOAA header: depth from top (mm), age (CE,
negative = BCE), and two isotope columns labelled `d13C` and `d18O`.

The table was transcribed through a text interface, not downloaded as a
file. Three independent checks reproduce numbers stated in the paper (the
1978-1997 mean d18O of -1.8 permil, the d18O-d13C r^2 of 0.52, and the
d18O maximum at about 523 CE), but single-digit transcription errors
cannot be excluded.

**Before publication, replace it with the original file:**

```bash
curl -o data/raw/fleitmann2022-stalagmite12.txt \
  https://www.ncei.noaa.gov/pub/data/paleo/speleothem/asia/oman/fleitmann2023/fleitmann2022-stalagmite12.txt
```

When `fleitmann2022-stalagmite12.txt` is present, the pipeline reads it
instead of the transcription.

## Column labels

In the NOAA header the two isotope labels appear in the opposite order to
their values. The pipeline does not trust the labels: it selects as d18O
the column whose 1978-1997 mean matches the published modern baseline
(-1.8 permil VPDB), and reports the result in `reports/01_data.txt`.
