# Reproducing the Step 10 (multi-authority) study

All commands run from the repository root with Python 3.10 or later. The pre-registration is
`docs/STEP10_PREREGISTRATION.md`; every rule below is defined there first.

## 0. Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pandas scikit-learn scipy matplotlib pyarrow pytest   # classical detectors, analysis, report
pip install torch                                                        # only for the LSTM / Transformer autoencoders
python3 -m pytest -q                                                     # regression tests for the evaluation conventions
```

`stumpy` is used only by `experiments/crosscheck_official.py` (matrix-profile cross-check).
Everything runs on CPU. Wall-clock on one machine: about 20 seconds per authority for the ten
classical detectors, about 45 minutes per authority for the two neural families (three seeds).

## 1. Series: EIA-930 hourly demand and the series rule

```bash
python3 scripts/step10_fetch_eia930.py --out data/step10/eia930
```

Downloads the public six-month EIA-930 bulk files, applies the series rule (coverage of at least
95 percent, no gap over 72 hours) and writes `series_manifest.csv` (57 authorities, 47 qualify) and
`raw_file_sha256.txt`. Raw files are deleted after filtering; the hashes are kept.

## 2. Events

Class B (rule-generated from NOAA NCEI Storm Events, mapped to authorities with EIA-861 territory data):

```bash
python3 scripts/step10_select_events.py \
  --storm-dir <dir with StormEvents_details-ftp_v1.0_dYYYY_c20261119.csv.gz> \
  --territory data/step10/ba_territory.parquet \
  --ba-codes data/step10/ba_codes.csv \
  --out results/step10/events_selected.csv
sha256sum -c results/step10/SHA256SUMS.txt      # run inside the storm directory: checks the NCEI files
```

Class A (operator- or regulator-declared, hand-entered with a source URL): `docs/step10_events.csv`.

## 3. Detector runs (blocked temporal cross-fitting)

```bash
# ten classical families, five seeds for the stochastic ones
python3 scripts/step10_run.py --out results/step10/runs_classical
# the two neural families, first 12 authorities in code order, three seeds (Amendment 3)
python3 scripts/step10_run.py --out results/step10/runs_neural \
  --families lstm_ae transformer --neural-first-n 12 --neural-seeds 3
```

The exact settings used for the reported runs are in `results/step10/runs_classical/run_config.json`
and `results/step10/runs_neural/run_config.json`. Runs are resumable per authority.

## 4. Analysis, label validity, perturbations, neural comparison, report

```bash
python3 scripts/step10_aggregate.py --runs results/step10/runs_classical --out results/step10/analysis_classical_prelim_lift --metric lift
python3 scripts/step10_label_validity.py
python3 scripts/step10_perturb_vus.py
python3 scripts/step10_neural_compare.py
python3 scripts/step10_report.py
```

Defaults (paths, 2,000 bootstrap replicates, seed 0, at least 30 storm groups for a non-descriptive
view) are the ones in the pre-registration. Tables and figures of the paper's Step 10 section come
from `step10_report.py`.

## 5. Check the headline counts

```bash
python3 scripts/step10_export_catalogue.py
```

Writes `results/step10/events_analysed.csv` and asserts the counts in the paper: 1,973 events on
45 authorities (10 Class A from 7 storms, 1,963 Class B) in 262 storm groups.
