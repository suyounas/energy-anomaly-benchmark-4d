# Revision log: changes

Status of each item in the 18-row issue list. **Done** = implemented and tested here.
**Partial** = implemented, but needs runs on real data / GPU that the authoring sandbox could not do.
**Open** = not done.

| # | Issue | Status | What was done / what is left |
|---|---|---|---|
| 1 | F1 mixed event recall with timestamp precision | Done | `ebench.metrics` separates event-level and timestamp-level; the paper uses the consistent same-day table. WTI DBSCAN F1 is 0.344 (consistent) not 0.695 (hybrid); WTI best is LSTM-AE 0.590. |
| 2 | "Detection gap" contradicted by GB 0.282 / FR line flow 0.408 | Done | Claim withdrawn. |
| 3 | FR line flow missing from Fig. 4 | Done | Fig. 4 removed with the claim; line flow is in the datasets table and results table. |
| 4 | DBSCAN not monotone | Done | Claim withdrawn; controlled sweep shows a U-shape in duration and ≤0.022 change from 30 min to 1 h. |
| 5 | ERCOT unlabelled summer 2023 | Partial | Tiered-label design in paper (Tier 1 pre-specified; Tier 2 = documented ERCOT records 27 Jun–10 Aug 2023, post hoc). Diagnostic now in `experiments/ercot_label_tiers.py` (tested). **To run:** regenerate `ercot_flags.csv` with the patched Study 4 pipeline, then `bash scripts/run_real_checks.sh --only tiers` (see `docs/RUNBOOK_REAL_DATA.md`). |
| 6 | Durations table, synthetic flag | Done | Table 1 gives dates, days, native samples, real/synthetic. |
| 7 | Confounded variables | Partial | Synthetic constant-prevalence sweep (2 event types × 3 resolutions × 6 durations × 10 seeds) done. **To run:** `experiments/real_resolution_ablation.py` on GB and ERCOT. |
| 8 | Only 5 events / 1–2 per dataset | Open | Real tables are explicitly descriptive. Adding documented events is future work. |
| 9 | Repeats and intervals | Partial | t-intervals over 10 seeds in the sweep. Day-block bootstrap implemented, not applied to real tables; neural repeats need a GPU. |
| 10 | Equal tuning, hyper-parameter table, threshold-free metrics | Partial | Hyper-parameter table done (from code); AUC-PR and F1 ceiling in `ebench`. `experiments/equal_budget_tuning.py` tested on classical families; neural families (`ebench/neural.py`) **untested, need torch**. |
| 11 | Related work, stronger baselines | Partial | Related work written from verified sources. Stronger recent detectors and VUS-PR not evaluated. |
| 12 | Novelty / ATLAS claim | Done | Acknowledgment claim removed; contribution reframed as evaluation methodology. |
| 13 | Code link, commit, DOI | Partial | Repo URL in paper. **To fill:** commit hash and Zenodo DOI after the final push. |
| 14 | Methods detail | Partial | Table 2 (settings from code) and metric contract in text; no pseudocode block. |
| 16 | `per_event_recall` excludes last day of event | Done | Fixed in Studies 3 and 4; strict-xfail test turned into a passing regression test. **To run:** regenerate `Recall_*` columns. |
| 17 | CMA-ES tuned on test set | Partial | Disclosed in Limitations; validation-split tuning implemented. The CMA-ES search itself was not re-run; the BayesAI poster must carry the caveat. |
| 18 | Protocol B window seam | Partial | `ebench.windows.make_windows` never spans a gap (tested). Study scripts unchanged; neural numbers not regenerated. |

## Additional findings made during the revision
* Isolation Forest, the sequence models and the `|return|` score flag exactly 5% of the test set; **DBSCAN flags whatever its noise set is** (fitted on the test set), so its alert rate floats (11–24% in the sweep).
* In Studies 2–4 the "rolling z-score" is `|return| > its 95th percentile on test`, not a rolling z > 3 as described in the earlier paper.
* The tuned-eps DBSCAN of Study 1 exists only in a diagnostics script; the main pipeline uses a guessed eps = 1.5.
* Sequence models and classical detectors were scored on different row sets in the same table.
* `inject_events` produced NaN events when called on a sub-array (scale from an empty slice) and the tuning loop silently fell back to defaults; both fixed with regression tests.

## How to reproduce / what you need to run
One-command version for the real-data steps: `bash scripts/run_real_checks.sh` (see `docs/RUNBOOK_REAL_DATA.md`).

```bash
pip install -r requirements-test.txt && pytest -q          # 61 tests
python experiments/synthetic_sweep.py --seeds 10 --kind shift --out results/synthetic/sweep_shift_mag3.csv
python experiments/synthetic_sweep.py --seeds 10 --kind burst --out results/synthetic/sweep_burst_mag3.csv
python experiments/analyse_sweep.py results/synthetic/sweep_shift_mag3.csv --tag shift
python paper/revised/make_tables.py && (cd paper/revised && pdflatex main.tex)

# On your data (GB; ERCOT analogous):
python experiments/real_resolution_ablation.py --csv neso_demand_full.csv --time-col Datetime \
  --value-col ND_MW --base-minutes 30 --event 2020-03-23 2020-05-31 COVID \
  --event 2022-12-08 2022-12-17 ColdSnap --train-window 2018-06-01 2020-02-01 \
  --factors 1 2 48 --out results/real/gb_resolution_ablation.csv
python experiments/equal_budget_tuning.py --csv neso_demand_full.csv --time-col Datetime \
  --value-col ND_MW --base-minutes 30 --factor 2 --event 2020-03-23 2020-05-31 COVID \
  --event 2022-12-08 2022-12-17 ColdSnap --train-window 2018-06-01 2020-02-01 \
  --families isolation_forest knn_inductive rolling_z lstm_ae transformer --out results/real/gb_equal_budget
```
Note: on the real data the `ebench` features are re-derived (return, rolling mean/std/skew over a time-fixed window), so they will not reproduce the published tables exactly.

## Real-data runs, 29 Oct 2026 (outputs in `results/real_data/`, summary in `docs/real_data_summary.md`)
* Resolution is a real confound on real data (ERCOT Isolation Forest F1 0.000 hourly vs 0.119 daily); the synthetic bound (<= 0.022) does not carry over.
* ERCOT Tier 1: neural detectors F1 = 0 under both protocols; their flags are 93-100% in Jul-Sep 2023. Tier 1+2 F1 0.43 (B) comes only from the post-hoc Tier 2 label.
* Validation on injected events is uninformative for kNN, LSTM-AE and Transformer (validation AUC-PR ~ prevalence for every config). `equal_budget_search` now keeps the default config in that case (`min_spread`), so "tuned" is not selection on noise. Existing runs predate this rule.
* `z_rolling` (z > 3) fires on 0-1% of real data; `z_rolling_top5` gives it the 5% budget of the other detectors.
* Taxonomy validation (`--val-suite taxonomy`, default) on real data: level-shift cells score at/below chance for Isolation Forest and kNN, bursts and pulses are found. The 4-feature recipe carries no level information. `--features resid` adds a causal seasonal-naive residual; choose features on validation lift only (see `docs/VALIDATION_DESIGN.md`).
* Resolution ablation with `--features resid` (30 Oct 2026): GB daily ts-F1 Isolation Forest 0.133 / kNN 0.103 / DBSCAN 0.065 (recipe: 0.010 / 0.031 / 0.083), so the ranking flips with the feature set; ERCOT daily DBSCAN 0.478 (= its F1 ceiling) vs Isolation Forest 0.190, kNN 0.119, but event-level F1 ranks kNN (0.348) above DBSCAN (0.250). Only 2 events per dataset: the ablation now reports paired 7-day-block bootstrap intervals and `reliably_worse_than_best` (`--n-boot`, `--ci-block`); read the intervals before the point estimates.
* `equal_budget_tuning.py` now also writes `<out>_ci.csv` (paired 7-day-block bootstrap of the seed-averaged timestamp F1 for every family/config, `--n-boot`, `--ci-block`), so neural and classical families are compared with intervals, not point estimates. Neural families on GB/ERCOT with the `recipe` features: validation macro lift 1.9 (GB) / 5.9 (ERCOT) with spread < 0.1, so the gate keeps their defaults ("tuned" == default); the earlier "tuned LSTM-AE 0.131" was selection on noise and must not be quoted.

## Step 7, 2 Nov 2026: stronger and simpler comparison detectors, range-aware metric
* New `ebench/stronger.py`: `diff_z` (Wu-Keogh one-liner), `seasonal_naive` (|resid| as a score; needs `--features resid`), `window_pca`, `window_knn`, `window_iforest` (same 30-step windows and features as the neural models), `matrix_profile` (AB-join, z-normalised, BLAS; tested against a brute-force reference), `spectral_residual`. Registered in `ebench.tuning.FAMILIES`, so `equal_budget_tuning.py --families ...` gives them the same budget, validation and bootstrap intervals as every other family. Families that need a feature the run does not have are skipped with a message.
* New `ebench/rangemetrics.py`: `range_auc_pr`, `vus_pr` (own re-implementation of the published definition; `experiments/crosscheck_official.py` compares with stumpy and TSB-AD on a machine that can install them).
* Correction in Related Work: TSB-AD detector count (the README claims 40 but lists 33).
* Tests: 107 pass (27 new). Not yet run on GB or ERCOT; no result from these detectors may be quoted until that run exists.
* Design doc: `docs/BENCHMARK_PLAN.md`.

## Step 7b, 2 Nov 2026: intervals for AUC-PR / VUS-PR, per-event coverage table, parameter-free note
* `equal_budget_tuning.py` now writes `<out>_ci_metrics.csv` (paired 7-day-block bootstrap of AUC-PR and VUS-PR for every detector, difference to the best, `reliably_worse_than_best`; `--ci-metrics`, `--n-boot-vus`, VUS is about 10x slower per replicate so it defaults to 100) and `<out>_coverage.csv` (per-event coverage by detector, `events_hit`, `min_event_coverage`, mean event/timestamp recall). Logic lives in `ebench/report.py` (tested).
* Parameter-free detectors (`diff_z`, `seasonal_naive`) print `INFO ... nothing to tune` instead of the misleading "validation cannot separate configs" note; trials CSV has a `parameter_free` column; the summary shows "n/a (parameter-free)".
* `experiments/summarise_real_results.py` takes `--dir`, `--suffix`, `--out` and adds the coverage table and the interval tables (F1, AUC-PR, VUS-PR) per dataset.
* Caveats stated in the code and the summary: the "best" detector is chosen post hoc among those listed; VUS-PR is our re-implementation; bootstrap replicates of a range metric can contain block seams.
* Tests: 116 pass (9 new, including one that runs the CLI end to end).
