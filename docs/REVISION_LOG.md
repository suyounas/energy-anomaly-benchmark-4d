# Revision log: response to IEEE BigData 2026 reviews (paper BigD340)

Status of each item in the 18-row issue register. **Done** = implemented and tested here.
**Partial** = implemented, but needs runs on real data / GPU that the authoring sandbox could not do.
**Open** = not done.

| # | Issue | Status | What was done / what is left |
|---|---|---|---|
| 1 | F1 mixed event recall with timestamp precision (R3-1) | Done | `ebench.metrics` separates event-level and timestamp-level; the paper uses the consistent same-day table. WTI DBSCAN F1 is 0.344 (consistent) not 0.695 (hybrid); WTI best is LSTM-AE 0.590. |
| 2 | "Detection gap" contradicted by GB 0.282 / FR line flow 0.408 (R3-3) | Done | Claim withdrawn. |
| 3 | FR line flow missing from Fig. 4 (R3-3) | Done | Fig. 4 removed with the claim; line flow is in the datasets table and results table. |
| 4 | DBSCAN not monotone (R3-2) | Done | Claim withdrawn; controlled sweep shows a U-shape in duration and ≤0.022 change from 30 min to 1 h. |
| 5 | ERCOT unlabelled summer 2023 (R3-4) | Partial | Tiered-label design in paper (Tier 1 pre-specified; Tier 2 = documented ERCOT records 27 Jun–10 Aug 2023, post hoc). Diagnostic now in `experiments/ercot_label_tiers.py` (tested). **To run:** regenerate `ercot_flags.csv` with the patched Study 4 pipeline, then `bash scripts/run_real_checks.sh --only tiers` (see `docs/RUNBOOK_REAL_DATA.md`). |
| 6 | Durations table, synthetic flag (R3-5) | Done | Table 1 gives dates, days, native samples, real/synthetic. |
| 7 | Confounded variables (W2.2) | Partial | Synthetic constant-prevalence sweep (2 event types × 3 resolutions × 6 durations × 10 seeds) done. **To run:** `experiments/real_resolution_ablation.py` on GB and ERCOT. |
| 8 | Only 5 events / 1–2 per dataset (W2.3, W3.1) | Open | Real tables are explicitly descriptive. Adding documented events is future work. |
| 9 | Repeats and intervals (W3.1) | Partial | t-intervals over 10 seeds in the sweep. Day-block bootstrap implemented, not applied to real tables; neural repeats need a GPU. |
| 10 | Equal tuning, hyper-parameter table, threshold-free metrics (W3.2) | Partial | Hyper-parameter table done (from code); AUC-PR and F1 ceiling in `ebench`. `experiments/equal_budget_tuning.py` tested on classical families; neural families (`ebench/neural.py`) **untested, need torch**. |
| 11 | Related work, stronger baselines (W3.3, R2) | Partial | Related work written from verified sources. Stronger recent detectors and VUS-PR not evaluated. |
| 12 | Novelty / ATLAS claim (W1) | Done | Acknowledgment claim removed; contribution reframed as evaluation methodology. |
| 13 | Code link, commit, DOI (O1) | Partial | Repo URL in paper. **To fill:** commit hash and Zenodo DOI after the final push. |
| 14 | Methods detail (R2, W3.2) | Partial | Table 2 (settings from code) and metric contract in text; no pseudocode block. |
| 15 | Scope / venue fit (R2) | Open | Venue choice is a decision for the author (see plan). |
| 16 | `per_event_recall` excludes last day of event | Done | Fixed in Studies 3 and 4; strict-xfail test turned into a passing regression test. **To run:** regenerate `Recall_*` columns. |
| 17 | CMA-ES tuned on test set | Partial | Disclosed in Limitations; validation-split tuning implemented. The CMA-ES search itself was not re-run; the BayesAI poster must carry the caveat. |
| 18 | Protocol B window seam | Partial | `ebench.windows.make_windows` never spans a gap (tested). Study scripts unchanged; neural numbers not regenerated. |

## Additional findings made during the revision (not raised by reviewers)
* Isolation Forest, the sequence models and the `|return|` score flag exactly 5% of the test set; **DBSCAN flags whatever its noise set is** (fitted on the test set), so its alert rate floats (11–24% in the sweep).
* In Studies 2–4 the "rolling z-score" is `|return| > its 95th percentile on test`, not a rolling z > 3 as described in the submitted paper.
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
