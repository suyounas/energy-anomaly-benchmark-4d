# Run-book: real-data checks (what to run, what to send back)

Everything here runs on your machine / Colab; the synthetic parts are already in the repo.
Each step writes to `results/real/` (git-ignored). A step whose input is missing is **skipped**, never counted as passed.

## 0. Setup
```bash
pip install -r requirements-test.txt scikit-learn pandas numpy   # torch only if NEURAL=1
bash scripts/run_real_checks.sh --dry-run                         # prints every command, runs nothing
```

## 1. Files the runner expects (repo root, or set the env var)
| File | Env var | Made by |
|---|---|---|
| `neso_demand_full.csv` (Datetime, ND_MW; 30-min) | `GB_CSV` | `code/study3_gb_neso/01_acquisition_and_verification.py` |
| `ercot_demand_full.csv` (Datetime, Demand_MW; hourly) | `ERCOT_CSV` | Study 4 acquisition step |
| `ercot_flags.csv` (protocol, detector, time, flag, y_tier1) | `ERCOT_FLAGS` | `code/study4_ercot_eia/02_full_pipeline.py` (now writes it; run in Colab/GPU, then download) |

## 2. Run
```bash
bash scripts/run_real_checks.sh                       # classical families, ~minutes
NEURAL=1 bash scripts/run_real_checks.sh --only tuning_gb,tuning_ercot   # adds lstm_ae + transformer (slow)
bash scripts/run_real_checks.sh --only tiers          # just the ERCOT label-tier diagnostic
```
Outputs: `gb_resolution_ablation.csv`, `ercot_resolution_ablation.csv`, `gb_equal_budget*/`, `ercot_equal_budget*/`, `ercot_tiers.csv`, plus one `.log` per step.

## 3. What each step answers
| Step | Reviewer point | Decision it informs |
|---|---|---|
| `ablation_*` | resolution is confounded with duration / prevalence (W2.2) | whether DBSCAN's real-data gap survives a fixed time window and resolution |
| `tuning_*` | tuned DBSCAN vs untuned neural baselines, test-set CMA-ES objective | whether "no universal detector" survives equal-budget, validation-based tuning |
| `tiers` | ERCOT summer-2023 flags are mostly unlabelled (R3-4) | how much of any ERCOT F1 depends on the post-hoc Tier 2 label |

## 4. Send back
`results/real/*.csv` and the final `== summary ==` block. The paper's `[TO RUN]` placeholders are then replaced with those numbers (and the commit hash and Zenodo DOI filled in).

## Caveats
* Features are re-derived by `ebench`, so numbers will not reproduce the submitted tables exactly.
* Tier 2 (27 Jun - 10 Aug 2023) was identified after seeing detector output: sensitivity analysis only, never the headline.
* The runner was tested here on synthetic stand-in files, not on your real data.
