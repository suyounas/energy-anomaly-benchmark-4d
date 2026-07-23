# Code

Two independent studies, each with their own run instructions.

## Study 1 — WTI / Henry Hub (`study1_wti_henryhub/`)

Run in Google Colab (PyTorch + unrestricted internet access needed).
Note: this script contains `!pip install` lines, which are Colab/Jupyter
cell magic syntax — valid when pasted into a Colab cell, but will raise
a SyntaxError if run directly via `python3 script.py`. This is intentional,
not a bug.

1. `01_main_training_pipeline.py` — downloads real FRED data, trains
   LSTM Autoencoder and Transformer Encoder for real, runs classical
   baselines, produces the first results table.
2. `diagnostics/` — run in order (01 through 06), in the same Colab
   session without restarting. Each depends on objects created by the
   previous script. This is the actual debugging trail (a guessed
   DBSCAN hyperparameter that silently broke it, a tolerance-metric
   denominator bug) — kept intact rather than collapsed into one clean
   script. See `paper/draft_paper.md` Section 5 for the full narrative
   behind each fix.

## Study 2 — Grid Network (`study2_grid_network/`)

Also Google Colab. Run phases in order:

1. **`phase1_data_acquisition/`** (run 01 → 07) — downloads load and
   line-flow data from the Gillioz et al. Zenodo dataset, verifies date
   alignment against an independently-known real event (Christmas Day
   2016), identifies France's cross-border interconnector lines directly
   from the network topology, and fixes the power-flow sign convention.
   Script 07 is a fully self-contained version safe to run in a fresh
   session (mounts Google Drive for persistence across disconnects).
2. **`phase2_ground_truth_features/`** (run 01 → 02) — labels ground
   truth events (French nuclear crisis, COVID-19), engineers features,
   and builds two different train/test split protocols for direct
   sensitivity comparison.
3. **`phase3_modelling/`** (run 01) — trains and evaluates all five
   methods (LSTM, Transformer, Isolation Forest, DBSCAN, rolling
   z-score) across both protocols and both data types (load, line flow).
4. **`diagnostics/`** — the verification trail: confirming COVID/nuclear
   crisis signals are genuinely present in this synthetic data before
   trusting them as ground truth, testing (and rejecting) a global
   sign-inversion hypothesis, and diagnosing why LSTM and Transformer
   produced near-identical results on line-flow data (genuine 99.5%
   convergence, not a bug — verified, not assumed).

Full methodology and honest limitations:
`docs/grid_network_methodology_and_caveats.md`.

## Why both studies keep their debugging history

Real bugs were found and fixed in both studies. The numbered diagnostic
scripts are the actual evidence trail, not a retrospectively cleaned-up
narrative. If you just want the finished pipeline without the debugging
history, the main pipeline scripts (Study 1: `01_main_training_pipeline.py`;
Study 2: phase1 → phase2 → phase3, skipping `diagnostics/`) are sufficient
— but the diagnostics are what make the final numbers trustworthy.
