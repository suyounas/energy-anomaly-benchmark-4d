# Validation design for equal-budget tuning

## Why the first design failed
The first validation set injected 1- and 3-day level shifts and variance bursts (3 sd) into the last
30% of the clean training window and selected configs by timestamp-level AUC-PR. On the real data
(29 Oct 2026 runs, `docs/real_data_summary.md`) kNN, LSTM-AE and Transformer scored a validation AUC-PR
equal to the validation prevalence for **every** config (GB 0.133-0.134 vs prevalence 0.132), so the
arg-max was selection on noise.

Likely contributing cause (to be confirmed on real data with `*_val_cells.csv`): the published 4-feature
recipe is `ret, roll_mean(30 d), roll_std(30 d), roll_skew` and contains **no raw level**. A short level shift
is then visible only at its two edges (a return spike); the timestamp labels cover the whole event, so a
detector that rings only at the edges has an AUC-PR near the prevalence. The synthetic stand-in shows the same
picture: every family has lift ~ 1 on shift cells.

## The taxonomy suite (default; `--val-suite taxonomy`)
* **Cells**: type x duration x magnitude, declared a priori and never derived from test events.
  Types: `shift` (level), `burst` (variance), `ramp` (drift), `pulse` (3 h). Durations 1, 3, 7 days
  (ERCOT: 1, 3; its clean window is too short for 7). Magnitude 1.5 clean-part sd.
* **Layout**: the validation region (last 35% of the clean window) is cut into one slot per cell, with
  >= 1 clean day after every event. Events are day-aligned (pulses are not).
* **Metric**: for each cell, AUC-PR with positives = that cell's event timestamps and negatives = clean
  timestamps only; reported as a lift over that cell's prevalence (1 = chance). Selection uses the
  **macro lift** (mean over cells), so one easy event type cannot decide the choice.
* **Gate**: a family is *informative* only if its best macro lift >= 1.3 **and** the macro lift varies
  by >= 0.1 across its configs. Otherwise the default config is kept and the run prints a NOTE.
  Per-cell lifts are written to the trials CSV (`lift_cell_<i>`) and the cell table to `*_val_cells.csv`.

## What this does and does not give you
* It makes "this detector cannot see this kind of event" **visible** (per-cell table) instead of hiding it
  inside one number, and it stops selection on noise.
* It does **not** prove that tuned configs transfer to real events: validation events are synthetic.
  Report validation lift and test results side by side, and say so in the paper.
* Check before trusting a neural result: if the per-cell lifts of a family are ~ 1 on every cell,
  the family is blind to the suite and "tuned == default" by design.
* kNN is nearly insensitive to `k` (the ranking by k-th neighbour distance barely changes between k = 3 and 50 when
  the training set is dense); on a stand-in series its validation macro lift varies by < 0.01 across k, matching the real runs.
  That is a property of the detector, not a bug, and the gate keeps the default for it.

## Real-data check of the level-blindness hypothesis (30 Oct 2026, taxonomy suite, `recipe` features)
Default-config validation lift (1 = chance), GB (1 h) / ERCOT (hourly):

| family | shift (GB 1/3/7 d) | burst (GB 1/3/7 d) | pulse |
|---|---|---|---|
| Isolation Forest | 0.66 / 0.59 / 0.59 | 7.9 / 10.8 / 6.0 | 1.2 |
| kNN | 0.75 / 0.80 / 0.65 | 3.2 / 6.3 / 6.5 | 0.7 |
| rolling z | 1.7 / 1.7 / 1.1 | 8.4 / 9.5 / 2.8 | 31.8 |

Level shifts score at or *below* chance for Isolation Forest and kNN (ERCOT the same pattern), while bursts are found. The
real test events (COVID, the Freeze, the cold snap) are mostly level-shift or ramp events, so the recipe is largely blind to
the event type the benchmark is about. This is a statement about the 4-feature recipe, not about the detectors.

## Feature ablation: `--features resid`
Adds one causal feature: value minus the median of the same time of the week over the previous 4 weeks (seasonal-naive
residual), scaled by a robust spread. It uses only past values, needs no fitting, and is 0 on a pure weekly pattern.
Choose between `recipe` and `resid` **on validation lift only**, before looking at test events. On a stand-in series
Isolation Forest's shift lift goes from ~1 to 6-11; kNN barely changes (a 3-sd residual is not far from its training
neighbours in 5-D), which is a property of kNN, not a bug. `FEATURES=resid` in `scripts/run_real_checks.sh`; outputs get a
`_resid` suffix. The published tables stay on `recipe`.
