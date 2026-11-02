# Step 9 result (factorial attribution, run 2 Nov 2026), reported as pre-registered

Design exactly as in `docs/STEP9_PREREGISTRATION.md`: 648 datasets (3 backgrounds x 3 event types x 3 durations x 3 magnitudes x 2 resolutions x 4 replicates),
9 detectors x 2 feature sets scored on identical rows, outcome = normalised AUC-PR. Full tables: `docs/step9_report.md`; raw rows: `results/factorial/grid.csv` (11,664 rows). No dataset failed.

## Variance in skill (percent of total; interval = bootstrap over the 4 replicates only)
| term | share | 95% interval |
|---|---|---|
| detector (main effect) | 19.9 | 19.5-20.4 |
| event type | 10.4 | 9.8-10.9 |
| duration x detector | 10.1 | 9.5-10.7 |
| event type x detector | 9.6 | 9.1-10.1 |
| event type x duration | 6.3 | 6.0-6.5 |
| magnitude | 4.5 | 3.9-5.2 |
| magnitude x detector | 4.1 | 3.8-4.5 |
| resolution x detector | 2.3 | 2.1-2.4 |
| features x detector | 0.5 | 0.3-0.6 |
| background x detector | 1.0 | 1.0-1.2 |
| remainder (higher-order terms and replicate noise) | 24.0 | 22.7-24.2 |

## Pre-registered quantities
1. Detector main effect 19.9%; everything involving the detector (main + two-way interactions) 47.4%, so 58% of detector-related variance is conditional on the setting.
2. Detector x (duration, event type, magnitude, resolution) interactions sum to about 26%, more than the detector main effect (19.9%): by the pre-registered reading, claims of the form "A beats B" are not portable across settings and must be stated per setting.
3. Ranking stability: the global best detector (seasonal_naive) wins 56% of 324 conditions (just above the half threshold); mean Spearman between a condition's ranking and the global ranking 0.54; 8 of 9 detectors win at least one condition.

## What the table shows (descriptive)
* Event type decides who wins: seasonal_naive wins 76-81% of shift and ramp conditions but only 11% of burst conditions, where the first-difference one-liner (diff_z) wins 44%.
* Level-blind detectors (matrix_profile) are near zero on ramps (0.02) and shifts (0.10) and reasonable on bursts (0.39); k-NN is strong on shifts (0.57) and weak on ramps (0.10).
* The residual feature raises isolation_forest (0.29 to 0.43) and k-NN (0.34 to 0.39) but leaves window_pca unchanged; the features x detector share is small (0.5%) because only 4 of 9 detectors read the feature columns and the effect is moderate here.

## Limits (some of them favour particular detectors)
* The backgrounds are exactly seasonal with additive noise, which is the situation a seasonal-naive residual is built for; its global win here says little about real grids. The real-data runs rank it first on ERCOT AUC-PR but not on GB timestamp F1.
* The four replicates only measure replicate noise. Different level choices (durations, magnitudes) would change the shares; they were fixed from the real-data runs and not tuned.
* Default configs, no neural models, three event types, one resolution step. Skill is AUC-PR based, so alert-budget F1 may rank differently (as on ERCOT).
* The shares describe this design, not energy data in general.
