# Step 8 result (TSB-AD-U, run 2 Nov 2026) — recorded as pre-registered

**Primary test: NOT SUPPORTED.** Partial Spearman between log median event length and the AUC-PR difference
(window_pca minus knn_inductive), controlling for log prevalence, stratified by source dataset, Fisher-z combined,
bootstrap over datasets (500 replicates): **rho = -0.047, 95% interval [-0.163, 0.139]**; 862 series, 23 source datasets,
17 with >= 8 series. The withdrawn rule (reconstruction gains over density on longer events) is not recovered on public data and stays withdrawn.

First run, before the harness fix below (822 series, 15 datasets): rho = -0.060, [-0.196, 0.125]. Same verdict.

## Exploratory sensitivity (not pre-registered; same statistic, final 862-series file)
| subset | series | datasets | rho | 95% interval |
|---|---|---|---|---|
| all (primary) | 862 | 17 | -0.047 | [-0.163, 0.139] |
| clean training prefix only | 691 | 13 | 0.008 | [-0.130, 0.188] |
| event length > 3 samples | 561 | 15 | -0.064 | [-0.207, 0.109] |
| clean prefix and length > 3 | 534 | 12 | -0.058 | [-0.195, 0.116] |
| period found (autocorrelation peak) only | 604 | 14 | -0.049 | [-0.176, 0.229] |
| datasets with >= 20 series only | 862 | 9 | -0.052 | [-0.178, 0.149] |

None excludes 0. Per-dataset signs disagree (first run: WSD +0.35, OPPORTUNITY +0.41, SMAP +0.39; YAHOO -0.15, NAB -0.43, IOPS -0.55),
so a single rule across sources is not supported either way. Window-PCA has higher AUC-PR than k-NN on 25.4% of series
(median difference -0.094): the stand-in pair is a poor proxy for "reconstruction vs density" on these sources.

## Deviations, stated openly
* 40 of 870 series errored with `StopIteration` in my period estimator (constant or degenerate training prefix). A harness bug, not a
  primary-detector error, so the pre-registered exclusion does not cover it. Fixed after the first run, without looking at how those series
  would change the test; re-run with `--retry-errors`. Both results are reported above. The re-run series have no detectable period (fallback used).
* 1 series excluded under the pre-registered rule (window_pca raised LinAlgError); 7 had no anomaly in the test part.
* 171 of the 862 analysed series have anomalies inside the training prefix (Stock, TAO, TODS, parts of YAHOO, LTDB): the clean-fit assumption is violated there; see the sensitivity table.
* Caveats from the pre-registration still apply: stand-in detectors, samples not time, known labelling problems (Wu and Keogh), VUS-PR is our re-implementation.
