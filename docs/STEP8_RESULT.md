# Step 8 result (TSB-AD-U, run 2 Nov 2026) — recorded as pre-registered

**Primary test: NOT SUPPORTED, in both runs.** Re-run with the 40 errored series restored (862 series, 17 datasets with >= 8 series): rho = -0.047, 95% interval [-0.163, 0.139]. First run (822 series, 15 datasets): rho = -0.060, [-0.196, 0.125]. The numbers below are from the first run. Partial Spearman between log median event length and the AUC-PR difference
(window_pca minus knn_inductive), controlling for log prevalence, stratified by source dataset, Fisher-z combined,
bootstrap over datasets: rho = -0.060, 95% interval [-0.196, 0.125]; 822 series, 23 source datasets, 15 with >= 8 series.
The withdrawn rule (reconstruction gains over density on longer events) is therefore not recovered on public data; it stays withdrawn.

Exploratory sensitivity (not pre-registered; same statistic): clean-training-prefix series only rho = -0.008 [-0.162, 0.200];
event length > 3 samples only -0.067 [-0.243, 0.112]; both -0.062 [-0.209, 0.127]; period found only -0.049 [-0.176, 0.229];
datasets with >= 20 series only -0.060 [-0.191, 0.151]. None excludes 0.
Per-dataset signs disagree (WSD +0.35, OPPORTUNITY +0.41, SMAP +0.39; YAHOO -0.15, NAB -0.43, IOPS -0.55), so a single rule across sources is not supported either way.
Window-PCA has higher AUC-PR than k-NN on only 25.5% of series (median difference -0.094): the stand-in pair is not a good proxy for "reconstruction vs density" here.

## Deviations, stated openly
* 40 of 870 series errored with `StopIteration` in my period estimator (constant or degenerate training prefix). This is a harness bug, not a
  primary-detector error, so the pre-registered exclusion does not cover it. It was fixed after the first run and the 40 series are re-run with
  `--retry-errors`. The first-run result above is without them; the re-run result must be reported next to it. The fix was made without looking
  at how those series would change the test.
* 1 series was excluded under the pre-registered rule (window_pca raised LinAlgError); 7 had no anomaly in the test part.
* Sources: 170 of 822 analysed series have anomalies inside the training prefix (Stock, TAO, TODS, parts of YAHOO, LTDB): the "clean" fit assumption is violated there; see the sensitivity row.
* Caveats from the pre-registration still apply (stand-in detectors, samples not time, known labelling problems, VUS-PR is our implementation).
