Colab GPU runs of 2 Nov 2026 (equal-budget tuning, 5 seeds, 8 trials; classical, stronger-baseline and neural families).
* `*_resid_*`: features = recipe + causal seasonal-naive residual (run with the code of step 7, before step 7b: no `_ci_metrics`/`_coverage` files exist for this feature set).
* no suffix: features = published 4-feature recipe (run with the code of step 7c).
* **`*_ci_metrics.csv`: the `vus_pr` rows were computed with the plain block bootstrap, which was wrong (point estimates outside their intervals; fixed 2 Nov 2026, see docs/REVISION_LOG.md). Use only the `auc_pr` rows. VUS-PR point values are valid; VUS-PR intervals must be recomputed.**
