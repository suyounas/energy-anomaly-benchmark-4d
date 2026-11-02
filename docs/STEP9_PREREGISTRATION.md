# Step 9 pre-registration: which design choices drive detector performance?

Written and committed before the factorial experiment is run. Nothing below is changed after seeing results; deviations go in `docs/STEP9_RESULT.md`.

## Question
Sample count, resolution, dataset, domain, normalisation and event type are confounded across real series, so no conclusion can be assigned to any one of them.
On real data they cannot be separated. Here they are varied independently and fully crossed on semi-synthetic energy-like series with injected labelled events,
so the share of outcome variance that belongs to each can be measured.

## Design (fully crossed, balanced)
| factor | levels |
|---|---|
| background (domain/dataset stand-in) | strong seasonality; weak seasonality, noisy; strong seasonality with trend |
| event type | level shift; variance burst; ramp |
| event duration (hours at the base 1 h grid) | 4, 24, 120 |
| magnitude (robust residual sd, peak) | 3, 6, 12 |
| resolution (aggregation of the base grid; event length in samples = duration / factor) | 1 h, 4 h |
| feature set (normalisation/representation) | published 4-feature recipe; recipe + causal seasonal-naive residual |
| detector | diff_z, rolling_z, seasonal_naive, isolation_forest, knn_inductive, window_pca, window_iforest, matrix_profile, spectral_residual (default configs, no tuning) |
Replicates: 4 independent random draws (background noise, event positions) per cell. 120 days clean training, 90 days test with 3 events of the given type.
Detectors that do not use the feature columns (diff_z, rolling_z, seasonal_naive, matrix_profile, spectral_residual) return identical scores under both feature levels: a structural zero, kept so the design stays balanced.

## Outcome
Normalised AUC-PR (skill) = (AUC-PR - prevalence) / (1 - prevalence), on rows where every detector has a score. Chosen because lift is bounded by 1/prevalence and prevalence changes with duration.

## Primary analysis
Sum-of-squares decomposition of the skill (balanced factorial, all main effects and all two-way interactions, remainder = higher-order terms and replicate noise).
Reported as share of total variance with a bootstrap interval over replicates (4 replicates only: the interval describes replicate noise, not the choice of levels).
Quantities pre-specified:
1. Share of variance for each main effect and for detector x factor interactions.
2. "Detector main effect share" versus "everything that involves the detector, interactions included": how much of detector performance is conditional on the setting.
3. Ranking stability: Spearman correlation between each condition's detector ranking and the global ranking, and the fraction of conditions in which the global-best detector is also best.

## What each outcome would mean (no pass/fail)
* If the detector main effect carries most of the variance and rankings are stable, a single "best detector" statement is defensible on this design.
* If detector x (event type, feature set, magnitude, duration) interactions are comparable to the detector main effect, or the global-best detector is best in fewer than half the conditions, then claims of the form "method A beats method B" are not portable and must be stated per setting.
Either result is reported.

## Limits stated in advance
Synthetic backgrounds and three event types are a model of energy data, not energy data. Default configs only (the equal-budget runs show tuning changed little).
Neural detectors are not in this grid (cost). Magnitudes are relative to the series' own residual spread. The levels were fixed from the real-data runs, so findings are conditional on this range.
