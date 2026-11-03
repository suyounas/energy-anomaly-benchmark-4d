# Label sensitivity report (experiments/label_sensitivity.py)

Series: neso_demand_full.csv; test rows 54884; detectors (default configs, mean score over 3 seeds): isolation_forest, knn_inductive, rolling_z, diff_z, seasonal_naive, window_pca, matrix_profile.

## A. Does the series deviate inside each labelled event? (causal seasonal-naive residual, robust z)

`mean_daily_dev` = mean over the event's days of the daily mean |z|; `pct_of_ref_days_below` = percentage of non-event days with a smaller daily deviation.

| event | n_timestamps | mean_daily_dev | share_above_cut | ref_mean_daily_dev | ref_share_above_cut | pct_of_ref_days_below |
|---|---|---|---|---|---|---|
| COVID | 1679 | 1.265 | 0.093 | 0.872 | 0.02 | 80.296 |
| ColdSnap | 240 | 2.664 | 0.242 | 0.872 | 0.02 | 97.748 |

## B. Unlabelled days that deviate at least as much as the weakest labelled event
546 of 2842 days outside every labelled event reach the weakest event's daily deviation (1.26). These are candidates for missing labels, **not** claimed anomalies (a day can deviate for reasons such as holidays or weather that nobody labelled as an event).

Candidate days by month (top 12):

| month | days |
|---|---|
| 2023-01 | 20 |
| 2017-12 | 18 |
| 2023-12 | 17 |
| 2022-12 | 17 |
| 2017-11 | 16 |
| 2017-01 | 15 |
| 2020-12 | 15 |
| 2018-04 | 15 |
| 2017-02 | 14 |
| 2021-11 | 14 |
| 2019-02 | 14 |
| 2016-11 | 14 |

## C. Detector ranking under label perturbations

### auc_pr
Baseline best: **matrix_profile**; unchanged in 95% of 19 perturbed label sets; Kendall tau of the full ranking vs baseline: median 1.00, minimum -0.24.

| scenario | tau_vs_baseline | best | best_unchanged |
|---|---|---|---|
| dilate 1d | 1.0 | matrix_profile | True |
| dilate 3d | 1.0 | matrix_profile | True |
| dilate 7d | 1.0 | matrix_profile | True |
| drop COVID | -0.238 | seasonal_naive | False |
| drop ColdSnap | 0.905 | matrix_profile | True |
| end earlier 1d | 1.0 | matrix_profile | True |
| end earlier 3d | 1.0 | matrix_profile | True |
| end earlier 7d | 0.905 | matrix_profile | True |
| end later 1d | 1.0 | matrix_profile | True |
| end later 3d | 1.0 | matrix_profile | True |
| end later 7d | 1.0 | matrix_profile | True |
| length x0.5 | 1.0 | matrix_profile | True |
| length x1.5 | 1.0 | matrix_profile | True |
| start earlier 1d | 1.0 | matrix_profile | True |
| start earlier 3d | 1.0 | matrix_profile | True |
| start earlier 7d | 0.905 | matrix_profile | True |
| start later 1d | 1.0 | matrix_profile | True |
| start later 3d | 1.0 | matrix_profile | True |
| start later 7d | 0.905 | matrix_profile | True |

| scenario | diff_z | isolation_forest | knn_inductive | matrix_profile | rolling_z | seasonal_naive | window_pca |
|---|---|---|---|---|---|---|---|
| baseline | 0.032 | 0.095 | 0.059 | 0.213 | 0.035 | 0.084 | 0.05 |
| dilate 1d | 0.033 | 0.094 | 0.061 | 0.227 | 0.037 | 0.086 | 0.051 |
| dilate 3d | 0.037 | 0.094 | 0.065 | 0.221 | 0.04 | 0.087 | 0.053 |
| dilate 7d | 0.044 | 0.098 | 0.073 | 0.204 | 0.047 | 0.096 | 0.057 |
| drop COVID | 0.004 | 0.004 | 0.003 | 0.003 | 0.004 | 0.046 | 0.003 |
| drop ColdSnap | 0.028 | 0.099 | 0.06 | 0.238 | 0.031 | 0.056 | 0.051 |
| end earlier 1d | 0.031 | 0.094 | 0.058 | 0.206 | 0.034 | 0.083 | 0.05 |
| end earlier 3d | 0.029 | 0.092 | 0.055 | 0.206 | 0.032 | 0.077 | 0.05 |
| end earlier 7d | 0.026 | 0.087 | 0.051 | 0.2 | 0.029 | 0.066 | 0.051 |
| end later 1d | 0.032 | 0.095 | 0.061 | 0.223 | 0.036 | 0.084 | 0.05 |
| end later 3d | 0.034 | 0.096 | 0.064 | 0.221 | 0.038 | 0.084 | 0.05 |
| end later 7d | 0.038 | 0.101 | 0.07 | 0.213 | 0.042 | 0.088 | 0.052 |
| length x0.5 | 0.016 | 0.042 | 0.025 | 0.103 | 0.017 | 0.035 | 0.021 |
| length x1.5 | 0.048 | 0.112 | 0.089 | 0.207 | 0.053 | 0.091 | 0.059 |
| start earlier 1d | 0.032 | 0.094 | 0.06 | 0.217 | 0.036 | 0.086 | 0.051 |
| start earlier 3d | 0.034 | 0.093 | 0.061 | 0.213 | 0.037 | 0.087 | 0.052 |
| start earlier 7d | 0.038 | 0.09 | 0.063 | 0.201 | 0.041 | 0.091 | 0.054 |
| start later 1d | 0.031 | 0.095 | 0.058 | 0.213 | 0.034 | 0.08 | 0.049 |
| start later 3d | 0.029 | 0.092 | 0.056 | 0.211 | 0.032 | 0.066 | 0.044 |
| start later 7d | 0.026 | 0.089 | 0.052 | 0.202 | 0.029 | 0.047 | 0.036 |

### ts_f1
Baseline best: **matrix_profile**; unchanged in 95% of 19 perturbed label sets; Kendall tau of the full ranking vs baseline: median 0.98, minimum -0.05.

| scenario | tau_vs_baseline | best | best_unchanged |
|---|---|---|---|
| dilate 1d | 0.976 | matrix_profile | True |
| dilate 3d | 0.976 | matrix_profile | True |
| dilate 7d | 0.976 | matrix_profile | True |
| drop COVID | -0.053 | seasonal_naive | False |
| drop ColdSnap | 0.878 | matrix_profile | True |
| end earlier 1d | 0.976 | matrix_profile | True |
| end earlier 3d | 0.9 | matrix_profile | True |
| end earlier 7d | 0.878 | matrix_profile | True |
| end later 1d | 0.976 | matrix_profile | True |
| end later 3d | 0.976 | matrix_profile | True |
| end later 7d | 0.976 | matrix_profile | True |
| length x0.5 | 0.878 | matrix_profile | True |
| length x1.5 | 0.878 | matrix_profile | True |
| start earlier 1d | 0.976 | matrix_profile | True |
| start earlier 3d | 0.976 | matrix_profile | True |
| start earlier 7d | 0.976 | matrix_profile | True |
| start later 1d | 0.878 | matrix_profile | True |
| start later 3d | 0.878 | matrix_profile | True |
| start later 7d | 0.878 | matrix_profile | True |

| scenario | diff_z | isolation_forest | knn_inductive | matrix_profile | rolling_z | seasonal_naive | window_pca |
|---|---|---|---|---|---|---|---|
| baseline | 0.02 | 0.17 | 0.02 | 0.331 | 0.027 | 0.176 | 0.106 |
| dilate 1d | 0.02 | 0.167 | 0.02 | 0.341 | 0.027 | 0.175 | 0.108 |
| dilate 3d | 0.024 | 0.164 | 0.021 | 0.332 | 0.03 | 0.171 | 0.104 |
| dilate 7d | 0.028 | 0.16 | 0.023 | 0.309 | 0.034 | 0.168 | 0.096 |
| drop COVID | 0.009 | 0.001 | 0.0 | 0.0 | 0.008 | 0.104 | 0.0 |
| drop ColdSnap | 0.015 | 0.179 | 0.021 | 0.349 | 0.023 | 0.116 | 0.111 |
| end earlier 1d | 0.02 | 0.171 | 0.019 | 0.326 | 0.026 | 0.177 | 0.107 |
| end earlier 3d | 0.019 | 0.172 | 0.019 | 0.328 | 0.024 | 0.165 | 0.109 |
| end earlier 7d | 0.016 | 0.169 | 0.017 | 0.322 | 0.022 | 0.144 | 0.114 |
| end later 1d | 0.02 | 0.169 | 0.02 | 0.337 | 0.026 | 0.175 | 0.105 |
| end later 3d | 0.021 | 0.169 | 0.022 | 0.333 | 0.028 | 0.171 | 0.103 |
| end later 7d | 0.022 | 0.171 | 0.024 | 0.321 | 0.03 | 0.172 | 0.099 |
| length x0.5 | 0.013 | 0.09 | 0.005 | 0.191 | 0.016 | 0.089 | 0.042 |
| length x1.5 | 0.029 | 0.168 | 0.024 | 0.297 | 0.036 | 0.156 | 0.091 |
| start earlier 1d | 0.02 | 0.169 | 0.02 | 0.336 | 0.027 | 0.177 | 0.109 |
| start earlier 3d | 0.023 | 0.165 | 0.019 | 0.331 | 0.029 | 0.176 | 0.107 |
| start earlier 7d | 0.026 | 0.159 | 0.019 | 0.318 | 0.031 | 0.172 | 0.103 |
| start later 1d | 0.019 | 0.172 | 0.02 | 0.33 | 0.026 | 0.168 | 0.107 |
| start later 3d | 0.018 | 0.168 | 0.016 | 0.326 | 0.026 | 0.142 | 0.094 |
| start later 7d | 0.015 | 0.164 | 0.013 | 0.31 | 0.023 | 0.106 | 0.068 |
