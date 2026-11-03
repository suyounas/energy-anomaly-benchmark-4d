# Label sensitivity report (experiments/label_sensitivity.py)

Series: ercot_demand_full.csv; test rows 34670; detectors (default configs, mean score over 3 seeds): isolation_forest, knn_inductive, rolling_z, diff_z, seasonal_naive, window_pca, matrix_profile.

## A. Does the series deviate inside each labelled event? (causal seasonal-naive residual, robust z)

`mean_daily_dev` = mean over the event's days of the daily mean |z|; `pct_of_ref_days_below` = percentage of non-event days with a smaller daily deviation.

| event | n_timestamps | mean_daily_dev | share_above_cut | ref_mean_daily_dev | ref_share_above_cut | pct_of_ref_days_below |
|---|---|---|---|---|---|---|
| Texas Freeze | 144 | 2.459 | 0.299 | 0.903 | 0.03 | 95.43 |
| Elliott spike | 120 | 3.423 | 0.508 | 0.903 | 0.03 | 98.513 |

## B. Unlabelled days that deviate at least as much as the weakest labelled event
69 of 1816 days outside every labelled event reach the weakest event's daily deviation (2.46). These are candidates for missing labels, **not** claimed anomalies (a day can deviate for reasons such as holidays or weather that nobody labelled as an event).

Candidate days by month (top 12):

| month | days |
|---|---|
| 2019-10 | 10 |
| 2023-10 | 9 |
| 2021-06 | 7 |
| 2023-06 | 6 |
| 2023-09 | 4 |
| 2020-06 | 4 |
| 2022-02 | 4 |
| 2022-05 | 4 |
| 2020-09 | 3 |
| 2021-02 | 3 |
| 2023-01 | 2 |
| 2019-11 | 2 |

## C. Detector ranking under label perturbations

### auc_pr
Baseline best: **seasonal_naive**; unchanged in 88% of 17 perturbed label sets; Kendall tau of the full ranking vs baseline: median 0.81, minimum 0.62.

| scenario | tau_vs_baseline | best | best_unchanged |
|---|---|---|---|
| dilate 1d | 0.81 | seasonal_naive | True |
| dilate 3d | 0.81 | seasonal_naive | True |
| dilate 7d | 0.714 | seasonal_naive | True |
| drop Elliott spike | 0.81 | matrix_profile | False |
| drop Texas Freeze | 0.905 | seasonal_naive | True |
| end earlier 1d | 1.0 | seasonal_naive | True |
| end earlier 3d | 0.905 | seasonal_naive | True |
| end later 1d | 0.81 | seasonal_naive | True |
| end later 3d | 0.81 | seasonal_naive | True |
| end later 7d | 0.81 | seasonal_naive | True |
| length x0.5 | 0.81 | seasonal_naive | True |
| length x1.5 | 0.905 | seasonal_naive | True |
| start earlier 1d | 0.905 | seasonal_naive | True |
| start earlier 3d | 0.905 | seasonal_naive | True |
| start earlier 7d | 0.905 | seasonal_naive | True |
| start later 1d | 0.81 | seasonal_naive | True |
| start later 3d | 0.619 | matrix_profile | False |

| scenario | diff_z | isolation_forest | knn_inductive | matrix_profile | rolling_z | seasonal_naive | window_pca |
|---|---|---|---|---|---|---|---|
| baseline | 0.009 | 0.012 | 0.014 | 0.133 | 0.012 | 0.239 | 0.08 |
| dilate 1d | 0.01 | 0.018 | 0.015 | 0.112 | 0.016 | 0.231 | 0.073 |
| dilate 3d | 0.013 | 0.023 | 0.018 | 0.106 | 0.022 | 0.192 | 0.064 |
| dilate 7d | 0.021 | 0.029 | 0.028 | 0.123 | 0.03 | 0.146 | 0.062 |
| drop Elliott spike | 0.01 | 0.012 | 0.013 | 0.16 | 0.011 | 0.064 | 0.116 |
| drop Texas Freeze | 0.003 | 0.003 | 0.005 | 0.018 | 0.004 | 0.23 | 0.008 |
| end earlier 1d | 0.009 | 0.012 | 0.015 | 0.144 | 0.01 | 0.277 | 0.083 |
| end earlier 3d | 0.011 | 0.013 | 0.016 | 0.084 | 0.011 | 0.31 | 0.117 |
| end later 1d | 0.01 | 0.013 | 0.014 | 0.117 | 0.015 | 0.213 | 0.079 |
| end later 3d | 0.011 | 0.013 | 0.015 | 0.11 | 0.02 | 0.171 | 0.076 |
| end later 7d | 0.014 | 0.017 | 0.02 | 0.094 | 0.023 | 0.14 | 0.074 |
| length x0.5 | 0.01 | 0.006 | 0.016 | 0.144 | 0.011 | 0.293 | 0.106 |
| length x1.5 | 0.01 | 0.023 | 0.016 | 0.11 | 0.016 | 0.246 | 0.069 |
| start earlier 1d | 0.009 | 0.018 | 0.015 | 0.123 | 0.012 | 0.253 | 0.071 |
| start earlier 3d | 0.01 | 0.024 | 0.016 | 0.112 | 0.014 | 0.237 | 0.059 |
| start earlier 7d | 0.014 | 0.025 | 0.02 | 0.129 | 0.018 | 0.173 | 0.048 |
| start later 1d | 0.009 | 0.008 | 0.014 | 0.14 | 0.012 | 0.211 | 0.09 |
| start later 3d | 0.002 | 0.003 | 0.003 | 0.055 | 0.005 | 0.014 | 0.019 |

### ts_f1
Baseline best: **matrix_profile**; unchanged in 47% of 17 perturbed label sets; Kendall tau of the full ranking vs baseline: median 0.81, minimum 0.49.

| scenario | tau_vs_baseline | best | best_unchanged |
|---|---|---|---|
| dilate 1d | 0.81 | seasonal_naive | False |
| dilate 3d | 0.714 | seasonal_naive | False |
| dilate 7d | 0.81 | matrix_profile | True |
| drop Elliott spike | 1.0 | matrix_profile | True |
| drop Texas Freeze | 0.488 | seasonal_naive | False |
| end earlier 1d | 1.0 | matrix_profile | True |
| end earlier 3d | 0.905 | seasonal_naive | False |
| end later 1d | 0.81 | seasonal_naive | False |
| end later 3d | 0.81 | matrix_profile | True |
| end later 7d | 0.714 | seasonal_naive | False |
| length x0.5 | 0.81 | matrix_profile | True |
| length x1.5 | 0.714 | seasonal_naive | False |
| start earlier 1d | 0.905 | seasonal_naive | False |
| start earlier 3d | 0.81 | seasonal_naive | False |
| start earlier 7d | 0.81 | matrix_profile | True |
| start later 1d | 0.81 | matrix_profile | True |
| start later 3d | 0.488 | matrix_profile | True |

| scenario | diff_z | isolation_forest | knn_inductive | matrix_profile | rolling_z | seasonal_naive | window_pca |
|---|---|---|---|---|---|---|---|
| baseline | 0.002 | 0.035 | 0.032 | 0.134 | 0.024 | 0.125 | 0.06 |
| dilate 1d | 0.003 | 0.056 | 0.031 | 0.135 | 0.033 | 0.156 | 0.072 |
| dilate 3d | 0.005 | 0.077 | 0.028 | 0.148 | 0.04 | 0.174 | 0.074 |
| dilate 7d | 0.012 | 0.073 | 0.024 | 0.184 | 0.047 | 0.166 | 0.064 |
| drop Elliott spike | 0.001 | 0.036 | 0.014 | 0.104 | 0.011 | 0.058 | 0.056 |
| drop Texas Freeze | 0.001 | 0.001 | 0.02 | 0.039 | 0.015 | 0.077 | 0.008 |
| end earlier 1d | 0.001 | 0.036 | 0.033 | 0.127 | 0.014 | 0.109 | 0.05 |
| end earlier 3d | 0.001 | 0.038 | 0.029 | 0.072 | 0.003 | 0.074 | 0.051 |
| end later 1d | 0.003 | 0.034 | 0.031 | 0.131 | 0.033 | 0.136 | 0.073 |
| end later 3d | 0.006 | 0.034 | 0.03 | 0.138 | 0.038 | 0.132 | 0.079 |
| end later 7d | 0.009 | 0.036 | 0.027 | 0.129 | 0.043 | 0.138 | 0.073 |
| length x0.5 | 0.001 | 0.012 | 0.031 | 0.088 | 0.014 | 0.079 | 0.051 |
| length x1.5 | 0.003 | 0.076 | 0.03 | 0.136 | 0.034 | 0.177 | 0.071 |
| start earlier 1d | 0.002 | 0.058 | 0.031 | 0.138 | 0.024 | 0.146 | 0.059 |
| start earlier 3d | 0.002 | 0.081 | 0.03 | 0.145 | 0.027 | 0.171 | 0.056 |
| start earlier 7d | 0.006 | 0.077 | 0.027 | 0.197 | 0.032 | 0.159 | 0.051 |
| start later 1d | 0.002 | 0.011 | 0.03 | 0.118 | 0.025 | 0.097 | 0.061 |
| start later 3d | 0.001 | 0.0 | 0.0 | 0.063 | 0.019 | 0.039 | 0.014 |
