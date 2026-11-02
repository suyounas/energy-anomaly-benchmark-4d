# Step 8 pre-registration: a held-out test of the withdrawn duration rule on public labelled series

Written on 2 Nov 2026, BEFORE any public-benchmark series has been run through this code. Anything not
listed under "Primary test" is exploratory and will be labelled so in the paper.

## Claim under test (from the earlier paper, withdrawn in the revision)
"The ranking of density-based and sequence-reconstruction detectors inverts as events span more native-resolution
samples: density-based detectors lose ground, reconstruction detectors gain, as event length in samples grows."

## Data
TSB-AD-U (Liu and Paparrizos, NeurIPS 2024): about 1,070 labelled univariate series from 40 source datasets,
each with a training prefix named in its file name (`..._tr_<N>_...`). Downloaded by the author from
`thedatum.org/datasets/TSB-AD-U.zip`; per-source licences differ and are respected (no redistribution of the data).
Series are used as given: no series is dropped except for the reasons listed under "Exclusions".

## Detectors (CPU, inductive: fitted on the training prefix only)
Stand-ins, because the neural models are too slow for ~1,000 series on CPU:
* density-based: `knn_inductive` (distance to the 5th nearest training point; the inductive analogue of DBSCAN)
* reconstruction-based: `window_pca` (linear autoencoder on 30-step windows; the linear analogue of the LSTM-AE)
Other detectors (isolation_forest, window_iforest, matrix_profile, seasonal_naive, diff_z, rolling_z,
spectral_residual) are reported descriptively. The LSTM-AE and Transformer can be added on a GPU later and are NOT part
of the primary test.

## Outcome and exposure
* Outcome per series s: d_s = AUC-PR(window_pca) - AUC-PR(knn_inductive) on the test part (rows scorable by all detectors).
* Exposure: log of the median labelled-event length in samples (test part).
* Confounders handled: dataset (stratification, one estimate per source dataset), prevalence (partial rank
  correlation controlling for log prevalence; AUC-PR depends on prevalence and prevalence co-varies with event length).

## Primary test (one hypothesis)
Per source dataset with at least 8 series: partial Spearman correlation rho_k between log median event length and d_s,
controlling for log prevalence. Combine datasets by Fisher-z with weights (n_k - 3). Interval: bootstrap over source
datasets (500 replicates, resample datasets with replacement), 95% percentile.
* The original claim is SUPPORTED if the combined rho > 0 (reconstruction gains on longer events) and its 95% interval
  excludes 0.
* It is CONTRADICTED if the interval excludes 0 with rho < 0.
* Otherwise it is NOT SUPPORTED (no evidence either way at this sample size); the paper will say "not supported".

## Exploratory (will be labelled as such)
Other detector pairs; tercile tables of mean rank by event length; per-dataset estimates; VUS-PR instead of AUC-PR
(our re-implementation, not the official code); event length in calendar time where a source resolution is known.

## Exclusions (decided now)
Series with no anomaly in the test part; series whose test part has fewer than 100 rows after the 29-row window warm-up;
series on which any primary detector raises an error (counted and reported, not silently dropped).

## Known limitations stated in advance
Linear and k-NN stand-ins do not represent LSTM-AE or DBSCAN exactly; event length in samples is not event length in
time; TSB-AD sources have known labelling problems (Wu and Keogh); window length is fixed at 30 samples for the window
models; stationarity-based features were chosen for energy data, not for these sources.
