# Evaluation design points: audit, sources, and what is implemented

Status key: **verified** = implemented and exercised by tests or a real-data diagnostic;
**implemented, not yet run** = code and tests exist, the real-data or public-benchmark run is still to do;
**open** = not implemented.

| Point | Concern | Verified so far | Still open |
|---|---|---|---|
| confounds | resolution moves with dataset, domain, normalisation, event type | Synthetic constant-prevalence sweep (resolution and duration varied independently, 2 event types, 10 seeds): resolution 30 min to 1 h moves DBSCAN F1 by <= 0.022. Real-data resolution ablation inside ONE series (same events resampled): resolution does matter on real data, and the ranking flips with the feature set (recipe vs `resid`), with paired block-bootstrap intervals. | Variance attribution across ALL factors at once (background series x event type x duration x resolution x detector), step 9. |
| five events | a selection rule on five events | The rule and the "detection gap" are withdrawn in the paper. | A held-out test of the withdrawn rule on hundreds of labelled public series (step 8). Operator-declared GB/ERCOT event lists to raise the energy event count (not verified, optional). |
| single split | 1-2 events, no repeats, no intervals | Seeds (5-10), paired 7-day-block bootstrap, `reliably_worse_than_best`; the 5-seed neural run shows seed sd (0.004) hides the real uncertainty (interval 0.065-0.361). | Series-level (cluster) bootstrap over public series (step 8). |
| equal terms | DBSCAN tuned, others fixed; missing details; mixed P/R | Equal-budget search (default + 8 random trials per family, validation on synthetic taxonomy events, informativeness gate), hyper-parameter table, separate event-level and timestamp-level metrics, AUC-PR next to F1. | DBSCAN is not a tuned family in the equal-budget harness (flag-only output); give it the same budget via a score variant. |
| benchmarks and stronger detectors | TSB-UAD, TimeEval, Wu and Keogh; stronger recent detectors | Related work written from verified sources. **Step 7 (this patch)**: Wu-Keogh one-liner (`diff_z`), seasonal-naive residual score, window-PCA / window-kNN / window-IForest on the SAME 30-step windows the neural models use, AB-join matrix profile, spectral residual; VUS-PR (own re-implementation). 107 tests. | Run step 7 on GB/ERCOT; cross-check with stumpy and TSB-AD (`experiments/crosscheck_official.py`); foundation-model zero-shot detector (Chronos / MOMENT, needs GPU) as an optional extra. |

## Sources used (all opened and read during this revision)

| Source | What it gives | Numbers verified against the paper |
|---|---|---|
| TSB-UAD, Paparrizos et al., PVLDB 15(8), 2022 (`vldb.org/pvldb/vol15/p1697-paparrizos.pdf`; code `github.com/TheDatumOrg/TSB-UAD`, Apache-2.0) | 13,766 univariate series: 18 public datasets (1,980 series), 126 artificial (958), 92 synthetic (10,828); 12 methods; 9 measures | Yes (abstract). The repository README says 12,686 series and lists 14 methods; the paper is the citation. |
| TSB-AD, Liu and Paparrizos, NeurIPS 2024 D&B (`github.com/TheDatumOrg/TSB-AD`, `pip install TSB-AD`, Apache-2.0) | about 1,070 series from 40 datasets (TSB-AD-U univariate, TSB-AD-M multivariate); statistical, neural and foundation-model detectors (33 listed); VUS-PR recommended; "simpler architectures and statistical methods often yield better performance" | Counts from the README only; the README says 40 algorithms but lists 33, so the paper text now says "33 are listed". Per-dataset licences differ; cite the source datasets. |
| TimeEval, Schmidl, Wenig, Papenbrock, PVLDB 15(8), 2022 (`vldb.org/pvldb/vol15/p1779-wenig.pdf`; toolkit `p3678-schmidl.pdf`) | 71 algorithms re-implemented, 976 series after filtering from 1,354 collected; no single winner; deep learning "not (yet) competitive"; algorithms "remarkably sensitive to their parameter settings"; extremum anomalies easiest, trend hardest | Yes. The study does not analyse performance by anomaly length (checked), so it cannot be cited for or against a duration effect. |
| Wu and Keogh, "Current time series anomaly detection benchmarks are flawed and are creating the illusion of progress" (arXiv 2009.13807; IEEE TKDE 2023) | triviality (86.1% of Yahoo solvable by a one-liner), unrealistic density, mislabelled ground truth, run-to-failure bias; UCR Anomaly Archive (one anomaly per series, training prefix); recommend simple baselines and invariance descriptions | Yes. |
| VUS, Paparrizos et al. (arXiv 2502.13318; PVLDB 2022 p2774) | range-aware, threshold-free VUS-PR | Definition read; our code is a re-implementation (see `ebench/rangemetrics.py`). |

Data access: the benchmark archives are hosted at `thedatum.org/datasets/` (TSB-AD-U.zip, TSB-UAD-Public.zip). The authoring sandbox cannot reach them (allow-list), so every public-benchmark run happens on the author's machine or Colab. No public-series result exists yet.

## What the detector literature recommends, and what we included
* Simple baselines first (Wu and Keogh; TimeEval; TSB-AD): `diff_z`, `seasonal_naive`.
* Distance and subsequence methods (TimeEval: Sub-LOF, kNN; TSB-UAD: Matrix Profile, NORMA): `window_knn`, `matrix_profile`.
* Linear reconstruction (TSB-AD Sub-PCA; TSB-UAD PCA/AE): `window_pca`. This is the control for the LSTM-AE: same windows, same features, linear model.
* Isolation forests on windows (TSB-AD Sub-IForest): `window_iforest`.
* Frequency-domain (TSB-AD SR): `spectral_residual` (offline within a block; documented).
* Not included: Series2Graph, SAND, NORMA, DWT-MLEAD (TimeEval's best cost/benefit), OmniAnomaly, USAD, TranAD, TimesNet, Chronos/MOMENT. They need dedicated packages or a GPU; the optional extra above is the single foundation-model detector.

## Planned steps after step 7
* **Step 8, public benchmark, held-out test of the withdrawn rule** (`experiments/public_benchmark.py`): TSB-AD-U series with a training prefix; each series gives (event length in samples, per-detector AUC-PR and VUS-PR). Test: does the pairwise advantage of one detector over another change with log event length, within dataset, with a bootstrap over series (cluster) and datasets? This is the first test of the original claim on more than five events.
* **Step 9, factor attribution** (`experiments/factor_attribution.py`): semi-synthetic factorial design on real backgrounds (GB, ERCOT, public series): event type x duration (days) x magnitude x resolution x detector, and a variance decomposition of AUC-PR into those factors, so "which factor produced the apparent inversion" has a number.
