# Model-Independent Unsupervised Anomaly Detection in Energy Markets: A Cross-Commodity Comparative Study

**Sulman Younas**
Independent Researcher, Edinburgh, UK
sulmanyounas93@gmail.com · github.com/suyounas · medium.com/@sulmanyounas93

*Status: working draft, in progress. Not yet submitted anywhere.*

---

## Abstract

We compare five unsupervised anomaly detection methods — LSTM Autoencoder,
Transformer Encoder, Isolation Forest, DBSCAN, and a rolling z-score baseline
— on real daily price data for two energy commodities, WTI crude oil and
Henry Hub natural gas (2010–2024, sourced from the Federal Reserve Economic
Data database). The motivation is a methodological analogy with
model-independent signal searches in particle physics: detecting rare,
structurally unusual events without specifying in advance what they look
like. We evaluate all methods against two real historical shocks reachable
within our test period — the February 2021 Texas deep freeze and the 2022
Russia-Ukraine energy crisis — under both a strict same-day matching
criterion and a disclosed ±10-day tolerance window with a corrected,
event-level recall formulation. We find no single method dominates: DBSCAN
is the only method to detect both events on both commodities, while the
neural approaches offer higher precision but miss events that unfold faster
than their 30-day feature window can register. We report this as a
commodity- and shock-dependent result rather than a general ranking, and
document in detail a methodological error we made and corrected during
evaluation, in the interest of transparency about how the final protocol
was arrived at.

---

## 1. Introduction

Detecting rare, anomalous events in large, noisy time-series data — without
prior knowledge of what the anomaly looks like — is a challenge shared
across scientific and industrial domains. In experimental particle physics,
this is the *model-independent search* problem: identifying statistically
unusual events without specifying in advance what new physics is being
looked for. The approach is to train a detector exclusively on well-understood
background behaviour, then flag events that are structurally incompatible
with it. This principle was used in a collaborative ATLAS Collaboration
analysis at CERN that the author contributed to [1], and underlies the
motivation for the present work: does the same train-on-normal,
flag-deviations principle transfer usefully to energy market time-series,
where the analogous challenge is detecting demand shocks, supply
disruptions, and price crises before they are retrospectively labelled?

This line of inquiry was directly inspired by Andrew Howe's account of
applied data science work at Wood Mackenzie's Innovation & Analytics Lab,
which describes using statistical and machine learning methods to detect
meaningful departures from expected patterns in commodity supply chain and
port congestion data [2]. That work demonstrated the practical value of
anomaly detection in commodity analytics; the present study asks a related
but distinct question at the price-signal level, using published,
reproducible methods rather than proprietary industry data.

A preliminary, non-technical version of this work was published as a blog
post [3]; the present document supersedes it with real trained models,
corrected evaluation methodology, and substantially revised conclusions —
in several respects the opposite of what the preliminary version reported,
for reasons explained in Section 5.

## 2. Methodology and Data

### 2.1 Data

We use the FRED series `DCOILWTICO` (WTI crude oil, $/barrel, Cushing,
Oklahoma) and `DHHNGSP` (Henry Hub natural gas, $/MMBtu), both published by
the U.S. Energy Information Administration and distributed by the Federal
Reserve Bank of St. Louis. Data covers January 2010 to December 2024. Both
series are real, directly downloaded observations — not synthetically
reconstructed.

### 2.2 Features

For each commodity, we compute: daily return, 30-day rolling mean, 30-day
rolling standard deviation, 30-day rolling skewness of returns, and a
rolling z-score of price relative to its 30-day mean and standard
deviation. All models receive the same four core features (return, rolling
mean, rolling std, rolling skew); the z-score is retained separately as a
diagnostic and as the basis of the rolling z-score baseline method.

### 2.3 Models

- **LSTM Autoencoder**: an LSTM encoder-decoder trained to reconstruct
  30-day input sequences. Reconstruction error is the anomaly score.
- **Transformer Encoder**: built from scratch in PyTorch (no pretrained
  weights), with sinusoidal positional encoding and causal masking,
  trained with the same reconstruction objective as the LSTM Autoencoder
  for direct comparability.
- **Isolation Forest**: an ensemble method that isolates anomalies by how
  few random partitions are needed to separate them from the bulk of the
  data. Contamination parameter fixed at 0.05.
- **DBSCAN**: density-based clustering; points falling outside any
  sufficiently dense cluster are flagged as anomalies. The neighbourhood
  radius (`eps`) was tuned via a k-distance elbow heuristic computed on the
  training set only, rather than a fixed guessed value (see Section 5.2 for
  why this mattered).
- **Rolling z-score**: a simple, interpretable statistical baseline
  flagging any day where price deviates more than 3 standard deviations
  from its 30-day rolling mean.

All models are trained exclusively on normal (unlabelled) historical data.
Ground-truth event labels are used only for evaluation, never for training,
preserving the unsupervised nature of the comparison throughout.

### 2.4 Ground-truth events and train/test split

Data was split chronologically per commodity: the first 70% of each series
(2010-01 to 2020-06) for training, the remaining 30% (2020-06 to 2024-12)
for testing. Four historically documented energy market disruptions were
originally selected as ground-truth events: the 2014–2016 OPEC price war,
the 2020 COVID-19 demand collapse, the February 2021 Texas deep freeze, and
the 2022 Russia-Ukraine energy crisis.

Only the latter two fall within the test period; the first two end more
than two months before the test period begins, and are therefore not
reachable by any method regardless of quality. We evaluate recall against
the two reachable events only, and report this explicitly rather than
silently diluting recall against unreachable ground truth (see Section
5.3).

### 2.5 Evaluation protocol

We report two regimes:

1. **Strict same-day matching**: a method must flag the exact calendar day
   of a documented event to count as a true positive.
2. **Event-level tolerance (±10 days)**: an event is counted as detected if
   at least one flagged day falls within 10 days of its documented start or
   end date. Precision remains computed at the day level (a flagged day is
   "correct" if it falls within tolerance of any event); recall is the
   fraction of the two reachable events detected. This distinction matters:
   an earlier version of this evaluation computed tolerance-adjusted recall
   by expanding the *day-level* ground-truth labels, which inflates the
   recall denominator and can cause recall to decrease as tolerance widens —
   the opposite of the intended effect. The event-level formulation used
   here is monotonic in tolerance window size by construction. This error
   and its correction are documented in full in Section 5.2, in the
   interest of transparency.

## 3. Results

### 3.1 Strict same-day performance

| Commodity | Method | Precision | Recall | F1 |
|---|---|---|---|---|
| WTI | LSTM Autoencoder | 0.836 | 0.455 | 0.590 |
| WTI | Transformer Encoder | 0.782 | 0.426 | 0.551 |
| WTI | Isolation Forest | 0.614 | 0.347 | 0.443 |
| WTI | DBSCAN | 0.452 | 0.277 | 0.344 |
| WTI | Rolling Z-score | 0.375 | 0.030 | 0.055 |
| Henry Hub | LSTM Autoencoder | 0.000 | 0.000 | 0.000 |
| Henry Hub | Transformer Encoder | 0.000 | 0.000 | 0.000 |
| Henry Hub | Isolation Forest | 0.175 | 0.099 | 0.127 |
| Henry Hub | DBSCAN | 0.190 | 0.762 | 0.304 |
| Henry Hub | Rolling Z-score | 0.250 | 0.040 | 0.068 |

Under strict evaluation, the LSTM Autoencoder leads on WTI, and both neural
methods score exactly zero on Henry Hub. Section 3.2 shows this headline
result is incomplete on its own.

### 3.2 Why the Henry Hub neural-network zero is not what it first appears

Examining the *percentile rank* of the true anomaly days within each
model's full reconstruction-error distribution — rather than only whether
they cleared the strict 95th-percentile flagging threshold — shows the
neural models are not blind to these events. True anomaly days sit at a
mean percentile rank of 61.5 (LSTM) and 65.2 (Transformer), with individual
days reaching the 93rd percentile: clearly elevated, but short of the cutoff
used for strict flagging. The earliest days each model actually flags
(2021-03-05 for LSTM, 2021-03-09 for Transformer) follow the Texas Freeze
event window (2021-02-10 to 2021-02-25) by 8–12 days. This is consistent
with a mechanistic explanation: features are computed over 30-day rolling
windows, so a sharp, short shock does not fully register in the rolling
statistics until weeks afterward. The strict metric is therefore better
understood as penalising a *lagged* signal than reflecting a genuinely
absent one.

### 3.3 Event-level performance (±10-day tolerance, corrected)

| Commodity | Method | Precision | Events detected (of 2 reachable) | F1 |
|---|---|---|---|---|
| WTI | LSTM Autoencoder | 0.836 | 1/2 | 0.626 |
| WTI | Transformer Encoder | 0.782 | 1/2 | 0.610 |
| WTI | Isolation Forest | 0.684 | 1/2 | 0.578 |
| WTI | **DBSCAN** | 0.532 | **2/2** | **0.695** |
| WTI | Rolling Z-score | 0.375 | 1/2 | 0.429 |
| Henry Hub | LSTM Autoencoder | 0.036 | 1/2 | 0.067 |
| Henry Hub | Transformer Encoder | 0.000 | 0/2 | 0.000 |
| Henry Hub | Isolation Forest | 0.211 | 2/2 | 0.348 |
| Henry Hub | DBSCAN | 0.207 | 2/2 | 0.343 |
| Henry Hub | Rolling Z-score | 0.312 | 1/2 | 0.385 |

**DBSCAN is the only method that detects both reachable events on both
commodities.** On WTI it flags the Texas Freeze 8 days after onset and
Russia-Ukraine 9 days after onset. On Henry Hub it flags the Texas Freeze
5 days *before* the documented window opens — plausibly picking up
pre-event volatility in the run-up to the cold snap — and Russia-Ukraine
10 days after onset.

The WTI Texas Freeze is missed by every method except DBSCAN. We believe
this is best explained as a ground-truth labelling limitation rather than a
model failure: the Texas Freeze was primarily a natural gas and electricity
event, and its transmission into WTI crude oil prices specifically is weak.
Applying one shared event list across both commodities without validating
commodity-specific signal strength likely penalises models for correctly
not reacting to a genuinely weak signal in that series.

Isolation Forest's Henry Hub detection of Russia-Ukraine is logged 79 days
after the event's documented start — this falls within, not after, the
approximately 130-day event window, and is a legitimate detection under our
criterion. It does expose a real limitation of the current metric, however:
a binary detected/not-detected outcome does not distinguish an onset-latency
detection (DBSCAN's +9/+10 days) from a coincidental mid-event flag. A
latency-weighted scoring extension is identified as future work.

## 4. Discussion

The central finding is that **no single method dominates across commodities
and shock types**, and the apparent ranking depends materially on which
evaluation protocol is applied to the same underlying trained models. Under
strict same-day matching, the LSTM Autoencoder appears best on WTI and both
neural methods appear to fail entirely on Henry Hub. Under event-level
evaluation with a modest, fully disclosed tolerance window, DBSCAN becomes
the most robust method overall, while the neural methods retain a genuine
precision advantage (0.78–0.84 vs. DBSCAN's 0.53) at the cost of narrower
event coverage.

We interpret this as a precision/recall operating-point trade-off rather
than a strict ordering of methods: reconstruction-error models (LSTM,
Transformer) flag fewer days with higher confidence, but their reliance on
30-day rolling features makes them structurally slow to respond to short,
sharp shocks. Density-based clustering (DBSCAN) is less selective day to
day, but — because it does not depend on temporal smoothing — is
structurally more responsive to shocks of any duration.

This is a materially different, and more specific, conclusion than either
"neural networks are better" or "classical methods are better" as a
blanket claim. It is also a different conclusion from an earlier,
non-rigorous version of this analysis (Section 5.1), underscoring the value
of running the actual comparison rather than assuming a result in advance.

## 5. A note on methodology and process

We include this section because it materially affects how the results
above should be read, and because we think it is more useful, not less, to
document it than to silently present only the final corrected numbers.

### 5.1 An earlier, invalid version of this comparison

A preliminary write-up of this work [3] reported that Isolation Forest
substantially outperformed both neural network approaches. Those numbers
were generated in an environment where PyTorch was unavailable; the LSTM
Autoencoder and Transformer Encoder results in that version were in fact
computed by a rolling z-score fallback standing in for the untrained neural
architectures, not by genuinely trained models. This was disclosed at the
time in the source code repository's documentation but was not adequately
reflected in the accompanying narrative write-up. Once the neural models
were properly trained (Section 2, Section 3), the finding reversed on WTI:
properly trained neural networks substantially outperformed Isolation
Forest under strict evaluation. We report this discrepancy explicitly as a
caution about the difference between a placeholder result and a genuine
one, and because we believe transparency about this correction is more
valuable than quietly superseding it.

### 5.2 A day-level tolerance metric that inflated its own denominator

An intermediate version of the tolerance-window evaluation computed recall
by extending the *day-level* ground-truth labels by ±10 days and comparing
against the same flagged-day predictions used for the strict metric.
Because recall's denominator (total labelled-positive days) grows under
this scheme as the tolerance window widens, while the numerator (correctly
flagged days) does not necessarily grow proportionally, recall could
mechanically *decrease* under a wider tolerance window — the opposite of
what a tolerance metric should do. We diagnosed this by checking whether
precision moved sensibly (it did) while recall did not, traced it to the
denominator construction, and corrected it to the event-level formulation
used in Section 3.3, where the denominator is the fixed count of reachable
ground-truth events rather than an expanding day count.

### 5.3 Verifying which ground-truth events were actually reachable

Initial inspection of test-set positive-label counts revealed that Henry
Hub's neural models scored exactly zero across precision, recall, and F1 —
an unusually clean result that prompted investigation rather than
acceptance. We verified date ranges and label coverage directly (Section
2.4) and confirmed that two of the four originally selected events fall
entirely within the training period, more than two months before the test
period begins, and are therefore unreachable by any method under this
train/test split. Recall reported in Section 3.3 is accordingly computed
against the two reachable events only.

## 6. Limitations

- Only two of four originally selected ground-truth events fall within the
  test period; the evaluation is correspondingly narrow, and results should
  not be over-generalised to a larger claim about anomaly detection in
  energy markets broadly.
- Ground-truth event windows were applied uniformly across both commodities
  without independently validating that each event produces a comparably
  strong signal in each price series.
- A single chronological train/test split was used; results are not
  cross-validated across multiple splits and may be sensitive to the
  particular boundary chosen.
- The event-level tolerance metric rewards detection within a window but
  does not currently weight detection latency within long events.
- With only two reachable ground-truth events, no confidence interval or
  significance test on the difference between methods is statistically
  meaningful; results should be read as a well-diagnosed case study rather
  than a general benchmark.

## 7. Conclusion and future work

Properly trained, this comparison shows that method choice for unsupervised
energy-market anomaly detection should depend on the shock dynamics of the
specific commodity and the operating trade-off desired between precision
and event coverage, rather than a single universally best architecture.
DBSCAN's consistency across both commodities and both event types — despite
lower day-level precision — is, in our view, the most practically useful
finding for a monitoring application where missing a real event is costlier
than an extra false alarm.

Future directions include: extending the ground-truth event set with
commodity-specific validation; a latency-weighted scoring function that
credits earlier detection within long events; cross-validation across
multiple chronological splits; and testing whether an ensemble combining
DBSCAN's coverage with the neural methods' precision improves on either
alone.

## References

1. ATLAS Collaboration. Anomaly detection search for new resonances
   decaying into a Higgs boson pair and a generic new scalar boson pair in
   final states with jets in pp collisions at √s = 13 TeV with the ATLAS
   detector. *Phys. Rev. D* 108, 052009 (2023).
2. Howe, A. Innovation & Analytics Lab at Wood Mackenzie. *Medium* (2024).
   medium.com/@ahowe42/innovation-analytics-lab-data-science-8c2d79deca79
3. Younas, S. Unsupervised anomaly detection in energy market time-series:
   a comparison of LSTM, Transformer, and classical methods. *Medium*
   (2026). medium.com/@sulmanyounas93
4. Liu, F.T., Ting, K.M., Zhou, Z.H. Isolation Forest. *Proceedings of the
   8th IEEE International Conference on Data Mining*, 413–422 (2008).
5. Hochreiter, S., Schmidhuber, J. Long short-term memory. *Neural
   Computation* 9(8), 1735–1780 (1997).
6. Vaswani, A. et al. Attention is all you need. *Advances in Neural
   Information Processing Systems* 30 (2017).
7. Ester, M. et al. A density-based algorithm for discovering clusters in
   large spatial databases with noise. *Proceedings of the 2nd ACM SIGKDD*,
   226–231 (1996).
8. U.S. Energy Information Administration. Crude Oil Prices: West Texas
   Intermediate (WTI) - Cushing, Oklahoma [DCOILWTICO], retrieved from
   FRED, Federal Reserve Bank of St. Louis.
9. U.S. Energy Information Administration. Henry Hub Natural Gas Spot Price
   [DHHNGSP], retrieved from FRED, Federal Reserve Bank of St. Louis.

## Acknowledgements

The author thanks Andrew Howe for his published account of applied
anomaly-detection work at Wood Mackenzie's Innovation & Analytics Lab [2],
which motivated the framing of this study. The author also thanks the
ATLAS Collaboration at CERN, whose published methodology on
model-independent signal detection [1] provided the statistical framework
this work adapts. All code is available in the accompanying repository.
This work is independent and not affiliated with any institution.

---

**See also:** `section8_grid_extension.md` in this same folder — an
extension of this methodology to network-structured power grid data,
testing whether the model-independent principle generalises beyond
univariate commodity price series to genuinely multi-node sensor data.
