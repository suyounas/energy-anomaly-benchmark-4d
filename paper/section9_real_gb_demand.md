# 9. Study 3: Real GB National Demand (NESO)

## 9.1 Motivation and data

Studies 1–2 covered real commodity prices and a validated synthetic grid.
Study 3 is the first fully-real grid dataset: Great Britain national
electricity demand from NESO (National Energy System Operator, formerly
National Grid ESO), half-hourly settlement-period resolution, 2016–2023
(140,240 rows). Files were discovered programmatically via NESO's CKAN
API rather than hardcoded URLs, and completeness was verified
arithmetically: the 16-row deficit against the naive 140,256 expectation
is exactly 8 years × 2 duplicate rows from October clock-change days
(50-period days), while the eight apparent "gaps" are the corresponding
March BST transitions (46-period days) — the data is complete; the
artefact is local-time settlement accounting, amounting to ~3 hours/year.

## 9.2 Ground truth, verified year-over-year

- **COVID-19 first UK lockdown** (23 Mar – 31 May 2020): mean demand
  22,922 MW vs 28,327 MW for the identical calendar window averaged over
  2017–2019 — a **-19.1%** year-over-year change, consistent with
  real-world GB reporting of roughly -10 to -20%. A naive
  before/after-lockdown comparison gives -27.8%, inflated by the normal
  winter→spring decline; the YoY figure is the defensible one.
- **December 2022 cold snap** (8–17 Dec, the first Demand Flexibility
  Service era): +18.3% vs the immediately preceding weeks, but only
  **+6.4%** YoY against 2018/19/21 same dates — partly because 2022's
  high prices suppressed baseline demand. This event is real but weak,
  and is treated as weak ground truth throughout.

As in Study 2, each study thus pairs one strong and one weak event.

## 9.3 Protocols

Protocol A (forward 70/30) places the COVID event entirely inside
training data — the same failure mode Study 2 exhibited for the nuclear
crisis, now replicated on real data — and additionally contaminates
"train on normal" with a major anomaly. Protocol B trains on a clean
window (2018-06-01 to 2020-02-01), chosen to end before COVID and to
*start after* the Feb/Mar 2018 "Beast from the East" cold event, a real
GB demand spike that would otherwise contaminate normal training. Beast
from the East consequently sits in the test period *unlabelled*, and is
used as a diagnostic: the share of each method's false positives falling
in that window measures whether "errors" are actually detections of a
real unlabelled event. (Result: no method showed enrichment above the
~1.2% base rate — the hypothesis was tested and rejected.)

## 9.4 Results

| Protocol | Method | P | R | F1 | Recall COVID | Recall cold snap |
|---|---|---|---|---|---|---|
| A | LSTM Autoencoder | 0.000 | 0.000 | 0.000 | n/a | 0.00 |
| A | Transformer Encoder | 0.000 | 0.000 | 0.000 | n/a | 0.00 |
| A | Isolation Forest | 0.000 | 0.000 | 0.000 | n/a | 0.00 |
| A | DBSCAN | 0.024 | 0.106 | 0.039 | n/a | 0.12 |
| A | Rolling Z-score | 0.012 | 0.052 | 0.019 | n/a | 0.06 |
| B | **LSTM Autoencoder** | 0.239 | 0.345 | **0.282** | **0.39** | 0.00 |
| B | Transformer Encoder | 0.120 | 0.174 | 0.142 | 0.19 | 0.00 |
| B | Isolation Forest | 0.094 | 0.135 | 0.111 | 0.15 | 0.00 |
| B | DBSCAN | 0.057 | 0.085 | 0.068 | 0.09 | 0.09 |
| B | Rolling Z-score | 0.023 | 0.033 | 0.027 | 0.03 | 0.05 |

Two headline observations. First, **the method ranking inverts relative
to Studies 1–2**: DBSCAN, best on three prior datasets (WTI 0.695,
Henry Hub 0.343, FR load 0.222/0.196), falls to fourth (0.068), while
the LSTM autoencoder leads at 0.282 — higher than any method achieved on
the synthetic FR load data. Second, the cold snap is essentially
undetected by every method (recalls at or near the ~5% chance level of
the flagging rate), a plainly-stated negative result on weak-event
detectability.

## 9.5 Diagnosis of the DBSCAN degradation

The degradation was diagnosed, not merely observed. Three findings:

1. **The eps tuning rule is exonerated.** The k-distance p90 rule gave
   eps=0.152 and a 5.1% flag rate against a 3.5% anomaly base rate — a
   sensible operating point. A 15-point sensitivity sweep over eps
   percentile × min_samples found a best F1 of only 0.097 anywhere on
   the grid (vs LSTM's 0.282): no tuning rescues the method, so the
   conclusion is robust to the operating point.
2. **The mechanism is duration × resolution.** COVID rows are
   unambiguous outliers versus training data (100% lie beyond eps of the
   training distribution; roll-skew z = -2.32, roll-mean z = -1.25), yet
   DBSCAN flagged only 9% of them — because DBSCAN labels *noise*, and
   3,311 mutually-similar half-hourly COVID rows form a large, dense
   cluster of their own (436 clusters were found in the test set
   overall). A sustained regime shift at half-hourly resolution is not
   sparse. The same event class at daily resolution (Study 1) is a
   handful of isolated points — exactly what noise-based detection is
   built for. Sampling resolution converts sustained anomalies from
   sparse noise into dense clusters, structurally inverting DBSCAN's
   suitability.
3. **Secondary effects.** The half-hourly `return` feature is 85%
   explained by the deterministic intraday cycle (R² of slot-of-day ×
   day-of-week group means = 0.853), so one of four feature dimensions
   mostly encodes time-of-day. And normal test rows sit at median 5-NN
   distance 0.44 from the training data versus the training set's
   internal 0.082 — genuine nonstationarity across the 8-year real
   series (secular demand decline, renewables growth, the 2022 price
   era) that the synthetic data largely lacked, inflating false-positive
   counts for all methods.

Conversely, the sequence models see precisely what pointwise density
cannot: a sustained level shift living in slowly-drifting rolling means
while individual half-hourly returns remain normal. The inversion is a
method-data-fit result, not a tuning accident.
