# 11. Four-Dataset Synthesis: What Governs Detectability

## 11.1 Correction of an earlier claim

Interim versions of this work reported DBSCAN as "the most robust method
across three independent datasets" (WTI, Henry Hub, FR synthetic load).
Studies 3–4 falsify that claim as a generalisation: DBSCAN falls to
fourth on real GB demand (F1 0.068 vs LSTM's 0.282) and, in the opposite
direction, is the only scoring method on ERCOT while the deep models
score zero. The corrected finding is stronger than the original: **no
method in this comparison is universally robust, and the ranking
inversions are systematic and mechanistically explainable, not noise.**

## 11.2 The four-dataset picture

| Study | Data (resolution) | Real? | Strong event | Best method (F1) | DBSCAN F1 |
|---|---|---|---|---|---|
| 1 | WTI crude (daily) | real | Russia-Ukraine 2022 | DBSCAN (0.695) | 0.695 |
| 1 | Henry Hub gas (daily) | real | Texas Freeze 2021 | DBSCAN (0.343) | 0.343 |
| 2 | FR load (hourly) | synthetic, validated | COVID-19 | DBSCAN (0.222 / 0.196) | 0.222 / 0.196 |
| 2 | FR line flow (hourly) | synthetic, validated | nuclear crisis 2016-17 | LSTM ≡ Transformer (0.408) | 0.148 |
| 3 | GB demand (half-hourly) | real | COVID-19 | LSTM (0.282) | 0.068 |
| 4 | ERCOT demand (hourly) | real | Freeze / Elliott | DBSCAN (0.097 / 0.047) | 0.097 / 0.047 |

## 11.3 Three governing variables, each verified by diagnostics

**(i) Event duration × sampling resolution decides what a "sustained
anomaly" looks like to density methods.** At daily resolution a
multi-day crisis is a handful of isolated points — sparse noise, the
exact object DBSCAN detects (Study 1). At half-hourly resolution a
10-week crisis is thousands of mutually-similar points — a dense
cluster, which DBSCAN by construction does *not* flag (Study 3
diagnostic: COVID rows 100% beyond eps of training data, yet only 9%
flagged; 436 clusters in the test set; no point on a 15-configuration
tuning grid exceeds F1 0.097). Hourly ERCOT with 5-day events sits
in between and behaves in between, as the mechanism predicts
out-of-sample (Study 4: DBSCAN the only scorer, Elliott recall 0.34).
Symmetrically, sequence models see what density methods cannot — a
level shift carried in slowly-drifting rolling means while pointwise
returns stay normal — which is why the LSTM leads exactly where DBSCAN
breaks (GB), and why both deep architectures independently converge on
the same anomalous hours where the event signal is dominant (Study 2
line flows: score correlation 0.9985, 99.5% flag overlap, verified
against a processing-bug explanation).

**(ii) Seasonal amplitude structure can hide entire event classes.** In
AC-dominated ERCOT, a +40% winter spike still sits below the annual mean
(Freeze roll-mean z = -0.74) while an ordinary Texas summer sits far
above it (Jul-22 heat z = +2.69): in annually-normalised feature space,
no winter event can outrank summer. This effect is absent in commodity
prices and in low-AC Britain, which is precisely why it first appears in
Study 4. Detectability is a property of the joint event-climate-feature
system, not of the event's newsworthiness.

**(iii) Label completeness bounds what "accuracy" means.** Twice, models
concentrated their false positives on real events we chose not to
label: Isolation Forest ~5×-enriched on ERCOT's July 2022 record heat,
and the LSTM spending 94% of its ERCOT flag budget on the
record-breaking summer of 2023 — arguably the correct answer to the
question "what is most anomalous in this data". The corresponding GB
check (Beast from the East) showed *no* enrichment, demonstrating the
diagnostic can return either verdict. With fixed-percentile flag
budgets, headline F1 against curated labels conflates "missed the event"
with "found a bigger unlabelled one"; per-event recall and
unlabelled-window diagnostics should be reported alongside.

## 11.4 Replicated methodological findings

- **Forward-chronological splits silenced a crisis event in three of
  four studies** (Study 2: nuclear crisis; Study 3: COVID; Study 4:
  Freeze), and in Studies 3–4 additionally contaminated "normal"
  training data. The clean-window protocol (B) is a disclosed departure
  from forecasting-style evaluation that this event class simply
  requires.
- **Real data is nonstationary in ways validated synthetic data was
  not** (GB normal test rows at 5× the training set's internal 5-NN
  distance), inflating false-positive counts for every method and
  arguing for periodic retraining in any deployment.
- **Every surprising number in all four studies was diagnosed before
  being reported** — the two ranking inversions, the deep-model tie, the
  zero scores, and the two label-artefact findings each have a dedicated
  verification script in `code/`, and two initially-attractive
  hypotheses (a global sign inversion in Study 2; Beast-from-the-East
  detection in Study 3) were tested and rejected rather than assumed
  either way.

## 11.5 Practical guidance

For practitioners monitoring energy systems: match the method to the
event class and data regime. Density-based methods for short, sharp
events at coarse resolution; sequence reconstruction models for
sustained regime shifts at fine resolution; deseasonalised or
seasonally-conditioned features wherever seasonal amplitude dominates
event amplitude (ERCOT-like systems); and label audits plus
unlabelled-window diagnostics before trusting any headline metric. No
single detector should be deployed alone across an energy portfolio
spanning prices, demand, and flows.
