## 8. Extension: Network-Structured Grid Data

### 8.1 Motivation

Sections 1-7 validate the model-independent anomaly detection principle
on univariate commodity price series (WTI crude oil, Henry Hub natural
gas). A natural question is whether the same principle generalises to
genuinely network-structured, multi-node data — the setting most relevant
to real-time infrastructure monitoring (e.g. Wood Mackenzie's Synoptic
platform, which flags anomalous load-shifting behaviour from a real-time
sensor network; SSEN Transmission's distributed acoustic sensing and
insulator-monitoring projects). This extension tests exactly that
question, using a large, peer-reviewed, real-topology synthetic dataset
of the continental European power transmission grid.

### 8.2 Data

Gillioz et al. (2024), "A large synthetic dataset for machine learning
applications in power transmission grids," *Nature Scientific Data*,
DOI: 10.5281/zenodo.13378476. Built on the real PanTaGruEl topology of
the continental European grid (4,097 buses, 8,375 lines, 815 generators,
25 countries), with synthetic generation calibrated against real
historical ENTSO-E data and validated peer-review. Code and network
files: github.com/GeeeHesso/PowerData.

We use two data types from this source: (a) national aggregate load
(demand) per country, and (b) individual transmission line flows,
programmatically restricted to France's 62 cross-border interconnector
lines to its six continental neighbours, identified directly from the
network topology and aggregated into 28 distinct physical crossing
points (accounting for parallel circuits — verified via matching
electrical parameters, not assumed).

### 8.3 Verification before modelling

Following the same discipline as Sections 2-5, several assumptions were
tested empirically before being trusted:

- **Date alignment**: each yearly file's exact calendar start date was
  not documented; verified via an independently-known fact (25 December
  2016 fell on a Sunday) showing the expected day-of-week and a real
  holiday demand dip (-12.6%).
- **Year-boundary continuity**: checked all four year-joins explicitly;
  found three continuous and one genuine 7-day structural gap
  (2018-12-30 to 2019-01-07), unrelated to either ground-truth event used.
- **Event realism**: rather than assume documented real-world events are
  reproduced in this synthetic data, we tested directly. The COVID-19
  demand collapse (16 Mar - 11 May 2020) shows a -20.7% load reduction in
  French national demand, matching real-world reporting (~15-20%) closely
  — strong evidence the generator preserves genuine historical anomalies,
  not just aggregate statistics.
- **Sign convention**: cross-border line flow direction was standardised
  programmatically from the network topology's own bus-country and
  branch f_bus/t_bus fields, following the documented PowerModels.jl
  convention this dataset's generation pipeline is built on.

### 8.4 Two evaluation protocols, compared directly

A standard forward-chronological 70/30 split (Protocol A) made only one
of two candidate ground-truth events reachable in the test set (the
French nuclear generation crisis, Oct 2016 - Jan 2017, fell entirely in
training data). We introduce a second protocol (Protocol B): train on a
clean window containing neither event, evaluate on everything outside it.
This is a deliberate, disclosed departure from a pure forecasting split,
justified by this work's "train on normal, flag deviations wherever they
occur" framing rather than next-step prediction. Both protocols' results
are reported, not just the more favourable one.

### 8.5 Results

| Dataset | Protocol | Method | Precision | Recall | F1 |
|---|---|---|---|---|---|
| FR Load | A (forward) | LSTM Autoencoder | 0.092 | 0.045 | 0.060 |
| FR Load | A (forward) | Transformer Encoder | 0.141 | 0.068 | 0.092 |
| FR Load | A (forward) | Isolation Forest | 0.044 | 0.022 | 0.029 |
| FR Load | A (forward) | **DBSCAN** | 0.203 | 0.244 | **0.222** |
| FR Load | A (forward) | Rolling Z-score | 0.084 | 0.041 | 0.055 |
| FR Load | B (clean window) | LSTM Autoencoder | 0.202 | 0.067 | 0.101 |
| FR Load | B (clean window) | Transformer Encoder | 0.102 | 0.034 | 0.051 |
| FR Load | B (clean window) | Isolation Forest | 0.155 | 0.051 | 0.077 |
| FR Load | B (clean window) | **DBSCAN** | 0.332 | 0.139 | **0.196** |
| FR Load | B (clean window) | Rolling Z-score | 0.150 | 0.050 | 0.075 |
| FR Line Flow | B (clean window) | LSTM Autoencoder | 0.713 | 0.286 | 0.408 |
| FR Line Flow | B (clean window) | Transformer Encoder | 0.713 | 0.286 | 0.408 |
| FR Line Flow | B (clean window) | Isolation Forest | 0.179 | 0.072 | 0.103 |
| FR Line Flow | B (clean window) | DBSCAN | 0.246 | 0.106 | 0.148 |
| FR Line Flow | B (clean window) | Rolling Z-score | 0.141 | 0.057 | 0.081 |

**DBSCAN is the strongest or most consistent method on load data under
both protocols** — directly consistent with, not merely similar to, the
finding on WTI and Henry Hub in Sections 3-4. At the time of this
study this appeared to be a robust three-dataset pattern; Sections 9-11
show it does NOT extend to real GB or US demand, and diagnose why — see
Section 11 for the corrected, stronger claim.

**LSTM and Transformer on line-flow data produced precision, recall and
F1 identical to three decimal places.** This was not accepted at face
value: a dedicated diagnostic (comparing raw scores and exact flagged
timestamps between the two independently-trained models) found raw score
correlation of 0.9985 and 99.5% overlap (1046/1051) in flagged hours —
genuine, strong convergence between architecturally different models on
which specific hours are anomalous, not a processing artefact. This is
notable evidence that the nuclear crisis produces a dominant,
architecture-independent signal in the line-flow features.

### 8.6 Honest limitations specific to this extension

Overall F1 magnitudes are lower here than on WTI/Henry Hub, plausibly
because ground-truth labels here are long windows (weeks to months)
rather than sharp discrete event days, which a percentile-threshold
flagging approach is structurally less suited to capturing fully. A
same-season comparison for the nuclear crisis event (controlling for
winter seasonality) produced an inconsistent result relative to a
summer-baseline comparison — this event should be treated as a weaker,
noisier ground-truth signal than COVID-19, not an equally strong one.
Bilateral (country-pair) flow averages should not be over-interpreted:
France's average net export to Italy across available data does not
match the well-documented real-world direction, most likely reflecting a
genuine limitation of the synthetic model's fidelity at the specific
country-pair level rather than a processing error (tested and
documented in full in `docs/grid_network_methodology_and_caveats.md`).

Full methodology, every verification step, and complete reasoning behind
each limitation are documented in
`docs/grid_network_methodology_and_caveats.md` — included in full rather
than summarised, in the same spirit of transparency as Section 5.
