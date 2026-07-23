# Anomaly Detection Across Energy Data Regimes: A Four-Dataset Benchmark
### From Particle-Physics Model-Independence to Prices, Grids, and Real National Demand

**Sulman Younas** — independent researcher; Physics PhD (CERN ATLAS)

## Abstract

Model-independent anomaly detection — training on "normal" data and
flagging deviations without assuming a signal model — originates in
particle physics searches. We benchmark five detectors (LSTM
autoencoder, Transformer encoder, Isolation Forest, DBSCAN, rolling
z-score) under an identical pipeline across four energy datasets
spanning three data regimes: real commodity prices (WTI, Henry Hub;
daily), a peer-reviewed synthetic model of the continental-European
transmission grid with real-event ground truth verified empirically
before use (hourly), and real national demand for Great Britain (NESO,
half-hourly) and Texas (ERCOT/EIA-930, hourly). Ten independently
documented crisis events — collapses, spikes, and one supply-constrained
hybrid — serve as ground truth. No method is universally robust: the
method ranking inverts twice across datasets, and we diagnose each
inversion mechanistically rather than reporting scores alone.
Detectability is governed by (i) event duration relative to sampling
resolution — sustained anomalies transition from sparse noise (where
density methods excel, F1 up to 0.695) to dense clusters (where they
fail, F1 0.068, with no tuning configuration recovering performance)
while sequence models show the opposite profile; (ii) seasonal amplitude
structure, which in AC-dominated Texas renders +40% winter spikes
invisible in annually-normalised features (roll-mean z=-0.74) beneath
ordinary summers (z=+2.69); and (iii) label completeness — in two of
four studies, models concentrated "false positives" on real unlabelled
events, including 94% of one model's flag budget on ERCOT's record 2023
summer. Forward-chronological splits silenced a crisis event in three of
four studies, motivating a disclosed clean-window protocol. All
verification and diagnostic code is released; two attractive hypotheses
tested along the way were rejected and are reported as such.

## Assembly map (repository sections → unified paper)

| Paper section | Source file |
|---|---|
| 1–7. Study 1: WTI / Henry Hub | `paper/draft_paper.md` |
| 8. Study 2: synthetic European grid | `paper/section8_grid_extension.md` |
| 9. Study 3: real GB demand (NESO) | `paper/section9_real_gb_demand.md` |
| 10. Study 4: real US demand (ERCOT) | `paper/section10_ercot_us_demand.md` |
| 11. Synthesis: what governs detectability | `paper/section11_four_dataset_synthesis.md` |
| Methodology & caveats appendix | `docs/grid_network_methodology_and_caveats.md` |

## Relationship to Gillioz, Dubuis & Jacquod (Sci Data 12, 168, 2025)

Study 2 uses their dataset (DOI 10.1038/s41597-025-04479-x; Zenodo
10.5281/zenodo.13378476), whose paper states it "will serve as a basis
for time series anomaly detection in transmission grids in a future work
of ours". This work is an independent realisation of that use case, and
additionally: verifies real-event ground truth empirically inside the
synthetic data before relying on it; traces two apparent anomalies of
the dataset (bilateral-flow direction, flow-side event attenuation) to
the paper's own documented annual-vs-hourly border-flow constraint; and
extends the evaluation to real GB and US demand where the synthetic
model's conclusions partially fail to transfer. Their figures validate a
generator against reality; ours evaluate detectors against verified
events — complementary halves of the same programme.

## Venue note

NeurIPS 2026 main-track submission closed May 2026. Targets, in order:
arXiv preprint (priority; the dataset authors' own anomaly-detection
follow-up is unpublished as of this writing — verify before submission),
NeurIPS 2026 workshop (time-series / ML-for-physical-systems /
climate-energy; deadlines ~Aug–Oct 2026), then NeurIPS 2027 or an
applied venue (e.g. Applied Energy) for an extended version.
