# 10. Study 4: Real US Demand — ERCOT (EIA-930)

## 10.1 Data and ground truth

Hourly ERCOT (Texas) demand from the EIA-930 Hourly Grid Monitor,
2019–2023 (43,824 rows — five complete years, zero gaps), using the
adjusted demand series, stream-filtered from the all-BA six-month
balance files. Two ground-truth events, both verified year-over-year
before use:

- **Texas Freeze / Winter Storm Uri** (14–19 Feb 2021): pre-shed cold
  ramp +42.1% vs early-February baseline; the shed period itself
  (15–18 Feb) still shows **+30.9%** vs the same dates in 2019/20 —
  despite ERCOT forcibly shedding up to ~20 GW. Recorded demand
  therefore *understates* true demand during the event; this is
  documented as a labelling caveat, not "corrected". Notably this is the
  same physical event Study 1 detected in Henry Hub gas prices — a
  cross-data-type closure between commodity and grid views of one
  crisis.
- **Winter Storm Elliott** (22–26 Dec 2022): **+49.4%** vs same dates
  2019/21, with no comparable load shed.

Unlike Studies 2–3, both events are strong; unlike GB, both are spikes,
and both are *short* (5 days ≈ 120–145 hourly points).

## 10.2 Protocols

Protocol A (forward 70/30) again buries the earlier event (the Freeze)
in training — the third study out of four in which a naive forward
split silences a crisis event. Protocol B trains on 2021-06-01 to
2022-06-01, after the Freeze and before Elliott. Real data offers no
pristine window: this one still contains ERCOT's June 2021 conservation
appeal and ordinary summer peaks, which is disclosed rather than hidden.
The July 2022 record heat (10–25 Jul) serves as the unlabelled-event
diagnostic, analogous to Beast from the East in Study 3.

## 10.3 Results

| Protocol | Method | P | R | F1 | Recall Freeze | Recall Elliott |
|---|---|---|---|---|---|---|
| A | LSTM Autoencoder | 0.000 | 0.000 | 0.000 | n/a | 0.00 |
| A | Transformer Encoder | 0.000 | 0.000 | 0.000 | n/a | 0.00 |
| A | Isolation Forest | 0.000 | 0.000 | 0.000 | n/a | 0.00 |
| A | **DBSCAN** | 0.058 | 0.281 | **0.097** | n/a | **0.34** |
| A | Rolling Z-score | 0.006 | 0.033 | 0.010 | n/a | 0.03 |
| B | LSTM Autoencoder | 0.000 | 0.000 | 0.000 | 0.00 | 0.00 |
| B | Transformer Encoder | 0.000 | 0.000 | 0.000 | 0.00 | 0.00 |
| B | Isolation Forest | 0.000 | 0.000 | 0.000 | 0.00 | 0.00 |
| B | **DBSCAN** | 0.029 | 0.135 | **0.047** | 0.07 | 0.26 |
| B | Rolling Z-score | 0.001 | 0.008 | 0.002 | 0.01 | 0.00 |

The ranking inverts a second time: DBSCAN — broken on GB — is here the
*only* method that detects anything, and its F1 exceeds its GB result
despite far rarer events. This is the out-of-sample behaviour predicted
by the Study 3 mechanism: 5-day events at hourly resolution remain
sparse enough to register as noise, where a 10-week event at half-hourly
resolution formed its own dense cluster. Additionally, Isolation
Forest's false positives under Protocol A are ~5×-enriched in the
July 2022 record heat (14.3% vs a ~2.7% window base rate) — the first
clear instance of a model concentrating "errors" on a real unlabelled
event (the equivalent GB check found no enrichment).

## 10.4 Diagnosis of the neural networks' zero scores

The LSTM's 0.000 (after leading GB at 0.282) was diagnosed with a
dedicated script; the answer is a ground-truth artefact plus a
feature-space mechanism, both verified:

1. **94% of LSTM flags land on the record summer of 2023.** Of 1,751
   flags, 1,639 fall in Jul–Sep 2023 — ERCOT's all-time-record heat and
   demand peak — and essentially none in 2019 (secular-drift hypothesis
   tested and rejected). Trained on Jun 2021–Jun 2022, the model ranked
   the genuinely largest anomaly in five years of Texas data above our
   two labelled winter events. The 0.000 F1 measures disagreement with
   our label choices at least as much as detection failure.
2. **Seasonal amplitude asymmetry makes winter spikes structurally
   invisible in annually-normalised features.** ERCOT is
   air-conditioning-dominated: the Freeze's 30-day rolling-mean z-score
   is **-0.74** — *negative*, because a +40% winter spike still sits
   below the summer-set annual mean — while the July 2022 heat sits at
   z=+2.69. Elliott's rows are *closer* to training density than normal
   rows are (median 5-NN distance 0.225 vs 0.543), and the LSTM ranked
   them at median percentile 4.7 — not near-missed at the 95th-percentile
   threshold but buried mid-distribution. Feature-window dilution
   contributes as designed-for arithmetic (a 5-day event moves a 30-day
   mean by ~5/30 of its amplitude), but the seasonal asymmetry — absent
   in commodity prices and in low-AC Britain — is the dominant effect.

The honest summary: on ERCOT, the deep models did not fail to find
anomalies; they found the biggest one, which our labels did not name.
This sharpens a general point about benchmark construction: with
percentile-budgeted flagging, an evaluation is only as meaningful as the
completeness of its event labels.
