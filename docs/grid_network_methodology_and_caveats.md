# Grid Anomaly Detection Extension — Methodology and Documented Caveats

## Data provenance

Source: Gillioz, M., Dubuis, G. & Jacquod, P. "A large synthetic dataset
for machine learning applications in power transmission grids,"
*Scientific Data* 12, 168 (2025), DOI: 10.1038/s41597-025-04479-x;
data at Zenodo, DOI: 10.5281/zenodo.13378476. Notably, the paper states
the dataset "will serve as a basis for time series anomaly detection in
transmission grids in a future work of ours" — the present study is an
independent realisation of exactly that use case, with real-event ground
truth verified empirically before use. Network topology and generation
pipeline: GeeeHesso/PowerData (github.com/GeeeHesso/PowerData), built on
PowerModels.jl (Julia) for optimal power flow.

**This is explicitly a synthetic dataset**, not real historical
measurements — generated via optimal power flow simulation on the real
PanTaGruEl topology of the continental European transmission grid, with
generation calibrated against real historical ENTSO-E data. This is a
fundamentally different, more rigorous kind of "synthetic" than an ad-hoc
statistical simulation: it is peer-reviewed, built on real network
topology, and — critically — we independently verified that at least one
genuine historical event (the COVID-19 demand collapse) is reproduced with
the correct sign and a magnitude close to real-world reporting (see below).

## What was verified, and how

**Date alignment.** Each yearly file is a 364-day period starting on a
Monday; the exact calendar start date per year was not stated in the
dataset's documentation and was independently confirmed by checking that
25 December 2016 fell on the day of the week the candidate alignment
implied ("Sunday" — correct) and showed the expected demand dip (-12.6%,
consistent with a major holiday landing on a weekend), plus a
full-year weekday/weekend load ratio of 1.128 in the expected direction.

**Year-boundary continuity.** Concatenating separate yearly files risks
gaps or overlaps at the joins. Checked explicitly for all four boundaries
across 2016-2020: three are perfectly continuous; **one genuine 7-day gap
exists between 2018-12-30 23:00 and 2019-01-07 00:00**, a structural
artifact of fitting 364-day files to a real calendar. Neither ground-truth
event used in this work falls near this gap.

**COVID-19 event, load data.** Directly tested: mean national load in
France fell 20.7% during the documented lockdown window (16 Mar - 11 May
2020) versus the immediately preceding period, matching the widely
reported real-world range of roughly 15-20% for French/European demand
during this period. This is strong evidence the dataset preserves genuine
historical demand anomalies, not just aggregate seasonal statistics.

**French nuclear crisis event, line-flow data.** France's 2016-17 nuclear
outage crisis (20 of 58 reactors offline, ~110 days as a net importer,
documented ~7.2 GW average export reduction) is fundamentally an
export/generation-balance story, not a demand story — confirmed empirically
by finding no visible signal in load data, as expected, and a real but
smaller-than-reported signal in the interconnector line-flow data (see
Known Limitations below).

## Interconnector identification methodology

France's 62 individual cross-border transmission lines to its 6
continental neighbours (DE, CH, IT, ES, BE, LU) were identified
programmatically from the network topology file (`europe_network.json`),
by matching each branch's `f_bus`/`t_bus` country against a bus→country
mapping built directly from the topology's own per-bus `country` field —
not inferred or hand-curated.

These 62 lines were found to represent only **28 distinct physical
crossing points**, with the remainder being parallel circuits at the same
two substations (e.g. 12 parallel circuits at the main France-Italy
crossing). This was verified as a genuine physical property, not a data
artefact, by confirming that every group of "parallel" lines shares
identical electrical parameters (reactance, thermal rating) — as real
parallel circuits between the same two substations should.

Flow direction sign was standardised so that positive consistently means
"France exporting to the neighbour," following each line's documented
`f_bus`/`t_bus` orientation from the network topology (10 of 62 lines
required a sign flip, where France was the `t_bus` rather than `f_bus`).
This follows the standard PowerModels.jl convention (confirmed via the
field-naming pattern matching PowerModels.jl's documented schema), which
is the framework this dataset's generation pipeline is built on.

## Known Limitations (disclosed, not resolved)

**1. Bilateral trade balances at the country-pair level are emergent
model outputs, not historical quantities — now confirmed by the dataset's
own documentation.** Under the standard, documented sign convention,
France's average net export to Italy comes out slightly negative
(-0.48 GW, 2017 average) across our available data, which does not match
the well-documented real-world reality that France was a large,
structurally positive net exporter to Italy throughout this period
(Italy had zero domestic nuclear generation since 1990 and was a large
structural importer). We tested whether a global sign inversion would
resolve this and found it does produce more plausible headline numbers,
but the empirical evidence for a genuine global inversion is weak
(43.7%/56.3% positive/negative split on the France-Italy line — not the
strongly one-sided pattern a true convention error would produce), and
the documented PowerModels.jl convention supports our original,
non-inverted correction.

**This is now explained, not merely conjectured.** The dataset's
accompanying paper (Gillioz, Dubuis & Jacquod, *Sci Data* 12, 168, 2025,
DOI: 10.1038/s41597-025-04479-x) states in its Methods that border flows
are deliberately constrained only in aggregate: *"we choose to only
constrain the annual total rather than the hour-by-hour border flow in
order to leave enough room for the dispatch procedure."* Hourly and
bilateral cross-border flows are therefore emergent outputs of the
dispatch optimisation (which minimises line loading plus a randomised
generation-cost term), not reproductions of historical flows. A specific
country-pair average such as France-Italy is consequently an artefact of
the cost model, with only each country's *annual national* import/export
total anchored to reality (their Table 2). Our observed discrepancy is
exactly what this construction predicts — a documented property of the
dataset, not a processing error in our pipeline.

**Practical consequence:** absolute bilateral flow magnitudes/averages
should not be over-interpreted. Relative deviations (anomaly detection,
the actual task here) remain valid provided the sign convention is at
least internally consistent, which was separately verified (all parallel
circuits within a crossing point agree in sign; no mixed-sign groups
found).

**2. The nuclear crisis signal is directionally correct but smaller than
the historically reported magnitude — and the same annual-vs-hourly
constraint explains why.** Same-season comparison (Oct-Jan 2016/17 vs.
Oct-Jan 2017/18, controlling for winter seasonality) found the crisis
period showed *less* export reduction than the following year, the
opposite of the expected direction under a simple two-period comparison —
though the broader multi-window analysis did show a real, correctly
directioned dip against a summer baseline (-0.70 GW vs. the ~-7.2 GW
reported historically). This inconsistency across comparison windows is
noted honestly rather than cherry-picking the comparison that best
matches the expected story.

The mechanism is the same documented design choice as in Limitation 1:
the crisis enters the dataset through two channels only — historical
nuclear generation profiles (used directly as OPF inputs) and the
*annual* national export balance — while hour-by-hour flows are free
dispatch outputs. A concentrated 110-day export collapse is therefore
structurally smeared across the annual constraint by the optimiser. This
also explains the asymmetry between our two ground-truth events: the
COVID-19 demand collapse is a *load-side* event, and national load
series are built directly from real ENTSO-E load statistics, so it
survives generation faithfully (we measured -20.7% vs. the real ~15-20%);
the nuclear crisis is a *generation/flow-side* event, and flows are
where the generator deliberately departs from history. Load-side events
in this dataset constitute strong ground truth; flow-side events are
structurally attenuated and should be treated as weaker, noisier ground
truth — a principled distinction, not an empirical guess.

**Notable positive finding despite this:** the LSTM Autoencoder and
Transformer Encoder, trained completely independently on this line-flow
data, converged on the same anomalous hours during the test period to a
striking degree — raw anomaly score correlation of 0.9985, and 99.5%
overlap (1046/1051) in which specific hours were flagged at the 95th
percentile threshold. This was verified directly (not assumed) after an
initial run showed the two models' precision/recall/F1 matching to three
decimal places, which prompted a dedicated check to rule out a processing
bug before trusting the result. Two architecturally different models
independently agreeing this strongly on which hours are anomalous is
meaningful evidence that the nuclear crisis produces a genuine, dominant,
architecture-independent deviation in the line-flow features — a stronger
form of validation than either model's individual performance metric
would suggest alone.

**3. Only 2016-2018 line-flow data has been processed.** Full 2016-2020
coverage (matching the load data) has not yet been downloaded, given each
year is ~2GB and the marginal value for the nuclear crisis event
specifically (which concludes in January 2017) is limited. This could be
extended if a broader line-flow analysis becomes valuable.

**4. Two evaluation protocols were used for load data, for different
reasons, and results should be compared with this in mind.** Protocol A
(forward-chronological 70/30 split) makes the COVID-19 event reachable but
not the nuclear crisis. Protocol B (clean training window with evaluation
on both sides) makes both events reachable but is a deliberate departure
from a pure forecasting-style split, justified by the "train on normal,
flag deviations wherever they occur" framing central to this work's
methodology (rather than a strict forecasting task). Both protocols'
results are reported side by side rather than selecting only the more
favourable one.

## What this extension demonstrates, for the purposes of further work

- The core "model-independent anomaly detection" methodology, previously
  validated on univariate commodity price series (WTI, Henry Hub),
  generalises to genuinely network-structured, multi-node sensor data —
  28 distinct physical grid locations, not an aggregate proxy.
- A real, independently-documented historical event (COVID-19) is
  detected with the correct magnitude in load data, providing a validated
  positive control.
- A second, structurally different event (nuclear generation crisis) is
  detected with the correct direction but attenuated magnitude in
  line-flow data, illustrating a genuine, honestly-reported limitation
  rather than an artificially clean result — and, on the same event, two
  independently-trained neural architectures converge strongly (99.5%
  agreement) on which hours are anomalous, a meaningful robustness signal.
- Two evaluation protocols were compared directly, in the same spirit as
  the earlier strict-vs-tolerance metric comparison on WTI/Henry Hub —
  methodological choices are shown to matter, and are disclosed rather
  than hidden behind a single reported number.
- DBSCAN (with a properly tuned neighbourhood radius, not a guessed one)
  was the strongest or most consistent method across the first three
  datasets (WTI, Henry Hub, this grid load data). **Superseded:**
  Studies 3-4 (real GB and US demand) show this pattern does not
  generalise — the ranking inverts twice, for diagnosed mechanistic
  reasons (event duration × sampling resolution; seasonal amplitude
  asymmetry). See `paper/section11_four_dataset_synthesis.md` for the
  corrected finding, which is stronger than the original claim.
