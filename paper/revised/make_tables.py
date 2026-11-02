"""Generate every table and every quoted number of the revised paper from CSVs.
Run from repo root:  python paper/revised/make_tables.py
Writes paper/revised/tables/*.tex and paper/revised/numbers.json (facts quoted in the text)."""
import json, os
import numpy as np, pandas as pd

R = "results"; OUT = "paper/revised"; T = f"{OUT}/tables"
DUR = ["2h", "12h", "1d", "3d", "7d", "14d"]
NAMES = {"isolation_forest": "IForest", "dbscan_published": "DBSCAN (as published)",
         "knn_inductive": "kNN-to-train", "z_published": "$|$return$|$ top-5\\%",
         "z_rolling": "Rolling $z>3$", "pca_window_recon": "PCA window recon."}
facts = {}

# ---------------------------------------------------------------- datasets / events
EV = [  # series, resolution, samples per day, period, real?, [(name, start, end)]
 ("WTI spot (FRED)", "daily", None, "2018--2023", "real", [("Texas Freeze", "2021-02-10", "2021-02-25"), ("Russia--Ukraine", "2022-02-20", "2022-06-30")]),
 ("Henry Hub (FRED)", "daily", None, "2019--2023", "real", [("Texas Freeze", "2021-02-10", "2021-02-25"), ("Russia--Ukraine", "2022-02-20", "2022-06-30")]),
 ("FR load (Gillioz)", "hourly", 24, "2016--2020", "synthetic", [("Nuclear crisis", "2016-10-10", "2017-01-27"), ("COVID-19", "2020-03-16", "2020-05-11")]),
 ("FR line flow (Gillioz)", "hourly", 24, "2016--2018", "synthetic", [("Nuclear crisis", "2016-10-10", "2017-01-27")]),
 ("GB demand (NESO)", "half-hourly", 48, "2016--2023", "real", [("COVID-19", "2020-03-23", "2020-05-31"), ("Cold snap", "2022-12-08", "2022-12-17")]),
 ("ERCOT demand (EIA-930)", "hourly", 24, "2019--2023", "real", [("Texas Freeze", "2021-02-14", "2021-02-19"), ("Elliott", "2022-12-22", "2022-12-26")]),
]
rows = []
for s, res, spd, per, kind, evs in EV:
    for n, a, b in evs:
        days = (pd.Timestamp(b) - pd.Timestamp(a)).days + 1
        samp = f"{days * spd:,}" if spd else f"$\\approx${int(round(days * 5 / 7))}\\,(business days)"
        rows.append(f"{s} & {res} & {kind} & {per} & {n} & {a} -- {b} & {days} & {samp} \\\\")
        facts[f"{s}|{n}|days"] = days
        if spd: facts[f"{s}|{n}|samples"] = days * spd
open(f"{T}/datasets.tex", "w").write(
 "\\begin{tabular}{llllllrr}\n\\toprule\nSeries & Resolution & Data & Period & Event & Dates (inclusive) & Days & Native samples \\\\\n\\midrule\n"
 + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- real-data tables (consistent metric)
def load_real():
    out = []
    s1 = pd.read_csv(f"{R}/study1_wti_henryhub/strict_same_day_results.csv").rename(columns={"Commodity": "Dataset"})
    s1["Protocol"] = "forward"; out.append(s1)
    s2 = pd.read_csv(f"{R}/study2_grid_network/phase3_final_results.csv")
    s3 = pd.read_csv(f"{R}/study3_gb_neso/neso_phase3_results.csv")
    s4 = pd.read_csv(f"{R}/study4_ercot_eia/ercot_phase3_results.csv")
    for d in (s2, s3, s4):
        d["Protocol"] = d["Protocol"].str.extract(r"^([AB])")[0]
        out.append(d[["Dataset", "Protocol", "Method", "Precision", "Recall", "F1"]])
    df = pd.concat(out, ignore_index=True)
    df["Dataset"] = df["Dataset"].replace({"GB Demand (NESO real)": "GB demand", "ERCOT Demand (EIA real)": "ERCOT demand"})
    return df
real = load_real()
M = ["LSTM Autoencoder", "Transformer Encoder", "Isolation Forest", "DBSCAN", "Rolling Z-score"]
DISP = {"LSTM Autoencoder": "LSTM-AE", "Transformer Encoder": "Transformer", "Isolation Forest": "IForest", "DBSCAN": "DBSCAN", "Rolling Z-score": "Rolling $z$"}
order = [("WTI", "forward"), ("Henry Hub", "forward"), ("FR Load", "A"), ("FR Load", "B"), ("FR Line Flow", "B"),
         ("GB demand", "A"), ("GB demand", "B"), ("ERCOT demand", "A"), ("ERCOT demand", "B")]
def cell(df, ds, pr, m, col):
    x = df[(df.Dataset == ds) & (df.Protocol == pr) & (df.Method == m)][col]
    return float(x.iloc[0]) if len(x) else np.nan
def table(cols, fname, bold_best=False):
    L = []
    for ds, pr in order:
        vals = [cell(real, ds, pr, m, cols) for m in M]
        best = np.nanmax(vals)
        cs = [(f"\\textbf{{{v:.3f}}}" if (bold_best and abs(v - best) < 1e-9 and best > 0) else f"{v:.3f}") for v in vals]
        L.append(f"{ds} & {pr} & " + " & ".join(cs) + " \\\\")
    open(f"{T}/{fname}.tex", "w").write("\\begin{tabular}{llccccc}\n\\toprule\nData & Split & LSTM & Trf. & IF & DBSCAN & $z$ \\\\\n\\midrule\n" + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}\n")
table("F1", "real_f1", bold_best=True); table("Precision", "real_precision"); table("Recall", "real_recall")
# how the headline changes vs the submitted (mixed-metric) table
sub = pd.read_csv(f"{R}/study1_wti_henryhub/final_corrected_results.csv")
facts["wti_dbscan_submitted_hybrid_f1"] = float(sub[(sub.Commodity == "WTI") & (sub.Method == "DBSCAN")].F1_corrected.iloc[0])
facts["wti_dbscan_consistent_f1"] = cell(real, "WTI", "forward", "DBSCAN", "F1")
facts["wti_lstm_consistent_f1"] = cell(real, "WTI", "forward", "LSTM Autoencoder", "F1")
facts["gb_B_lstm_f1"] = cell(real, "GB demand", "B", "LSTM Autoencoder", "F1")
facts["dbscan_f1_by_dataset"] = {f"{d}|{p}": cell(real, d, p, "DBSCAN", "F1") for d, p in order}

# per-event coverage (GB, ERCOT, Protocol B) -- pre-fix per_event_recall
g = pd.read_csv(f"{R}/study3_gb_neso/neso_phase3_results.csv"); e = pd.read_csv(f"{R}/study4_ercot_eia/ercot_phase3_results.csv")
g = g[g.Protocol.str.startswith("B")]; e = e[e.Protocol.str.startswith("B")]
L = []
for m in M:
    a = g[g.Method == m].iloc[0]; b = e[e.Method == m].iloc[0]
    L.append(f"{DISP[m]} & {a.Recall_COVID:.2f} & {a.Recall_ColdSnap:.2f} & {b.Recall_TexasFreeze:.2f} & {b.Recall_Elliott:.2f} \\\\")
open(f"{T}/coverage.tex", "w").write("\\begin{tabular}{lcccc}\n\\toprule\n & \\multicolumn{2}{c}{GB (half-hourly)} & \\multicolumn{2}{c}{ERCOT (hourly)} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nDetector & COVID-19 & Cold snap & Texas Freeze & Elliott \\\\\n\\midrule\n" + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- synthetic sweep
for tag in ("shift", "burst"):
    raw = pd.read_csv(f"{R}/synthetic/sweep_{tag}_mag3.csv"); S = pd.read_csv(f"{R}/synthetic/summary_{tag}.csv")
    db = raw[raw.detector == "dbscan_published"]
    facts[f"{tag}|prevalence_range"] = [float(raw.prevalence.min()), float(raw.prevalence.max())]
    facts[f"{tag}|dbscan_alert_rate_cell_range"] = [float(S[S.detector == 'dbscan_published'].alert_rate_mean.min()), float(S[S.detector == 'dbscan_published'].alert_rate_mean.max())]
    facts[f"{tag}|corr_dbscan_f1_alert_rate"] = float(db.ts_f1.corr(db.alert_rate))
    p = S[S.detector == "dbscan_published"].pivot(index="duration", columns="resolution", values="ts_f1_mean")
    facts[f"{tag}|dbscan_f1_30min_vs_1h_max_abs_diff"] = float((p["30min"] - p["1h"]).abs().max())
    facts[f"{tag}|dbscan_f1_range"] = [float(S[S.detector == 'dbscan_published'].ts_f1_mean.min()), float(S[S.detector == 'dbscan_published'].ts_f1_mean.max())]
    zr = S[(S.detector == "z_rolling")]
    facts[f"{tag}|z_rolling_alert_rate_max"] = float(zr.alert_rate_mean.max())
    # best detector by AUC-PR at 30 min, plus the AUC-PR itself
    s = S[S.resolution == "30min"]
    L = []
    for d in DUR:
        x = s[s.duration == d].sort_values("auc_pr_mean", ascending=False)
        b = x.iloc[0]; b2 = x.iloc[1]
        tie = (b.auc_pr_mean - b.auc_pr_ci95) <= (b2.auc_pr_mean + b2.auc_pr_ci95)
        dag = "$^\\dagger$" if tie else ""
        facts[f"{tag}|tie|{d}"] = bool(tie)
        L.append(f"{d}{dag} & {NAMES[b.detector]} & {b.auc_pr_mean:.2f}$\\pm${b.auc_pr_ci95:.2f} & {NAMES[b2.detector]} & {b2.auc_pr_mean:.2f}$\\pm${b2.auc_pr_ci95:.2f} & {x[x.detector=='dbscan_published'].auc_pr_mean.iloc[0]:.2f} & {x[x.detector=='dbscan_published'].alert_rate_mean.iloc[0]*100:.0f}\\% \\\\")
    open(f"{T}/sweep_best_{tag}.tex", "w").write("\\begin{tabular}{llcllcc}\n\\toprule\nDuration & Best (AUC-PR) & AUC-PR & 2nd best & AUC-PR & DBSCAN AUC-PR & DBSCAN alert rate \\\\\n\\midrule\n" + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}\n")
    facts[f"{tag}|best_by_duration_30min"] = {d: s[s.duration == d].sort_values("auc_pr_mean", ascending=False).iloc[0].detector for d in DUR}
    # threshold-free vs thresholded disagreement at 1d resolution
    zz = S[(S.detector == "z_rolling") & (S.resolution == "1d")][["duration", "ts_f1_mean", "auc_pr_mean"]]
    facts[f"{tag}|z_rolling_1d_res"] = zz.round(3).to_dict("records")
    facts[f"{tag}|n_runs"] = int(raw.shape[0]); facts[f"{tag}|n_seeds"] = int(raw.seed.nunique())

# ---------------------------------------------------------------- real-data equal-budget tables (stronger baselines, intervals)
import sys
sys.path.insert(0, ".")
from ebench.report import coverage_table
from ebench.factorial import ss_decomposition
from ebench.crossseries import stratified_partial_spearman
NEW = f"{R}/real_data/2026-11-02_stronger_neural"
DET = [("matrix_profile", "Matrix profile"), ("seasonal_naive", "Seasonal-naive resid."), ("lstm_ae", "LSTM-AE"), ("transformer", "Transformer"),
       ("window_pca", "Window PCA"), ("isolation_forest", "Isolation Forest"), ("window_iforest", "Window IForest"),
       ("knn_inductive", "kNN-to-train"), ("window_knn", "Window kNN"), ("diff_z", "$|\\Delta x|$ one-liner"),
       ("rolling_z", "Rolling $z$"), ("spectral_residual", "Spectral residual")]
def iv(p, lo, hi): return f"{p:.3f} [{lo:.3f}, {hi:.3f}]"
def real_table(ds, title_ev):
    fr = {sfx: pd.read_csv(f"{NEW}/{ds}_equal_budget{sfx}_final.csv") for sfx in ("", "_resid")}
    ci = {sfx: pd.read_csv(f"{NEW}/{ds}_equal_budget{sfx}_ci.csv").set_index("detector") for sfx in ("", "_resid")}
    cm = pd.read_csv(f"{NEW}/{ds}_equal_budget_ci_metrics.csv"); cm = cm[cm.metric == "auc_pr"].set_index("detector")
    prev = float(fr[""].prevalence.iloc[0])
    L = []
    for fam, nm in DET:
        cells = []
        for sfx in ("", "_resid"):
            f = fr[sfx]; d = f[(f.family == fam) & (f.config == "default")]
            if d.empty:
                cells += ["--", "--"]; continue
            ap = d.auc_pr.mean(); k = f"{fam}|default"
            if sfx == "" and k in cm.index:
                a = iv(cm.loc[k, "point"], cm.loc[k, "lo"], cm.loc[k, "hi"])
            else:
                a = f"{ap:.3f}"
            f1 = iv(ci[sfx].loc[k, "ts_f1"], ci[sfx].loc[k, "ts_f1_lo"], ci[sfx].loc[k, "ts_f1_hi"]) if k in ci[sfx].index else f"{d.ts_f1.mean():.3f}"
            cells += [a, f1 + ("$^\\ast$" if k in ci[sfx].index and bool(ci[sfx].loc[k, "reliably_worse_than_best"]) else "")]
            facts[f"{ds}|{fam}|{'resid' if sfx else 'recipe'}|auc_pr"] = float(ap); facts[f"{ds}|{fam}|{'resid' if sfx else 'recipe'}|ts_f1"] = float(d.ts_f1.mean())
        L.append(f"{nm} & " + " & ".join(cells) + " \\\\")
    facts[f"{ds}|prevalence"] = prev
    head = ("\\begin{tabular}{lcccc}\n\\toprule\n & \\multicolumn{2}{c}{4-feature recipe} & \\multicolumn{2}{c}{recipe + seasonal residual} \\\\\n"
            "Detector & AUC-PR [95\\%] & F1 [95\\%] & AUC-PR & F1 [95\\%] \\\\\n\\midrule\n")
    open(f"{T}/real_{ds}.tex", "w").write(head + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}\n")
real_table("gb", ""); real_table("ercot", "")
# per-event coverage (residual feature set), default configs
for ds in ("gb", "ercot"):
    f = pd.read_csv(f"{NEW}/{ds}_equal_budget_resid_final.csv"); f = f[f.config == "default"]
    ct = coverage_table(f).set_index("family"); cov = [c for c in ct.columns if c.startswith("cov_")]
    L = []
    for fam, nm in DET:
        if fam in ct.index:
            L.append(f"{nm} & " + " & ".join(f"{ct.loc[fam, c]:.2f}" for c in cov) + f" & {int(ct.loc[fam, 'events_hit'])}/{int(ct.loc[fam, 'n_events'])} \\\\")
            for c in cov: facts[f"{ds}|{fam}|{c}"] = float(ct.loc[fam, c])
    names = " & ".join(c[4:].replace("Texas Freeze", "Texas Freeze").replace("_", " ") for c in cov)
    open(f"{T}/coverage_{ds}.tex", "w").write("\\begin{tabular}{l" + "c" * (len(cov) + 1) + "}\n\\toprule\nDetector & " + names + " & Events hit \\\\\n\\midrule\n" + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- factorial attribution (step 9)
G = f"{R}/factorial/grid.csv"
if os.path.exists(G):
    g = pd.read_csv(G)
    FACT = ["background", "event", "dur_h", "mag", "res_h", "features", "detector"]
    ssd = ss_decomposition(g, FACT, "skill", max_order=2)
    nice = {"detector": "Detector (main effect)", "event": "Event type", "dur_h x detector": "Duration $\\times$ detector", "event x detector": "Event type $\\times$ detector",
            "event x dur_h": "Event type $\\times$ duration", "mag": "Magnitude", "mag x detector": "Magnitude $\\times$ detector", "res_h x detector": "Resolution $\\times$ detector",
            "features x detector": "Feature set $\\times$ detector", "background x detector": "Background $\\times$ detector"}
    L = []
    for k, v in nice.items():
        sh = float(ssd.loc[ssd.term == k, "share"].iloc[0]) * 100; facts[f"factorial|{k}"] = sh
        L.append(f"{v} & {sh:.1f} \\\\")
    rs = float(ssd.loc[ssd.term.str.startswith("residual"), "share"].iloc[0]) * 100; facts["factorial|residual"] = rs
    L.append(f"Remainder (higher-order terms, replicate noise) & {rs:.1f} \\\\")
    det_all = float(ssd.loc[ssd.term.str.contains("detector"), "share"].sum()) * 100; facts["factorial|detector_all"] = det_all
    facts["factorial|n_datasets"] = int(g[["background", "event", "dur_h", "mag", "res_h", "rep"]].drop_duplicates().shape[0])
    open(f"{T}/factorial.tex", "w").write("\\begin{tabular}{lr}\n\\toprule\nSource & \\% of variance in skill \\\\\n\\midrule\n" + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}\n")
    wins = g.groupby(["background", "event", "dur_h", "mag", "res_h", "features", "detector"]).skill.mean().unstack("detector")
    facts["factorial|global_best"] = wins.mean().idxmax(); facts["factorial|global_best_wins_share"] = float((wins.idxmax(axis=1) == wins.mean().idxmax()).mean())
    ev_win = wins.idxmax(axis=1).reset_index(name="w").groupby("event").w.value_counts(normalize=True)
    for ev in ("shift", "ramp", "burst"):
        facts[f"factorial|win_share|{ev}|seasonal_naive"] = float(ev_win.get((ev, "seasonal_naive"), 0)); facts[f"factorial|win_share|{ev}|diff_z"] = float(ev_win.get((ev, "diff_z"), 0))

# ---------------------------------------------------------------- public benchmark (step 8)
P = f"{R}/public/tsbad_u.csv"
if os.path.exists(P):
    raw = pd.read_csv(P); ok = raw[raw.status == "ok"].copy()
    ok["log_len"] = np.log(ok.median_len); ok["log_prev"] = np.log(ok.prevalence); ok["d"] = ok.auc_pr_window_pca - ok.auc_pr_knn_inductive
    r_, lo_, hi_, per = stratified_partial_spearman(ok, "log_len", "d", "log_prev", n_boot=500, seed=0)
    facts.update({"public|n_series": int(len(ok)), "public|n_datasets": int(ok.dataset.nunique()), "public|n_datasets_min8": int(len(per)),
                  "public|rho": r_, "public|lo": lo_, "public|hi": hi_, "public|pca_beats_knn": float((ok.d > 0).mean()), "public|median_d": float(ok.d.median()),
                  "public|n_excluded": int(len(raw) - len(ok))})
    sens = []
    for nm, sub in (("All series (primary)", ok), ("Clean training prefix only", ok[ok.train_contaminated == 0]), ("Event length $>3$ samples", ok[ok.median_len > 3]),
                    ("Period found only", ok[ok.period_found == 1])):
        a_, l_, h_, p_ = stratified_partial_spearman(sub, "log_len", "d", "log_prev", n_boot=500, seed=0)
        sens.append(f"{nm} & {len(sub)} & {len(p_)} & {a_:.3f} & [{l_:.3f}, {h_:.3f}] \\\\")
    open(f"{T}/public_sens.tex", "w").write("\\begin{tabular}{lrrrc}\n\\toprule\nSubset & Series & Datasets & $\\rho$ & 95\\% interval \\\\\n\\midrule\n" + "\n".join(sens) + "\n\\bottomrule\n\\end{tabular}\n")

json.dump(facts, open(f"{OUT}/numbers.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in facts.items() if "|" not in k or "corr" in k or "range" in k or "diff" in k}, indent=1, default=float)[:3500])
