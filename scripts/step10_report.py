#!/usr/bin/env python3
"""Step 10 report: LaTeX tables and a detector-by-view rank figure from the saved analysis outputs."""
import argparse, glob, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {"isolation_forest": "Isolation forest", "knn_inductive": "kNN (inductive)", "rolling_z": "Rolling z", "lstm_ae": "LSTM autoencoder",
         "transformer": "Transformer", "diff_z": "Differenced z", "seasonal_naive": "Seasonal naive", "window_pca": "Window PCA",
         "window_knn": "Window kNN", "window_iforest": "Window iForest", "matrix_profile": "Matrix profile", "spectral_residual": "Spectral residual"}
VIEWS = ["pooled", "class_A", "class_B", "family_heat", "family_cold", "family_tropical", "duration_short", "duration_medium", "duration_long"]

def tex_escape(s): return str(s).replace("_", r"\_")

def load_views(d, label):
    out = {}
    for f in glob.glob(f"{d}/{label}__*__ranking.csv"):
        out[os.path.basename(f).split("__")[1]] = pd.read_csv(f).set_index("detector")
    return out

def pooled_table(r, caption, label):
    r = r.sort_values("mean_rank")
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{lrrr}", r"\toprule",
             r"Detector & Mean lift [95\% CI] & Mean rank [95\% CI] & Share of events won [95\% CI] \\", r"\midrule"]
    for d, x in r.iterrows():
        lines.append(f"{NAMES.get(d, tex_escape(d))} & {x.mean_metric:.2f} [{x.mean_metric_lo:.2f}, {x.mean_metric_hi:.2f}] & "
                     f"{x.mean_rank:.2f} [{x.mean_rank_lo:.2f}, {x.mean_rank_hi:.2f}] & {100*x.share_won:.1f}\\% [{100*x.share_won_lo:.1f}, {100*x.share_won_hi:.1f}] \\\\")
    n, g = int(r.n_events.iloc[0]), int(r.n_groups.iloc[0])
    lines += [r"\bottomrule", r"\end{tabular}", f"\\caption{{{caption} ({n} events, {g} storm groups; intervals from a cluster bootstrap over storm groups).}}", f"\\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)

def rank_figure(views, path, title, order=None):
    cols = [v for v in VIEWS if v in views]
    M = pd.DataFrame({v: views[v].mean_rank for v in cols}); N = {v: int(views[v].n_groups.iloc[0]) for v in cols}
    M = M.loc[[d for d in order if d in M.index]] if order else (M.loc[M["pooled"].sort_values().index] if "pooled" in M else M)
    fig, ax = plt.subplots(figsize=(1.0 * len(cols) + 3, 0.42 * len(M) + 1.6))
    im = ax.imshow(M.values, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([f"{c.replace('_', ' ')}{'†' if N[c] < 30 else ''}\n({N[c]} groups)" for c in cols], fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(len(M))); ax.set_yticklabels([NAMES.get(d, d) for d in M.index], fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M.values[i, j]:.1f}", ha="center", va="center", fontsize=8, color="white" if M.values[i, j] > M.values.mean() else "black")
    for j, c in enumerate(cols):
        if N[c] < 30: pass
    grp = [c.split("_")[0] for c in cols]
    for j in range(1, len(cols)):
        if grp[j] != grp[j - 1]: ax.axvline(j - .5, color="white", lw=4)
    ax.set_title(title, fontsize=10); fig.colorbar(im, ax=ax, label="mean rank (1 = best)", shrink=0.8)
    fig.tight_layout(); fig.savefig(path + ".pdf"); fig.savefig(path + ".png", dpi=200); plt.close(fig)

def simple_table(df, cols, heads, caption, label, fmt):
    lines = [r"\begin{table}[t]", r"\centering", r"\small", "\\begin{tabular}{" + "l" + "r" * (len(cols) - 1) + "}", r"\toprule", " & ".join(heads) + r" \\", r"\midrule"]
    for _, x in df.iterrows():
        lines.append(" & ".join(fmt[c](x[c]) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="results/step10/analysis_classical_prelim_lift"); ap.add_argument("--label", default="classical_all_bas")
    ap.add_argument("--validated", default="results/step10/analysis_classical_validated_noSN")
    ap.add_argument("--validity", default="results/step10/label_validity/summary.csv")
    ap.add_argument("--perturb", default="results/step10/perturb_vus/perturbation_summary.csv")
    ap.add_argument("--out", default="results/step10/report")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    V = load_views(a.analysis, a.label)
    open(f"{a.out}/table_pooled.tex", "w").write(pooled_table(V["pooled"], "Pooled detector comparison on all events", "tab:step10-pooled"))
    order = list(V["pooled"].sort_values("mean_rank").index)
    rank_figure(V, f"{a.out}/fig_rank_by_view", "Mean per-event rank by event view († fewer than 30 storm groups, descriptive only)", order)
    if os.path.isdir(a.validated):
        VV = load_views(a.validated, a.label)
        open(f"{a.out}/table_validated.tex", "w").write(pooled_table(VV["pooled"], "Pooled comparison on validated events only, seasonal-naive excluded", "tab:step10-validated"))
        rank_figure(VV, f"{a.out}/fig_rank_by_view_validated", "Validated events only, seasonal naive excluded († fewer than 30 storm groups, descriptive only)", order)
    if os.path.exists(a.validity):
        s = pd.read_csv(a.validity)
        open(f"{a.out}/table_label_validity.tex", "w").write(simple_table(s, ["view", "n_events", "n_storm_groups", "share_validated", "median_pct_rank", "median_ratio"],
            ["View", "Events", "Storm groups", "Validated", "Median pct. rank", "Median ratio"],
            "Label validity: share of events whose mean daily $|z|$ exceeds the 95th percentile of days outside events.", "tab:step10-validity",
            {"view": lambda v: tex_escape(v), "n_events": lambda v: f"{int(v)}", "n_storm_groups": lambda v: f"{int(v)}", "share_validated": lambda v: f"{100*v:.1f}\\%",
             "median_pct_rank": lambda v: f"{v:.2f}", "median_ratio": lambda v: f"{v:.2f}"}))
    if os.path.exists(a.perturb):
        p = pd.read_csv(a.perturb)
        open(f"{a.out}/table_perturbation.tex", "w").write(simple_table(p, ["perturbation", "spearman_vs_baseline", "kendall_tau", "best", "top4_same_set"],
            ["Perturbation", "Spearman", "Kendall $\\tau$", "Best detector", "Same top four"],
            "Label perturbations: agreement of the pooled ranking with the baseline ranking.", "tab:step10-perturb",
            {"perturbation": lambda v: tex_escape(v), "spearman_vs_baseline": lambda v: f"{v:.2f}", "kendall_tau": lambda v: "--" if pd.isna(v) else f"{v:.2f}",
             "best": lambda v: NAMES.get(v, tex_escape(v)), "top4_same_set": lambda v: "--" if pd.isna(v) else ("yes" if v else "no")}))
    print("wrote:", sorted(os.listdir(a.out)))

if __name__ == "__main__":
    main()
