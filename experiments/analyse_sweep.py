"""Summarise the synthetic sweep: mean and 95% t-interval over seeds, plus three figures.
Usage: python experiments/analyse_sweep.py results/synthetic/sweep_shift_mag3.csv --tag shift
"""
import argparse, os
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DUR = ["2h", "12h", "1d", "3d", "7d", "14d"]
RES = ["30min", "1h", "1d"]
DETS = ["isolation_forest", "dbscan_published", "knn_inductive", "z_published", "z_rolling", "pca_window_recon"]
LABEL = {"isolation_forest": "Isolation Forest", "dbscan_published": "DBSCAN (as published)",
         "knn_inductive": "kNN-to-train (inductive)", "z_published": "|return| top-5% (as published 'z')",
         "z_rolling": "Rolling z > 3 (as paper text)", "pca_window_recon": "PCA window reconstruction"}
COL = dict(zip(DETS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
MARK = dict(zip(DETS, ["o", "s", "^", "D", "v", "P"]))
LS = dict(zip(DETS, ["-", "--", "-.", ":", "-", "--"]))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) < 2:
        return np.nan
    return stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


def summarise(df):
    g = df.groupby(["resolution", "duration", "detector"])
    cols = ["ts_f1", "auc_pr", "auc_pr_lift", "alert_rate", "ev_f1", "ev_recall", "ev_precision", "f1_ceiling", "prevalence"]
    out = g[cols].agg(["mean", ci]).round(4)
    out.columns = [f"{a}_{'mean' if b=='mean' else 'ci95'}" for a, b in out.columns]
    out["n_seeds"] = g.size()
    return out.reset_index()


def panel_fig(S, metric, ylabel, path, hline=None, hline_label=None, ylim=None):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    x = np.arange(len(DUR))
    for ax, r in zip(axes, RES):
        ax.set_facecolor("#fcfcfb")
        for d in DETS:
            s = S[(S.resolution == r) & (S.detector == d)].set_index("duration").reindex(DUR)
            m, c = s[f"{metric}_mean"].to_numpy(), s[f"{metric}_ci95"].to_numpy()
            ax.plot(x, m, color=COL[d], ls=LS[d], marker=MARK[d], ms=4.5, lw=1.5, label=LABEL[d])
            ax.fill_between(x, m - np.nan_to_num(c), m + np.nan_to_num(c), color=COL[d], alpha=0.12, lw=0)
        if hline is not None:
            ax.axhline(hline, color=MUTED, lw=0.9, ls=(0, (2, 3)))
        ax.set_xticks(x); ax.set_xticklabels(DUR, fontsize=8)
        ax.set_title(f"sampled at {r}", fontsize=9, color=INK)
        ax.set_xlabel("event duration (fixed prevalence)", fontsize=8, color=MUTED)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=8, colors=MUTED)
        if ylim: ax.set_ylim(*ylim)
        sub = S[(S.resolution == r)]
        too_short = [d for d in DUR if sub[sub.duration == d].empty]
        for d in too_short:
            ax.text(DUR.index(d), 0.02 if ylim else 0, "n/a", ha="center", fontsize=7, color=MUTED)
    axes[0].set_ylabel(ylabel, fontsize=9, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.12))
    if hline_label:
        axes[0].text(0.02, hline, hline_label, fontsize=7, color=MUTED, va="bottom", transform=axes[0].get_yaxis_transform())
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight"); fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", default="results/synthetic")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    S = summarise(df)
    S.to_csv(os.path.join(a.outdir, f"summary_{a.tag}.csv"), index=False)
    pi = df.prevalence.mean()
    panel_fig(S, "auc_pr", "AUC-PR (threshold-free)", os.path.join(a.outdir, f"fig_{a.tag}_aucpr.png"),
              hline=pi, hline_label=f"chance = prevalence ({pi:.3f})")
    panel_fig(S, "ts_f1", "timestamp-level F1 (as published rule)", os.path.join(a.outdir, f"fig_{a.tag}_f1.png"),
              hline=0.75, hline_label="F1 ceiling at a 5% budget", ylim=(0, 1))
    panel_fig(S, "alert_rate", "realised alert rate", os.path.join(a.outdir, f"fig_{a.tag}_alertrate.png"),
              hline=0.05, hline_label="5% budget")
    print(S[S.detector == "dbscan_published"][["resolution", "duration", "alert_rate_mean", "ts_f1_mean", "auc_pr_mean", "n_seeds"]].to_string())
