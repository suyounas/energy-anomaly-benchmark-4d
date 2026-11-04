"""Regenerate every figure of the paper from stored result files.

    python3 paper/revised/make_figures.py        (run from the repository root)

Inputs : results/synthetic/summary_{shift,burst}.csv, results/label_sens/*_candidate_days.csv
Outputs: paper/revised/figures/*.pdf
The year-on-year changes in fig_yoy are the values recorded in paper/data_records.md (they are not recomputed here).
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/revised/figures"
OUT.mkdir(exist_ok=True, parents=True)
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})

DUR = ["2h", "12h", "1d", "3d", "7d", "14d"]
RES = [("30min", "30-min sampling"), ("1h", "1-hour sampling"), ("1d", "daily sampling")]
DET = [("z_rolling", "Rolling z > 3", "#C2185B", "v", "-"),
       ("isolation_forest", "Isolation Forest", "#1F5FBF", "o", "-"),
       ("knn_inductive", "kNN to training rows", "#138A63", "^", "-."),
       ("pca_window_recon", "Window PCA", "#2E7D32", "P", "--"),
       ("z_published", "|return| top-5%", "#E69500", "D", ":"),
       ("dbscan_published", "DBSCAN (as published)", "#D8572A", "s", "--"),
       ("lstm_ae", "LSTM-AE", "#6B3FA0", "X", "-."),
       ("transformer", "Transformer", "#7A4A1D", "*", ":")]


def sweep_panel(df, col, ci, ylabel, axes, ref=None, ref_label=None, only=None, ylim=(-0.02, 1.02)):
    for ax, (res, title) in zip(axes, RES):
        d = df[df.resolution == res]
        for key, lab, c, m, ls in DET:
            if only and key not in only:
                continue
            if key not in set(d.detector):
                continue
            s = d[d.detector == key].set_index("duration").reindex(DUR)
            x = np.arange(len(DUR)); y = s[col].to_numpy(); e = s[ci].to_numpy()
            ax.plot(x, y, color=c, marker=m, ls=ls, lw=1.5, ms=4.5, label=lab)
            ax.fill_between(x, y - e, y + e, color=c, alpha=0.13, lw=0)
        if ref is not None:
            ax.axhline(ref, color="0.35", ls=(0, (2, 2)), lw=1)
        ax.set_xticks(range(len(DUR))); ax.set_xticklabels(DUR)
        ax.set_title(title); ax.set_ylim(*ylim); ax.grid(axis="y", color="0.9", lw=0.7)
        ax.set_xlabel("event duration")
        missing = d.duration.unique()
        gone = [i for i, du in enumerate(DUR) if du not in missing]
        if gone:
            ax.axvspan(-0.4, max(gone) + 0.4, color="0.94", lw=0)
            ax.text(np.mean([-0.4, max(gone) + 0.4]), ylim[1] * 0.5, "event <\n1 sample",
                    ha="center", va="center", fontsize=7.5, color="0.4")
    axes[0].set_ylabel(ylabel)


def fig_aucpr():
    for kind, lab in (("shift", "level shifts"), ("burst", "variance bursts")):
        df = pd.read_csv(ROOT / f"results/synthetic/summary_{kind}.csv")
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
        sweep_panel(df, "auc_pr_mean", "auc_pr_ci95", "AUC-PR", axes, ref=0.03)
        axes[0].text(5.2, 0.05, "chance (3%)", ha="right", va="bottom", fontsize=7.5, color="0.35")
        fig.suptitle(f"AUC-PR against event duration, {lab} (prevalence fixed at 3%)", y=1.0, fontsize=9.5, fontweight="bold")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.09), columnspacing=1.2, handletextpad=0.4)
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        fig.savefig(OUT / f"fig_{kind}_aucpr.pdf", bbox_inches="tight"); plt.close(fig)


def fig_alert_f1():
    df = pd.read_csv(ROOT / "results/synthetic/summary_shift.csv")
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True)
    sweep_panel(df, "alert_rate_mean", "alert_rate_ci95", "share of points flagged", axes[0], ref=0.05, ylim=(0, 0.3))
    sweep_panel(df, "ts_f1_mean", "ts_f1_ci95", "timestamp-level F1", axes[1], ref=0.75, ylim=(-0.02, 1.0))
    for ax in axes[0]:
        ax.set_xlabel("")
    axes[0, 0].text(5.2, 0.057, "5% budget", ha="right", va="bottom", fontsize=7.5, color="0.35")
    axes[1, 0].text(5.2, 0.77, "F1 ceiling at 5% budget (0.75)", ha="right", va="bottom", fontsize=7.5, color="0.35")
    h, l = axes[1, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.05), columnspacing=1.2, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "fig_shift_alertf1.pdf", bbox_inches="tight"); plt.close(fig)


def box(ax, x, y, w, h, text, fc="#E8EEF8", ec="#1F5FBF", fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.25)


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 2.35)); ax.set_xlim(0, 12.4); ax.set_ylim(0, 4); ax.axis("off")
    w, h, y = 1.75, 1.35, 1.35
    steps = [("1  Series\n+ dated events\n(fixed before\nmodelling)", "#F3F3F3", "0.4"),
             ("2  Resolution\nand features\n(return, rolling,\nseasonal residual)", "#E8EEF8", "#1F5FBF"),
             ("3  Split\nforward 70/30 or\nclean window;\nscalers on train", "#E8EEF8", "#1F5FBF"),
             ("4  Fit and score\n11 detectors,\nequal tuning\nbudget", "#E8EEF8", "#1F5FBF"),
             ("5  Alerts\ntop 5% of test\nscores (alert\nbudget)", "#E8EEF8", "#1F5FBF"),
             ("6  Metrics\nevent- and\ntimestamp-level,\nAUC-PR, VUS-PR", "#E6F3EC", "#138A63")]
    xs = [0.05 + i * 2.06 for i in range(6)]
    for x, (t, fc, ec) in zip(xs, steps):
        box(ax, x, y, w, h, t, fc, ec, fs=6.6)
    for a, b in zip(xs[:-1], xs[1:]):
        ax.add_patch(FancyArrowPatch((a + w + 0.04, y + h / 2), (b - 0.04, y + h / 2), arrowstyle="-|>", mutation_scale=9, color="0.25", lw=1))
    box(ax, 2.1, 0.0, 3.9, 0.95, "Label-validity checks (Sec. 4)\nresidual inside events, unlabelled days,\nranking under label perturbations", "#FFF4E0", "#E69500", fs=6.8)
    box(ax, 6.4, 0.0, 3.9, 0.95, "Uncertainty (Sec. 5.3)\npaired 7-day block bootstrap,\nper-event coverage, reliably-worse flag", "#FFF4E0", "#E69500", fs=6.8)
    box(ax, 0.05, 3.15, 12.0, 0.62, "Controlled studies reuse steps 2-6:  duration sweep (Sec. 6)  ·  648-dataset factorial (Sec. 7)  ·  GB and ERCOT (Sec. 8)  ·  862 public series (Sec. 9)", "#F7F7F7", "0.6", fs=7.2)
    fig.savefig(OUT / "fig_pipeline.pdf", bbox_inches="tight"); plt.close(fig)


EVENTS = {"gb": [("2020-03-23", "2020-05-31", "COVID-19\nlockdown"), ("2022-12-08", "2022-12-17", "cold\nsnap")],
          "ercot": [("2021-02-14", "2021-02-19", "Texas\nFreeze"), ("2022-12-22", "2022-12-26", "Winter Storm\nElliott")]}


def fig_label_gap():
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.9))
    for ax, key, name, cut in ((axes[0], "gb", "GB demand", 1.26), (axes[1], "ercot", "ERCOT demand", 2.46)):
        d = pd.read_csv(ROOT / f"results/label_sens/{key}_candidate_days.csv", parse_dates=["day"])
        ax.vlines(d.day, 0, d.daily_dev, color="#7A7A7A", lw=0.9, label="unlabelled day at or above the weakest event's level")
        ax.axhline(cut, color="0.35", ls=(0, (2, 2)), lw=1)
        for s, e, lab in EVENTS[key]:
            ax.axvspan(pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta(days=1), color="#D8572A", alpha=0.28, lw=0, label="labelled event" if lab.startswith(("COVID", "Texas")) else None)
            ax.text(pd.Timestamp(s) + (pd.Timestamp(e) - pd.Timestamp(s)) / 2, ax.get_ylim()[1], "", ha="center")
        ax.set_ylabel("daily mean |z|"); ax.set_title(f"{name}: {len(d)} unlabelled candidate days", loc="left")
        ymax = d.daily_dev.max() * 1.18
        ax.set_ylim(0, ymax)
        for s, e, lab in EVENTS[key]:
            ax.text(pd.Timestamp(s) + (pd.Timestamp(e) - pd.Timestamp(s)) / 2, ymax * 0.97, lab, ha="center", va="top", fontsize=7.5, color="#8A2E10")
        ax.grid(axis="y", color="0.92", lw=0.7)
    axes[1].annotate("11–13 Feb 2021: three unlabelled days\nimmediately before the labelled start", xy=(pd.Timestamp("2021-02-12"), 4.1),
                     xytext=(pd.Timestamp("2019-03-01"), 4.4), fontsize=7.5, arrowprops=dict(arrowstyle="-", color="0.4", lw=0.8))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "fig_label_gap.pdf", bbox_inches="tight"); plt.close(fig)


def fig_yoy():
    ev = [("GB lockdown\n(2020)", -19.1), ("GB cold snap\n(Dec 2022)", 6.4), ("ERCOT Freeze\n(Feb 2021)", 30.9), ("ERCOT Elliott\n(Dec 2022)", 49.4)]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    v = [x[1] for x in ev]
    bars = ax.bar(range(4), v, color=["#1F5FBF" if t < 0 else "#D8572A" for t in v], width=0.62)
    for b, t in zip(bars, v):
        ax.text(b.get_x() + b.get_width() / 2, t + (2 if t >= 0 else -2), f"{t:+.1f}%", ha="center", va="bottom" if t >= 0 else "top", fontsize=8)
    ax.axhline(0, color="0.2", lw=0.8)
    ax.set_xticks(range(4)); ax.set_xticklabels([x[0] for x in ev], fontsize=7)
    ax.set_ylabel("demand vs same dates\nin other years (%)"); ax.set_ylim(-30, 60)
    ax.grid(axis="y", color="0.92", lw=0.7)
    fig.tight_layout(); fig.savefig(OUT / "fig_yoy.pdf", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig_aucpr(); fig_alert_f1(); fig_pipeline(); fig_label_gap(); fig_yoy()
    print("figures written to", OUT)
