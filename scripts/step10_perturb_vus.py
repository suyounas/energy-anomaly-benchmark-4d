#!/usr/bin/env python3
"""Step 10 from saved scores: (1) Analysis C, label perturbations (boundary shifts of 1, 3, 7 days in both directions, dilation by
1 and 3 days) and leave-one-storm-group-out; (2) VUS-PR (secondary outcome). Reads <runs>/scores/<BA>_scores.npz and events_final.csv.
Scores of a detector are averaged over seeds before per-event AUC-PR (baseline is recomputed the same way, so comparisons are like with like).
Per-event lift = per-event AUC-PR / (event rows / (event rows + rows outside every window)). Ranking = mean per-event rank over detectors."""
import argparse, glob, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import average_precision_score

PERTS = {"baseline": (0, 0), "shift_-7": (-7, -7), "shift_-3": (-3, -3), "shift_-1": (-1, -1),
         "shift_+1": (1, 1), "shift_+3": (3, 3), "shift_+7": (7, 7), "dilate_1": (-1, 1), "dilate_3": (-3, 3)}
G = {}

def load_scores(path):
    z = np.load(path); v = z["t"].astype("int64"); mx = np.abs(v).max()    # t was saved as DatetimeIndex.asi8; the unit depends on the pandas version
    unit = "s" if mx < 4e10 else "ms" if mx < 4e13 else "us" if mx < 4e16 else "ns"
    t = pd.DatetimeIndex(pd.to_datetime(v, unit=unit)); by = {}
    for k in z.files:
        if k == "t": continue
        d, _ = k.rsplit("__", 1); by.setdefault(d, []).append(z[k].astype(float))
    return t, {d: np.mean(v, axis=0) for d, v in by.items()}

def masks(t, e, ds, de):
    from ebench import event_mask
    return [event_mask(t, r.start + pd.Timedelta(days=ds), r.end + pd.Timedelta(days=de)) for r in e.itertuples()]

def work(args):
    ba, runs, events = args
    t, S = load_scores(os.path.join(runs, "scores", f"{ba}_scores.npz"))
    e = events[events.ba == ba].reset_index(drop=True); rows = []
    for name, (ds, de) in PERTS.items():
        M = masks(t, e, ds, de); anyev = np.any(M, axis=0)
        for i, m in enumerate(M):
            if m.sum() < 2: continue
            sel = m | ~anyev; prev = m.sum() / sel.sum()
            for d, sc in S.items():
                ap = average_precision_score(m[sel].astype(int), sc[sel])
                rows.append(dict(ba=ba, event_id=e.event_id[i], storm_group=e.storm_group[i], pert=name, detector=d, lift=ap / prev))
    return pd.DataFrame(rows)

def work_vus(args):
    ba, runs, events = args
    from ebench.rangemetrics import vus_pr
    t, S = load_scores(os.path.join(runs, "scores", f"{ba}_scores.npz"))
    e = events[events.ba == ba]
    y = np.any(masks(t, e, 0, 0), axis=0).astype(int)
    seg = np.concatenate([[0], np.cumsum(np.diff(t.asi8) != 3600 * 10**9)])
    return pd.DataFrame([dict(ba=ba, detector=d, vus_pr=vus_pr(y, sc, segments=seg), prevalence=float(y.mean())) for d, sc in S.items()])

def mean_rank(df):
    W = df.pivot_table(index=["ba", "event_id", "storm_group"], columns="detector", values="lift").dropna()
    return W, W.rank(axis=1, ascending=False).mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="."); ap.add_argument("--runs", default="results/step10/runs_classical")
    ap.add_argument("--events-final", default="results/step10/runs_classical/events_final.csv")
    ap.add_argument("--out", default="results/step10/perturb_vus"); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--skip-vus", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, a.repo)
    ev = pd.read_csv(a.events_final, parse_dates=["start", "end"])
    bas = sorted(os.path.basename(f).split("_scores")[0] for f in glob.glob(os.path.join(a.runs, "scores", "*_scores.npz")))
    jobs = [(b, a.runs, ev) for b in bas]
    with Pool(a.workers) as p:
        P = pd.concat(p.map(work, jobs), ignore_index=True)
        P.to_csv(f"{a.out}/perturbed_event_lift.csv.gz", index=False)
        _, base = mean_rank(P[P.pert == "baseline"]); out = []; table = {"baseline": base}
        for name in PERTS:
            _, r = mean_rank(P[P.pert == name]); table[name] = r
            top = set(base.nsmallest(4).index) == set(r.nsmallest(4).index)
            out.append(dict(perturbation=name, n_events=len(_), spearman_vs_baseline=spearmanr(base, r.reindex(base.index)).statistic,
                            kendall_tau=kendalltau(base, r.reindex(base.index)).statistic, best=r.idxmin(), top4_same_set=top))
        W, base = mean_rank(P[P.pert == "baseline"]); R = W.rank(axis=1, ascending=False)
        gids = W.index.get_level_values("storm_group"); sp = []
        for g in pd.unique(gids):
            keep = gids != g; sp.append(spearmanr(base, R[keep].mean().reindex(base.index)).statistic)
        out.append(dict(perturbation="leave_one_storm_group_out", n_events=len(W), spearman_vs_baseline=float(np.mean(sp)),
                        kendall_tau=np.nan, best=base.idxmin(), top4_same_set=np.nan))
        pd.DataFrame(table).round(3).to_csv(f"{a.out}/perturbation_mean_ranks.csv")
        s = pd.DataFrame(out); s.to_csv(f"{a.out}/perturbation_summary.csv", index=False); print(s.round(3).to_string(index=False))
        print("min leave-one-group-out Spearman:", round(float(np.min(sp)), 4))
        if not a.skip_vus:
            V = pd.concat(p.map(work_vus, jobs), ignore_index=True); V.to_csv(f"{a.out}/vus_pr_by_ba.csv", index=False)
            V["rank"] = V.groupby("ba").vus_pr.rank(ascending=False)
            g = V.groupby("detector").agg(mean_vus_pr=("vus_pr", "mean"), mean_rank=("rank", "mean")).sort_values("mean_rank")
            g.to_csv(f"{a.out}/vus_pr_summary.csv"); print(g.round(4).to_string())

if __name__ == "__main__":
    main()
