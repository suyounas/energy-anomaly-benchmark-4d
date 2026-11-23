#!/usr/bin/env python3
"""Step 10 primary analysis (pre-registered): per-event ranks, cluster bootstrap over storm groups,
rank stability, conditional structure, source class. Reads <BA>_events.csv from the run folders and events_final.csv."""
import argparse, glob, json, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr

def load(run_dirs, events_final):
    parts = [pd.read_csv(f) for d in run_dirs for f in glob.glob(os.path.join(d, "*_events.csv")) if not f.endswith("events_final.csv")]
    r = pd.concat(parts, ignore_index=True)
    r = r.groupby(["ba", "event_id", "detector"], as_index=False)[["n_pos", "auc_pr_event", "coverage", "detected"]].mean()  # mean over seeds
    neg = {}
    for d in run_dirs:
        for f in glob.glob(os.path.join(d, "*_metrics.csv")):
            m = pd.read_csv(f).iloc[0]; neg[m["ba"]] = m["n_rows"] * (1 - m["prevalence"])
    r["n_neg"] = r.ba.map(neg)
    r["lift"] = r.auc_pr_event / (r.n_pos / (r.n_pos + r.n_neg))   # per-event AUC-PR over its own prevalence
    ev = pd.read_csv(events_final, parse_dates=["start", "end"])
    ev["duration_days"] = (ev.end - ev.start).dt.days + 1
    return r.merge(ev[["ba", "event_id", "storm_group", "source_class", "family", "duration_days"]], on=["ba", "event_id"], how="inner")

def wide(df, metric):
    return df.pivot_table(index=["ba", "event_id", "storm_group", "source_class", "family", "duration_days"],
                          columns="detector", values=metric).dropna().reset_index()

def stats(W, dets, groups, nboot, rng, refs):
    """W: events x detectors AUC matrix; groups: storm_group per event."""
    A = W[dets].to_numpy()
    R = pd.DataFrame(A).rank(axis=1, ascending=False).to_numpy()          # 1 = best
    top = (A == A.max(axis=1, keepdims=True)); Wn = top / top.sum(axis=1, keepdims=True)   # ties split
    g = pd.factorize(groups)[0]; G = g.max() + 1
    def agg(M): 
        out = np.zeros((G, M.shape[1])); np.add.at(out, g, M); return out
    sA, sR, sW = agg(A), agg(R), agg(Wn); n = np.bincount(g, minlength=G).astype(float)
    def summarise(idx):
        N = n[idx].sum(); return sA[idx].sum(0) / N, sR[idx].sum(0) / N, sW[idx].sum(0) / N
    full = np.arange(G); a, r, w = summarise(full)
    res = pd.DataFrame({"detector": dets, "mean_metric": a, "mean_rank": r, "share_won": w, "n_events": len(A), "n_groups": G})
    B = np.array([summarise(rng.integers(0, G, G)) for _ in range(nboot)])          # nboot x 3 x D
    for k, name in enumerate(["mean_metric", "mean_rank", "share_won"]):
        res[name + "_lo"], res[name + "_hi"] = np.percentile(B[:, k, :], [2.5, 97.5], axis=0)
    pair = []
    for ref in refs:
        if ref not in dets: continue
        j = dets.index(ref)
        for i, d in enumerate(dets):
            if i == j: continue
            diff = a[i] - a[j]; bd = B[:, 0, i] - B[:, 0, j]
            pair.append(dict(detector=d, reference=ref, diff_metric=diff, lo=np.percentile(bd, 2.5), hi=np.percentile(bd, 97.5)))
    # rank stability
    gm = np.where(n[:, None] > 0, sA / np.maximum(n[:, None], 1), 0)
    gr = pd.DataFrame(gm).rank(axis=1, ascending=False).to_numpy()
    m, k = gr.shape; S = ((gr.sum(0) - gr.sum(0).mean()) ** 2).sum()
    kendall_w = 12 * S / (m ** 2 * (k ** 3 - k)) if m > 1 else np.nan
    best = int(np.argmin(r)); frac_best = float((gr[:, best] == 1).mean())
    loo = []
    for gi in range(G):
        idx = np.delete(full, gi); loo.append(spearmanr(summarise(idx)[1], r).statistic)
    stab = dict(kendall_w=float(kendall_w), best_detector=dets[best], frac_groups_best_is_best=frac_best,
                loo_spearman_mean=float(np.nanmean(loo)), loo_spearman_min=float(np.nanmin(loo)))
    return res, pd.DataFrame(pair), stab

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["results/step10/runs_classical", "results/step10/runs_neural"])
    ap.add_argument("--events-final", default="results/step10/runs_classical/events_final.csv")
    ap.add_argument("--out", default="results/step10/analysis")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--metric", default="auc_pr_event", choices=["auc_pr_event", "lift"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-ba", nargs="*", default=[], help="robustness: e.g. ERCO")
    ap.add_argument("--min-groups", type=int, default=30, help="descriptive-only below this many storm groups")
    ap.add_argument("--exclude-detectors", nargs="*", default=[], help="drop these detectors before ranking")
    ap.add_argument("--only-validated", default=None, help="path to event_validity.csv; keep only events with validated == True")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed)
    df = load(a.runs, a.events_final); df = df[~df.ba.isin(a.exclude_ba)]
    df = df[~df.detector.isin(a.exclude_detectors)]
    if a.only_validated:
        v = pd.read_csv(a.only_validated); v = v[v.validated.astype(str).str.lower() == "true"][["ba", "event_id"]]
        df = df.merge(v, on=["ba", "event_id"], how="inner")
    neural = ["lstm_ae", "transformer"]
    nbas = sorted(df[df.detector.isin(neural)].ba.unique())
    classical = sorted(set(df.detector) - set(neural))
    sets = {"classical_all_bas": (df[df.detector.isin(classical)], classical),
            "all_detectors_neural_bas": (df[df.ba.isin(nbas)], sorted(set(df.detector)))}
    summary = {}
    for label, (d, dets) in sets.items():
        W = wide(d, a.metric)
        if W.empty: continue
        groups_n = W.storm_group.nunique()
        summary[label] = dict(n_events=len(W), n_storm_groups=int(groups_n), n_bas=int(W.ba.nunique()),
                              descriptive_only=bool(groups_n < a.min_groups))
        views = {"pooled": W, **{f"class_{c}": W[W.source_class == c] for c in ["A", "B"]},
                 **{f"family_{f}": W[W.family.str.contains(f)] for f in ["heat", "cold", "tropical"]}}
        W = W.assign(dur=pd.qcut(W.duration_days, 3, labels=["short", "medium", "long"], duplicates="drop"))
        views.update({f"duration_{t}": W[W.dur == t] for t in W.dur.dropna().unique()})
        for v, Wv in views.items():
            if Wv.storm_group.nunique() < 2: continue
            res, pair, stab = stats(Wv, dets, Wv.storm_group.to_numpy(), a.nboot, rng, ["matrix_profile", "seasonal_naive"])
            res.to_csv(f"{a.out}/{label}__{v}__ranking.csv", index=False)
            pair.to_csv(f"{a.out}/{label}__{v}__pairwise.csv", index=False)
            summary[label][v] = dict(stab, n_events=len(Wv), n_storm_groups=int(Wv.storm_group.nunique()))
    json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=2)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, dict)} for k, v in summary.items()}, indent=2))

if __name__ == "__main__":
    main()
