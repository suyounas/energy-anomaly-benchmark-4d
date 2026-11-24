#!/usr/bin/env python3
"""Neural vs classical on the same authorities AND the same rows. The classical and neural runs each score only rows that all their own
detectors can score, so the row sets can differ slightly. Here every detector is evaluated on the intersection of timestamps, using the saved scores
(seed-averaged), with per-event lift as in the main analysis. Output files follow the naming of step10_aggregate.py (label all_detectors_neural_bas)."""
import argparse, glob, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="."); ap.add_argument("--classical", default="results/step10/runs_classical")
    ap.add_argument("--neural", default="results/step10/runs_neural"); ap.add_argument("--events-final", default="results/step10/runs_classical/events_final.csv")
    ap.add_argument("--out", default="results/step10/neural_compare"); ap.add_argument("--nboot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-groups", type=int, default=30)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [a.repo, here, os.path.join(a.repo, "scripts")]
    from ebench import event_mask
    from step10_perturb_vus import load_scores
    from step10_aggregate import stats
    ev = pd.read_csv(a.events_final, parse_dates=["start", "end"]); ev["duration_days"] = (ev.end - ev.start).dt.days + 1
    bas = sorted(os.path.basename(f).split("_scores")[0] for f in glob.glob(os.path.join(a.neural, "scores", "*_scores.npz")))
    rows, kept = [], []
    for ba in bas:
        tc, Sc = load_scores(os.path.join(a.classical, "scores", f"{ba}_scores.npz")); tn, Sn = load_scores(os.path.join(a.neural, "scores", f"{ba}_scores.npz"))
        common = tc.intersection(tn); ic, jn = tc.get_indexer(common), tn.get_indexer(common)
        kept.append(dict(ba=ba, classical_rows=len(tc), neural_rows=len(tn), common_rows=len(common)))
        S = {d: s[ic] for d, s in Sc.items()}; S.update({d: s[jn] for d, s in Sn.items()})
        e = ev[ev.ba == ba].reset_index(drop=True)
        M = [event_mask(common, r.start, r.end) for r in e.itertuples()]; anyev = np.any(M, axis=0)
        for i, m in enumerate(M):
            if m.sum() < 2: continue
            sel = m | ~anyev; prev = m.sum() / sel.sum()
            for d, sc in S.items():
                rows.append(dict(ba=ba, event_id=e.event_id[i], storm_group=e.storm_group[i], source_class=e.source_class[i], family=e.family[i],
                                 duration_days=e.duration_days[i], detector=d, lift=average_precision_score(m[sel].astype(int), sc[sel]) / prev))
    pd.DataFrame(kept).to_csv(f"{a.out}/row_intersection.csv", index=False)
    L = pd.DataFrame(rows); L.to_csv(f"{a.out}/neural_compare_event_lift.csv.gz", index=False)
    W = L.pivot_table(index=["ba", "event_id", "storm_group", "source_class", "family", "duration_days"], columns="detector", values="lift").dropna().reset_index()
    dets = sorted(set(L.detector)); rng = np.random.default_rng(a.seed)
    print(f"authorities: {len(bas)} | events scored on common rows: {len(W)} | storm groups: {W.storm_group.nunique()} | rows kept: "
          f"{pd.DataFrame(kept).common_rows.sum()} of {pd.DataFrame(kept).classical_rows.sum()} classical rows")
    W["dur"] = pd.qcut(W.duration_days, 3, labels=["short", "medium", "long"], duplicates="drop")
    views = {"pooled": W, **{f"class_{c}": W[W.source_class == c] for c in ["A", "B"]}, **{f"family_{f}": W[W.family.str.contains(f)] for f in ["heat", "cold", "tropical"]},
             **{f"duration_{t}": W[W.dur == t] for t in W.dur.dropna().unique()}}
    for v, Wv in views.items():
        if Wv.storm_group.nunique() < 2: continue
        res, pair, stab = stats(Wv, dets, Wv.storm_group.to_numpy(), a.nboot, rng, ["matrix_profile", "seasonal_naive", "window_iforest"])
        res.to_csv(f"{a.out}/all_detectors_neural_bas__{v}__ranking.csv", index=False); pair.to_csv(f"{a.out}/all_detectors_neural_bas__{v}__pairwise.csv", index=False)
        if v == "pooled":
            print(res.sort_values("mean_rank")[["detector", "mean_metric", "mean_rank", "mean_rank_lo", "mean_rank_hi", "share_won"]].round(2).to_string(index=False))
            print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in stab.items()}, "| descriptive only:", Wv.storm_group.nunique() < a.min_groups)

if __name__ == "__main__":
    main()
