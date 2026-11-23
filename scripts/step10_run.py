#!/usr/bin/env python3
"""Step 10 runner: blocked temporal cross-fitting on every qualifying balancing-authority (BA) series.

Protocol (docs/STEP10_PREREGISTRATION.md, Amendments 1 and 2):
  * series: hourly demand from results of step10_fetch_eia930.py, qualifying BAs that have >= 1 event;
  * features: the 'resid' set of the equal-budget runs (published 4 features + causal seasonal-naive residual),
    30-day rolling window, built once per BA on the full hourly grid (causal, so no future leakage);
  * 4 contiguous equal-length time blocks; for each held-out block the detector is fitted on the other blocks
    with every event window +/- 14 days removed, then scores the held-out block;
  * default configuration of every detector family (no tuning), 5 seeds for stochastic families, 1 for the rest;
  * scores rank-normalised within block over rows every detector can score; 5% alert budget applied within block.

Writes, per BA, <out>/<BA>_metrics.csv (BA-level) and <out>/<BA>_events.csv (per event). Resumable.
"""
import argparse, os, sys, json
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

WIN_START, WIN_END = pd.Timestamp("2019-07-01 00:00"), pd.Timestamp("2026-07-31 23:00")
N_BLOCKS, MARGIN_DAYS, SPD, WINDOW_DAYS = 4, 14, 24, 30.0
STOCHASTIC = {"isolation_forest", "window_iforest", "lstm_ae", "transformer"}
NEURAL = ["lstm_ae", "transformer"]
DEFAULT_FAMILIES = ["isolation_forest", "knn_inductive", "rolling_z", "lstm_ae", "transformer", "diff_z",
                    "seasonal_naive", "window_pca", "window_knn", "window_iforest", "matrix_profile",
                    "spectral_residual"]

def load_events(path_b, path_a):
    rows = []
    for path, cls in ((path_b, "B"), (path_a, "A")):
        if not path or not os.path.exists(path):
            continue
        d = pd.read_csv(path, dtype=str).dropna(subset=["event_id", "start_date", "end_date"])
        for r in d.itertuples():
            for ba in str(r.BAs_mapped).replace(",", ";").split(";"):
                ba = ba.strip()
                if ba and ba != "nan":
                    rows.append(dict(ba=ba, event_id=r.event_id, storm_group=r.storm_group, source_class=cls,
                                     start=pd.Timestamp(r.start_date), end=pd.Timestamp(r.end_date)))
    return pd.DataFrame(rows)

def load_series(eia_dir, ba):
    long = pd.read_csv(os.path.join(eia_dir, "eia930_demand_long.csv.gz"), parse_dates=["datetime_utc"])
    s = long[long.ba == ba].set_index("datetime_utc")["demand_mw"].sort_index()
    return s.reindex(pd.date_range(WIN_START, WIN_END, freq="h"))

def block_ids(index):
    edges = pd.date_range(WIN_START, WIN_END + pd.Timedelta(hours=1), periods=N_BLOCKS + 1)
    return np.clip(np.searchsorted(edges, index, side="right") - 1, 0, N_BLOCKS - 1)

def flag_top(score, pct=95):
    return (score > np.percentile(score, pct)).astype(int)

def run_ba(ba, series, ev, families, n_seeds, ebench):
    em, FAMILIES, build_features, feature_cols = (ebench[k] for k in ("event_mask", "FAMILIES", "build_features", "feature_cols"))
    FEATS = feature_cols("resid")
    F = build_features(series.index, series.to_numpy(float), SPD, WINDOW_DAYS, with_resid=True)
    tt = F.index
    events = [(r.start, r.end, r.event_id) for r in ev.itertuples()]
    y = np.zeros(len(F), int)
    ev_masks = []
    for s, e, _ in events:
        m = em(tt, s, e); ev_masks.append(m)
        y[m] = 1
    margin = np.zeros(len(F), bool)
    for s, e, _ in events:
        margin |= em(tt, s, e, pd.Timedelta(days=MARGIN_DAYS))
    blk = block_ids(tt)
    jobs = [(n, sd) for n in families for sd in range(n_seeds if n in STOCHASTIC else 1)]
    pieces = {j: [] for j in jobs}
    meta = []
    for k in range(N_BLOCKS):
        te = blk == k
        tr = (blk != k) & ~margin
        if te.sum() < 24 * 30 or tr.sum() < 24 * 60:
            continue
        Ftr, Fte = F[tr], F[te]
        raw = {}
        for n, sd in jobs:
            raw[(n, sd)] = np.asarray(FAMILIES[n]["fn"](FAMILIES[n]["default"], Ftr, Fte, FEATS, sd, spd=SPD), float)
        common = np.ones(te.sum(), bool)
        for sc in raw.values():
            common &= np.isfinite(sc)
        if common.sum() < 24 * 14:
            continue
        for j, sc in raw.items():
            pieces[j].append(rankdata(sc[common]) / common.sum())
        meta.append(dict(block=k, t=tt[te][common], y=y[te][common], ev=[m[te][common] for m in ev_masks],
                         any=(np.any([m[te][common] for m in ev_masks], axis=0) if ev_masks else np.zeros(common.sum(), bool))))
    if not meta:
        return None, None
    t = pd.DatetimeIndex(np.concatenate([m["t"] for m in meta]))
    yy = np.concatenate([m["y"] for m in meta])
    anyev = np.concatenate([m["any"] for m in meta])
    evm = [np.concatenate([m["ev"][i] for m in meta]) for i in range(len(events))]
    rows, erows = [], []
    for j in jobs:
        sc = np.concatenate(pieces[j])
        flag = np.concatenate([flag_top(p) for p in pieces[j]])
        n, sd = j
        tsm = ebench["timestamp_level_prf"](yy, flag)
        evl = ebench["event_level_prf"](t, flag, events, tol=pd.Timedelta(days=1))
        rows.append(dict(ba=ba, detector=n, seed=sd, n_rows=len(yy), prevalence=round(float(yy.mean()), 5),
                         auc_pr=ebench["auc_pr"](yy, sc), ts_f1=tsm["f1"], ts_precision=tsm["precision"], ts_recall=tsm["recall"],
                         ev_recall=evl["recall"], ev_precision=evl["precision"], n_events_scored=evl["n_events"]))
        for (s, e, eid), m in zip(events, evm):
            if m.sum() < 2:
                continue
            sel = m | ~anyev                                  # this event's rows vs rows outside every event
            ap = float(average_precision_score(m[sel].astype(int), sc[sel])) if m[sel].any() else np.nan
            erows.append(dict(ba=ba, event_id=eid, detector=n, seed=sd, n_pos=int(m.sum()), auc_pr_event=ap,
                              coverage=float(flag[m].mean()), detected=int(flag[m].any())))
    return pd.DataFrame(rows), pd.DataFrame(erows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--eia", default="data/step10/eia930")
    ap.add_argument("--manifest", default="results/step10/series_manifest.csv")
    ap.add_argument("--events-b", default="results/step10/events_selected.csv")
    ap.add_argument("--events-a", default="docs/step10_events.csv")
    ap.add_argument("--out", default="results/step10/runs")
    ap.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--bas", nargs="*", help="restrict to these BA codes (testing / batching)")
    ap.add_argument("--neural-first-n", type=int, default=None,
                    help="run neural families only on the first N BAs in code order (pre-registered subset rule)")
    a = ap.parse_args()
    sys.path.insert(0, a.repo); sys.path.insert(0, os.path.join(a.repo, "experiments"))
    from ebench import event_mask, auc_pr, timestamp_level_prf, event_level_prf
    from ebench.tuning import FAMILIES
    from real_resolution_ablation import build_features, feature_cols
    eb = dict(event_mask=event_mask, auc_pr=auc_pr, timestamp_level_prf=timestamp_level_prf, event_level_prf=event_level_prf,
              FAMILIES=FAMILIES, build_features=build_features, feature_cols=feature_cols)
    os.makedirs(a.out, exist_ok=True)
    man = pd.read_csv(a.manifest)
    ev = load_events(a.events_b, a.events_a)
    bas = sorted(set(man[man.qualifies].ba) & set(ev.ba))
    if a.bas:
        bas = [b for b in bas if b in a.bas]
    print(len(bas), "BAs to run:", bas)
    for i, ba in enumerate(bas):
        fm = os.path.join(a.out, f"{ba}_metrics.csv")
        if os.path.exists(fm):
            print(ba, "done, skipping"); continue
        fams = [f for f in a.families if not (f in NEURAL and a.neural_first_n is not None and i >= a.neural_first_n)]
        print(f"[{i+1}/{len(bas)}] {ba}: {int((ev.ba == ba).sum())} events, families {len(fams)}", flush=True)
        m, e = run_ba(ba, load_series(a.eia, ba), ev[ev.ba == ba], fams, a.n_seeds, eb)
        if m is None:
            print(ba, "no scoreable blocks, skipped"); continue
        e.to_csv(os.path.join(a.out, f"{ba}_events.csv"), index=False)
        m.to_csv(fm, index=False)
    json.dump(dict(families=a.families, n_seeds=a.n_seeds, blocks=N_BLOCKS, margin_days=MARGIN_DAYS,
                   neural_first_n=a.neural_first_n), open(os.path.join(a.out, "run_config.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
