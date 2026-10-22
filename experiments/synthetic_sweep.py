"""Controlled duration x resolution sweep (register rows 6, 7, 9, 10).

Holds anomaly PREVALENCE constant (default 3% of the test split) while event DURATION varies,
analyses the SAME series at three resolutions, injects anomalies only in the test split, and
repeats over seeds.  Reports timestamp-level F1 under the published fixed top-5% alert budget,
the budget-free AUC-PR, the F1 ceiling implied by the alert rate actually used, and DBSCAN's
realised alert rate.
Usage:  python experiments/synthetic_sweep.py --seeds 10 --out results/synthetic/sweep.csv
"""
import argparse, os, sys, time, itertools
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ebench import (timestamp_level_prf, event_level_prf, auc_pr, f1_ceiling_at_budget)
from ebench.synthetic import base_series, inject_events, resample, STEPS_PER_DAY
from ebench.detectors import features, run_classical, pca_window_reconstruction

DURATIONS = {"2h": 4, "12h": 24, "1d": 48, "3d": 144, "7d": 336, "14d": 672}   # base 30-min steps
RESOLUTIONS = {"30min": 1, "1h": 2, "1d": 48}                                    # base steps / sample


def run_one(seed, dname, rname, mag, prevalence, n_steps, kind):
    rng = np.random.default_rng(seed)
    dsteps, f = DURATIONS[dname], RESOLUTIONS[rname]
    if dsteps < f:
        return []                                                # event shorter than one sample
    t, x = base_series(n_steps, rng)
    test_start = int(0.6 * n_steps); test_start -= test_start % STEPS_PER_DAY
    x2, y, ev = inject_events(x, t, test_start, dsteps, prevalence, mag, rng, kind)
    t, x2, y = resample(t, x2, y, f)
    ts = test_start // f
    spd = STEPS_PER_DAY // f
    F = features(x2, spd)
    Ftr, Fte = F.iloc[:ts], F.iloc[ts:]
    yte, tte = y[ts:], t[ts:]
    pi = float(yte.mean())
    events = [(a, b, n) for a, b, n in ev]
    rows = []
    res = run_classical(Ftr, Fte, seed=seed)
    pw_score, pw_flag, idx = pca_window_reconstruction(x2[:ts], t[:ts], x2[ts:], tte)
    # sequence-style detectors are scored on windowed samples only (as in the published tables)
    res["pca_window_recon"] = (pw_score, np.where(np.isnan(pw_score), 0, 0))
    full_flag = np.zeros(len(yte), int); full_flag[idx] = pw_flag
    res["pca_window_recon"] = (np.nan_to_num(pw_score), full_flag)
    for det, (score, flag) in res.items():
        flag = np.asarray(flag).astype(int)
        ts_m = timestamp_level_prf(yte, flag)
        ev_m = event_level_prf(tte, flag, events, tol=pd.Timedelta(days=1))
        rate = float(flag.mean())
        rows.append(dict(seed=seed, duration=dname, resolution=rname, magnitude_sd=mag, kind=kind,
                         detector=det, n_test=len(yte), prevalence=pi, n_events=ev_m["n_events"],
                         alert_rate=rate,
                         ts_precision=ts_m["precision"], ts_recall=ts_m["recall"], ts_f1=ts_m["f1"],
                         f1_ceiling=f1_ceiling_at_budget(pi, rate) if rate > 0 else 0.0,
                         auc_pr=auc_pr(yte, score), auc_pr_lift=auc_pr(yte, score) / pi,
                         ev_precision=ev_m["precision"], ev_recall=ev_m["recall"], ev_f1=ev_m["f1"]))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--prevalence", type=float, default=0.03)
    ap.add_argument("--mags", type=float, nargs="+", default=[3.0])
    ap.add_argument("--kind", default="shift")
    ap.add_argument("--years", type=float, default=9.6)
    ap.add_argument("--durations", nargs="+", default=list(DURATIONS))
    ap.add_argument("--resolutions", nargs="+", default=list(RESOLUTIONS))
    ap.add_argument("--out", default="results/synthetic/sweep.csv")
    a = ap.parse_args()
    n_steps = int(a.years * 365.25 * STEPS_PER_DAY)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    allrows, t0 = [], time.time()
    grid = list(itertools.product(range(a.seeds), a.durations, a.resolutions, a.mags))
    for i, (seed, d, r, m) in enumerate(grid):
        allrows += run_one(seed, d, r, m, a.prevalence, n_steps, a.kind)
        if (i + 1) % 10 == 0:
            print(f"{i+1}/{len(grid)}  {time.time()-t0:.0f}s", flush=True)
            pd.DataFrame(allrows).to_csv(a.out, index=False)
    pd.DataFrame(allrows).to_csv(a.out, index=False)
    print("wrote", a.out, len(allrows), "rows")
