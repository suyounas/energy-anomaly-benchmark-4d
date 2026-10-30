"""Real-data resolution ablation (register rows 7, 9): run the SAME classical detectors and the
SAME events at several resolutions of the SAME real series.  If a detector's score falls with
resolution while events, series and domain are held fixed, resolution is implicated; if not,
the earlier cross-dataset 'resolution' trend was confounded with dataset/event type.

Runs offline on CPU (no torch).  Example (GB NESO, Protocol B):

  python experiments/real_resolution_ablation.py \
     --csv neso_demand_full.csv --time-col Datetime --value-col ND_MW --base-minutes 30 \
     --event 2020-03-23 2020-05-31 COVID --event 2022-12-08 2022-12-17 ColdSnap \
     --train-window 2018-06-01 2020-02-01 --window-days 30 --factors 1 2 48 \
     --out results/real/gb_resolution_ablation.csv

ERCOT:  --csv <hourly csv> --base-minutes 60 --factors 1 24 and the Study 4 event list.
Features are re-derived (return, rolling mean/std/skew over a window fixed in TIME), mirroring
the published 4-feature recipe, so differences vs the published tables are expected.
"""
import argparse, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ebench import (timestamp_level_prf, event_level_prf, per_event_coverage, auc_pr,
                    f1_ceiling_at_budget, event_mask, paired_block_bootstrap, summarise_paired)
from ebench.detectors import run_classical
from ebench.synthetic import resample

FEATS = ["ret", "roll_mean", "roll_std", "roll_skew"]         # the published recipe: no level information
FEATS_RESID = FEATS + ["resid"]                                 # + causal seasonal-naive residual


def feature_cols(name):
    return {"recipe": FEATS, "resid": FEATS_RESID}[name]


def seasonal_residual(s, samples_per_day, n_weeks=4, min_lags=2):
    """Level information that is causal and needs no fitting: value minus the median of the same
    time of the week over the previous `n_weeks` weeks (seasonal-naive forecast residual), divided by
    the series' own robust spread of that residual (median absolute deviation over the whole input,
    which is a scale only).  A level shift that lasts days stays visible for as long as it lasts; a
    pure weekly pattern gives 0.  Uses only past values, so it cannot leak test information."""
    lag = 7 * samples_per_day
    past = np.column_stack([s.shift(k * lag).to_numpy() for k in range(1, n_weeks + 1)])
    ok = np.isfinite(past).sum(1) >= min_lags
    base = np.full(len(s), np.nan)
    base[ok] = np.nanmedian(past[ok], axis=1)
    r = s.to_numpy() - base
    mad = np.nanmedian(np.abs(r - np.nanmedian(r))) * 1.4826
    floor = 1e-3 * float(np.nanstd(s.to_numpy())) or 1.0          # never divide float noise by ~0
    return r / (mad if np.isfinite(mad) and mad > floor else floor)


def build_features(t, x, samples_per_day, window_days, with_resid=False):
    w = max(8, int(window_days * samples_per_day))
    s = pd.Series(x, index=t)
    ret = s.pct_change()
    F = pd.DataFrame({"value": s.values, "ret": ret.values,
                      "roll_mean": s.rolling(w, min_periods=max(2, w // 30)).mean().values,
                      "roll_std": s.rolling(w, min_periods=max(2, w // 30)).std().values,
                      "roll_skew": ret.rolling(w, min_periods=max(2, w // 30)).skew().values}, index=t)
    if with_resid:
        F["resid"] = seasonal_residual(s, samples_per_day)
    return F.replace([np.inf, -np.inf], np.nan).dropna()


def main(a):
    df = pd.read_csv(a.csv, parse_dates=[a.time_col]).sort_values(a.time_col)
    t0 = pd.DatetimeIndex(df[a.time_col]); x0 = df[a.value_col].to_numpy(float)
    events = [(s, e, n) for s, e, n in a.event]
    tr0, tr1 = pd.Timestamp(a.train_window[0]), pd.Timestamp(a.train_window[1])
    rows = []
    for f in a.factors:
        # aggregate by mean over f base steps (labels are derived from dates, so no label aggregation)
        t, x, _ = resample(t0, x0, np.zeros(len(x0), int), f)
        spd = int(round(24 * 60 / (a.base_minutes * f)))
        F = build_features(t, x, spd, a.window_days, with_resid=a.features == "resid")
        tt = F.index
        y = np.zeros(len(F), int)
        for s, e, n in events:
            y[event_mask(tt, s, e)] = 1
        in_train = (tt >= tr0) & (tt < tr1)
        Ftr, Fte = F[in_train], F[~in_train]
        yte, tte = y[~in_train], tt[~in_train]
        res = run_classical(Ftr, Fte, seed=a.seed, cols=feature_cols(a.features))
        pi = float(yte.mean())
        for det, (score, flag) in res.items():
            flag = np.asarray(flag).astype(int)
            tsm = timestamp_level_prf(yte, flag)
            evm = event_level_prf(tte, flag, events, tol=pd.Timedelta(days=a.tol_days))
            cov = per_event_coverage(tte, flag, events)
            r = dict(factor=f, resolution_min=a.base_minutes * f, n_train=len(Ftr), n_test=len(Fte),
                     detector=det, prevalence=pi, alert_rate=float(flag.mean()),
                     f1_ceiling=f1_ceiling_at_budget(pi, flag.mean()) if flag.mean() > 0 else 0.0,
                     ts_precision=tsm["precision"], ts_recall=tsm["recall"], ts_f1=tsm["f1"],
                     auc_pr=auc_pr(yte, score), auc_pr_lift=auc_pr(yte, score) / pi if pi else np.nan,
                     ev_precision=evm["precision"], ev_recall=evm["recall"], ev_f1=evm["f1"],
                     n_events=evm["n_events"])
            for name, c in cov.items():
                r[f"cov_{name}"] = c["coverage"]
            rows.append(r)
        if a.n_boot > 0:
            flags = {d: np.asarray(fl).astype(int) for d, (_, fl) in res.items()}
            fns = {d: (lambda idx, fl=fl: timestamp_level_prf(yte[idx], fl[idx])["f1"]) for d, fl in flags.items()}
            reps = paired_block_bootstrap(tte, fns, n_boot=a.n_boot, seed=a.seed, block=a.ci_block)
            point = {d: timestamp_level_prf(yte, fl)["f1"] for d, fl in flags.items()}
            ci, best, diffs = summarise_paired(reps, point)
            for r in rows:
                if r["factor"] == f:
                    d = r["detector"]
                    r.update(ts_f1_lo=ci[d][0], ts_f1_hi=ci[d][1], best_detector=best,
                             diff_vs_best=diffs[d][0], diff_vs_best_lo=diffs[d][1], diff_vs_best_hi=diffs[d][2],
                             reliably_worse_than_best=bool(diffs[d][2] < 0))
        print(f"factor {f}: n_train={len(Ftr)} n_test={len(Fte)} prevalence={pi:.3f}", flush=True)
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(a.out, index=False)
    pd.set_option("display.width", 220)
    show = ["factor", "detector", "alert_rate", "ts_f1", "f1_ceiling", "auc_pr_lift", "ev_f1"]
    if a.n_boot > 0:
        show += ["ts_f1_lo", "ts_f1_hi", "reliably_worse_than_best"]
    print(out[show].round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--time-col", required=True)
    ap.add_argument("--value-col", required=True); ap.add_argument("--base-minutes", type=int, required=True)
    ap.add_argument("--event", nargs=3, action="append", metavar=("START", "END", "NAME"), required=True)
    ap.add_argument("--train-window", nargs=2, required=True)
    ap.add_argument("--window-days", type=float, default=30.0)
    ap.add_argument("--factors", type=int, nargs="+", default=[1, 2, 48])
    ap.add_argument("--tol-days", type=float, default=1.0)
    ap.add_argument("--features", choices=["recipe", "resid"], default="recipe",
                    help="recipe = published 4 features (default); resid = + causal seasonal-naive residual")
    ap.add_argument("--n-boot", type=int, default=300, help="paired block-bootstrap replicates for ts_f1 (0 = off)")
    ap.add_argument("--ci-block", default="7D", help="bootstrap block (pandas offset); 7D keeps events mostly intact")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/real/resolution_ablation.csv")
    main(ap.parse_args())
