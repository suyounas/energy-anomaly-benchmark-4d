"""INTERNAL exploration (not part of the paper): can the alert stage of Section 4B-style analyses produce fewer false positives?

Question: unlabelled holiday and weather days make seasonal-naive and matrix-profile scores fire outside the labels.
Four post-processing strategies are applied to the SAME scores at the SAME 5% alert budget before alerts are formed:

  base         top-5% of test scores (as in the paper)
  holiday      scores on 24 Dec - 2 Jan and on a given list of public-holiday dates are set to the series' own median score (no alert)
  persistence  the score is replaced by its centred rolling median over --persist-hours (a flag must persist, single spikes vanish)
  both         holiday then persistence
  daily        the score is aggregated to the daily mean and every timestamp of the day inherits it (alerts are whole days)

For each strategy and detector it reports timestamp F1, event F1 (1-day tolerance), AUC-PR and the number of alert RUNS that do not overlap any
labelled window (the false-alert count the labels can see).  Nothing here changes the paper's tables; if one strategy clearly helps, it would be
added as a documented, pre-specified variant before any new claim is made.

  python3 experiments/fp_mitigation_internal.py --csv neso_demand_full.csv --time-col Datetime --value-col ND_MW --base-minutes 30 --factor 2 \
      --event 2020-03-23 2020-05-31 COVID --event 2022-12-08 2022-12-17 ColdSnap --train-window 2018-06-01 2020-02-01 \
      --families seasonal_naive matrix_profile isolation_forest --out results/fp_internal/gb
"""
import argparse, os, sys, warnings
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ebench import timestamp_level_prf, event_level_prf, auc_pr
from ebench.metrics import event_mask
from ebench.synthetic import resample
from ebench.tuning import FAMILIES
from real_resolution_ablation import build_features, feature_cols


def holiday_mask(idx, extra):
    d = pd.DatetimeIndex(idx)
    m = ((d.month == 12) & (d.day >= 24)) | ((d.month == 1) & (d.day <= 2))
    if extra:
        days = set(pd.to_datetime(extra).normalize())
        m |= d.normalize().isin(days)
    return np.asarray(m)


def runs_outside(flag, t, events):
    """Number of maximal alert runs that do not overlap any labelled window."""
    lab = np.zeros(len(t), bool)
    for s, e, _ in events:
        lab |= event_mask(t, s, e).astype(bool)
    n, i = 0, 0
    f = np.asarray(flag).astype(bool)
    while i < len(f):
        if f[i]:
            j = i
            while j + 1 < len(f) and f[j + 1]:
                j += 1
            if not lab[i:j + 1].any():
                n += 1
            i = j + 1
        else:
            i += 1
    return n


def main(a):
    df = pd.read_csv(a.csv, parse_dates=[a.time_col]).sort_values(a.time_col)
    t, x, _ = resample(pd.DatetimeIndex(df[a.time_col]), df[a.value_col].to_numpy(float), np.zeros(len(df), int), a.factor)
    spd = int(round(24 * 60 / (a.base_minutes * a.factor)))
    events = [(s, e, n) for s, e, n in a.event]
    lo, hi = pd.Timestamp(a.train_window[0]), pd.Timestamp(a.train_window[1])
    F = build_features(t, x, spd, a.window_days, with_resid=True)
    in_tr = (F.index >= lo) & (F.index < hi)
    Ftr, Fte = F[in_tr], F[~in_tr]
    cols = feature_cols(a.features)
    scores = {}
    for name in a.families:
        fam = FAMILIES[name]
        use = cols if name != "seasonal_naive" else feature_cols("resid")
        per = [np.asarray(fam["fn"](dict(fam["default"]), Ftr, Fte, use, s, spd=spd), float) for s in range(a.n_seeds)]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            scores[name] = np.nanmean(per, axis=0)
    ok = np.ones(len(Fte), bool)
    for s in scores.values():
        ok &= np.isfinite(s)
    tc = Fte.index[ok]
    y = np.zeros(len(tc), int)
    for s, e, _ in events:
        y[event_mask(tc, s, e).astype(bool)] = 1
    hol = holiday_mask(tc, a.holidays)
    w = max(1, int(a.persist_hours * 60 / (a.base_minutes * a.factor)))
    day = pd.DatetimeIndex(tc).normalize()
    rows = []
    for name, sc in scores.items():
        sc = sc[ok]
        med = float(np.median(sc))
        variants = {}
        variants["base"] = sc
        h = sc.copy(); h[hol] = med; variants["holiday"] = h
        variants["persistence"] = pd.Series(sc).rolling(w, center=True, min_periods=1).median().to_numpy()
        variants["both"] = pd.Series(h).rolling(w, center=True, min_periods=1).median().to_numpy()
        variants["daily"] = pd.Series(sc).groupby(day).transform("mean").to_numpy()
        for vname, v in variants.items():
            flag = (v >= np.quantile(v, 1 - a.budget)).astype(int)
            ts = timestamp_level_prf(y, flag)
            ev = event_level_prf(tc, flag, events, tol=pd.Timedelta(days=1))
            rows.append(dict(detector=name, variant=vname, ts_f1=ts["f1"], ev_f1=ev["f1"], ev_recall=ev["recall"], ev_precision=ev["precision"],
                             auc_pr=auc_pr(y, v), alert_runs_outside_labels=runs_outside(flag, tc, events), flagged_in_holiday_window=int(flag[hol].sum())))
    out = pd.DataFrame(rows).round(4)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.to_csv(a.out + "_fp_variants.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--time-col", required=True); ap.add_argument("--value-col", required=True)
    ap.add_argument("--base-minutes", type=int, required=True); ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--event", nargs=3, action="append", required=True, metavar=("START", "END", "NAME"))
    ap.add_argument("--train-window", nargs=2, required=True)
    ap.add_argument("--features", default="resid"); ap.add_argument("--window-days", type=float, default=30)
    ap.add_argument("--families", nargs="+", default=["seasonal_naive", "matrix_profile"])
    ap.add_argument("--n-seeds", type=int, default=3); ap.add_argument("--budget", type=float, default=0.05)
    ap.add_argument("--persist-hours", type=float, default=6.0)
    ap.add_argument("--holidays", nargs="*", default=[], help="extra public-holiday dates (YYYY-MM-DD)")
    ap.add_argument("--out", default="results/fp_internal/run")
    main(ap.parse_args())
