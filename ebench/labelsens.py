"""Annotation-validity diagnostics: how far can the event labels be trusted?

Two detector-independent questions and one detector-dependent one:

1. event_effects      Does the series actually deviate from its own seasonal expectation inside each labelled event?
                      (daily mean |robust z| of the causal seasonal-naive residual, inside vs outside all events)
2. unlabelled_days    How many days OUTSIDE every labelled event deviate at least as much as the weakest labelled event?
                      These are candidates, not claimed anomalies: they quantify label incompleteness without using any detector.
3. scenario ranking   If the labels are imperfect (boundaries off by days, an event dropped, events longer or shorter), do the detector
                      rankings change?  evaluate_scenarios / rank_robustness.

Pure numpy/pandas.  Events are (start, end, name) with inclusive dates.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from .metrics import auc_pr, event_mask, timestamp_level_prf


def _d(x):
    return pd.Timestamp(x).normalize()


def perturbations(events, shifts=(1, 3, 7), scales=(0.5, 1.5)):
    """List of (scenario name, kind, events).  Dates stay whole days; a perturbation that would empty an event is skipped."""
    ev = [(_d(s), _d(e), n) for s, e, n in events]
    out = [("baseline", "baseline", ev)]

    def put(name, kind, new):
        if all(s <= e for s, e, _ in new) and new:
            out.append((name, kind, new))

    for d in shifts:
        td = pd.Timedelta(days=d)
        put(f"start earlier {d}d", "start_shift", [(s - td, e, n) for s, e, n in ev])
        put(f"start later {d}d", "start_shift", [(s + td, e, n) for s, e, n in ev])
        put(f"end earlier {d}d", "end_shift", [(s, e - td, n) for s, e, n in ev])
        put(f"end later {d}d", "end_shift", [(s, e + td, n) for s, e, n in ev])
        put(f"dilate {d}d", "dilate", [(s - td, e + td, n) for s, e, n in ev])
    for f in scales:
        new = []
        for s, e, n in ev:
            half = (e - s) / 2
            mid = s + half
            new.append((_d(mid - half * f), _d(mid + half * f), n))
        put(f"length x{f}", "scale", new)
    if len(ev) > 1:
        for i, (_, _, n) in enumerate(ev):
            put(f"drop {n}", "drop", [x for j, x in enumerate(ev) if j != i])
    return out


def evaluate_scenarios(times, scores, events, budget_pct=95, **kw):
    """scores: {detector: score array on `times`}.  One row per (scenario, detector): AUC-PR, timestamp F1 at the
    top-(100-budget_pct)% alert budget, prevalence."""
    t = pd.DatetimeIndex(times)
    rows = []
    for name, kind, evs in perturbations(events, **kw):
        y = np.zeros(len(t), int)
        for s, e, _ in evs:
            y[event_mask(t, s, e)] = 1
        if y.sum() == 0 or y.sum() == len(y):
            continue
        for det, sc in scores.items():
            sc = np.asarray(sc, float)
            flag = (sc > np.percentile(sc, budget_pct)).astype(int)
            rows.append(dict(scenario=name, kind=kind, detector=det, prevalence=float(y.mean()),
                             auc_pr=auc_pr(y, sc), ts_f1=timestamp_level_prf(y, flag)["f1"]))
    return pd.DataFrame(rows)


def rank_robustness(df, metric="auc_pr"):
    """Per scenario: Kendall tau between the detector ranking and the baseline ranking, and whether the baseline best detector
    is still best.  Returns (summary dict, per-scenario DataFrame).  Ties in the baseline are broken by the metric value."""
    w = df.pivot(index="scenario", columns="detector", values=metric)
    base = w.loc["baseline"]
    best = base.idxmax()
    rows = []
    for sc, r in w.iterrows():
        if sc == "baseline":
            continue
        tau = kendalltau(base.to_numpy(), r.to_numpy())[0] if r.nunique() > 1 and base.nunique() > 1 else float("nan")
        rows.append(dict(scenario=sc, tau_vs_baseline=tau, best=r.idxmax(), best_unchanged=bool(r.idxmax() == best)))
    per = pd.DataFrame(rows)
    return dict(baseline_best=best, n_scenarios=len(per), share_best_unchanged=float(per.best_unchanged.mean()),
                median_tau=float(np.nanmedian(per.tau_vs_baseline)), min_tau=float(np.nanmin(per.tau_vs_baseline))), per


def daily_deviation(times, resid):
    """Daily mean of |robust z| of a residual (z = resid / MAD-sigma over the finite values)."""
    r = np.asarray(resid, float)
    ok = np.isfinite(r)
    sig = np.median(np.abs(r[ok] - np.median(r[ok]))) * 1.4826 if ok.any() else np.nan
    z = pd.Series(np.where(ok, np.abs(r) / (sig if sig and sig > 0 else 1.0), np.nan), index=pd.DatetimeIndex(times))
    return z.groupby(z.index.normalize()).mean()


def event_effects(times, resid, events, z_cut=3.0):
    """Per event: mean daily deviation inside the event, share of its timestamps with |z|>z_cut, and the same two numbers for all
    timestamps outside every event (the reference)."""
    t = pd.DatetimeIndex(times)
    r = np.asarray(resid, float)
    ok = np.isfinite(r)
    sig = np.median(np.abs(r[ok] - np.median(r[ok]))) * 1.4826
    z = np.abs(r) / (sig if sig > 0 else 1.0)
    dd = daily_deviation(t, r)
    inside_any = np.zeros(len(t), bool)
    masks = []
    for s, e, n in events:
        m = event_mask(t, s, e) & ok
        masks.append(m)
        inside_any |= m
    ref = ~inside_any & ok
    ref_days = dd[~dd.index.isin(pd.DatetimeIndex(t[inside_any]).normalize().unique())]
    rows = []
    for (s, e, n), m in zip(events, masks):
        days = pd.date_range(_d(s), _d(e))
        d_in = dd.reindex(days).dropna()
        rows.append(dict(event=n, n_timestamps=int(m.sum()), mean_daily_dev=float(d_in.mean()) if len(d_in) else np.nan,
                         share_above_cut=float((z[m] > z_cut).mean()) if m.any() else np.nan,
                         ref_mean_daily_dev=float(ref_days.mean()), ref_share_above_cut=float((z[ref] > z_cut).mean()),
                         pct_of_ref_days_below=float((ref_days < d_in.mean()).mean() * 100) if len(d_in) else np.nan))
    return pd.DataFrame(rows)


def unlabelled_days(times, resid, events, weakest=None):
    """Days outside every event whose daily deviation is >= that of the weakest labelled event (or `weakest`).
    Returns (n_days, n_outside_days, DataFrame of those days) -- candidates for missing labels, not claimed anomalies."""
    t = pd.DatetimeIndex(times)
    dd = daily_deviation(t, resid)
    in_days = set()
    for s, e, _ in events:
        in_days |= set(pd.date_range(_d(s), _d(e)))
    ev_dev = [dd.reindex(pd.date_range(_d(s), _d(e))).dropna().mean() for s, e, _ in events]
    thr = weakest if weakest is not None else float(np.nanmin(ev_dev))
    out = dd[~dd.index.isin(in_days)]
    cand = out[out >= thr].rename("daily_dev").rename_axis("day").reset_index()   # index name varies (e.g. "Datetime")
    return len(cand), len(out), cand, thr
