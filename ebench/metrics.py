"""Metrics for event-structured time-series anomaly benchmarks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

# ---------------------------------------------------------------- event helpers

def event_end_exclusive(end):
    """Events are specified by inclusive calendar dates; return the exclusive upper bound
    (midnight at the START of the day after `end`).  This is the fix for the
    last-day-of-event exclusion bug (register row 16)."""
    return pd.Timestamp(end).normalize() + pd.Timedelta(days=1)


def event_mask(times, start, end, tol=pd.Timedelta(0)):
    """Boolean mask of timestamps inside [start, end] (whole days, inclusive) widened by `tol`."""
    t = pd.DatetimeIndex(times)
    lo = pd.Timestamp(start).normalize() - tol
    hi = event_end_exclusive(end) + tol
    return np.asarray((t >= lo) & (t < hi))


def _runs(flag):
    """(start_idx, end_idx_inclusive) of contiguous runs of 1s."""
    f = np.asarray(flag).astype(int)
    if f.size == 0:
        return []
    d = np.diff(np.concatenate([[0], f, [0]]))
    s = np.where(d == 1)[0]
    e = np.where(d == -1)[0] - 1
    return list(zip(s, e))


# ------------------------------------------------------- per-event (not pooled)

def per_event_coverage(times, flag, events):
    """For each event: n timestamps, fraction flagged (timestamp recall for that event) and
    whether it was hit at all.  Events with no timestamps in `times` (e.g. outside the test
    split) are returned with n=0 and are excluded from any average -- never counted as misses."""
    flag = np.asarray(flag).astype(int)
    out = {}
    for start, end, name in events:
        m = event_mask(times, start, end)
        n = int(m.sum())
        out[name] = {"n": n,
                     "coverage": float(flag[m].mean()) if n else None,
                     "hit": bool(flag[m].any()) if n else None}
    return out


# --------------------------------------------------------------- event-level PRF

def event_level_prf(times, flag, events, tol=pd.Timedelta(0), merge_gap=0):
    """EVENT-level precision/recall/F1.

    alert  = maximal run of flags (runs separated by <= merge_gap unflagged steps are merged)
    recall = events with >=1 flag inside [start-tol, end+tol]  / events present in `times`
    precision = alerts overlapping some event window widened by tol / alerts
    Both are counted in EVENTS/ALERTS, so the F1 is a single, internally consistent object.
    """
    t = pd.DatetimeIndex(times)
    flag = np.asarray(flag).astype(int)
    f = flag.copy()
    if merge_gap > 0:
        for (s, e), (s2, _) in zip(_runs(flag)[:-1], _runs(flag)[1:]):
            if s2 - e - 1 <= merge_gap:
                f[e + 1:s2] = 1
    alerts = _runs(f)
    ev_masks = [event_mask(t, s, e, tol) for s, e, _ in events]
    present = [m.any() for m in ev_masks]
    n_events = int(sum(present))
    hit = sum(bool(f[m].any()) for m, p in zip(ev_masks, present) if p)
    any_ev = np.zeros(len(t), bool)
    for m in ev_masks:
        any_ev |= m
    tp_alerts = sum(bool(any_ev[s:e + 1].any()) for s, e in alerts)
    rec = hit / n_events if n_events else float("nan")
    prec = tp_alerts / len(alerts) if alerts else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1,
            "n_events": n_events, "n_hit": int(hit), "n_alerts": len(alerts)}


# ---------------------------------------------------------- timestamp-level PRF

def timestamp_level_prf(y_true, flag):
    """TIMESTAMP-level precision/recall/F1 (pooled over all timestamps; use per_event_coverage
    for the per-event breakdown)."""
    y = np.asarray(y_true).astype(bool)
    f = np.asarray(flag).astype(bool)
    tp = int((y & f).sum()); fp = int((~y & f).sum()); fn = int((y & ~f).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def auc_pr(y_true, scores):
    """Threshold-free average precision.  Chance level equals the prevalence."""
    y = np.asarray(y_true).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(average_precision_score(y, np.nan_to_num(np.asarray(scores, float))))


def f1_ceiling_at_budget(prevalence, budget):
    """Best timestamp-level F1 ANY detector can reach when it must flag exactly a fraction
    `budget` of the test set and the anomaly prevalence is `prevalence`:
        F1_max = 2*min(pi, b) / (pi + b).
    Reviewer W1/R3: a fixed top-5% alert budget makes F1 depend on prevalence alone."""
    pi, b = float(prevalence), float(budget)
    return 2 * min(pi, b) / (pi + b)


# ------------------------------------------------------------- uncertainty

def block_bootstrap_ci(times, metric_fn, n_boot=1000, seed=0, alpha=0.05, block="D"):
    """Block bootstrap over calendar blocks (default days).  `metric_fn(index_array)` must return
    a float computed on those row indices.  Returns (point, lo, hi, n_blocks)."""
    t = pd.DatetimeIndex(times)
    keys = t.floor(block)
    uniq = pd.unique(keys)
    groups = {k: np.where(keys == k)[0] for k in uniq}
    rng = np.random.default_rng(seed)
    point = metric_fn(np.arange(len(t)))
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx = np.concatenate([groups[uniq[i]] for i in pick])
        v = metric_fn(idx)
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan"), len(uniq)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi), len(uniq)


def paired_block_bootstrap(times, metric_fns, n_boot=300, seed=0, alpha=0.05, block="7D"):
    """Block bootstrap that evaluates SEVERAL metrics on the SAME resampled rows, so differences between
    detectors are paired (a replicate that is hard for one detector is hard for all).  `block` is a
    pandas offset ('7D' keeps the serial dependence inside a week and, for multi-day events, mostly
    keeps an event together; plain days understate the uncertainty when there are only a few events).
    `metric_fns` is {name: f(index_array) -> float}.  Returns {name: array of replicate values}."""
    t = pd.DatetimeIndex(times)
    keys = t.floor(block)
    uniq = pd.unique(keys)
    groups = {k: np.where(keys == k)[0] for k in uniq}
    rng = np.random.default_rng(seed)
    out = {k: [] for k in metric_fns}
    for _ in range(n_boot):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx = np.concatenate([groups[uniq[i]] for i in pick])
        for k, fn in metric_fns.items():
            out[k].append(fn(idx))
    return {k: np.asarray(v, float) for k, v in out.items()}


def summarise_paired(reps, point, alpha=0.05):
    """From replicate arrays and point estimates, return per-name (lo, hi) and the difference to the best
    point estimate: (best_name, {name: (diff_point, diff_lo, diff_hi)}).  A diff interval entirely below 0
    means the detector is reliably worse than the best one; one that contains 0 means 'cannot tell'."""
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    ci = {k: tuple(np.nanpercentile(v, [lo_q, hi_q])) if np.isfinite(v).any() else (np.nan, np.nan)
          for k, v in reps.items()}
    best = max(point, key=lambda k: point[k])
    diffs = {}
    for k in reps:
        d = reps[k] - reps[best]
        ok = np.isfinite(d)
        diffs[k] = (point[k] - point[best],) + (tuple(np.percentile(d[ok], [lo_q, hi_q])) if ok.any() else (np.nan, np.nan))
    return ci, best, diffs

