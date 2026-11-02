"""Reporting helpers shared by the experiment scripts (tested in tests/test_report.py).

* coverage_table      per-event coverage by detector (what share of each labelled event's timestamps
                      is flagged), plus how many events each detector touches at all.
* metric_intervals    paired block-bootstrap intervals for threshold-free metrics (AUC-PR, VUS-PR),
                      same resampled rows for every detector, difference to the best detector and
                      `reliably_worse_than_best`, exactly like the F1 intervals.

Range metric and the bootstrap.  A plain block bootstrap glues blocks together in random order, which
(i) cuts a long event into pieces and (ii) lets the buffer ramp of vus_pr cross the seams.  Measured on
a synthetic series with two events and a random scorer this pushed the whole interval above the point
estimate (point 0.147, interval 0.17-0.44), so it is NOT used for vus_pr.  Instead, for vus_pr:
  * the resampling units are 7-day blocks, except that every labelled event, together with the margin its
    buffer can reach (max_buffer // 2 rows each side), is one indivisible unit;
  * each unit in a replicate is its own segment: ramps and the existence reward never cross a unit edge.
Because the margin covers the longest ramp, the segment-wise estimate on the original series equals the
ordinary one, so the point estimate lies inside the distribution it is compared with.  AUC-PR is rank based
(no adjacency) and keeps the ordinary block bootstrap.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .metrics import auc_pr, paired_block_bootstrap, summarise_paired
from .rangemetrics import vus_pr, _runs


def coverage_table(final):
    """`final` is the per-seed results table (columns family, config, seed, cov_<event>, ev_recall,
    ts_recall).  Returns one row per (family, config): mean coverage per event over seeds, the number
    of events with non-zero mean coverage, the smallest per-event coverage, and mean event / timestamp
    recall.  Coverage is the share of the event's timestamps that carry a flag, so a detector can 'hit'
    an event with 1% coverage; read both columns."""
    cov = [c for c in final.columns if c.startswith("cov_")]
    if not cov:
        raise ValueError("no cov_<event> columns in the results table")
    g = final.groupby(["family", "config"], sort=False)
    t = g[cov].mean()
    out = t.copy()
    out["events_hit"] = (t > 0).sum(axis=1)
    out["n_events"] = len(cov)
    out["min_event_coverage"] = t.min(axis=1)
    for c in ("ev_recall", "ts_recall"):
        if c in final.columns:
            out[c] = g[c].mean()
    return out.reset_index()


def _unique_variants(score_list):
    """Seeds of a deterministic detector give identical scores; keep one copy of each distinct variant
    (the metric is still the mean over seeds, weighted by multiplicity)."""
    uniq = []
    for s in score_list:
        for u in uniq:
            if np.array_equal(u[0], s, equal_nan=True):
                u[1] += 1
                break
        else:
            uniq.append([s, 1])
    return uniq


def event_units(times, y, margin_rows=0, block="7D"):
    """Resampling units for range metrics: 7-day blocks, but each labelled event plus `margin_rows` rows on
    each side (merged where they overlap) is one unit.  Returns a list of index arrays in time order."""
    t = pd.DatetimeIndex(times)
    if not t.is_monotonic_increasing:
        raise ValueError("times must be sorted")
    y = np.asarray(y).astype(int)
    n = len(y)
    bid = pd.factorize(t.floor(block))[0]
    ivs = sorted((int(bid[max(s - margin_rows, 0)]), int(bid[min(e + margin_rows, n - 1)])) for s, e in _runs(y))
    merged = []
    for lo, hi in ivs:
        if merged and lo <= merged[-1][1] + 0:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    unit_of_block = np.arange(bid.max() + 1)
    for u, (lo, hi) in enumerate(merged):
        unit_of_block[lo:hi + 1] = -1 - u            # negative ids mark event units
    uid = unit_of_block[bid]
    return [np.where(uid == k)[0] for k in pd.unique(uid)]


def _paired_unit_bootstrap(units, fns, n_boot, seed):
    """fns: {name: f(idx, segments) -> float}.  Units are drawn with replacement, concatenated, and each
    drawn copy gets its own segment id."""
    rng = np.random.default_rng(seed)
    out = {k: [] for k in fns}
    for _ in range(n_boot):
        pick = rng.integers(0, len(units), len(units))
        idx = np.concatenate([units[i] for i in pick])
        seg = np.concatenate([np.full(len(units[i]), j) for j, i in enumerate(pick)])
        for k, f in fns.items():
            out[k].append(f(idx, seg))
    return {k: np.asarray(v, float) for k, v in out.items()}


def metric_intervals(times, y, groups, metric, n_boot=200, seed=0, block="7D", alpha=0.05,
                     max_buffer=None, n_buffers=5):
    """`groups` = {label: [score arrays, one per seed]} on the SAME rows as `y` and `times`.
    metric in {'auc_pr','vus_pr'}.  Returns a tidy DataFrame, one row per label."""
    y = np.asarray(y).astype(int)
    if metric not in ("auc_pr", "vus_pr"):
        raise ValueError(metric)
    if metric == "vus_pr":
        if max_buffer is None:
            runs = _runs(y)
            med = int(np.median([e - s + 1 for s, e in runs])) if runs else 2
            max_buffer = int(np.clip(med, 2, max(2, len(y) // 10)))
        units = event_units(times, y, margin_rows=max_buffer // 2 + 1, block=block)
        unit_id = np.zeros(len(y), int)
        for j, u in enumerate(units):
            unit_id[u] = j

        def mk(variants):
            w = sum(m for _, m in variants)
            return lambda idx, seg: float(sum(
                m * vus_pr(y[idx], s[idx], max_buffer=max_buffer, n_buffers=n_buffers, segments=seg)
                for s, m in variants) / w)

        fns = {k: mk(_unique_variants(v)) for k, v in groups.items()}
        point = {k: fn(np.arange(len(y)), unit_id) for k, fn in fns.items()}
        reps = _paired_unit_bootstrap(units, fns, n_boot, seed)
    else:
        def mk(variants):
            w = sum(m for _, m in variants)
            return lambda idx: float(sum(m * auc_pr(y[idx], s[idx]) for s, m in variants) / w)

        fns = {k: mk(_unique_variants(v)) for k, v in groups.items()}
        point = {k: fn(np.arange(len(y))) for k, fn in fns.items()}
        reps = paired_block_bootstrap(times, fns, n_boot=n_boot, seed=seed, alpha=alpha, block=block)
    ci, best, diffs = summarise_paired(reps, point, alpha=alpha)
    return pd.DataFrame([dict(metric=metric, detector=k, point=point[k], lo=ci[k][0], hi=ci[k][1],
                              best=best, diff_vs_best=diffs[k][0], diff_lo=diffs[k][1], diff_hi=diffs[k][2],
                              reliably_worse_than_best=bool(diffs[k][2] < 0), n_boot=n_boot)
                         for k in groups])
