"""Reporting helpers shared by the experiment scripts (tested in tests/test_report.py).

* coverage_table      per-event coverage by detector (what share of each labelled event's timestamps
                      is flagged), plus how many events each detector touches at all.
* metric_intervals    paired block-bootstrap intervals for threshold-free metrics (AUC-PR, VUS-PR),
                      same resampled rows for every detector, difference to the best detector and
                      `reliably_worse_than_best`, exactly like the F1 intervals.

Caveat for the range metric: block resampling concatenates calendar blocks, so a replicate can contain
seams (an event cut at a block edge, or two anomalous blocks joined).  The buffer ramp of vus_pr can
cross such a seam.  The effect is small for week-long blocks and long events, but VUS intervals are
therefore approximate; AUC-PR (rank based, no adjacency) has no such issue.
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


def metric_intervals(times, y, groups, metric, n_boot=200, seed=0, block="7D", alpha=0.05,
                     max_buffer=None, n_buffers=5):
    """`groups` = {label: [score arrays, one per seed]} on the SAME rows as `y` and `times`.
    metric in {'auc_pr','vus_pr'}.  Returns a tidy DataFrame, one row per label."""
    y = np.asarray(y).astype(int)
    if metric == "auc_pr":
        f = lambda yy, ss: auc_pr(yy, ss)
    elif metric == "vus_pr":
        if max_buffer is None:
            runs = _runs(y)
            med = int(np.median([e - s + 1 for s, e in runs])) if runs else 2
            max_buffer = int(np.clip(med, 2, max(2, len(y) // 10)))
        f = lambda yy, ss: vus_pr(yy, ss, max_buffer=max_buffer, n_buffers=n_buffers)
    else:
        raise ValueError(metric)

    def mk(variants):
        w = sum(m for _, m in variants)
        return lambda idx: float(sum(m * f(y[idx], s[idx]) for s, m in variants) / w)

    fns = {k: mk(_unique_variants(v)) for k, v in groups.items()}
    point = {k: fn(np.arange(len(y))) for k, fn in fns.items()}
    reps = paired_block_bootstrap(times, fns, n_boot=n_boot, seed=seed, alpha=alpha, block=block)
    ci, best, diffs = summarise_paired(reps, point, alpha=alpha)
    return pd.DataFrame([dict(metric=metric, detector=k, point=point[k], lo=ci[k][0], hi=ci[k][1],
                              best=best, diff_vs_best=diffs[k][0], diff_lo=diffs[k][1], diff_hi=diffs[k][2],
                              reliably_worse_than_best=bool(diffs[k][2] < 0), n_boot=n_boot)
                         for k in groups])
