"""Statistics for many-series studies (step 8): stratified partial rank correlation with a cluster
bootstrap over source datasets.  Pure numpy/pandas/scipy."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def partial_spearman(x, y, z=None):
    """Spearman correlation of x and y after removing the (rank-)linear effect of z."""
    x, y = rankdata(x), rankdata(y)
    if z is not None:
        z = rankdata(z)
        if np.std(z) > 0:
            A = np.column_stack([np.ones_like(z), z])
            x = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
            y = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _combine(rs, ns):
    rs, ns = np.asarray(rs, float), np.asarray(ns, float)
    ok = np.isfinite(rs) & (ns > 3)
    if not ok.any():
        return float("nan")
    w = ns[ok] - 3
    z = np.arctanh(np.clip(rs[ok], -0.999999, 0.999999))
    return float(np.tanh(np.sum(w * z) / np.sum(w)))


def stratified_partial_spearman(df, x, y, z=None, group="dataset", min_n=8, n_boot=500, seed=0, alpha=0.05):
    """Per-group partial Spearman between columns x and y (controlling for z), combined by Fisher-z with
    weights n-3; 95% interval from a bootstrap that resamples GROUPS (source datasets) with replacement.
    Returns (combined_r, lo, hi, per_group DataFrame)."""
    rows = []
    for g, d in df.dropna(subset=[x, y] + ([z] if z else [])).groupby(group):
        if len(d) >= min_n:
            rows.append(dict(group=g, n=len(d), r=partial_spearman(d[x], d[y], d[z] if z else None)))
    per = pd.DataFrame(rows)
    if per.empty:
        return float("nan"), float("nan"), float("nan"), per
    point = _combine(per.r, per.n)
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(per), len(per))
        reps.append(_combine(per.r.to_numpy()[pick], per.n.to_numpy()[pick]))
    reps = np.asarray(reps)
    reps = reps[np.isfinite(reps)]
    lo, hi = (np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if len(reps) else (np.nan, np.nan))
    return point, float(lo), float(hi), per


def mean_rank_by_bin(df, detectors, metric_col_fmt, bin_col, bins):
    """Mean within-series rank (1 = best) of each detector inside each bin of `bin_col`."""
    M = np.column_stack([df[metric_col_fmt.format(d)].to_numpy(float) for d in detectors])
    R = np.apply_along_axis(lambda r: rankdata(-r, method="average"), 1, M)
    out = pd.DataFrame(R, columns=detectors, index=df.index)
    out[bin_col] = pd.cut(df[bin_col], bins=bins, include_lowest=True)
    return out.groupby(bin_col, observed=True)[detectors].mean()
