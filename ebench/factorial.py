"""Fully crossed semi-synthetic design for step 9 (docs/STEP9_PREREGISTRATION.md).

Pure numpy/pandas.  Three pieces: `make_background` / `inject_events` (data), `ss_decomposition`
(variance shares in a balanced factorial) and `ranking_stability`.
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd

BACKGROUNDS = {          # amplitudes in units of the signal sd before scaling to MW
    "strong": dict(daily=1.0, harm2=0.35, weekly=0.4, annual=0.8, phi=0.90, noise=0.15, trend=0.0),
    "weak": dict(daily=0.4, harm2=0.1, weekly=0.15, annual=0.3, phi=0.95, noise=0.35, trend=0.0),
    "trend": dict(daily=1.0, harm2=0.35, weekly=0.4, annual=1.2, phi=0.90, noise=0.25, trend=0.6),
}
EVENT_TYPES = ("shift", "burst", "ramp")


def make_background(kind, n_hours, rng, start="2015-01-01"):
    p = BACKGROUNDS[kind]
    t = pd.date_range(start, periods=n_hours, freq="h")
    k = np.arange(n_hours)
    d = 2 * np.pi * k / 24
    sig = (p["daily"] * np.sin(d - 1.0) + p["harm2"] * np.sin(2 * d) + p["weekly"] * np.sin(2 * np.pi * k / 168)
           + p["annual"] * np.sin(2 * np.pi * k / (365.25 * 24)) + p["trend"] * k / (365.25 * 24))
    eps = rng.standard_t(df=6, size=n_hours) * p["noise"]
    ar = np.empty(n_hours); ar[0] = eps[0]
    for i in range(1, n_hours):
        ar[i] = p["phi"] * ar[i - 1] + eps[i]
    return t, 30000.0 * (1.0 + 0.1 * (sig + ar))          # positive, MW-like, so percentage returns are defined


def residual_scale(x_train, lag=168):
    """Robust sd of the weekly-lag innovation: the unit in which magnitudes are expressed."""
    d = x_train[lag:] - x_train[:-lag]
    return float(np.median(np.abs(d - np.median(d))) * 1.4826 / np.sqrt(2))


def inject_events(x, test_start, kind, dur_h, mag, scale, rng, n_events=3, align=4):
    """Inject `n_events` events of `dur_h` hours into x[test_start:].  Starts are snapped to `align` hours
    so labels survive 4 h aggregation.  Returns (x_new, y)."""
    n = len(x)
    n_test = n - test_start
    slot = n_test // n_events
    if slot < dur_h + 2 * align:
        raise ValueError("duration too long for the number of events")
    x2, y = x.copy(), np.zeros(n, int)
    for i in range(n_events):
        lo = test_start + i * slot + align
        hi = test_start + (i + 1) * slot - dur_h - align
        s = lo + int(rng.integers(0, max(hi - lo, 1)))
        s -= (s - test_start) % align
        e = s + dur_h
        if kind == "shift":
            x2[s:e] += mag * scale
        elif kind == "burst":
            x2[s:e] += mag * scale * rng.standard_normal(dur_h)
        elif kind == "ramp":
            x2[s:e] += mag * scale * np.linspace(0.0, 1.0, dur_h)
        else:
            raise ValueError(kind)
        y[s:e] = 1
    return x2, y


# ------------------------------------------------------------------ variance decomposition
def ss_decomposition(df, factors, y, max_order=2):
    """Sums of squares of a BALANCED full factorial.  Terms = all subsets of `factors` up to `max_order`;
    `residual` = total minus the listed terms (higher-order interactions and replicate noise).
    Returns a DataFrame term, order, ss, share."""
    counts = df.groupby(factors).size()
    levels = [sorted(df[f].unique()) for f in factors]
    if len(counts) != int(np.prod([len(l) for l in levels])) or counts.nunique() != 1:
        raise ValueError("design is not balanced: every factor combination needs the same number of observations")
    d = df.copy()
    d["_rep"] = d.groupby(factors).cumcount()
    idx = pd.MultiIndex.from_product(levels + [sorted(d["_rep"].unique())], names=factors + ["_rep"])
    A = d.set_index(factors + ["_rep"])[y].reindex(idx).to_numpy(float).reshape([len(l) for l in levels] + [-1])
    k = len(factors)
    grand = A.mean()
    total = float(((A - grand) ** 2).sum())

    def marginal(T):                       # mean over every axis not in T (incl. replicates), dims kept
        axes = tuple(i for i in range(k + 1) if i not in T)
        return A.mean(axis=axes, keepdims=True) if axes else A

    rows = []
    for r in range(1, max_order + 1):
        for S in itertools.combinations(range(k), r):
            eff = np.zeros([1] * (k + 1))
            for q in range(0, r + 1):
                for T in itertools.combinations(S, q):
                    eff = eff + (-1) ** (r - q) * (marginal(T) if q else grand)
            ss = float((np.broadcast_to(eff, A.shape) ** 2).sum())
            rows.append(dict(term=" x ".join(factors[i] for i in S), order=r, ss=ss))
    out = pd.DataFrame(rows)
    out["share"] = out.ss / total
    resid = total - out.ss.sum()
    return pd.concat([out, pd.DataFrame([dict(term="residual (higher-order + replicates)", order=99, ss=resid, share=resid / total)])],
                     ignore_index=True)


def ranking_stability(df, condition_cols, det_col, y):
    """Per condition (mean over replicates) ranking of detectors vs the global ranking (mean over conditions).
    Returns (summary dict, DataFrame of per-condition winner and Spearman)."""
    from scipy.stats import spearmanr
    cm = df.groupby(condition_cols + [det_col])[y].mean().unstack(det_col)
    glob = cm.mean(axis=0)
    best = glob.idxmax()
    rows = []
    for idx, r in cm.iterrows():
        rho = spearmanr(r.to_numpy(), glob.to_numpy())[0] if r.nunique() > 1 and glob.nunique() > 1 else float('nan')
        rows.append(dict(**dict(zip(condition_cols, idx if isinstance(idx, tuple) else (idx,))),
                         winner=r.idxmax(), spearman_vs_global=rho))
    per = pd.DataFrame(rows)
    return dict(global_best=best, share_conditions_global_best_wins=float((per.winner == best).mean()),
                mean_spearman=float(np.nanmean(per.spearman_vs_global)) if per.spearman_vs_global.notna().any() else float('nan'), n_conditions=len(per),
                n_distinct_winners=int(per.winner.nunique())), per
