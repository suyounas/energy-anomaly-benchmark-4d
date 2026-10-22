"""Windowing that never spans a gap in the time axis (register row 18: Protocol-B seam)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def contiguous_segments(times, step=None, tol=1.5):
    """Split the index into segments whose successive gaps are <= tol*step.
    `step` defaults to the median gap.  Returns list of (start_idx, end_idx_exclusive)."""
    t = pd.DatetimeIndex(times)
    if len(t) < 2:
        return [(0, len(t))]
    gaps = np.diff(t.values).astype("timedelta64[s]").astype(float)
    step = float(np.median(gaps)) if step is None else pd.Timedelta(step).total_seconds()
    cuts = np.where(gaps > tol * step)[0] + 1
    bounds = [0, *cuts.tolist(), len(t)]
    return list(zip(bounds[:-1], bounds[1:]))


def make_windows(X, times, seq_len, step=None):
    """Return (windows, end_index) where windows[i] = X[end_index[i]-seq_len+1 : end_index[i]+1]
    and every window lies inside ONE contiguous segment.  The window is scored AT its last
    timestamp (end_index), so score alignment is explicit instead of an implicit offset."""
    X = np.asarray(X)
    wins, ends = [], []
    for s, e in contiguous_segments(times, step=step):
        for j in range(s + seq_len - 1, e):
            wins.append(X[j - seq_len + 1: j + 1])
            ends.append(j)
    if not wins:
        return np.empty((0, seq_len) + X.shape[1:]), np.empty(0, int)
    return np.stack(wins), np.asarray(ends, int)
