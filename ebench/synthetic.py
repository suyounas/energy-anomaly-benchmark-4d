"""Synthetic energy-like series with injected anomalies of controlled duration.

Purpose (register rows 6, 7): break the confound between event duration, sampling
resolution, dataset and domain.  The SAME underlying series is analysed at several
resolutions, anomaly PREVALENCE is held constant while duration varies, and anomalies are
injected only into the test split (train is clean, as in the benchmark protocol).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASE_FREQ_MIN = 30
STEPS_PER_DAY = 24 * 60 // BASE_FREQ_MIN          # 48


def base_series(n_steps, rng, start="2010-01-01"):
    """Daily + weekly + annual seasonality, AR(1) residual, occasional heavy-tailed noise."""
    t = pd.date_range(start, periods=n_steps, freq=f"{BASE_FREQ_MIN}min")
    k = np.arange(n_steps)
    day = 2 * np.pi * k / STEPS_PER_DAY
    week = 2 * np.pi * k / (7 * STEPS_PER_DAY)
    year = 2 * np.pi * k / (365.25 * STEPS_PER_DAY)
    seas = 1.0 * np.sin(day - 1.0) + 0.35 * np.sin(2 * day) + 0.4 * np.sin(week) + 0.8 * np.sin(year)
    eps = rng.standard_t(df=6, size=n_steps) * 0.25
    ar = np.empty(n_steps); ar[0] = eps[0]
    for i in range(1, n_steps):                      # AR(1), phi=0.97 (slow-moving residual)
        ar[i] = 0.97 * ar[i - 1] + eps[i] * 0.25
    x = seas + ar
    return t, x


def inject_events(x, t, test_start, duration_steps, prevalence, magnitude_sd, rng, kind="shift", scale=None):
    """Inject n non-overlapping events of `duration_steps` into x[test_start:] so that the
    anomalous fraction of the TEST split equals `prevalence` (up to rounding).
    Events start on calendar-day boundaries so labels are identical at every coarser resolution.
    Returns (x_new, label_array_full_length, events_list)."""
    N = len(x)
    n_test = N - test_start
    n_events = max(3, int(round(prevalence * n_test / duration_steps)))
    slot = n_test // n_events
    sd = float(scale) if scale is not None else float(x[:test_start].std())
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError('injection scale is not finite/positive; pass scale= explicitly')
    x2 = x.copy(); y = np.zeros(N, int); events = []
    for i in range(n_events):
        lo = test_start + i * slot
        hi = lo + slot - duration_steps
        if hi <= lo:
            raise ValueError("duration too long for the number of events")
        s = lo + int(rng.integers(0, hi - lo + 1))
        s -= (s - test_start) % STEPS_PER_DAY        # snap to a day boundary
        s = max(s, lo)
        e = s + duration_steps
        if kind == "shift":
            x2[s:e] += magnitude_sd * sd
        elif kind == "burst":                          # variance burst, zero-mean
            x2[s:e] += magnitude_sd * sd * rng.standard_normal(e - s)
        else:
            raise ValueError(kind)
        y[s:e] = 1
        events.append((t[s], t[e - 1], f"ev{i}"))
    return x2, y, events


def resample(t, x, y, factor):
    """Aggregate by mean over `factor` base steps; label = majority of the block.
    Blocks are aligned to the start of the series."""
    if factor == 1:
        return t, x, y
    n = (len(x) // factor) * factor
    xs = x[:n].reshape(-1, factor).mean(1)
    ys = (y[:n].reshape(-1, factor).mean(1) >= 0.5).astype(int)
    ts = t[:n:factor]
    return ts, xs, ys
