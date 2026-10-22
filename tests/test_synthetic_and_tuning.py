"""Tests for the controlled-experiment machinery (synthetic injection, tuning protocol)."""
import numpy as np
import pandas as pd
import pytest

from ebench.synthetic import base_series, inject_events, resample, STEPS_PER_DAY
from ebench.tuning import inject_into_series, equal_budget_search, sample_configs, FAMILIES
from ebench.detectors import features, run_classical


def _series(years=1.2, seed=0):
    rng = np.random.default_rng(seed)
    return base_series(int(years * 365.25 * STEPS_PER_DAY), rng)


@pytest.mark.parametrize("dur_steps", [4, 48, 336])
def test_injection_holds_prevalence_constant_across_durations(dur_steps):
    t, x = _series(years=9.6)         # long enough that >=3 events of the longest duration fit at 3%
    ts = int(0.6 * len(x)); ts -= ts % STEPS_PER_DAY
    _, y, ev = inject_events(x, t, ts, dur_steps, 0.03, 3.0, np.random.default_rng(1))
    assert y[:ts].sum() == 0                                   # train split stays clean
    assert y[ts:].mean() == pytest.approx(0.03, abs=0.006)
    assert len(ev) >= 3


def test_events_start_on_day_boundaries_and_do_not_overlap():
    t, x = _series()
    ts = int(0.6 * len(x)); ts -= ts % STEPS_PER_DAY
    _, y, ev = inject_events(x, t, ts, 24, 0.03, 3.0, np.random.default_rng(2))
    assert all(a.hour == 0 and a.minute == 0 for a, _, _ in ev)
    assert y.sum() == 24 * len(ev)                              # no overlap


def test_injection_on_subarray_needs_finite_scale_regression():
    """Regression: x[:0].std() is NaN, which silently turned events into NaN rows."""
    t, x = _series(0.5)
    with pytest.raises(ValueError):
        inject_events(x, t, 0, 48, 0.05, 3.0, np.random.default_rng(0))
    x2, y, _ = inject_events(x, t, 0, 48, 0.05, 3.0, np.random.default_rng(0), scale=float(x.std()))
    assert np.isfinite(x2).all() and y.sum() > 0


def test_inject_into_series_only_touches_validation_part():
    t, x = _series(1.0)
    vs = int(0.7 * len(x)); vs -= vs % STEPS_PER_DAY
    x2, y = inject_into_series(t, x, vs, (1, 3), 0.08, 3.0, seed=0)
    assert y[:vs].sum() == 0 and np.allclose(x2[:vs], x[:vs])
    assert y[vs:].sum() > 0 and np.isfinite(x2).all()


def test_resample_label_is_majority_and_values_are_means():
    t = pd.date_range("2020-01-01", periods=96, freq="30min")
    x = np.arange(96, dtype=float); y = np.zeros(96, int); y[:30] = 1
    ts, xs, ys = resample(t, x, y, 48)
    assert len(xs) == 2 and xs[0] == pytest.approx(x[:48].mean())
    assert list(ys) == [1, 0]                                   # 30/48 >= 0.5 -> 1 ; 0/48 -> 0


def test_equal_budget_search_gives_every_family_the_same_number_of_trials():
    t, x = _series(1.0)
    vs = int(0.7 * len(x)); vs -= vs % STEPS_PER_DAY
    x2, y = inject_into_series(t, x, vs, (1, 3), 0.08, 3.0, seed=0)
    F = features(x2, STEPS_PER_DAY)
    cols = ["value", "ret", "roll_mean", "roll_std"]
    Ftr, Fva, yva = F.iloc[:vs], F.iloc[vs:], y[vs:]
    fams = ["isolation_forest", "knn_inductive", "rolling_z"]
    trials, chosen = equal_budget_search(fams, Ftr, Fva, yva, cols, n_trials=3, seed=0)
    counts = trials.groupby("family").size()
    assert counts.nunique() == 1 and counts.iloc[0] == 4          # default + 3 random
    for f in fams:
        best = trials[trials.family == f].val_auc_pr.max()
        chosen_val = trials[(trials.family == f)].val_auc_pr.max()
        assert chosen[f] is not None and best == chosen_val


def test_equal_budget_search_refuses_validation_without_events():
    t, x = _series(0.6)
    F = features(x, STEPS_PER_DAY)
    cols = ["value", "ret", "roll_mean", "roll_std"]
    with pytest.raises(ValueError):
        equal_budget_search(["knn_inductive"], F.iloc[:5000], F.iloc[5000:], np.zeros(len(F) - 5000, int), cols)


def test_sample_configs_are_unique_and_within_space():
    space = FAMILIES["isolation_forest"]["space"]
    cfgs = sample_configs(space, 8, np.random.default_rng(0))
    assert len({tuple(sorted(c.items())) for c in cfgs}) == len(cfgs) == 8
    assert all(c["n_estimators"] in space["n_estimators"] for c in cfgs)


def test_classical_detectors_return_aligned_scores_and_binary_flags():
    t, x = _series(1.0)
    F = features(x, STEPS_PER_DAY)
    res = run_classical(F.iloc[:12000], F.iloc[12000:15000])
    for name, (score, flag) in res.items():
        assert len(score) == len(flag) == 3000, name
        assert set(np.unique(flag)) <= {0, 1}, name
