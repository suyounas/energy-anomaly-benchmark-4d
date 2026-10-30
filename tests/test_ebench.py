"""Tests for the shared evaluation contract (ebench)."""
import numpy as np
import pandas as pd
import pytest

from ebench import (event_level_prf, timestamp_level_prf, per_event_coverage, auc_pr,
                    f1_ceiling_at_budget, make_windows, contiguous_segments,
                    block_bootstrap_ci, event_mask)


def hourly(start, end):
    return pd.date_range(start, end, freq="h", inclusive="left")


EVENTS = [("2020-01-10", "2020-01-12", "A"), ("2020-02-01", "2020-02-02", "B")]


def test_event_end_day_is_fully_inside_event():
    t = hourly("2020-01-12", "2020-01-14")
    m = event_mask(t, "2020-01-10", "2020-01-12")
    assert m[:24].all() and not m[24:].any()


def test_event_level_f1_is_harmonic_mean_of_event_precision_and_recall():
    t = hourly("2020-01-01", "2020-03-01")
    flag = np.zeros(len(t), int)
    flag[(t >= "2020-01-11") & (t < "2020-01-11 06:00")] = 1      # hits A
    flag[(t >= "2020-02-20") & (t < "2020-02-20 06:00")] = 1      # false alert
    r = event_level_prf(t, flag, EVENTS)
    assert r["recall"] == pytest.approx(0.5) and r["precision"] == pytest.approx(0.5)
    assert r["f1"] == pytest.approx(0.5)
    assert r["n_events"] == 2 and r["n_hit"] == 1 and r["n_alerts"] == 2


def test_event_level_never_mixes_units_one_long_alert_counts_once():
    """A single alert covering event A for 3 days is ONE alert, not 72 timestamps."""
    t = hourly("2020-01-01", "2020-03-01")
    flag = ((t >= "2020-01-10") & (t < "2020-01-13")).astype(int)
    r = event_level_prf(t, flag, EVENTS)
    assert r["n_alerts"] == 1 and r["precision"] == 1.0 and r["recall"] == 0.5


def test_tolerance_widens_the_window_for_both_precision_and_recall():
    t = hourly("2020-01-01", "2020-03-01")
    flag = ((t >= "2020-01-15") & (t < "2020-01-15 03:00")).astype(int)   # 3 days after A
    assert event_level_prf(t, flag, EVENTS)["recall"] == 0.0
    r = event_level_prf(t, flag, EVENTS, tol=pd.Timedelta(days=5))
    assert r["recall"] == 0.5 and r["precision"] == 1.0


def test_events_outside_the_data_are_not_counted_as_misses():
    t = hourly("2020-01-01", "2020-01-20")                      # event B is not in this split
    flag = ((t >= "2020-01-11") & (t < "2020-01-11 06:00")).astype(int)
    r = event_level_prf(t, flag, EVENTS)
    assert r["n_events"] == 1 and r["recall"] == 1.0
    cov = per_event_coverage(t, flag, EVENTS)
    assert cov["B"]["n"] == 0 and cov["B"]["coverage"] is None


def test_timestamp_level_counts():
    y = np.array([0, 1, 1, 0, 0, 1]); f = np.array([0, 1, 0, 1, 0, 1])
    r = timestamp_level_prf(y, f)
    assert (r["tp"], r["fp"], r["fn"]) == (2, 1, 1)
    assert r["f1"] == pytest.approx(2 / 3)


def test_auc_pr_perfect_and_chance_level():
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.05).astype(int)
    assert auc_pr(y, y.astype(float)) == pytest.approx(1.0)
    assert auc_pr(y, rng.random(len(y))) == pytest.approx(y.mean(), abs=0.01)
    assert np.isnan(auc_pr(np.zeros(10), np.arange(10)))


@pytest.mark.parametrize("pi,b,expected", [
    (0.05, 0.05, 1.0), (0.01, 0.05, 2 * 0.01 / 0.06), (0.20, 0.05, 2 * 0.05 / 0.25)])
def test_f1_ceiling_matches_closed_form(pi, b, expected):
    assert f1_ceiling_at_budget(pi, b) == pytest.approx(expected)


def test_f1_ceiling_is_attained_by_an_oracle_flagging_top_b_of_scores():
    rng = np.random.default_rng(1)
    for pi in (0.01, 0.05, 0.2):
        y = (rng.random(50000) < pi).astype(int)
        scores = y + 1e-3 * rng.random(len(y))                  # oracle
        flag = (scores > np.percentile(scores, 95)).astype(int)
        f1 = timestamp_level_prf(y, flag)["f1"]
        assert f1 == pytest.approx(f1_ceiling_at_budget(y.mean(), 0.05), abs=0.01)


def test_windows_never_span_a_gap():
    t1 = hourly("2018-01-01", "2018-01-03"); t2 = hourly("2020-06-01", "2020-06-03")
    t = t1.append(t2)
    X = np.arange(len(t), dtype=float)[:, None]
    segs = contiguous_segments(t)
    assert len(segs) == 2
    W, ends = make_windows(X, t, seq_len=10)
    for w, j in zip(W, ends):
        span = t[j] - t[j - 9]
        assert span == pd.Timedelta(hours=9)                   # no window crosses the 2.4-year gap
    assert len(W) == (len(t1) - 9) + (len(t2) - 9)


def test_windows_short_segments_are_skipped_not_padded():
    t = hourly("2020-01-01", "2020-01-01 05:00")
    W, ends = make_windows(np.zeros((len(t), 1)), t, seq_len=10)
    assert len(W) == 0 and len(ends) == 0


def test_block_bootstrap_ci_contains_point_estimate_and_is_reproducible():
    t = hourly("2020-01-01", "2020-03-01")
    rng = np.random.default_rng(2)
    x = rng.normal(size=len(t))
    f = lambda idx: float(x[idx].mean())
    a = block_bootstrap_ci(t, f, n_boot=200, seed=3)
    b = block_bootstrap_ci(t, f, n_boot=200, seed=3)
    assert a == b and a[1] <= a[0] <= a[2] and a[3] == 60


def test_paired_bootstrap_difference_is_reliable_for_a_clearly_better_detector():
    from ebench import paired_block_bootstrap, summarise_paired
    t = hourly("2020-01-01", "2020-04-01")
    y = np.zeros(len(t), int); y[500:560] = 1; y[1500:1580] = 1; y[2000:2050] = 1
    good = y.copy(); bad = np.zeros(len(t), int); bad[100:160] = 1; bad[3000:3030] = 1
    fns = {"good": lambda idx: timestamp_level_prf(y[idx], good[idx])["f1"],
           "bad": lambda idx: timestamp_level_prf(y[idx], bad[idx])["f1"]}
    reps = paired_block_bootstrap(t, fns, n_boot=100, seed=1, block="7D")
    point = {k: fn(np.arange(len(t))) for k, fn in fns.items()}
    ci, best, diffs = summarise_paired(reps, point)
    assert best == "good" and diffs["good"][0] == 0.0
    assert diffs["bad"][2] < 0                                   # reliably worse
    assert ci["good"][0] <= point["good"] <= ci["good"][1]


def test_paired_bootstrap_cannot_separate_identical_detectors_and_is_reproducible():
    from ebench import paired_block_bootstrap, summarise_paired
    t = hourly("2020-01-01", "2020-03-01")
    y = np.zeros(len(t), int); y[300:360] = 1
    f = y.copy()
    fns = {"a": lambda idx: timestamp_level_prf(y[idx], f[idx])["f1"], "b": lambda idx: timestamp_level_prf(y[idx], f[idx])["f1"]}
    r1 = paired_block_bootstrap(t, fns, n_boot=50, seed=2)
    r2 = paired_block_bootstrap(t, fns, n_boot=50, seed=2)
    assert np.array_equal(r1["a"], r2["a"], equal_nan=True)
    ci, best, diffs = summarise_paired(r1, {"a": 1.0, "b": 1.0})
    assert diffs["b"][1] <= 0 <= diffs["b"][2]
