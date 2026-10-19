"""Tests for the evaluation helpers in Study 3 (GB demand): the code that turns
anomaly scores into flags and computes per-event recall.  If these are wrong,
every published recall number is wrong, so they are tested on hand-built cases.
"""
import numpy as np
import pandas as pd
import pytest

from _load import load

NS = load("code/study3_gb_neso/03_modelling.py",
          ["flag_from_scores", "per_event_recall", "beast_fp_share",
           "GROUND_TRUTH_EVENTS", "BEAST_WINDOW"])
flag_from_scores = NS["flag_from_scores"]
per_event_recall = NS["per_event_recall"]
beast_fp_share = NS["beast_fp_share"]


def half_hourly(start, end):
    return pd.Series(pd.date_range(start, end, freq="30min"))


def test_flag_from_scores_flags_about_five_percent():
    scores = np.arange(1000, dtype=float)
    flag = flag_from_scores(scores, pct=95)
    assert set(np.unique(flag)) <= {0, 1}
    assert flag.sum() == 50                      # strictly above the 95th percentile
    assert flag[-50:].all() and not flag[:-50].any()


def test_flag_from_scores_is_scale_and_shift_invariant():
    s = np.random.default_rng(0).normal(size=500)
    assert np.array_equal(flag_from_scores(s), flag_from_scores(3.0 * s + 10.0))


def test_per_event_recall_all_flagged_and_none_flagged():
    dates = half_hourly("2020-03-01", "2020-06-30")
    y = np.zeros(len(dates), int)
    all_flag = np.ones(len(dates), int)
    none_flag = np.zeros(len(dates), int)
    r_all = per_event_recall(dates, y, all_flag)
    r_none = per_event_recall(dates, y, none_flag)
    assert r_all["COVID collapse"] == pytest.approx(1.0)
    assert r_none["COVID collapse"] == pytest.approx(0.0)


def test_per_event_recall_is_none_when_event_absent_from_test_data():
    dates = half_hourly("2019-01-01", "2019-03-01")        # contains neither event
    r = per_event_recall(dates, np.zeros(len(dates), int), np.ones(len(dates), int))
    assert r["COVID collapse"] is None and r["Cold snap spike"] is None


def test_per_event_recall_half_flagged():
    dates = half_hourly("2022-12-01", "2022-12-31")
    flag = np.zeros(len(dates), int)
    mask = ((dates >= "2022-12-08") & (dates <= "2022-12-17")).values
    idx = np.flatnonzero(mask)
    flag[idx[: len(idx) // 2]] = 1
    r = per_event_recall(dates, np.zeros(len(dates), int), flag)
    assert r["Cold snap spike"] == pytest.approx(0.5, abs=0.01)


@pytest.mark.xfail(
    strict=True,
    reason="KNOWN INCONSISTENCY: labelling treats an event's end date as inclusive of "
           "the whole day (Date <= end + 1 day), but per_event_recall compares against "
           "the end date at 00:00, so the final day of each event is excluded from the "
           "recall denominator. Effect is about 1 day in 10 (cold snap) or 1 in 70 (COVID).")
def test_event_end_date_is_inclusive_of_whole_day():
    dates = half_hourly("2022-12-17", "2022-12-18")
    flag = np.ones(len(dates), int)
    flag[dates < "2022-12-17 12:00"] = 0                    # only the last day's afternoon
    r = per_event_recall(dates, np.zeros(len(dates), int), flag)
    # Rows on 17 Dec that belong to the event: all 48. Flagged: 24. Expect 0.5.
    assert r["Cold snap spike"] == pytest.approx(0.5, abs=0.01)


def test_beast_fp_share_counts_only_false_positives():
    dates = half_hourly("2018-02-01", "2018-03-31")
    y = np.zeros(len(dates), int)
    flag = np.zeros(len(dates), int)
    in_beast = ((dates >= "2018-02-15") & (dates <= "2018-03-15")).values
    flag[in_beast] = 1                                       # every FP inside the window
    share, n_fp = beast_fp_share(dates, y, flag)
    assert share == pytest.approx(1.0) and n_fp == in_beast.sum()


def test_beast_fp_share_ignores_true_positives_and_handles_no_fp():
    dates = half_hourly("2018-02-01", "2018-03-31")
    y = np.ones(len(dates), int)                             # everything is "anomalous"
    share, n_fp = beast_fp_share(dates, y, np.ones(len(dates), int))
    assert (share, n_fp) == (0.0, 0)
