"""Tests for experiments/ercot_label_tiers.py using constructed flags with known answers."""
import numpy as np, pandas as pd
import pytest
from ercot_label_tiers import tier_report


def _frame(flag_fn, start="2021-01-01", end="2023-12-31", protocol="A", detector="D"):
    t = pd.date_range(start, end, freq="h")
    return pd.DataFrame({"protocol": protocol, "detector": detector, "time": t, "flag": flag_fn(t)})


def _in(t, s, e):
    return ((t >= pd.Timestamp(s)) & (t < pd.Timestamp(e) + pd.Timedelta(days=1))).astype(int)


def test_flags_only_in_tier2_have_zero_tier1_f1_and_full_share():
    df = _frame(lambda t: _in(t, "2023-06-27", "2023-08-10"))
    r = tier_report(df, n_boot=0).iloc[0]
    assert r["share_flags_in_tier2"] == pytest.approx(1.0)
    assert r["share_flags_outside_labels"] == pytest.approx(0.0)
    assert r["tier1_f1"] == 0.0
    assert r["tier1_plus_tier2_f1"] > 0.4          # all flags are now true positives
    assert r["cov_2023 record demand"] == pytest.approx(1.0)


def test_flags_outside_all_labels_are_counted_as_outside():
    df = _frame(lambda t: _in(t, "2022-03-01", "2022-03-10"))
    r = tier_report(df, n_boot=0).iloc[0]
    assert r["share_flags_outside_labels"] == pytest.approx(1.0)
    assert r["tier1_f1"] == 0.0 and r["tier1_plus_tier2_f1"] == 0.0


def test_event_last_day_is_inside_tier1():
    # flag only the final calendar day of the Freeze (2021-02-19); coverage must be 1/6 of the event
    df = _frame(lambda t: _in(t, "2021-02-19", "2021-02-19"))
    r = tier_report(df, n_boot=0).iloc[0]
    assert r["cov_Texas Freeze"] == pytest.approx(1 / 6)
    assert r["share_flags_in_tier1"] == pytest.approx(1.0)


def test_tier2_outside_test_span_is_flagged_not_zero():
    df = _frame(lambda t: _in(t, "2021-02-14", "2021-02-19"), start="2021-01-01", end="2022-06-30",
                protocol="B")
    r = tier_report(df, n_boot=0).iloc[0]
    assert not r["in_test_span_tier2"] and r["in_test_span_tier1"]
    assert r["tier1_plus_tier2_prevalence"] == pytest.approx(r["tier1_prevalence"])
    assert r["tier1_f1"] == pytest.approx(1.0)


def test_summer_2023_count_matches_manual_count():
    flag = lambda t: _in(t, "2023-07-05", "2023-07-14") | _in(t, "2022-05-01", "2022-05-02")
    df = _frame(flag)
    r = tier_report(df, n_boot=0).iloc[0]
    assert r["n_flags_jul_sep_2023"] == 10 * 24
    assert r["n_flags"] == 12 * 24
    assert r["share_flags_jul_sep_2023"] == pytest.approx(10 / 12)


def test_bootstrap_interval_brackets_point_estimate_and_is_reproducible():
    df = _frame(lambda t: _in(t, "2021-02-15", "2021-02-19") | _in(t, "2022-01-01", "2022-01-05"),
                end="2022-12-31")
    a = tier_report(df, n_boot=100, seed=3).iloc[0]
    b = tier_report(df, n_boot=100, seed=3).iloc[0]
    assert a["tier1_f1_lo"] == b["tier1_f1_lo"] and a["tier1_f1_hi"] == b["tier1_f1_hi"]
    assert a["tier1_f1_lo"] <= a["tier1_f1"] <= a["tier1_f1_hi"]


def test_missing_columns_raise():
    with pytest.raises(ValueError):
        tier_report(pd.DataFrame({"time": [], "flag": []}))


def test_groups_are_reported_separately():
    a = _frame(lambda t: _in(t, "2023-07-01", "2023-07-02"), detector="LSTM")
    b = _frame(lambda t: _in(t, "2021-02-14", "2021-02-19"), detector="DBSCAN")
    out = tier_report(pd.concat([a, b]), n_boot=0).set_index("detector")
    assert out.loc["LSTM", "share_flags_in_tier2"] == pytest.approx(1.0)
    assert out.loc["DBSCAN", "share_flags_in_tier1"] == pytest.approx(1.0)
