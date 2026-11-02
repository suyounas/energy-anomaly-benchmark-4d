import os, subprocess, sys
import numpy as np, pandas as pd
from ebench.labelsens import perturbations, evaluate_scenarios, rank_robustness, event_effects, unlabelled_days, daily_deviation

EV = [("2020-06-01", "2020-06-10", "A"), ("2020-11-01", "2020-11-05", "B")]


def _series(seed=0, lift=4.0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2020-01-01", periods=24 * 400, freq="h")
    r = rng.normal(size=len(t))
    for s, e, _ in EV:
        r[(t >= s) & (t < pd.Timestamp(e) + pd.Timedelta(days=1))] += lift
    return t, r


def test_perturbations_are_valid_and_include_drops_and_baseline():
    ps = perturbations(EV)
    names = [p[0] for p in ps]
    assert names[0] == "baseline" and "drop A" in names and "drop B" in names and "dilate 3d" in names
    for _, _, evs in ps:
        assert all(s <= e for s, e, _ in evs)
    base = dict((n, e) for n, _, e in ps)["baseline"]
    assert base[0][0] == pd.Timestamp("2020-06-01") and base[0][1] == pd.Timestamp("2020-06-10")
    longer = dict((n, e) for n, _, e in ps)["length x1.5"][0]
    assert (longer[1] - longer[0]) > (base[0][1] - base[0][0])


def test_event_shorter_than_shift_is_skipped_not_inverted():
    ps = perturbations([("2020-06-01", "2020-06-02", "S")], shifts=(3,))
    assert all(s <= e for _, _, evs in ps for s, e, _ in evs)
    assert "start later 3d" not in [p[0] for p in ps]


def test_event_effects_and_unlabelled_days_flag_a_hidden_event():
    t, r = _series()
    r2 = r.copy()
    r2[(t >= "2020-09-01") & (t < "2020-09-04")] += 6          # an unlabelled event clearly stronger than the weakest labelled one
    ee = event_effects(t, r2, EV)
    assert (ee.mean_daily_dev > 3).all() and (ee.ref_mean_daily_dev < 1.5).all() and (ee.pct_of_ref_days_below > 99).all()
    n, outside, cand, thr = unlabelled_days(t, r2, EV)
    assert n == 3 and set(pd.to_datetime(cand.day).dt.strftime("%Y-%m-%d")) == {"2020-09-01", "2020-09-02", "2020-09-03"}
    n0, _, _, _ = unlabelled_days(t, r, EV)
    assert n0 == 0


def test_rank_robustness_stable_for_good_scorer_and_reports_changes():
    t, r = _series()
    rng = np.random.default_rng(1)
    sc = {"good": r + rng.normal(size=len(t)), "mid": 0.3 * r + rng.normal(size=len(t)), "bad": rng.normal(size=len(t))}
    df = evaluate_scenarios(t, sc, EV)
    assert {"auc_pr", "ts_f1", "prevalence"} <= set(df.columns) and df.scenario.nunique() >= 15
    s, per = rank_robustness(df)
    assert s["baseline_best"] == "good" and s["share_best_unchanged"] == 1.0 and s["median_tau"] > 0.9
    # a scorer that only sees event A loses its lead when A is dropped
    only_a = (r * ((t >= "2020-06-01") & (t < "2020-06-11")))
    df2 = evaluate_scenarios(t, {"onlyA": only_a + 0.1 * rng.normal(size=len(t)), "both": r + 3 * rng.normal(size=len(t))}, EV)
    w = df2.pivot(index="scenario", columns="detector", values="auc_pr")
    assert w.loc["drop B", "onlyA"] > w.loc["baseline", "onlyA"]


def test_daily_deviation_handles_nan():
    t = pd.date_range("2020-01-01", periods=48, freq="h")
    r = np.r_[np.full(24, np.nan), np.random.default_rng(0).normal(size=24)]
    d = daily_deviation(t, r)
    assert np.isnan(d.iloc[0]) and np.isfinite(d.iloc[1])


def test_cli(tmp_path):
    rng = np.random.default_rng(1)
    n = 24 * 330
    t = pd.date_range("2019-01-01", periods=n, freq="h")
    x = 100 + 10 * np.sin(2 * np.pi * np.arange(n) / 24) + 3 * np.sin(2 * np.pi * np.arange(n) / 168) + rng.normal(0, 1, n)
    x[(t >= "2019-10-01") & (t < "2019-10-04")] += 12
    x[(t >= "2019-11-10") & (t < "2019-11-11")] += rng.normal(0, 15, 24) + 15
    csv = tmp_path / "s.csv"
    pd.DataFrame({"Datetime": t, "V": x}).to_csv(csv, index=False)
    root = os.path.dirname(os.path.dirname(__file__))
    out = tmp_path / "o"
    r = subprocess.run([sys.executable, os.path.join(root, "experiments", "label_sensitivity.py"), "--csv", str(csv), "--time-col", "Datetime",
                        "--value-col", "V", "--base-minutes", "60", "--event", "2019-10-01", "2019-10-03", "Shift",
                        "--event", "2019-11-10", "2019-11-10", "Burst", "--train-window", "2019-01-01", "2019-09-20",
                        "--families", "diff_z", "seasonal_naive", "window_pca", "--n-seeds", "1", "--out", str(out)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    txt = open(str(out) + "_report.md").read()
    assert "A. Does the series deviate" in txt and "B. Unlabelled days" in txt and "C. Detector ranking" in txt
    assert os.path.exists(str(out) + "_scenarios.csv")
