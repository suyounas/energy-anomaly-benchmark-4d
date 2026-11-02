"""Coverage table, metric intervals, parameter-free note (ebench.report, ebench.tuning)."""
import os, subprocess, sys
import numpy as np
import pandas as pd
import pytest

from ebench import coverage_table, metric_intervals
from ebench.tuning import is_parameter_free


def toy_final():
    rows = []
    for fam, covs in (("a", (0.5, 0.0)), ("b", (0.0, 0.0)), ("c", (0.2, 0.4))):
        for seed in range(3):
            rows.append(dict(family=fam, config="default", seed=seed, **{"cov_Big event": covs[0], "cov_Spike": covs[1]},
                             ev_recall=float(sum(c > 0 for c in covs)) / 2, ts_recall=0.1))
    return pd.DataFrame(rows)


def test_coverage_table_counts_events_hit_and_min_coverage():
    t = coverage_table(toy_final()).set_index("family")
    assert list(t.events_hit) == [1, 0, 2] and (t.n_events == 2).all()
    assert t.loc["a", "cov_Big event"] == pytest.approx(0.5) and t.loc["c", "min_event_coverage"] == pytest.approx(0.2)
    assert t.loc["b", "ev_recall"] == 0.0


def test_coverage_table_requires_coverage_columns():
    with pytest.raises(ValueError):
        coverage_table(pd.DataFrame(dict(family=["a"], config=["d"])))


def series(n_days=120, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2020-01-01", periods=n_days * 24, freq="h")
    y = np.zeros(len(t), int)
    for d0 in (30, 80):
        y[d0 * 24: d0 * 24 + 36] = 1
    good = y * 3.0 + rng.normal(size=len(t))
    poor = rng.normal(size=len(t))
    return t, y, good, poor


@pytest.mark.parametrize("metric", ["auc_pr", "vus_pr"])
def test_metric_intervals_rank_a_good_detector_above_noise(metric):
    t, y, good, poor = series()
    d = metric_intervals(t, y, {"good": [good], "poor": [poor]}, metric, n_boot=40, seed=1).set_index("detector")
    assert d.loc["good", "point"] > d.loc["poor", "point"]
    assert d.loc["good", "best"] == "good" and d.loc["good", "diff_vs_best"] == 0
    assert d.loc["poor", "reliably_worse_than_best"]
    assert d.loc["good", "lo"] <= d.loc["good", "point"] <= d.loc["good", "hi"] + 1e-9


def test_metric_intervals_are_deterministic_and_collapse_identical_seeds():
    t, y, good, poor = series()
    a = metric_intervals(t, y, {"g": [good, good, good], "p": [poor]}, "auc_pr", n_boot=30, seed=3)
    b = metric_intervals(t, y, {"g": [good], "p": [poor]}, "auc_pr", n_boot=30, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_metric_intervals_average_over_different_seeds():
    t, y, good, poor = series()
    d = metric_intervals(t, y, {"x": [good, poor]}, "auc_pr", n_boot=10, seed=0).iloc[0]
    from ebench import auc_pr
    assert d.point == pytest.approx((auc_pr(y, good) + auc_pr(y, poor)) / 2)


def test_unknown_metric_is_rejected():
    t, y, good, _ = series()
    with pytest.raises(ValueError):
        metric_intervals(t, y, {"g": [good]}, "f1")


def test_parameter_free_families_are_recognised():
    assert is_parameter_free("diff_z") and is_parameter_free("seasonal_naive")
    assert not is_parameter_free("window_pca") and not is_parameter_free("isolation_forest")


def test_cli_writes_coverage_and_metric_intervals_and_prints_info_not_note(tmp_path):
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
    r = subprocess.run([sys.executable, os.path.join(root, "experiments", "equal_budget_tuning.py"), "--csv", str(csv),
                        "--time-col", "Datetime", "--value-col", "V", "--base-minutes", "60", "--factor", "1",
                        "--event", "2019-10-01", "2019-10-03", "Shift", "--event", "2019-11-10", "2019-11-10", "Burst",
                        "--train-window", "2019-01-01", "2019-09-20", "--features", "resid", "--suite-days-per-cell", "2",
                        "--families", "diff_z", "seasonal_naive", "window_pca", "--n-trials", "2", "--n-seeds", "2",
                        "--n-boot", "15", "--n-boot-vus", "8", "--suite-durations", "1", "3", "--out", str(out)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "INFO diff_z: parameter-free" in r.stdout and "NOTE diff_z" not in r.stdout
    assert "NOTE seasonal_naive" not in r.stdout
    cov = pd.read_csv(str(out) + "_coverage.csv")
    assert {"cov_Shift", "cov_Burst", "events_hit", "min_event_coverage"} <= set(cov.columns)
    mi = pd.read_csv(str(out) + "_ci_metrics.csv")
    assert set(mi.metric) == {"auc_pr", "vus_pr"} and mi.point.notna().all()
    assert "auc_pr: paired 7D-block bootstrap" in r.stdout


def _two_event_series(n_days=730):
    import numpy as np, pandas as pd
    t = pd.date_range("2020-01-01", periods=24 * n_days, freq="h")
    y = np.zeros(len(t), int)
    y[24 * 300:24 * 370] = 1
    y[24 * 560:24 * 569] = 1
    return t, y


def test_vus_interval_contains_point_for_random_and_good_scorers():
    import numpy as np
    from ebench.report import metric_intervals
    t, y = _two_event_series()
    rng = np.random.default_rng(0)
    g = {f"rand{i}": [rng.normal(size=len(t))] for i in range(3)}
    g["good"] = [rng.normal(size=len(t)) + 2 * y]
    d = metric_intervals(t, y, g, "vus_pr", n_boot=60)
    assert ((d.point >= d.lo) & (d.point <= d.hi)).all(), d[["detector", "point", "lo", "hi"]]


def test_event_units_keep_events_whole_and_partition_rows():
    import numpy as np
    from ebench.report import event_units
    t, y = _two_event_series()
    units = event_units(t, y, margin_rows=100)
    allidx = np.concatenate(units)
    assert sorted(allidx.tolist()) == list(range(len(y)))                  # a partition
    for s, e in ((24 * 300, 24 * 370 - 1), (24 * 560, 24 * 569 - 1)):
        owner = {k for k, u in enumerate(units) if s in set(u.tolist())}
        assert len(owner) == 1
        k = owner.pop()
        u = set(units[k].tolist())
        assert all(i in u for i in range(s - 100, e + 101))               # event plus margin in one unit


def test_segmentwise_point_equals_plain_vus():
    import numpy as np
    from ebench.report import event_units
    from ebench.rangemetrics import vus_pr
    t, y = _two_event_series()
    rng = np.random.default_rng(1)
    sc = rng.normal(size=len(t)) + y
    mb = 300
    units = event_units(t, y, margin_rows=mb // 2 + 1)
    seg = np.zeros(len(y), int)
    for j, u in enumerate(units):
        seg[u] = j
    assert abs(vus_pr(y, sc, max_buffer=mb, n_buffers=5, segments=seg) - vus_pr(y, sc, max_buffer=mb, n_buffers=5)) < 1e-12
