"""Every number quoted in paper/revised/main.tex prose must match what make_tables.py derives
from the CSVs.  Fails if the text and the results drift apart."""
import json, os, re, subprocess, sys
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
TEX = open(os.path.join(ROOT, "paper/revised/main.tex"), encoding="utf-8").read()


def facts():
    subprocess.run([sys.executable, "-W", "ignore", "paper/revised/make_tables.py"], cwd=ROOT,
                   check=True, stdout=subprocess.DEVNULL)
    return json.load(open(os.path.join(ROOT, "paper/revised/numbers.json")))


F = facts()


def test_headline_f1_values_quoted_in_text():
    assert "0.695" in TEX and F["wti_dbscan_submitted_hybrid_f1"] == 0.695
    assert "0.344" in TEX and F["wti_dbscan_consistent_f1"] == 0.344
    assert "0.590" in TEX and abs(F["wti_lstm_consistent_f1"] - 0.590) < 1e-9
    assert "0.282" in TEX and abs(F["gb_B_lstm_f1"] - 0.282) < 1e-9


def test_dbscan_cross_dataset_series_quoted_in_text():
    d = F["dbscan_f1_by_dataset"]
    quoted = ["0.344, 0.304", "0.222/0.196", "0.148", "0.039/0.068", "0.097/0.047"]
    for q in quoted:
        assert q in TEX, q
    assert (d["WTI|forward"], d["Henry Hub|forward"]) == (0.344, 0.304)
    assert (d["FR Load|A"], d["FR Load|B"], d["FR Line Flow|B"]) == (0.222, 0.196, 0.148)
    assert (d["GB demand|A"], d["GB demand|B"], d["ERCOT demand|A"], d["ERCOT demand|B"]) == (0.039, 0.068, 0.097, 0.047)


def test_sweep_numbers_quoted_in_text():
    lo, hi = F["shift|dbscan_alert_rate_cell_range"]
    assert f"{round(lo*100):d}--{round(hi*100):d}\\%" in TEX
    lo, hi = F["burst|dbscan_alert_rate_cell_range"]
    assert f"{round(lo*100):d}--{round(hi*100):d}\\%" in TEX
    assert "$-0.92$" in TEX
    assert round(F["shift|corr_dbscan_f1_alert_rate"], 2) == -0.92
    assert "$+0.10$" in TEX and round(F["burst|corr_dbscan_f1_alert_rate"], 2) == 0.10
    assert "0.021" in TEX and round(F["shift|dbscan_f1_30min_vs_1h_max_abs_diff"], 3) == 0.021
    assert "0.022" in TEX and round(F["burst|dbscan_f1_30min_vs_1h_max_abs_diff"], 3) == 0.022
    lo = min(F["shift|prevalence_range"][0], F["burst|prevalence_range"][0])
    hi = max(F["shift|prevalence_range"][1], F["burst|prevalence_range"][1])
    assert f"{lo:.4f}--{hi:.4f}" in TEX
    assert round(F["shift|z_rolling_alert_rate_max"] * 100, 1) == 2.0 and "at most 2.0\\%" in TEX


def test_dbscan_u_shape_and_aucpr_range_quoted_in_text():
    S = pd.read_csv(os.path.join(ROOT, "results/synthetic/summary_shift.csv"))
    s = S[(S.resolution == "30min") & (S.detector == "dbscan_published")].set_index("duration")
    f1 = [round(s.loc[d, "ts_f1_mean"], 2) for d in ["2h", "12h", "1d", "3d", "7d", "14d"]]
    assert f1 == [0.35, 0.24, 0.26, 0.32, 0.35, 0.32]
    assert "0.35, 0.24, 0.26, 0.32, 0.35, 0.32" in TEX
    ap = s.auc_pr_mean
    assert round(ap.min(), 2) == 0.14 and round(ap.max(), 2) == 0.20 and "0.14--0.20" in TEX


def test_best_detector_claims_match_sweep():
    b = F["shift|best_by_duration_30min"]
    assert [b[d] for d in ["2h", "12h", "1d"]] == ["z_rolling"] * 3
    assert [b[d] for d in ["3d", "7d", "14d"]] == ["isolation_forest"] * 3
    assert F["shift|tie|14d"] is True and F["shift|tie|3d"] is False
    S = pd.read_csv(os.path.join(ROOT, "results/synthetic/summary_shift.csv"))
    s = S[(S.resolution == "30min") & (S.detector == "z_rolling")].set_index("duration").auc_pr_mean
    assert [round(s[d], 2) for d in ["2h", "12h", "1d", "3d", "7d"]] == [0.89, 0.97, 0.71, 0.33, 0.10]
    assert "(AUC-PR 0.89, 0.97, 0.71)" in TEX and "(0.33 at 3\\,d, 0.10 at 7\\,d)" in TEX


def test_daily_resolution_rolling_z_f1_is_exactly_zero_and_auc_pr_097():
    rec = {r["duration"]: r for r in F["shift|z_rolling_1d_res"]}
    assert all(r["ts_f1_mean"] == 0 for r in rec.values())
    assert round(rec["1d"]["auc_pr_mean"], 2) == 0.97 and "AUC-PR is 0.97 at the 1-day event" in TEX


def test_event_sample_counts_quoted_in_text():
    assert F["GB demand (NESO)|COVID-19|samples"] == 3360 and "3{,}360" in TEX
    assert F["GB demand (NESO)|Cold snap|samples"] == 480
    assert F["FR load (Gillioz)|Nuclear crisis|samples"] == 2640
    assert F["ERCOT demand (EIA-930)|Texas Freeze|samples"] == 144
