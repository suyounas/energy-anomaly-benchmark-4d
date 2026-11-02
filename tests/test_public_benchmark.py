import importlib.util, os, subprocess, sys
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
spec = importlib.util.spec_from_file_location("public_benchmark", os.path.join(ROOT, "experiments", "public_benchmark.py"))
pb = importlib.util.module_from_spec(spec); spec.loader.exec_module(pb)


def make_series(n=1500, P=50, tr=300, ev=(900, 940), seed=0):
    rng = np.random.default_rng(seed)
    x = np.sin(2 * np.pi * np.arange(n) / P) + 0.1 * rng.normal(size=n)
    y = np.zeros(n, int)
    x[ev[0]:ev[1]] += 3.0
    y[ev[0]:ev[1]] = 1
    return x, y, tr


def write(dirpath, name, x, y, cols=("Data", "Label")):
    pd.DataFrame({cols[0]: x, cols[1]: y}).to_csv(os.path.join(dirpath, name), index=False)


def test_parse_name():
    assert pb.parse_name("/a/001_UCR_id_1_Facility_tr_1000_1st_1500.csv") == ("UCR", 1000)
    assert pb.parse_name("x.csv")[1] is None


def test_period_estimate():
    x, _, tr = make_series()
    P, found = pb.estimate_period(x[:tr])
    assert found and abs(P - 50) <= 2
    P2, found2 = pb.estimate_period(np.random.default_rng(1).normal(size=300))
    assert not found2


def test_features_are_causal():
    x, _, tr = make_series()
    F1 = pb.public_features(x, 50, tr)
    x2 = x.copy(); x2[1200:] += 100
    F2 = pb.public_features(x2, 50, tr)
    assert np.allclose(F1.iloc[:1200].to_numpy(), F2.iloc[:1200].to_numpy())


def test_evaluate_file_detects_obvious_event(tmp_path):
    x, y, tr = make_series()
    write(tmp_path, "001_SYN_id_1_Sensor_tr_300_1st_400.csv", x, y)
    row = pb.evaluate_file(str(tmp_path / "001_SYN_id_1_Sensor_tr_300_1st_400.csv"))
    assert row["status"] == "ok", row
    assert row["dataset"] == "SYN" and row["n_train"] == 300 and row["n_events"] == 1
    prev = row["prevalence"]
    assert row["auc_pr_diff_z"] >= 0
    assert row["auc_pr_seasonal_naive"] > 3 * prev
    assert row["auc_pr_window_pca"] > 3 * prev
    assert not any(k.startswith("err_") for k in row)


def test_no_anomaly_and_short_are_reported(tmp_path):
    x, y, _ = make_series()
    write(tmp_path, "a_X_id_1_s_tr_300_1st_400.csv", x, np.zeros_like(y))
    assert pb.evaluate_file(str(tmp_path / "a_X_id_1_s_tr_300_1st_400.csv"))["status"] == "no_anomaly_in_test"
    write(tmp_path, "b_X_id_2_s_tr_30_1st_40.csv", x[:100], y[:100])
    assert pb.evaluate_file(str(tmp_path / "b_X_id_2_s_tr_30_1st_40.csv"))["status"] == "too_short"


def test_cli_run_resume_and_analyse(tmp_path):
    d = tmp_path / "data"; d.mkdir()
    for ds in ("AAA", "BBB"):
        for i in range(9):
            x, y, _ = make_series(seed=i, ev=(800 + 10 * i, 830 + 40 * i))
            write(d, f"{i:03d}_{ds}_id_{i}_S_tr_300_1st_400.csv", x, y, cols=("data", "label"))
    out = tmp_path / "r.csv"
    run = [sys.executable, os.path.join(ROOT, "experiments", "public_benchmark.py")]
    env = dict(os.environ, OMP_NUM_THREADS="1")
    r = subprocess.run(run + ["run", "--data", str(d), "--out", str(out), "--chunk", "5"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    df = pd.read_csv(out); assert len(df) == 18 and (df.status == "ok").all()
    r2 = subprocess.run(run + ["run", "--data", str(d), "--out", str(out)], capture_output=True, text=True, env=env)
    assert "0 to run" in r2.stdout and len(pd.read_csv(out)) == 18
    rep = tmp_path / "rep.md"
    r3 = subprocess.run(run + ["analyse", "--results", str(out), "--out", str(rep), "--n-boot", "50"], capture_output=True, text=True, env=env)
    assert r3.returncode == 0, r3.stderr[-2000:]
    txt = rep.read_text()
    assert "PRIMARY (pre-registered)" in txt and "rho =" in txt and "Series accounting" in txt


def test_missing_data_dir_fails_loudly(tmp_path):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "public_benchmark.py"), "run", "--data", str(tmp_path), "--out", str(tmp_path / "o.csv")], capture_output=True, text=True)
    assert r.returncode != 0 and "no CSV files" in (r.stderr + r.stdout)
