import importlib.util, itertools, os, subprocess, sys
import numpy as np, pandas as pd
from ebench.factorial import make_background, residual_scale, inject_events, ss_decomposition, ranking_stability

ROOT = os.path.join(os.path.dirname(__file__), "..")
spec = importlib.util.spec_from_file_location("fa", os.path.join(ROOT, "experiments", "factor_attribution.py"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
fa = importlib.util.module_from_spec(spec); spec.loader.exec_module(fa)


def test_background_positive_and_reproducible():
    for kind in ("strong", "weak", "trend"):
        t, x = make_background(kind, 24 * 60, np.random.default_rng(0))
        assert (x > 0).all() and len(t) == len(x)
    assert np.allclose(make_background("strong", 500, np.random.default_rng(1))[1], make_background("strong", 500, np.random.default_rng(1))[1])


def test_injection_labels_alignment_and_isolation():
    rng = np.random.default_rng(0)
    t, x = make_background("strong", 24 * 210, rng)
    scale = residual_scale(x[:24 * 120])
    for kind in ("shift", "burst", "ramp"):
        for dur in (4, 24, 120):
            x2, y = inject_events(x, 24 * 120, kind, dur, 6, scale, np.random.default_rng(2))
            assert y[:24 * 120].sum() == 0 and (x2[:24 * 120] == x[:24 * 120]).all()
            assert y.sum() == 3 * dur
            assert (np.flatnonzero(np.diff(np.r_[0, y]) == 1) % 4 == 0).all()          # snapped to 4 h
            assert (x2[y == 0] == x[y == 0]).all()
            assert np.abs(x2[y == 1] - x[y == 1]).max() > 0


def test_ss_decomposition_recovers_known_structure():
    rng = np.random.default_rng(0)
    rows = [dict(A=a, B=b, C=c, y=2.0 * a + 1.0 * b * (a == 2) + 0.3 * rng.normal())
            for a, b, c, _ in itertools.product(range(3), range(2), range(4), range(6))]
    d = pd.DataFrame(rows)
    out = ss_decomposition(d, ["A", "B", "C"], "y", max_order=3).set_index("term")
    assert abs(out.share.sum() - 1) < 1e-9                              # shares are exhaustive
    assert out.share["A"] > 0.9 and out.share["C"] < 0.01 and out.share["A x B"] > out.share["B x C"]
    # a pure-noise factor has ~0 share; a factor-free outcome has no structure
    d["y"] = rng.normal(size=len(d))
    assert ss_decomposition(d, ["A", "B", "C"], "y", max_order=2).set_index("term").share[["A", "B", "C"]].max() < 0.05


def test_ss_decomposition_rejects_unbalanced():
    d = pd.DataFrame(dict(A=[0, 0, 1], B=[0, 1, 0], y=[1.0, 2.0, 3.0]))
    try:
        ss_decomposition(d, ["A", "B"], "y")
    except ValueError as e:
        assert "balanced" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_ranking_stability_detects_flip():
    rows = []
    for ev in ("shift", "burst"):
        for det, v in (("a", 1.0 if ev == "shift" else 0.0), ("b", 0.0 if ev == "shift" else 1.0), ("c", 0.5)):
            for rep in range(3):
                rows.append(dict(event=ev, detector=det, skill=v + 0.01 * rep))
    s, per = ranking_stability(pd.DataFrame(rows), ["event"], "detector", "skill")
    assert s["n_distinct_winners"] == 2 and s["share_conditions_global_best_wins"] == 0.5


def test_run_dataset_signal_and_structural_zero():
    rows = fa.run_dataset(dict(background="strong", event="shift", dur_h=24, mag=12, res_h=1, rep=0))
    df = pd.DataFrame(rows)
    assert len(df) == 2 * len(fa.DETECTORS) and df.skill.notna().all()
    w = df.pivot(index="detector", columns="features", values="skill")
    for d in set(fa.DETECTORS) - fa.USES_FEATURES:
        assert w.loc[d, "recipe"] == w.loc[d, "resid"]                    # structural zero
    assert w.loc["seasonal_naive"].max() > 0.5                            # an obvious shift is found
    assert df.prevalence.nunique() == 1                                   # same rows for every detector


def test_cli_quick_run_resume_and_analyse(tmp_path):
    run = [sys.executable, os.path.join(ROOT, "experiments", "factor_attribution.py")]
    env = dict(os.environ, OMP_NUM_THREADS="1")
    out = tmp_path / "g.csv"
    r = subprocess.run(run + ["run", "--out", str(out), "--quick", "--reps", "2", "--jobs", "2", "--chunk", "8"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-1500:]
    df = pd.read_csv(out)
    assert len(df) == 2 ** 5 * 2 * 18 and "datasets with no usable rows or errors: 0" in r.stdout
    r2 = subprocess.run(run + ["run", "--out", str(out), "--quick", "--reps", "2"], capture_output=True, text=True, env=env)
    assert "0 to run" in r2.stdout and len(pd.read_csv(out)) == len(df)
    rep = tmp_path / "rep.md"
    r3 = subprocess.run(run + ["analyse", "--results", str(out), "--out", str(rep), "--n-boot", "20"], capture_output=True, text=True, env=env)
    assert r3.returncode == 0, r3.stderr[-1500:]
    txt = rep.read_text()
    assert "Share of total variance" in txt and "Ranking stability" in txt and "Detector main effect" in txt
