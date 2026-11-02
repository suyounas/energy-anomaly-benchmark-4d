"""Guards the hand-typed numbers in paper/revised/main.tex against the generated numbers.json."""
import json, os, pytest
ROOT = os.path.join(os.path.dirname(__file__), "..", "paper", "revised")
NJ = os.path.join(ROOT, "numbers.json")
pytestmark = pytest.mark.skipif(not os.path.exists(NJ), reason="numbers.json not generated")


def test_quoted_real_data_numbers():
    f = json.load(open(NJ))
    a = lambda k: f[k]
    assert round(a("gb|matrix_profile|recipe|ts_f1"), 2) == 0.33 and round(a("gb|matrix_profile|recipe|auc_pr"), 2) == 0.21
    assert round(a("gb|lstm_ae|resid|ts_f1"), 2) == 0.23 and round(a("gb|lstm_ae|recipe|ts_f1"), 3) == 0.125
    assert round(a("ercot|seasonal_naive|resid|auc_pr"), 2) == 0.24
    assert round(a("ercot|lstm_ae|resid|auc_pr"), 2) == 0.03 and round(a("ercot|transformer|resid|auc_pr"), 2) == 0.02
    assert a("ercot|lstm_ae|recipe|ts_f1") == 0 and a("ercot|transformer|recipe|ts_f1") == 0
    assert round(a("ercot|window_pca|recipe|auc_pr"), 3) == 0.124 and round(a("ercot|window_pca|resid|auc_pr"), 3) == 0.080
    assert round(a("gb|seasonal_naive|cov_ColdSnap"), 2) == 0.65
    for fam in ("matrix_profile", "lstm_ae", "transformer"):
        assert a(f"gb|{fam}|cov_ColdSnap") == 0


def test_quoted_factorial_and_public_numbers():
    f = json.load(open(NJ))
    assert round(f["factorial|detector"], 1) == 19.9 and round(f["factorial|detector_all"], 1) == 47.4
    inter = sum(f[f"factorial|{k}"] for k in ("dur_h x detector", "event x detector", "mag x detector", "res_h x detector"))
    assert 25.5 < inter < 26.5 and inter > f["factorial|detector"]
    assert f["factorial|global_best"] == "seasonal_naive" and round(100 * f["factorial|global_best_wins_share"]) == 56
    assert (f["public|n_series"], f["public|n_datasets"], f["public|n_datasets_min8"]) == (862, 23, 17)
    assert round(f["public|rho"], 3) == -0.047 and f["public|lo"] < 0 < f["public|hi"]
