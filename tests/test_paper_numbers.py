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


def test_quoted_label_validity_numbers():
    f = json.load(open(NJ))
    assert round(f["labels|gb|COVID|mean_dev"], 2) == 1.26 or round(f["labels|gb|COVID|mean_dev"], 2) == 1.27
    assert round(f["labels|gb|COVID|pct_below"]) == 80 and round(f["labels|gb|ColdSnap|pct_below"]) == 98
    assert (f["labels|gb|cand"], f["labels|gb|outside"], f["labels|gb|cand_christmas_ny"]) == (546, 2842, 61)
    assert (f["labels|ercot|cand"], f["labels|ercot|outside"]) == (69, 1816)
    n_lab = f["labels|gb|COVID|n_ts"] + f["labels|gb|ColdSnap|n_ts"]
    assert (f["labels|gb|COVID|n_ts"], n_lab) == (1679, 1919) and round(100 * 1679 / n_lab) == 87
    assert round(100 * f["labels|gb|prevalence"], 1) == 3.4 and round(100 * f["labels|gb|prevalence_drop_first"], 2) == 0.44
    assert (f["labels|gb|auc_pr|n"], f["labels|gb|auc_pr|n_changed"], f["labels|gb|ts_f1|n_changed"]) == (19, 1, 1)
    assert (f["labels|ercot|auc_pr|n"], f["labels|ercot|auc_pr|n_changed"], f["labels|ercot|ts_f1|n_changed"]) == (17, 2, 9)
    assert round(f["labels|gb|auc_pr|min_tau"], 2) == -0.24 and round(f["labels|ercot|auc_pr|min_tau"], 2) == 0.62 and round(f["labels|ercot|ts_f1|min_tau"], 2) == 0.49


def test_quoted_tier_numbers():
    f = json.load(open(NJ))
    assert f["tiers|B|LSTM-AE|n_flags"] == 1751 and round(100 * f["tiers|B|LSTM-AE|share_jul_sep"]) == 95 and round(100 * f["tiers|B|LSTM-AE|share_tier2"]) == 38
    assert round(100 * f["tiers|B|Transformer|share_jul_sep"]) == 93 and round(100 * f["tiers|B|Transformer|share_tier2"]) == 37
    assert f["tiers|B|LSTM-AE|f1_t1"] == 0 and round(f["tiers|B|LSTM-AE|f1_t12"], 2) == 0.43 and round(f["tiers|B|Transformer|f1_t12"], 2) == 0.42
    assert round(f["tiers|B|LSTM-AE|f1_t12_lo"], 2) == 0.32 and round(f["tiers|B|LSTM-AE|f1_t12_hi"], 2) == 0.53
    assert f["tiers|A|LSTM-AE|n_flags"] == 656 and f["tiers|A|LSTM-AE|share_jul_sep"] == 1.0 and f["tiers|A|LSTM-AE|share_tier2"] < 0.02 and f["tiers|A|LSTM-AE|f1_t12"] < 0.01
