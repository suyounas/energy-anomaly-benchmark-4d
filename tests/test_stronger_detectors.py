"""Stronger / simpler comparison detectors (ebench.stronger)."""
import numpy as np
import pandas as pd
import pytest

from ebench import auc_pr
from ebench.tuning import FAMILIES, equal_budget_search
from real_resolution_ablation import build_features, feature_cols

SPD = 24
NEW = ["diff_z", "seasonal_naive", "window_pca", "window_knn", "window_iforest", "matrix_profile",
       "spectral_residual"]


def make(seed=0, shift=True):
    """150 days hourly, weekly + daily pattern + noise.  Train = first 90 days.  Test events:
    a 3-day level shift (+4 sd) at day 110 and a 6-hour spike burst at day 130."""
    rng = np.random.default_rng(seed)
    n = 150 * SPD
    t = pd.date_range("2020-01-01", periods=n, freq="h")
    base = (10 * np.sin(2 * np.pi * np.arange(n) / SPD) + 4 * np.sin(2 * np.pi * np.arange(n) / (7 * SPD))
            + rng.normal(0, 1, n) + 100)
    y_shift = np.zeros(n, int); y_burst = np.zeros(n, int)
    x = base.copy()
    if shift:
        s = 110 * SPD; x[s:s + 3 * SPD] += 8.0; y_shift[s:s + 3 * SPD] = 1
    b = 130 * SPD; x[b:b + 6] += rng.normal(0, 12, 6) + 15; y_burst[b:b + 6] = 1
    F = build_features(t, x, SPD, 7, with_resid=True)
    ys = pd.Series(y_shift, index=t).reindex(F.index).to_numpy(); yb = pd.Series(y_burst, index=t).reindex(F.index).to_numpy()
    tr = F.index < t[90 * SPD]
    return F[tr], F[~tr], ys[~tr], yb[~tr]


COLS = feature_cols("resid")


@pytest.mark.parametrize("name", NEW)
def test_shape_and_nan_convention(name):
    Ftr, Fev, _, _ = make()
    sc = FAMILIES[name]["fn"](FAMILIES[name]["default"], Ftr, Fev, COLS, 0, spd=SPD)
    assert sc.shape == (len(Fev),)
    assert np.isfinite(sc[200:]).all()
    # only the leading incomplete window may be NaN
    assert np.isnan(sc).sum() <= 30


@pytest.mark.parametrize("name", [n for n in NEW if n != "spectral_residual"])
def test_scores_are_causal_in_the_evaluation_window(name):
    """The score at row i must not depend on rows after i (spectral_residual is the one offline
    exception: it sees its whole block, like DBSCAN fitted on the test set)."""
    Ftr, Fev, _, _ = make()
    half = len(Fev) // 2
    F2 = Fev.copy()
    for c in F2.columns:
        F2.iloc[half:, F2.columns.get_loc(c)] = F2[c].iloc[half:] * 3 + 5
    f = FAMILIES[name]
    a = f["fn"](f["default"], Ftr, Fev, COLS, 0, spd=SPD)[:half]
    b = f["fn"](f["default"], Ftr, F2, COLS, 0, spd=SPD)[:half]
    np.testing.assert_allclose(a, b, equal_nan=True)


def lift(y, s):
    ok = ~np.isnan(s)
    return auc_pr(y[ok], s[ok]) / y[ok].mean()


def test_seasonal_naive_and_windowed_residual_models_see_the_level_shift():
    Ftr, Fev, ys, _ = make()
    sn = FAMILIES["seasonal_naive"]["fn"]({}, Ftr, Fev, COLS, 0)
    pca = FAMILIES["window_pca"]["fn"](FAMILIES["window_pca"]["default"], Ftr, Fev, COLS, 0)
    assert lift(ys, sn) > 3
    assert lift(ys, pca) > 3


def test_matrix_profile_is_level_blind_but_sees_a_burst():
    """Z-normalisation removes level: the shift is found far less well than the burst."""
    Ftr, Fev, ys, yb = make()
    f = FAMILIES["matrix_profile"]
    mp = f["fn"](dict(window_days=0.5), Ftr, Fev, COLS, 0, spd=SPD)
    assert lift(yb, mp) > 2 * lift(ys, mp)


def test_diff_z_is_a_one_liner_with_clean_window_statistics():
    Ftr, Fev, _, _ = make()
    s = FAMILIES["diff_z"]["fn"]({}, Ftr, Fev, COLS, 0)
    mu, sd = Ftr["ret"].mean(), Ftr["ret"].std(ddof=0)
    np.testing.assert_allclose(s, np.abs(Fev["ret"].to_numpy() - mu) / sd, rtol=1e-6)


def test_seasonal_naive_requires_resid_feature():
    Ftr, Fev, _, _ = make()
    with pytest.raises(ValueError, match="resid"):
        FAMILIES["seasonal_naive"]["fn"]({}, Ftr.drop(columns="resid"), Fev.drop(columns="resid"), COLS[:-1], 0)
    assert FAMILIES["seasonal_naive"]["requires"] == ["resid"]


def test_equal_budget_search_runs_the_new_families():
    Ftr, Fev, ys, _ = make()
    fams = ["diff_z", "window_pca", "window_knn", "matrix_profile"]
    trials, chosen = equal_budget_search(fams, Ftr, Fev, ys, COLS, n_trials=2, seed=0, spd=SPD)
    assert set(chosen) == set(fams)
    assert (trials.groupby("family").size() >= 2).all()
    assert trials.val_auc_pr.notna().all()


def test_matrix_profile_matches_brute_force_ab_join():
    """Independent O(n*m*w) reference: z-normalise every subsequence, take the minimum Euclidean
    distance to the reference series' subsequences; scored at the window END."""
    rng = np.random.default_rng(3)
    m = 12
    xtr = np.cumsum(rng.normal(size=150)); xev = np.cumsum(rng.normal(size=80))
    t = pd.date_range("2020-01-01", periods=230, freq="h")
    Ftr = pd.DataFrame({"value": xtr}, index=t[:150]); Fev = pd.DataFrame({"value": xev}, index=t[150:])
    got = FAMILIES["matrix_profile"]["fn"](dict(window_days=m / 24), Ftr, Fev, ["value"], 0, spd=24)

    def z(w):
        return (w - w.mean()) / w.std()
    ref = np.array([z(xtr[i:i + m]) for i in range(len(xtr) - m + 1)])
    want = np.full(len(xev), np.nan)
    for j in range(m - 1, len(xev)):
        q = z(xev[j - m + 1:j + 1])
        want[j] = np.sqrt(((ref - q) ** 2).sum(1)).min()
    assert np.isnan(got[:m - 1]).all()
    np.testing.assert_allclose(got[m - 1:], want[m - 1:], atol=1e-6)
